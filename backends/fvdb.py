import torch
import torch.nn as nn
import fvdb
import fvdb.nn as fvnn

from utils.tensor import FvdbTensor


def _plan(source_grid, target_grid, kernel_size, stride=1):
    # fvdb sparse convolutions need an explicit neighborhood mapping between
    # the source and target sparse grids.
    return fvdb.ConvolutionPlan.from_grid_batch(
        kernel_size=kernel_size,
        stride=stride,
        source_grid=source_grid,
        target_grid=target_grid,
    )


def _jagged_like(grid, jdata: torch.Tensor):
    return grid.jagged_like(jdata.contiguous())


def _ensure_same_grid(a: FvdbTensor, b: FvdbTensor):
    if not a.grid.is_same(b.grid):
        raise ValueError("FvdbTensor grids must match for this operation")


def _replace_data(x: FvdbTensor, new_data):
    # Most pointwise ops only change the feature matrix and keep the sparse
    # voxel topology unchanged.
    if isinstance(new_data, fvdb.JaggedTensor):
        return FvdbTensor(x.grid, new_data)
    return FvdbTensor(x.grid, _jagged_like(x.grid, new_data))


def _add(a: FvdbTensor, b: FvdbTensor) -> FvdbTensor:
    _ensure_same_grid(a, b)
    return FvdbTensor(a.grid, _jagged_like(a.grid, a.data.jdata + b.data.jdata))


def _cat(*tensors: FvdbTensor) -> FvdbTensor:
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided")
    base = tensors[0]
    for t in tensors[1:]:
        _ensure_same_grid(base, t)
    jdata = torch.cat([t.data.jdata for t in tensors], dim=1)
    return FvdbTensor(base.grid, _jagged_like(base.grid, jdata))


class SparseELU(nn.ELU):
    def forward(self, input):
        if isinstance(input, FvdbTensor):
            return _replace_data(input, super().forward(input.data.jdata))
        return super().forward(input)


class SparsePReLU(nn.PReLU):
    def forward(self, input):
        if isinstance(input, FvdbTensor):
            return _replace_data(input, super().forward(input.data.jdata))
        return super().forward(input)


class SparseDropout(nn.Dropout):
    def forward(self, input):
        if isinstance(input, FvdbTensor):
            return _replace_data(input, super().forward(input.data.jdata))
        return super().forward(input)


class SparseLinear(nn.Module):
    """Pointwise channel mixer for FvdbTensor features."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: FvdbTensor) -> FvdbTensor:
        return _replace_data(x, self.linear(x.data.jdata))


def ELUCons(elu: bool, nchan: int) -> nn.Module:
    if elu:
        return SparseELU(inplace=True)
    return SparsePReLU(num_parameters=nchan)


class LUConv(nn.Module):
    def __init__(self, nchan: int, elu: bool):
        super().__init__()
        self.conv = fvnn.SparseConv3d(nchan, nchan, kernel_size=5, stride=1, bias=True)
        self.bn1 = fvnn.BatchNorm(nchan)
        self.relu1 = ELUCons(elu, nchan)

    def forward(self, x: FvdbTensor) -> FvdbTensor:
        # This is a same-resolution sparse convolution, so the plan maps the
        # grid back onto itself.
        plan = _plan(x.grid, x.grid, kernel_size=5, stride=1)
        out = self.conv(x.data, plan)
        out = self.bn1(out, x.grid)
        return self.relu1(FvdbTensor(x.grid, out))


def _make_nConv(nchan: int, depth: int, elu: bool) -> nn.Sequential:
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels: int, elu: bool, num_features: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.conv1 = fvnn.SparseConv3d(in_channels, num_features, kernel_size=5, stride=1, bias=True)
        self.bn1 = fvnn.BatchNorm(num_features)
        self.relu1 = ELUCons(elu, num_features)

    def forward(self, x: FvdbTensor) -> FvdbTensor:
        plan = _plan(x.grid, x.grid, kernel_size=5, stride=1)
        out = self.conv1(x.data, plan)
        out = self.bn1(out, x.grid)
        out = FvdbTensor(x.grid, out)

        if self.num_features == self.in_channels:
            return self.relu1(_add(out, x))

        repeat_rate = self.num_features // self.in_channels
        if repeat_rate * self.in_channels != self.num_features:
            raise ValueError(
                f"num_features={self.num_features} not multiple of in_channels={self.in_channels}"
            )

        # Match the original VNet residual behavior by repeating the input
        # channels when the first sparse block widens the feature count.
        x_expanded = _replace_data(x, x.data.jdata.repeat(1, repeat_rate))
        return self.relu1(_add(out, x_expanded))


class DownTransition(nn.Module):
    def __init__(self, inChans: int, nConvs: int, elu: bool, dropout: bool = False):
        super().__init__()
        outChans = 2 * inChans
        self.channel_fan_out = SparseLinear(inChans, outChans, bias=True)
        self.bn1 = fvnn.BatchNorm(outChans)
        self.relu1 = ELUCons(elu, outChans)
        self.do1 = SparseDropout(p=0.5) if dropout else nn.Identity()
        self.ops = _make_nConv(outChans, nConvs, elu)
        self.relu2 = ELUCons(elu, outChans)

    def forward(self, x: FvdbTensor) -> FvdbTensor:
        # Downsampling in fvdb changes the sparse topology first, then pools
        # features onto the coarsened grid.
        coarse_grid = x.grid.coarsened_grid(2)
        pooled_data, coarse_grid = x.grid.max_pool(pool_factor=2, data=x.data, coarse_grid=coarse_grid)

        down = FvdbTensor(coarse_grid, pooled_data)
        down = self.channel_fan_out(down)
        down = FvdbTensor(coarse_grid, self.bn1(down.data, coarse_grid))
        down = self.relu1(down)

        out = self.do1(down)
        out = self.ops(out)
        out = _add(out, down)
        return self.relu2(out)


class UpTransition(nn.Module):
    def __init__(self, inChans: int, outChans: int, nConvs: int, elu: bool, dropout: bool = False):
        super().__init__()
        self.channel_fan_in = SparseLinear(inChans, outChans // 2, bias=True)
        self.bn1 = fvnn.BatchNorm(outChans // 2)
        self.relu1 = ELUCons(elu, outChans // 2)
        self.do1 = SparseDropout(p=0.5) if dropout else nn.Identity()
        self.do2 = SparseDropout(p=0.5)
        self.ops = _make_nConv(outChans, nConvs, elu)
        self.relu2 = ELUCons(elu, outChans)

    def forward(self, x: FvdbTensor, skipx: FvdbTensor) -> FvdbTensor:
        out = self.do1(x)
        skipxdo = self.do2(skipx)

        out = self.channel_fan_in(out)
        out = FvdbTensor(out.grid, self.bn1(out.data, out.grid))
        # Upsampling refines the coarse sparse grid back onto the skip
        # connection topology before concatenation.
        out_data, fine_grid = x.grid.refine(subdiv_factor=2, data=out.data, fine_grid=skipx.grid)
        out = self.relu1(FvdbTensor(fine_grid, out_data))

        xcat = _cat(out, skipxdo)
        out = self.ops(xcat)
        out = _add(out, xcat)
        return self.relu2(out)


class OutputTransition(nn.Module):
    def __init__(self, in_channels: int, classes: int, elu: bool):
        super().__init__()
        self.conv1 = fvnn.SparseConv3d(in_channels, classes, kernel_size=5, stride=1, bias=True)
        self.bn1 = fvnn.BatchNorm(classes)
        self.relu1 = ELUCons(elu, classes)
        self.conv2 = SparseLinear(classes, classes, bias=True)

    def forward(self, x: FvdbTensor) -> FvdbTensor:
        plan5 = _plan(x.grid, x.grid, kernel_size=5, stride=1)
        out = self.conv1(x.data, plan5)
        out = self.bn1(out, x.grid)
        out = self.relu1(FvdbTensor(x.grid, out))
        # The final channel projection is pointwise, so a linear layer is
        # enough and avoids changing the sparse topology.
        return self.conv2(out)


class ChannelAdjuster(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = SparseLinear(in_channels, out_channels, bias=True)

    def forward(self, x: FvdbTensor) -> FvdbTensor:
        return self.linear(x)


Fvdb = {
    "InputTransition": InputTransition,
    "DownTransition": DownTransition,
    "UpTransition": UpTransition,
    "OutputTransition": OutputTransition,
    "ChannelAdjuster": ChannelAdjuster,
}

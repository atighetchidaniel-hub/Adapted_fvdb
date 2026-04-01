import torch
import torch.nn as nn
import fvdb
import fvdb.nn as fvnn

from utils.tensor import FvdbTensor


def _plan(source_grid, target_grid, kernel_size, stride=1):
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

        x_expanded = _replace_data(x, x.data.jdata.repeat(1, repeat_rate))
        return self.relu1(_add(out, x_expanded))


class DownTransition(nn.Module):
    def __init__(self, inChans: int, nConvs: int, elu: bool, dropout: bool = False):
        super().__init__()
        outChans = 2 * inChans
        self.down_conv = fvnn.SparseConv3d(inChans, outChans, kernel_size=2, stride=2, bias=True)
        self.bn1 = fvnn.BatchNorm(outChans)
        self.relu1 = ELUCons(elu, outChans)
        self.do1 = SparseDropout(p=0.5) if dropout else nn.Identity()
        self.ops = _make_nConv(outChans, nConvs, elu)
        self.relu2 = ELUCons(elu, outChans)

    def forward(self, x: FvdbTensor) -> FvdbTensor:
        coarse_grid = x.grid.coarsened_grid(2)
        plan = _plan(x.grid, coarse_grid, kernel_size=2, stride=2)

        down = self.down_conv(x.data, plan)
        down = self.bn1(down, coarse_grid)
        down = self.relu1(FvdbTensor(coarse_grid, down))

        out = self.do1(down)
        out = self.ops(out)
        out = _add(out, down)
        return self.relu2(out)


class UpTransition(nn.Module):
    def __init__(self, inChans: int, outChans: int, nConvs: int, elu: bool, dropout: bool = False):
        super().__init__()
        self.up_conv = fvnn.SparseConvTranspose3d(
            inChans, outChans // 2, kernel_size=2, stride=2, bias=True
        )
        self.bn1 = fvnn.BatchNorm(outChans // 2)
        self.relu1 = ELUCons(elu, outChans // 2)
        self.do1 = SparseDropout(p=0.5) if dropout else nn.Identity()
        self.do2 = SparseDropout(p=0.5)
        self.ops = _make_nConv(outChans, nConvs, elu)
        self.relu2 = ELUCons(elu, outChans)

    def forward(self, x: FvdbTensor, skipx: FvdbTensor) -> FvdbTensor:
        out = self.do1(x)
        skipxdo = self.do2(skipx)

        plan = _plan(x.grid, skipx.grid, kernel_size=2, stride=2)
        out_data = self.up_conv(out.data, plan)
        out_data = self.bn1(out_data, skipx.grid)
        out = self.relu1(FvdbTensor(skipx.grid, out_data))

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
        self.conv2 = fvnn.SparseConv3d(classes, classes, kernel_size=1, stride=1, bias=True)

    def forward(self, x: FvdbTensor) -> FvdbTensor:
        plan5 = _plan(x.grid, x.grid, kernel_size=5, stride=1)
        out = self.conv1(x.data, plan5)
        out = self.bn1(out, x.grid)
        out = self.relu1(FvdbTensor(x.grid, out))

        plan1 = _plan(x.grid, x.grid, kernel_size=1, stride=1)
        out = self.conv2(out.data, plan1)
        return FvdbTensor(x.grid, out)


class ChannelAdjuster(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = fvnn.SparseConv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x: FvdbTensor) -> FvdbTensor:
        plan = _plan(x.grid, x.grid, kernel_size=1, stride=1)
        out = self.conv(x.data, plan)
        return FvdbTensor(x.grid, out)


Fvdb = {
    "InputTransition": InputTransition,
    "DownTransition": DownTransition,
    "UpTransition": UpTransition,
    "OutputTransition": OutputTransition,
    "ChannelAdjuster": ChannelAdjuster,
}

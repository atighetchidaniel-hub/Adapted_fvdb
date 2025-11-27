import spconv.pytorch as spconv
import torch
import torch.nn as nn

from utils.train import trunc_normal_


def ELUCons(elu: bool, nchan: int) -> nn.Module:
    """Choose between ELU or PReLU activations on features."""
    if elu:
        return SparseELU(inplace=True)
    else:
        return SparsePReLU(num_parameters=nchan)


class SparseELU(nn.ELU):
    """this module is exists only for torch.fx transformation for quantization.
    """

    def forward(self, input):
        if isinstance(input, spconv.SparseConvTensor):
            return input.replace_feature(super().forward(input.features))
        return super().forward(input)


class SparsePReLU(nn.PReLU):
    """this module is exists only for torch.fx transformation for quantization.
    """

    def forward(self, input):
        if isinstance(input, spconv.SparseConvTensor):
            return input.replace_feature(super().forward(input.features))
        return super().forward(input)


class SparseDropout(nn.Dropout):
    """this module is exists only for torch.fx transformation for quantization.
    """

    def forward(self, input):
        if isinstance(input, spconv.SparseConvTensor):
            return input.replace_feature(super().forward(input.features))
        return super().forward(input)


def sparse_cat(*tensors: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
    """
    Concatenate features of N SparseConvTensors by zero-padding each tensor's features
    into disjoint blocks and then summing.

    All tensors must share the same coordinates, spatial_shape, batch_size, and coordinate_manager.
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided")
    # extract feature dims and compute offsets
    dims = [t.features.size(1) for t in tensors]
    total_dim = sum(dims)

    # build zero-padded versions
    padded = []
    for i, t in enumerate(tensors):
        before = sum(dims[:i])
        after = total_dim - before - dims[i]
        # create zero pads
        pad_before = torch.zeros((t.features.size(0), before),
                                 dtype=t.features.dtype, device=t.features.device)
        pad_after = torch.zeros((t.features.size(0), after),
                                dtype=t.features.dtype, device=t.features.device)
        # concat: [ zeros_before | original_feats | zeros_after ]
        new_feats = torch.cat([pad_before, t.features, pad_after], dim=-1)
        padded.append(t.replace_feature(new_feats))

    # sum them all up
    result = padded[0]
    for t in padded[1:]:
        result = spconv.functional.sparse_add(result, t)

    return result


class LUConv(nn.Module):
    """
    Single 'LUConv' block:
      conv -> bn -> activation on sparse tensor
    """

    def __init__(self, nchan: int, elu: bool):
        super(LUConv, self).__init__()
        self.conv = spconv.SubMConv3d(
            in_channels=nchan,
            out_channels=nchan,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )
        self.bn1 = spconv.SparseBatchNorm(nchan)
        self.relu1 = ELUCons(elu, nchan)
        self.initialize()

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu1(out)
        return out

    def initialize(self):
        trunc_normal_(self.conv.weight, std=0.02)
        nn.init.constant_(self.conv.bias, -0.01)
        nn.init.kaiming_normal_(
            self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)


def _make_nConv(nchan: int, depth: int, elu: bool) -> nn.Sequential:
    """Stack of LUConv layers."""
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    """
    Mimics:
      conv -> bn -> out + x_expanded -> activation
    """

    def __init__(self, in_channels: int, elu: bool, num_features: int = 16):
        super(InputTransition, self).__init__()
        self.in_channels = in_channels
        self.num_features = num_features

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            num_features,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )
        self.bn1 = spconv.SparseBatchNorm(num_features)
        self.relu1 = ELUCons(elu, num_features)
        self.initialize()

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        out = self.conv1(x)
        out = self.bn1(out)

        if self.num_features == self.in_channels:
            out = spconv.functional.sparse_add(out, x)
            out = self.relu1(out)
            return out

        repeat_rate = self.num_features // self.in_channels
        if repeat_rate * self.in_channels != self.num_features:
            raise ValueError(
                f"num_features={self.num_features} not multiple of in_channels={self.in_channels}"
            )

        x_reps = [x] * repeat_rate
        x_expanded = sparse_cat(*x_reps)
        out = spconv.functional.sparse_add(out, x_expanded)
        out = self.relu1(out)
        return out

    def initialize(self):
        trunc_normal_(self.conv1.weight, std=0.02)
        nn.init.constant_(self.conv1.bias, -0.01)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)


class DownTransition(nn.Module):
    """
    down = conv + bn + act
    out = dropout(down) -> _make_nConv -> residual + act
    """

    def __init__(self, inChans: int, nConvs: int, elu: bool, dropout: bool = False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans

        self.down_conv = spconv.SparseConv3d(
            in_channels=inChans,
            out_channels=outChans,
            kernel_size=2,
            stride=2,
            bias=True,
        )
        self.bn1 = spconv.SparseBatchNorm(outChans)
        self.relu1 = ELUCons(elu, outChans)
        self.do1 = SparseDropout(
            p=0.5) if dropout else spconv.SparseIdentity()
        self.ops = _make_nConv(outChans, nConvs, elu)
        self.relu2 = ELUCons(elu, outChans)
        self.initialize()

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        down = self.down_conv(x)
        down = self.bn1(down)
        down = self.relu1(down)

        out = self.do1(down)
        out = self.ops(out)

        out = spconv.functional.sparse_add(out, down)
        out = self.relu2(out)
        return out

    def initialize(self):
        trunc_normal_(self.down_conv.weight, std=0.02)
        nn.init.constant_(self.down_conv.bias, -0.01)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)


class UpTransition(nn.Module):
    """
    out = dropout(x) -> up_conv + bn + act
    xcat = cat(out, dropout(skipx)) -> ops -> residual + act
    """

    def __init__(self, inChans: int, outChans: int, nConvs: int, elu: bool, dropout: bool = False):
        super(UpTransition, self).__init__()
        self.up_conv = spconv.SparseConvTranspose3d(
            in_channels=inChans,
            out_channels=outChans // 2,
            kernel_size=2,
            stride=2,
            bias=True,
        )
        self.bn1 = spconv.SparseBatchNorm(outChans // 2)
        self.relu1 = ELUCons(elu, outChans // 2)
        self.do1 = SparseDropout(
            p=0.5) if dropout else spconv.SparseIdentity()
        self.do2 = SparseDropout(p=0.5)
        self.ops = _make_nConv(outChans, nConvs, elu)
        self.relu2 = ELUCons(elu, outChans)
        self.initialize()

    def forward(self, x: spconv.SparseConvTensor, skipx: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        out = self.do1(x)
        skipxdo = self.do2(skipx)

        # conv -> bn -> relu
        out = self.up_conv(out)
        out = self.bn1(out)
        out = self.relu1(out)

        xcat = sparse_cat(out, skipxdo)

        out = self.ops(xcat)
        out = spconv.functional.sparse_add(out, xcat)
        out = self.relu2(out)
        return out

    def initialize(self):
        trunc_normal_(self.up_conv.weight, std=0.02)
        nn.init.constant_(self.up_conv.bias, -0.01)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)


class OutputTransition(nn.Module):
    """
    out = conv1 -> bn -> act -> conv2
    """

    def __init__(self, in_channels: int, classes: int, elu: bool):
        super(OutputTransition, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            in_channels,
            classes,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )
        self.bn1 = spconv.SparseBatchNorm(classes)
        self.relu1 = ELUCons(elu, classes)
        self.conv2 = spconv.SubMConv3d(
            classes,
            classes,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.initialize()

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        return out

    def initialize(self):
        trunc_normal_(self.conv1.weight, std=0.02)
        nn.init.constant_(self.conv1.bias, -0.01)
        trunc_normal_(self.conv2.weight, std=0.02)
        nn.init.constant_(self.conv2.bias, 1.0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)


class ChannelAdjuster(nn.Module):
    """Simple 1x1 conv to adjust channel dimensions."""

    def __init__(self, in_channels: int, out_channels: int):
        super(ChannelAdjuster, self).__init__()
        self.conv = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        return self.conv(x)


# Mapping dictionary for backward compatibility
Spconv = {
    "InputTransition": InputTransition,
    "DownTransition": DownTransition,
    "UpTransition": UpTransition,
    "OutputTransition": OutputTransition,
    "ChannelAdjuster": ChannelAdjuster,
}

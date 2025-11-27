import torch.nn as nn
import torch

"""
Implementation of this model is borrowed and modified
(to support multi-channels and latest pytorch version)
from here:
https://github.com/Dawn90/V-Net.pytorch
"""


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(nchan)

        # self.initialize()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

    def initialize(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)

        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels, elu, num_features=16):
        super(InputTransition, self).__init__()
        self.num_features = num_features
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(
            self.in_channels, self.num_features, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(self.num_features)

        self.relu1 = ELUCons(elu, self.num_features)

        # self.initialize()

    def forward(self, x):
        out = self.conv1(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn1(out)
        x16 = x.repeat(1, repeat_rate, 1, 1, 1)
        return self.relu1(torch.add(out, x16))

    def initialize(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)

        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(outChans)

        self.do1 = nn.Identity()
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

        # self.initialize()

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out

    def initialize(self):
        nn.init.kaiming_normal_(self.down_conv.weight)
        nn.init.constant_(self.down_conv.bias, 0)

        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False, extra_scale=1, skip_scale=1):
        """
        Args:
            inChans: Number of channels from the lower-resolution input.
            outChans: Number of output channels after this up transition.
            nConvs: Number of convolution operations in the conv block.
            elu: Boolean to choose between ELU or PReLU activations.
            dropout: Whether to apply dropout on the skip connection.
            extra_scale: Additional upscaling factor to apply after the standard ×2 upsampling.
                         For example, extra_scale=n will upscale further by a factor of n.
            skip_scale: Additional upscaling factor to apply to the skip connection.
                        This is useful when the skip connection has fewer dims than the main branch.
        """
        super(UpTransition, self).__init__()
        self.extra_scale = extra_scale
        self.skip_scale = skip_scale

        self.up_conv = nn.ConvTranspose3d(
            inChans, outChans // 2, kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm3d(outChans // 2)
        self.do1 = nn.Dropout3d() if dropout else nn.Identity()
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        self.ops = _make_nConv(outChans, nConvs, elu)

        # If an extra upscale is needed, apply it to both the upsampled branch and the skip connection.
        if self.extra_scale > 1:
            self.extra_up = nn.ConvTranspose3d(
                outChans // 2, outChans // 2, kernel_size=self.extra_scale, stride=self.extra_scale)
            # For the skip connection, we assume it originally has outChans//2 channels.
            self.skip_up = nn.ConvTranspose3d(
                outChans // 2, outChans // 2, kernel_size=self.extra_scale * self.skip_scale, stride=self.extra_scale * self.skip_scale)
            self.skip_bn = nn.BatchNorm3d(outChans // 2)
            self.skip_relu = ELUCons(elu, outChans // 2)
        else:
            self.extra_up = None
            self.skip_up = None
            self.skip_bn = None
            self.skip_relu = None
        
        # self.initialize()

    def forward(self, x, skipx):
        out = self.do1(x)
        skipx = self.do2(skipx)
        # Standard upsampling: double the spatial dimensions.
        # out now has shape (B, outChans//2, D*2, H*2, W*2)
        out = self.up_conv(out)
        if self.extra_scale > 1:
            # Further upscale the main branch.
            out = self.extra_up(out)
            # Also upscale the skip connection so that they match.
            skipx = self.skip_up(skipx)
            skipx = self.skip_bn(skipx)
            skipx = self.skip_relu(skipx)
        out = self.bn1(out)
        out = self.relu1(out)
        # Concatenate along the channel dimension.
        xcat = torch.cat((out, skipx), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out

    def initialize(self):
        nn.init.kaiming_normal_(self.up_conv.weight)
        nn.init.constant_(self.up_conv.bias, 0)

        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

        self.do2.p = 0.5
        if isinstance(self.do1, nn.Dropout3d):
            self.do1.p = 0.5


class OutputTransition(nn.Module):
    def __init__(self, in_channels, classes, elu):
        super(OutputTransition, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(in_channels, classes, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(classes)

        self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.relu1 = ELUCons(elu, classes)

        # self.initialize()

    def forward(self, x):
        # convolve 32 down to channels as the desired classes
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out

    def initialize(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)

        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)

        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)


class ModePooling(nn.Module):
    def __init__(self, kernel_size):
        super(ModePooling, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        out = x.unfold(2, self.kernel_size, self.kernel_size).unfold(
            3, self.kernel_size, self.kernel_size).unfold(4, self.kernel_size, self.kernel_size)
        # Take dimensions of pooled volume and combine sub-volumes into 1
        out = out.contiguous().view(out.size()[:5] + (-1,))
        out = out.quantile(0.75, dim=-1)
        return out


class ChannelAdjuster(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAdjuster, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


TorchNN = {
    "InputTransition": InputTransition,
    "DownTransition": DownTransition,
    "UpTransition": UpTransition,
    "OutputTransition": OutputTransition,
    'ChannelAdjuster': ChannelAdjuster,
}

import torch

try:
    from torchinfo import summary
except Exception:
    summary = None

from modules.interleaver import Interleaver, Deinterleaver
from backends import backends


class VNet(torch.nn.Module):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(self, elu=True, in_channels=1, classes=4, backend_type='torchnn'):
        super(VNet, self).__init__()
        self.backend_type = backend_type
        self.backend = backends[backend_type]
        self.classes = classes
        self.in_channels = in_channels

        self.forward_times = {
            "total": 0,
            "count": 0
        }

        self.init_model(in_channels, classes, elu)

    def init_model(self, in_channels, classes, elu, embedded_channels=32):
        self.in_adjuster = self.backend['ChannelAdjuster'](
                in_channels, embedded_channels)

        self.in_tr = self.backend['InputTransition'](
            embedded_channels, num_features=64, elu=elu)

        self.down_tr128 = self.backend['DownTransition'](
            64, 3, elu, dropout=True)
        self.down_tr256 = self.backend['DownTransition'](
            128, 2, elu, dropout=True)
        self.up_tr256 = self.backend['UpTransition'](
            256, 256, 2, elu, dropout=True)
        self.up_tr128 = self.backend['UpTransition'](
            256, 128, 2, elu, dropout=True)
        self.out_tr = self.backend['OutputTransition'](128, embedded_channels, elu)

        self.out_ch_adjuster = self.backend['ChannelAdjuster'](
            embedded_channels, classes)

    def forward(self, x, data={}):
        x = self.in_adjuster(x)
        out64 = self.in_tr(x)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.out_tr(out)
        out = self.out_ch_adjuster(out)
        return out

    def test(self, device='cpu'):
        input_tensor = torch.rand(1, self.in_channels, 32, 32, 32)
        ideal_out = torch.rand(1, self.classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        if summary is not None:
            if summary is not None:
                summary(self.to(torch.device(device)),
                        (self.in_channels, 32, 32, 32), device=device)
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("Vnet test is complete")


class VNetInterleaved(VNet):
    """
    Interleaved version of VNet that applies interleaving and deinterleaving operations before and after the network.
    Motivated by DeepFocus: https://doi.org/10.1145/3272127.3275032
    """

    def __init__(self, elu=True, in_channels=1, classes=4, r=2, backend_type='torchnn'):
        self.r = r
        super().__init__(
            elu, in_channels, classes, backend_type)

    def init_model(self, in_channels, classes, elu, embedded_channels=32):
        interleaved_in_channels = in_channels * (self.r ** 3)
        interleaved_classes = classes * (self.r ** 3)

        self.interleaver = Interleaver(self.r)

        self.in_adjuster = self.backend['ChannelAdjuster'](
                interleaved_in_channels, embedded_channels)

        self.in_tr = self.backend['InputTransition'](
            embedded_channels, num_features=64, elu=elu)

        self.down_tr128 = self.backend['DownTransition'](
            64, 3, elu, dropout=True)
        self.down_tr256 = self.backend['DownTransition'](
            128, 2, elu, dropout=True)
        self.up_tr256 = self.backend['UpTransition'](
            256, 256, 2, elu, dropout=True)
        self.up_tr128 = self.backend['UpTransition'](
            256, 128, 2, elu, dropout=True)
        
        self.out_tr = self.backend['OutputTransition'](
            128, embedded_channels, elu)

        self.out_ch_adjuster = self.backend['ChannelAdjuster'](
            embedded_channels, interleaved_classes)

        self.deinterleaver = Deinterleaver(self.r)

    def forward(self, x, data={}):
        x = self.interleaver(x)
        x = self.in_adjuster(x)
        out64 = self.in_tr(x)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.out_tr(out)
        out = self.out_ch_adjuster(out)
        out = self.deinterleaver(out)
        return out

class VNetLighter(torch.nn.Module):
    """
    A lighterer version of Vnet that uses less channels in order to reduce time and space complexity
    """

    def __init__(self, elu=True, in_channels=1, classes=4, backend_type='torchnn'):
        super(VNetLighter, self).__init__()
        self.backend_type = backend_type
        self.backend = backends[backend_type]
        self.classes = classes
        self.in_channels = in_channels
        self.in_tr = self.backend['InputTransition'](
            in_channels, elu, num_features=4)
        self.down_tr8 = self.backend['DownTransition'](4, 1, elu)
        self.down_tr16 = self.backend['DownTransition'](8, 2, elu)
        self.down_tr32 = self.backend['DownTransition'](
            16, 3, elu, dropout=True)
        self.down_tr64 = self.backend['DownTransition'](
            32, 2, elu, dropout=True)
        self.up_tr64 = self.backend['UpTransition'](
            64, 64, 2, elu, dropout=True)
        self.up_tr32 = self.backend['UpTransition'](
            64, 32, 2, elu, dropout=True)
        self.up_tr16 = self.backend['UpTransition'](32, 16, 1, elu)
        self.up_tr8 = self.backend['UpTransition'](16, 8, 1, elu)
        self.out_tr = self.backend['OutputTransition'](8, classes, elu)

    def forward(self, x):
        out4 = self.in_tr(x)
        out8 = self.down_tr8(out4)
        out16 = self.down_tr16(out8)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out = self.up_tr64(out64, out32)
        out = self.up_tr32(out, out16)
        out = self.up_tr16(out, out8)
        out = self.up_tr8(out, out4)
        out = self.out_tr(out)
        return out

    def test(self, device='cpu'):
        pass


class VNetLight(torch.nn.Module):
    """
    A lighter version of Vnet that skips down_tr256 and up_tr256 in order to reduce time and space complexity
    """

    def __init__(self, elu=True, in_channels=1, classes=4, backend_type='torchnn'):
        super(VNetLight, self).__init__()
        self.backend_type = backend_type
        self.backend = backends[backend_type]
        self.classes = classes
        self.in_channels = in_channels
        self.in_tr = self.backend['InputTransition'](in_channels, elu)
        self.down_tr32 = self.backend['DownTransition'](16, 1, elu)
        self.down_tr64 = self.backend['DownTransition'](32, 2, elu)
        self.down_tr128 = self.backend['DownTransition'](
            64, 3, elu, dropout=True)
        self.up_tr128 = self.backend['UpTransition'](
            128, 128, 2, elu, dropout=True)
        self.up_tr64 = self.backend['UpTransition'](128, 64, 1, elu)
        self.up_tr32 = self.backend['UpTransition'](64, 32, 1, elu)
        self.out_tr = self.backend['OutputTransition'](32, classes, elu)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out = self.up_tr128(out128, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

    def test(self, device='cpu'):
        if self.backend_type == 'spconv':
            # Test with spconv backend
            input_tensor = torch.rand(1, self.in_channels, 32, 32, 32)
            from utils.tensor import dense_to_spconv
            input_sparse = dense_to_spconv(input_tensor)
            out = self.forward(input_sparse)
            # spconv output is also sparse, check features shape
            assert out.features.shape[1] == self.classes
            print("VNet spconv test is complete")
        elif self.backend_type == 'torchnn':
            input_tensor = torch.rand(1, self.in_channels, 32, 32, 32)
            ideal_out = torch.rand(1, self.classes, 32, 32, 32)
            out = self.forward(input_tensor)
            assert ideal_out.shape == out.shape
            if summary is not None:
                summary(self.to(torch.device(device)),
                        (self.in_channels, 32, 32, 32), device=device)

        print("Vnet light test is complete")

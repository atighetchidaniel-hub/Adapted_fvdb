from .torchnn import TorchNN
from .spconv import Spconv

backends = {
    'torchnn': TorchNN,
    'spconv': Spconv,
}
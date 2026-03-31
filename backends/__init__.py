from .torchnn import TorchNN

try:
    from .spconv import Spconv
except Exception:
    Spconv = None

backends = {
    'torchnn': TorchNN,
}

if Spconv is not None:
    backends['spconv'] = Spconv

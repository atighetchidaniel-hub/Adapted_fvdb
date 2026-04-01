from .torchnn import TorchNN

try:
    from .spconv import Spconv
except Exception:
    Spconv = None

try:
    from .fvdb import Fvdb
except Exception:
    Fvdb = None

backends = {
    'torchnn': TorchNN,
}

if Spconv is not None:
    backends['spconv'] = Spconv

if Fvdb is not None:
    backends['fvdb'] = Fvdb

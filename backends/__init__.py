from .torchnn import TorchNN

try:
    from .fvdb import Fvdb
except Exception:
    Fvdb = None

backends = {
    'torchnn': TorchNN,
}

if Fvdb is not None:
    backends['fvdb'] = Fvdb

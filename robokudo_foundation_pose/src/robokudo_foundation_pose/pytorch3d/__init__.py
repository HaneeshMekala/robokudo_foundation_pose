# Alias 'robokudo_foundation_pose.pytorch3d' to top-level 'pytorch3d'
import importlib, sys as _sys
_mod = importlib.import_module("pytorch3d")
globals().update(_mod.__dict__)
_sys.modules[__name__] = _mod

# Provide compat.eigh if the version doesn't expose it
try:
    import torch as _torch
    import pytorch3d.common.compat as _compat
    if not hasattr(_compat, "eigh"):
        def _eigh(A):  # returns (eigenvalues, eigenvectors)
            return _torch.linalg.eigh(A)
        _compat.eigh = _eigh
except Exception:
    pass

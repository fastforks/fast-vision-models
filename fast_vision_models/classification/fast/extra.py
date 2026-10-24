"""Loader for csrc/extra.cu (torch.ops.clfx.*: rope_qk, swiglu_ln); compiled once per source hash, like fused.py."""
import os


from ..._build import jit_load

from .fused import HERE

SRC = os.path.join(HERE, "csrc", "extra.cu")
_loaded = [False]


def load():
    if _loaded[0]:
        return True
    from .fused import load as _lf

    _lf()
    jit_load("clf_extra", [SRC])
    _loaded[0] = True
    return True

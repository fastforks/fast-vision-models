"""Shared build helpers for the custom CUDA ops of the cuda-kernels tier.

Extensions are JIT-compiled once per source hash and GPU arch with torch.utils.cpp_extension into the user cache
(``$FVM_CACHE`` or ``~/.cache/fast-vision-models``); nothing is compiled per input size. The CUTLASS headers are not
vendored: set ``CUTLASS_DIR`` to a CUTLASS checkout (v3.x/v4.x, header-only).
"""

import hashlib
import os

import torch

CUDA_FLAGS = [
    "-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]


def cache_dir(*parts):
    d = os.path.join(
        os.environ.get("FVM_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "fast-vision-models")), *parts
    )
    os.makedirs(d, exist_ok=True)
    return d


def cutlass_include():
    root = os.environ.get("CUTLASS_DIR")
    if not root or not os.path.isdir(os.path.join(root, "include", "cutlass")):
        raise RuntimeError(
            "The cuda-kernels tier needs the CUTLASS headers: git clone https://github.com/NVIDIA/cutlass and set CUTLASS_DIR "
            "(or use the portable tier)."
        )
    return os.path.join(root, "include")


def lib_dir():
    """A linker dir with libcudart.so / libcublasLt.so (pip CUDA wheels ship only the versioned .so.13 files)."""
    d = cache_dir("lib")
    home = os.environ.get("CUDA_HOME", "")
    for name in ("libcudart", "libcublasLt"):
        for cand in (os.path.join(home, "lib", f"{name}.so.13"), os.path.join(home, "lib64", f"{name}.so")):
            link = os.path.join(d, f"{name}.so")
            if os.path.exists(cand) and not os.path.lexists(link):
                os.symlink(cand, link)
                break
    return d


def arch():
    """GPU arch of device 0; FVM_ARCH=12.0 (etc.) builds for that arch without touching the GPU (precompile)."""
    a = os.environ.get("FVM_ARCH")
    if a:
        major, minor = a.split(".")
        return int(major), int(minor)
    return torch.cuda.get_device_capability()


def src_tag(paths, n=12):
    h = hashlib.sha1()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:n]


def jit_load(prefix, sources, deps=(), extra_cflags=(), extra_ldflags=()):
    """Compile (or load the cached build of) a torch-op extension; returns the build dir."""
    tag = src_tag(list(sources) + list(deps))
    major, minor = arch()
    name = f"{prefix}_{tag}"
    bdir = cache_dir(f"{name}_sm{major}{minor}")
    so = os.path.join(bdir, f"{name}.so")
    if os.path.exists(so):
        torch.ops.load_library(so)
        return bdir
    from torch.utils.cpp_extension import load

    load(
        name=name,
        sources=list(sources),
        build_directory=bdir,
        verbose=False,
        is_python_module=False,
        extra_include_paths=[cutlass_include()],
        extra_ldflags=[f"-L{lib_dir()}", *extra_ldflags],
        extra_cuda_cflags=[*CUDA_FLAGS, *extra_cflags, f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
    )
    return bdir

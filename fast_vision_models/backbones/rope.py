"""Fused in-place RoPE (DINOv3) for the packed qkv GEMM output: one kernel (fatbin sm_80..120 + PTX, built once with the nvcc of CUDA_HOME into
~/.cache/fast-vision-models/rope), any size, nothing compiled per shape. Plain-PyTorch reference: rope_torch."""
import ctypes
import hashlib
import math
import os
import struct
import subprocess

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "csrc", "rope.cu")
_PRE = os.path.join(os.environ.get("FVM_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "fast-vision-models")), "rope")
_ST = [None]


def _tag():
    return hashlib.sha1(open(_SRC, "rb").read()).hexdigest()[:12]


def build_prebuilt(archs=(80, 86, 87, 89, 90, 110, 120)):
    os.makedirs(_PRE, exist_ok=True)
    out = os.path.join(_PRE, f"rope_{_tag()}.fatbin")
    ch = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    gen = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs] + ["-gencode=arch=compute_80,code=compute_80"]
    subprocess.run([os.path.join(ch, "bin", "nvcc"), "-fatbin", "-O3", "-std=c++17", *gen, f"-I{ch}/include", "-o", out + ".tmp", _SRC], check=True)
    os.replace(out + ".tmp", out)
    return out


def _load():
    if _ST[0] is None:
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        fat = os.path.join(_PRE, f"rope_{_tag()}.fatbin")
        if not os.path.exists(fat):
            fat = build_prebuilt()
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        mod, f = ctypes.c_void_p(), ctypes.c_void_p()
        img = open(fat, "rb").read()
        assert lib.cuModuleLoadData(ctypes.byref(mod), img) == 0
        assert lib.cuModuleGetFunction(ctypes.byref(f), mod, b"rope_qk_h") == 0
        S = struct.Struct("<QQiiiif")
        buf = ctypes.create_string_buffer(64)
        sz = ctypes.c_size_t(S.size)
        extra = (ctypes.c_void_p * 5)(1, ctypes.addressof(buf), 2, ctypes.addressof(sz), 0)
        _ST[0] = (lib, mod, f, S, buf, sz, extra, img)
    return _ST[0]


def rope_table(h, w, head_dim=64, theta=100.0, device="cuda"):
    """fp32 [h*w, 2, head_dim//2] (cos, sin) -- same math as HF DINOv3ViTRopePositionEmbedding (coords in [-1, 1], centers)."""
    ch = (torch.arange(0.5, h, dtype=torch.float32, device=device) / h)
    cw = (torch.arange(0.5, w, dtype=torch.float32, device=device) / w)
    coords = 2.0 * torch.stack(torch.meshgrid(ch, cw, indexing="ij"), -1).flatten(0, 1) - 1.0
    inv = 1 / theta ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32, device=device)
    ang = (2 * math.pi * coords[:, :, None] * inv[None, None, :]).flatten(1, 2)      # [Np, head_dim/2]
    return torch.stack([ang.cos(), ang.sin()], 1).contiguous()


def rope_inplace(qkv, cs, P, qscale=1.0):
    """qkv [B, N, 3, H, 64] contiguous fp16, cs from rope_table -> rotates q and k of tokens >= P in place (q additionally multiplied by qscale)."""
    B, N, _, H, D = qkv.shape
    assert D == 64 and qkv.is_contiguous() and qkv.dtype == torch.float16 and cs.shape[0] == N - P
    lib, mod, f, S, buf, sz, extra, _ = _load()
    total = B * N * 2 * H * 16
    S.pack_into(buf, 0, qkv.data_ptr(), cs.data_ptr(), N, P, H, total, float(qscale))
    blocks = min((total + 255) // 256, 65535 * 8)
    r = lib.cuLaunchKernel(f, blocks, 1, 1, 256, 1, 1, 0, torch.cuda.current_stream().cuda_stream, None, extra)
    if r:
        raise RuntimeError(f"rope launch error {r}")
    return qkv


def rope_torch(qkv, cs, P):
    """Plain PyTorch (exportable) version; returns a new tensor with q, k of tokens >= P rotated (any dtype)."""
    cos = cs[:, 0].to(qkv.dtype).repeat(1, 2)[None, :, None, :]       # [1, Np, 1, 64]
    sin = cs[:, 1].to(qkv.dtype).repeat(1, 2)[None, :, None, :]
    q, k, v = qkv.unbind(2)                                            # [B, N, H, 64]

    def rot(x):
        xp = x[:, P:]
        r = torch.cat((-xp[..., 32:], xp[..., :32]), -1)
        return torch.cat((x[:, :P], xp * cos + r * sin), 1)
    return torch.stack([rot(q), rot(k), v], 2)

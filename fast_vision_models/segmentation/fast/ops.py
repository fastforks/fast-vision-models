"""Shared general ops of exp 009 (any size, prebuilt kernels)."""
import contextlib

import torch
import torch.nn.functional as F

from . import kernels as _k

_KERN = [None]
# (launch-overhead pass) max path: prebuilt kernels as C++ torch ops (torch.ops.segx, seg_fast/segx.py) instead of ctypes
# launches: ~3x less CPU per launch and visible to a TorchScript trace (seg_fast/trace.py). SEG_CX=0 = round-2 ctypes path.
CX = __import__("os").environ.get("SEG_CX", "1") == "1"
_CX_LOADED = [False]


def cx():
    """True if the segx C++ ops are available (built / loaded on first use)."""
    if not CX:
        return False
    if not _CX_LOADED[0]:
        from . import segx
        segx.load()
        _CX_LOADED[0] = True
    return True


def kern():
    if _KERN[0] is None:
        _KERN[0] = _k.Kernels.get()
    return _KERN[0]


def up_argmax(s, H, W, hin=None, win=None, align_corners=False):
    """s: fp16 scores [B, h, w, C] (NHWC view, C contiguous) -> int32 [B, H, W] = argmax(bilinear(s[:, :hin, :win], (H, W))).
    Never materializes the full-resolution scores (150 x H x W)."""
    B, h, w, C = s.shape
    if cx() and hin is None and win is None:
        return torch.ops.segx.up_argmax(s, H, W, bool(align_corners))
    hin, win = hin or h, win or w
    if s.dtype != torch.float16 or s.stride(3) != 1 or s.stride(1) != w * C or s.stride(2) != C or C % 2:
        s = s.half().contiguous()
        if C % 2:
            s = F.pad(s, (0, 1), value=float("-inf"))
            C += 1
    out = torch.empty((B, H, W), device=s.device, dtype=torch.int32)
    n = B * H * W
    kern().launch("up_argmax_h", ((min((n + 255) // 256, 65535 * 8), 1, 1), (256, 1, 1)), s, out, B, hin, win,
                  h * w, w, C, H, W, *(((hin - 1) / max(H - 1, 1), (win - 1) / max(W - 1, 1), 1) if align_corners
                                       else (hin / H, win / W, 0)))
    return out


@contextlib.contextmanager
def fp16_accumulate(enabled=True):
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


def upsample_argmax(s, H, W, align_corners=False, fused=True):
    """NHWC scores [B, h, w, C] -> int class map [B, H, W]. fused = prebuilt kernel (max version); otherwise plain PyTorch
    (fp16 bilinear interpolate + argmax, ONNX / TensorRT exportable; the upstream-friendly version)."""
    if fused:
        return up_argmax(s, H, W, align_corners=align_corners)
    s = s.permute(0, 3, 1, 2)
    if tuple(s.shape[2:]) != (H, W):
        s = F.interpolate(s, (H, W), mode="bilinear", align_corners=align_corners)
    return s.argmax(1)


def dwconv(x, conv, with_bias=True, act=0):
    """depthwise nn.Conv2d (groups == C) on an NHWC fp16 tensor with the prebuilt dwconv_h kernel -> channels-last output.
    conv._wt (tap-major weights) is made on first use."""
    wt = getattr(conv, "_wt", None)
    C = conv.in_channels
    kh, kw = conv.kernel_size
    if wt is None:
        wt = conv.weight.detach().reshape(C, kh * kw).t().contiguous().half()
        conv.register_buffer("_wt", wt, persistent=False)   # registered buffer: moves with the module, traced as a constant
    if cx():
        b = conv.bias if (with_bias and conv.bias is not None) else None
        if C % 8 == 0 and tuple(conv.dilation) == (1, 1) and kh <= 7 and kw <= 7:
            # exp 008's register-tiled kernel (8 channels x 2 columns per thread, 32-bit index math): dwconv_h's 64-bit
            # divisions made it ~10x slower on SegFormer-B5's wide Mix-FFN convs (10 ms of 28 at 512x512)
            wr = getattr(conv, "_wrsc", None)
            if wr is None:
                wr = conv.weight.detach().reshape(C, kh, kw).permute(1, 2, 0).contiguous().half()
                conv.register_buffer("_wrsc", wr, persistent=False)
            return torch.ops.segx.dwconv8(x, wr, b, list(conv.stride), list(conv.padding), 5 if act == 1 else 0)
        return torch.ops.segx.dwconv(x, wt, b, kh, kw, list(conv.stride), list(conv.padding), list(conv.dilation), act)
    x = x.half().contiguous(memory_format=torch.channels_last)
    N, _, H, W = x.shape
    (sh, sw), (ph, pw), (dh, dw) = conv.stride, conv.padding, conv.dilation
    Ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    Wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    y = torch.empty((N, C, Ho, Wo), device=x.device, dtype=torch.float16, memory_format=torch.channels_last)
    b = conv.bias if (with_bias and conv.bias is not None) else 0
    n = N * Ho * Wo * C // 2
    kern().launch_packed("dwconv_h", (min((n + 255) // 256, 65535 * 8), 1, 1), (256, 1, 1), x, wt, b, y,
                         N, H, W, C, Ho, Wo, kh, kw, sh, sw, ph, pw, dh, dw, act)
    return y


class GraphCache:
    """CUDA-graph replay for a fixed-shape callable (max + graphs version only: SIZE-SPECIALIZED, one capture per input
    shape; removes the CPU launch overhead of models with hundreds of small kernels). Nothing is compiled: a shape is
    captured the second time it is seen (first call eager: warms cuDNN plans / caches), replayed afterwards; at most `keep`
    shapes are kept (LRU), all in one memory pool shared across sizes.
    Memory (round 2): `trim` empties the allocator cache before a capture (otherwise the eager blocks of the first call stay
    reserved next to the graph pool -> ~2x reserved), and the side-stream warm-up is skipped by default (it caches a third
    copy of the activations in the side stream's pool; the eager first call already warmed everything).
    `max_pixels`: shapes with more input pixels run eager (launch overhead only matters for small inputs).
    fn(x, out_size) -> tensor; the returned tensor is a fresh copy."""

    def __init__(self, fn, keep=3, pre_replay=(), trim=True, warm=False, max_pixels=None):
        self.fn, self.keep, self.pre_replay = fn, keep, list(pre_replay)  # pre_replay: callables run before each replay
        self.trim, self.warm, self.max_pixels = trim, warm, max_pixels
        self.seen, self.graphs = set(), {}
        self.pool = None

    def __call__(self, x, out_size=None):
        if self.max_pixels is not None and x.shape[-1] * x.shape[-2] > self.max_pixels:
            return self.fn(x, out_size)
        key = (tuple(x.shape), x.dtype, tuple(out_size) if out_size is not None else None)
        e = self.graphs.get(key)
        if e is None:
            if key not in self.seen:
                self.seen.add(key)
                return self.fn(x, out_size)
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            xin = x.clone()
            rng = torch.get_rng_state()                                 # warm-up / capture must not consume CPU draws
            if self.warm:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    self.fn(xin, out_size)                              # warm-up on the capture stream
                torch.cuda.current_stream().wait_stream(s)
            if self.trim:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                out = self.fn(xin, out_size)
            torch.set_rng_state(rng)
            e = self.graphs[key] = (g, xin, out)
            while len(self.graphs) > self.keep:
                self.graphs.pop(next(iter(self.graphs)))
        else:
            self.graphs[key] = self.graphs.pop(key)                   # LRU order
        g, xin, out = e
        xin.copy_(x)
        for h in self.pre_replay:
            h()
        g.replay()
        return out.clone()


_F_INTERP = F.interpolate


def _fast_interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None,
                      recompute_scale_factor=None, antialias=False):
    """F.interpolate with bilinear NHWC fp16 4-D inputs (C even) on the prebuilt kernel; everything else -> PyTorch."""
    x = input
    if (mode != "bilinear" or antialias or x.dim() != 4 or x.dtype != torch.float16 or x.shape[1] % 2 or not x.is_cuda
            or not x.is_contiguous(memory_format=torch.channels_last) or recompute_scale_factor):
        return _F_INTERP(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
    B, C, h, w = x.shape
    if cx():
        al = bool(align_corners)
        if size is not None:
            H, W = (size, size) if not isinstance(size, (tuple, list, torch.Size)) else (size[0], size[1])
            if torch.jit.is_tracing() and (isinstance(H, torch.Tensor) or isinstance(W, torch.Tensor)):
                # traced sizes (0-dim tensors while tracing): pass them as a tensor, an int argument would be frozen
                sz = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in (H, W)])
                return torch.ops.segx.upsample_bilinear_t(x, sz, al)
            return torch.ops.segx.upsample_bilinear(x, H, W, None, None, al)
        sf = (scale_factor, scale_factor) if not isinstance(scale_factor, (tuple, list)) else tuple(scale_factor)
        return torch.ops.segx.upsample_bilinear(x, -1, -1, float(sf[0]), float(sf[1]), al)  # size from the scale, in C++
    if size is not None:
        H, W = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
        sf = None
    else:
        sf = (scale_factor, scale_factor) if not isinstance(scale_factor, (tuple, list)) else tuple(scale_factor)
        H, W = int(h * sf[0]), int(w * sf[1])
    al = bool(align_corners)
    if al:
        sh, sw = ((h - 1) / (H - 1) if H > 1 else 0.0), ((w - 1) / (W - 1) if W > 1 else 0.0)
    elif sf is not None:
        sh, sw = 1.0 / sf[0], 1.0 / sf[1]
    else:
        sh, sw = h / H, w / W
    out = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
    n = B * H * W * C // 2
    kern().launch("upsample_bilinear_h", ((min((n + 255) // 256, 65535 * 8), 1, 1), (256, 1, 1)), x, out,
                  B, h, w, C, H, W, sh, sw, int(al))
    return out


@contextlib.contextmanager
def fast_interpolate(enabled=True):
    """route F.interpolate (bilinear, NHWC fp16) to the prebuilt kernel inside the block (max version)."""
    if not enabled:
        yield
        return
    F.interpolate = _fast_interpolate
    try:
        yield
    finally:
        F.interpolate = _F_INTERP

"""General fast path for CNN segmentation nets (mmseg classics, SegNet): exp 006 fastbb CNN optimizer (copy in
seg_fast/cnn.py) = BN folded into convs, fp16 channels-last, fp16-accumulate cuDNN convs incl. dilated ones (precompiled
engines only; seg_fast/cudnn_conv.py adds dilation), fp16-acc GEMMs; then the fused upsample-argmax kernel on the
low-resolution NHWC logits (no full-resolution 150-channel scores)."""
import os

import torch
import torch.nn.functional as F

from .cnn import optimize
from .ops import fast_interpolate, upsample_argmax


FP32_ISLANDS = ("NMF2D",)  # SegNeXt Ham head: multiplicative-update matrix factorization (ratios of small numbers)


def _pinned_bases(self, B, S, D, R, device=None):
    """NMF2D._build_bases, CUDA-graph capturable with the upstream random stream: the bases are drawn with torch.rand on
    the CPU exactly as upstream (one draw per forward) into a pinned buffer and copied from there. Eager calls draw
    here; inside a graph only the copy is recorded and `refill` (a pre-replay hook) draws before each replay."""
    if os.environ.get("SEG_GRAPH", "0") != "1":         # graph-free max (round 2): no host sync, caching host allocator
        return F.normalize(torch.rand((B * S, D, R)).pin_memory().to("cuda", non_blocking=True), dim=1)
    pin = getattr(self, "_pin", None)
    if pin is None or tuple(pin.shape) != (B * S, D, R):
        pin = self._pin = torch.empty((B * S, D, R)).pin_memory()
    if not torch.cuda.is_current_stream_capturing():
        torch.cuda.current_stream().synchronize()   # a previous replay may still read the buffer
        pin.copy_(torch.rand((B * S, D, R)))
    return F.normalize(pin.to("cuda", non_blocking=True), dim=1)


def _nmf_forward_cx(self, x, return_bases=False):
    """NMF2D.forward (eval, rand_init) with the iterations in one C++ op (segx.nmf2d; same math and op order)."""
    B, C, H, W = x.shape
    D = C // self.S
    x = x.reshape(B * self.S, D, -1)
    bases = self._build_bases(B, self.S, D, self.R, device=x.device)
    return torch.ops.segx.nmf2d(x, bases, self.eval_steps, float(self.inv_t)).view(B, C, H, W)


def _refill(mods):
    def hook():
        torch.cuda.current_stream().synchronize()
        for m in mods:
            if getattr(m, "_pin", None) is not None:
                m._pin.copy_(torch.rand(m._pin.shape))
    return hook


def _fp32_island(mod):
    f = mod.forward

    def fwd(x, *a, **k):
        y = f(x.float().contiguous(), *a, **k)
        return y.to(x.dtype) if isinstance(y, torch.Tensor) else y
    mod.forward = fwd


class FastMMSeg:
    def __init__(self, net, spec=os.environ.get("SEG_CNN_SPEC", "fold-hacc-gacc-fuse-dwk-fint"), align=False, fused=True):
        self.align, self.fused = align, fused
        islands = [m for m in net.modules() if m.__class__.__name__ in FP32_ISLANDS]
        segnext = os.environ.get("SEG_SEGNEXT", "1") == "1" and any(type(m).__name__ == "MSCAN" for m in net.modules())
        if segnext:                                     # round 2: SegNeXt block fast path (seg_fast/segnext.py)
            from .segnext import finalize_segnext, patch_segnext
            self.segnext = patch_segnext(net, fused)
        self.f = optimize(net, spec=spec, sample_hw=(64, 64))
        if segnext:
            finalize_segnext(net)
        if "fuse" in spec.split("-"):
            from .cnn_fuse import fuse_epilogues
            self.fused_epilogues = fuse_epilogues(self.f.model)
        for m in islands:
            _fp32_island(m)
            if fused:                                   # max version: capturable bases, same CPU random stream
                import types
                m._build_bases = types.MethodType(_pinned_bases, m)
                from .ops import cx
                if cx() and type(m).__name__ == "NMF2D" and m.rand_init and not m.training \
                        and os.environ.get("SEG_NMF_CX", "1") == "1":
                    m.forward = types.MethodType(_nmf_forward_cx, m)   # re-wrapped in the fp32 island below
                    _fp32_island(m)
        self.pre_replay = [_refill(islands)] if (islands and fused) else []
        from .ops import cx
        if fused and cx() and os.environ.get("SEG_GRAPH", "0") != "1":   # launch-overhead pass: C++ executor
            from .trace import Traced
            self.f.traced[0] = Traced(self.f.model, name=type(net).__name__)
        self.info = {"opt": "cnn-fast", "fused_up_argmax": fused, "spec": spec, "fp32_islands": len(islands), "epilogues": getattr(self, "fused_epilogues", None), **getattr(self.f, "info", {}), **({"segnext": self.segnext} if segnext else {})}

    def _run(self, x):
        with fast_interpolate("fint" in self.info["spec"].split("-")):
            return self.f(x)

    def logits(self, x):
        return self._run(x)

    @property
    def trace_info(self):
        t = self.f.traced[0]
        return t.info if t is not None else {"trace": "off"}

    def predict(self, x, out_size=None):
        s = self._run(x)                                # [B, C, h, w] fp16 (channels-last when it comes from a conv)
        H, W = out_size or x.shape[2:]
        return upsample_argmax(s.permute(0, 2, 3, 1), H, W, align_corners=self.align, fused=self.fused)

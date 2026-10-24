"""General fast inference paths for the exp 008 classification zoo.

General = any batch size and input size, nothing compiled / autotuned per size, prebuilt kernels only:
- BN folded into the preceding conv (fp32 fold via torch.fx `fuse`, then cast), fx graph = no nn.Module call overhead
- pure fp16 weights + activations, channels-last
- fp16-accumulate cuBLAS GEMMs (`allow_fp16_accumulation`, 2x tensor-core rate on GeForce) unless the spec says otherwise
- optional fp16-accumulate cuDNN convs (groups=1, precompiled cudnn-frontend plans, no NVRTC engines): CLF_HACC=1
- attention: torch SDPA (flash for fp16) — timm uses it already when fused_attn is on

Modes: "max" (everything above) | "friendly" (upstream-friendly: only plain-PyTorch, ONNX/TensorRT-exportable changes —
BN fold, fp16, channels-last, SDPA, fp16-acc cuBLAS flag; no cudnn-frontend convs, no fastbb kernels).

Env switches (A/B tests): CLF_FUSE=0, CLF_GACC=0, CLF_HACC=1, CLF_CL=0.
"""
import contextlib
import os

import torch

FUSE = os.environ.get("CLF_FUSE", "1") == "1"
GACC = os.environ.get("CLF_GACC", "1") == "1"
HACC = os.environ.get("CLF_HACC", "auto")          # 1 = always, 0 = never, auto = only when batch*H*W >= HACC_PIX
HACC_PIX = int(os.environ.get("CLF_HACC_PIX", str(8 * 224 * 224)))
CL = os.environ.get("CLF_CL", "1") == "1"
FUSED = os.environ.get("CLF_FUSED", "1") == "1"   # max path: CUTLASS fused conv+bias(+res)+act (clf_fast/fused.py)
TRACE = os.environ.get("CLF_TRACE", "1") == "1"    # max path: TorchScript trace (C++ executor, fusers off), validated at 1st call
GCONV = os.environ.get("CLF_GCONV", "1") == "1"    # max path: grouped convs on clfmb.gconv
DWFFN = os.environ.get("CLF_DWFFN", "0") == "1"   # dw-first FFN kernel (FastViT): slower than unfused, A/B only
XDW = os.environ.get("CLF_XDW", "1") == "1"        # expand 1x1 + dw fused (clfmb.xdw) for Cin <= 64
MBFULL = os.environ.get("CLF_MBFULL", "0") == "1"  # max path: whole MBConv block in one kernel (slower, kept for A/B)
MB = os.environ.get("CLF_MB", "1") == "1"
SHUF_BIG = os.environ.get("CLF_SHUF_BIG", "1") == "1"   # ShuffleNet: standard-kernel net when batch*H*W >= HACC_PIX          # max path: one-kernel MBConv blocks (clf_fast/mbconv.py)


@contextlib.contextmanager
def fp16_accumulate(enabled=True):
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


class Fast(torch.nn.Module):
    """fp32 NCHW in -> fp16 channels-last net -> fp32 logits. `big` (optional) = the same net (shared weights) with dense
    convs on fp16-accumulate cuDNN; used when batch * H * W >= HACC_PIX (pure input-shape rule: small inputs are CPU-launch
    bound and the cudnn-frontend dispatch costs more than the 2x tensor-core rate saves)."""

    def __init__(self, net, gacc=True, cl=True, big=None, trace=False):
        super().__init__()
        self.net, self.gacc, self.cl = net, gacc, cl
        self.big = [big]  # list: not a registered submodule (weights are shared with net)
        self.trace = trace          # max path: TorchScript trace of net (C++ executor, fusers off) built at the first call
        self.traced = [None]
        self.pad8 = getattr(net, "pad8", False)

    def _traced(self, x):
        """trace on the first call (any shape), validate against eager at another batch size; None if it differs."""
        if self.traced[0] is None:
            _jit_plain()
            with torch.no_grad():
                self.net(x)  # eager warm-up first: shape caches (e.g. Swin window orders) are built outside the trace
                tr = torch.jit.trace(self.net, x, check_trace=False)
                # validate on fresh random inputs (traced first, so stale memory from earlier eager calls cannot match:
                # an op invisible to the tracer, e.g. a raw ctypes launch, shows up as a mismatch)
                ok = True
                for b in (x.shape[0], 2 if x.shape[0] == 1 else 1):
                    xi = torch.randn((b,) + tuple(x.shape[1:]), device=x.device).to(x.dtype).contiguous(
                        memory_format=torch.channels_last if x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last)
                        else torch.contiguous_format)
                    t = tr(xi).float()
                    r = self.net(xi).float()
                    if r.shape != t.shape or not bool(((r - t).abs().max() <= 1e-2 * (r.abs().max() + 1e-3)).item()):
                        ok = False
                self.traced[0] = tr if ok else False
                if not ok:
                    print("[clf_fast] trace mismatch -> eager")
        return self.traced[0] or None

    def forward(self, x):
        net = self.big[0] if self.big[0] is not None and x.shape[0] * x.shape[2] * x.shape[3] >= HACC_PIX else self.net
        if getattr(net, "raw_input", False):
            pass                                    # the stem kernel reads the fp32 NCHW input itself
        elif getattr(net, "pad8", False):
            x = torch.ops.clf.input_nhwc8(x)        # one kernel: fp32 NCHW -> fp16 NHWC, C zero-padded 3 -> 8
        else:
            x = x.to(torch.float16, memory_format=torch.channels_last if self.cl else torch.contiguous_format)
        with fp16_accumulate(self.gacc):
            if self.trace and net is self.net:
                tr = self._traced(x)
                if tr is not None:
                    net = tr
            y = net(x)
        return y.float()


def _jit_plain():
    """TorchScript executor without fusers / codegen (nothing compiled per shape): plain aten dispatch from C++."""
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    if hasattr(torch._C, "_jit_set_nvfuser_enabled"):
        try:
            torch._C._jit_set_nvfuser_enabled(False)
        except Exception:  # noqa: BLE001
            pass


def _conv_fwd(self, x):
    from .cudnn_conv import conv_half_acc
    return conv_half_acc(x, self.weight, self.bias, list(self.padding), list(self.stride))


def hacc_copy(model):
    """shallow copy of the (fx) module tree whose dense convs (groups 1) run as fp16-accumulate cuDNN convs; parameters are
    shared with `model` (no extra memory)."""
    import copy
    memo = {}

    def clone(mod):
        if id(mod) in memo:
            return memo[id(mod)]
        c = copy.copy(mod)
        c._modules = {k: (clone(v) if v is not None else None) for k, v in mod._modules.items()}
        memo[id(mod)] = c
        if type(mod) is torch.nn.Conv2d and mod.groups == 1 and mod.dilation == (1, 1) and mod.padding_mode == "zeros" \
                and not isinstance(mod.padding, str) and mod.weight.dtype == torch.half:
            mod.weight.data = mod.weight.data.contiguous(memory_format=torch.channels_last)
            c.forward = _conv_fwd.__get__(c)
        return c
    return clone(model)


class _Fp32Out(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).float()


def reparam(model):
    """timm re-parameterizable blocks (FastViT / MobileOne / RepViT ...: `reparameterize` / `fuse` / `switch_to_deploy`)."""
    if any(hasattr(m, a) for m in model.modules() for a in ("reparameterize", "switch_to_deploy")):
        from timm.utils.model import reparameterize_model
        return reparameterize_model(model, inplace=True), True
    return model, False


def fold_bn_hooks(model, size=64):
    """Fold every eval BatchNorm2d whose input is exactly the output of a Conv2d (found by one tiny fp32 forward with hooks,
    no tracing: works for timm BatchNormAct2d and untraceable models). The BN module is replaced by its activation
    (timm BatchNormAct2d: drop + act) or nn.Identity. Returns the number folded."""
    import torch.nn as nn
    convs, pairs, hooks = {}, [], []

    def conv_hook(m, inp, out):
        convs[id(out)] = (m, out)

    def bn_hook(m, inp, out):
        c = convs.get(id(inp[0]))
        if c is not None and c[1] is inp[0]:
            pairs.append((c[0], m))
    for m in model.modules():
        if type(m) is nn.Conv2d:
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(bn_hook))
    dev = next(model.parameters()).device
    x0 = None
    try:
        for sz in (size, 224):
            try:
                x0 = torch.randn(1, 3, sz, sz, device=dev, generator=torch.Generator(dev).manual_seed(0))
                with torch.inference_mode():
                    ref = model(x0)
                break
            except Exception:  # noqa: BLE001 - fixed-size models
                convs.clear(); pairs.clear(); x0 = None
    finally:
        for h in hooks:
            h.remove()
    if x0 is None:
        return 0
    parent_of = {}
    for parent in model.modules():
        for k, ch in parent._modules.items():
            if ch is not None:
                parent_of.setdefault(id(ch), []).append((parent, k))
    nbn, ncv = {}, {}
    for c, bn in pairs:
        nbn[id(bn)] = nbn.get(id(bn), 0) + 1
        ncv[id(c)] = ncv.get(id(c), 0) + 1
    cand, seen = [], set()
    for c, bn in pairs:
        if nbn[id(bn)] == 1 and ncv[id(c)] == 1 and id(bn) not in seen:
            seen.add(id(bn)); cand.append((c, bn))

    def apply(c, bn):
        old_w, old_b = c.weight.data, c.bias
        with torch.no_grad():
            sc = (bn.weight.float() if bn.affine else 1) / torch.sqrt(bn.running_var.float() + bn.eps)
            b = (bn.bias.float() if bn.affine else 0) - bn.running_mean.float() * sc
            c.weight.data = (c.weight.float() * sc.view(-1, 1, 1, 1)).to(c.weight.dtype)
            cb = (c.bias.float() * sc + b) if c.bias is not None else b
            c.bias = nn.Parameter(cb.to(c.weight.dtype))
        act, drop = getattr(bn, "act", None), getattr(bn, "drop", None)
        if act is not None and not isinstance(act, nn.Identity):
            r = act if drop is None or isinstance(drop, nn.Identity) else nn.Sequential(drop, act)
        else:
            r = nn.Identity()
        for parent, k in parent_of.get(id(bn), []):
            parent._modules[k] = r

        def undo():
            c.weight.data, c.bias = old_w, old_b
            for parent, k in parent_of.get(id(bn), []):
                parent._modules[k] = bn
        return undo

    def ok():
        with torch.inference_mode():
            y = model(x0)
        return bool(((y - ref).abs().max() <= 1e-3 * (ref.abs().max() + 1e-6)).item())
    undos = [apply(c, bn) for c, bn in cand]
    if ok():
        return len(cand)
    for u in reversed(undos):  # some conv output has a second consumer (e.g. RepMixer): fold one by one, keep the exact ones
        u()
    n = 0
    for c, bn in cand:
        u = apply(c, bn)
        if ok():
            n += 1
        else:
            u()
    return n


FX_LEAVES = ("LiteMLA",)   # modules with data-dependent Python control flow: kept as eager calls inside the fx graph


def flatten(model):
    """fx symbolic trace (no nn.Module call overhead), dropping nn.Identity calls; modules in FX_LEAVES stay leaf calls;
    module unchanged if untraceable."""
    import torch.fx as fx

    class _Tracer(fx.Tracer):
        def is_leaf_module(self, m, qualname):
            return type(m).__name__ in FX_LEAVES or super().is_leaf_module(m, qualname)
    try:
        graph = _Tracer().trace(model)
        gm = fx.GraphModule(model, graph)
    except Exception as e:  # noqa: BLE001
        print(f"[clf_fast] fx trace failed ({type(e).__name__}: {str(e)[:60]}), eager modules kept")
        return model, False
    mods = dict(gm.named_modules())
    for n in list(gm.graph.nodes):
        if n.op == "call_module" and isinstance(mods.get(n.target), torch.nn.Identity):
            n.replace_all_uses_with(n.args[0])
            gm.graph.erase_node(n)
    gm.graph.lint()
    gm.recompile()
    return gm, True


def fold_bn(model):
    """reparam (timm) + hook BN fold + fx flatten (+ fx conv/bn fuse for anything left that fx can see)."""
    from torch.fx.experimental.optimization import fuse
    model, _ = reparam(model)
    n = fold_bn_hooks(model)
    model, traced = flatten(model)
    if traced:
        try:
            model = fuse(model, inplace=True)
        except Exception:  # noqa: BLE001
            pass
    return model, n


def optimize(model, spec, mode="max"):
    model = model.eval()
    friendly = mode == "friendly"
    be = os.environ.get("CLF_BACKEND", "")
    if be.startswith("fastbb") and spec.family in ("cnn", "dwcnn", "hybrid"):
        from fastbb.cnn import optimize as fb_opt  # exp 006 (reuse; spec after ':' e.g. fastbb:fold-gacc)
        f = fb_opt(model, spec=be.partition(":")[2] or "fold-hacc-gacc")
        return _Fp32Out(f)
    fused = False
    names = {type(m).__name__ for m in model.modules()}
    kind = "convnext" if "ConvNeXtBlock" in names else "swin" if "SwinTransformerStage" in names else spec.family
    if type(model).__name__ == "ShuffleNetV2" and not friendly and FUSED and MB:
        from .shufflenet import build           # one fused kernel per unit, shuffle folded into the channel layout
        small, big = build(model)
        f = Fast(small, gacc=GACC, cl=CL, trace=TRACE, big=big if SHUF_BIG else None)
        f.big_owner = big                        # keeps the big-batch net registered for .to(device)
        f.fused, f.nfused = True, -1
        return f
    if "_DenseBlock" in names and not friendly and FUSED:
        from .densenet import patch_densenet
        patch_densenet(model)          # one buffer per dense block, conv2 writes its slice (plain-torch version was slower)
    if FUSE and kind in ("cnn", "dwcnn", "hybrid"):
        model, fused = fold_bn(model)
    model = model.half()
    if CL:
        model = model.to(memory_format=torch.channels_last)
    if kind == "convnext":
        from .convnext import patch_convnext
        patch_convnext(model, "friendly" if friendly or not FUSED else "max")
    elif kind == "swin":
        from .swin import patch_swin_friendly, patch_swin_max
        patch_swin_friendly(model)
        if not friendly and FUSED:
            patch_swin_max(model)
    if "SiglipEncoderLayer" in names and not friendly and FUSED:
        try:
            from .hf_siglip import patch_hf_siglip
            patch_hf_siglip(model)
        except Exception as e:  # noqa: BLE001
            print(f"[clf_fast] SigLIP fast path unavailable ({type(e).__name__}: {str(e)[:100]})")
    if "EvaBlock" in names and not friendly and FUSED:
        try:
            from .eva import patch_eva
            patch_eva(model)
        except Exception as e:  # noqa: BLE001
            print(f"[clf_fast] EVA fast path unavailable ({type(e).__name__}: {str(e)[:100]})")
    if spec.family == "vit":
        from .vit import ATTN, patch_vit_blocks  # noqa: F401
        if friendly:
            patch_vit_blocks(model, attn="sdpa")
        else:
            patch_vit_blocks(model, attn=ATTN, fused=FUSED)
    gacc = GACC and spec.extra.get("gacc", True)
    big = None
    nfused = 0
    if not friendly and FUSED and isinstance(model, torch.fx.GraphModule) and kind in ("cnn", "dwcnn", "hybrid"):
        try:
            from .fused import fuse_all
            nfused, _ = fuse_all(model)
            if MB:
                from .mbconv import fuse_dwffn, fuse_mbconv
                model.nffn = fuse_dwffn(model) if DWFFN else 0   # dw -> fc1 -> fc2 in one kernel (FastViT)
                model.nxdw = 0
                if XDW:
                    from .mbconv import fuse_xdw
                    model.nxdw = fuse_xdw(model)      # expand + dw in one kernel, project stays CUTLASS
                model.nmb = fuse_mbconv(model) if MBFULL else 0    # fused inverted-residual blocks (csrc/mbconv.cu)
                if "LiteMLA" in names:
                    from .efficientvit import patch_litemla
                    model.nmla = patch_litemla(model)
            if GCONV:
                from .mbconv import fuse_gconv
                model.ngconv = fuse_gconv(model)  # grouped convs (RegNet): own fp16-acc implicit GEMM
                from .mbconv import fuse_dwmult
                model.ndwm = fuse_dwmult(model)   # depthwise with channel multiplier (FastViT patch embed)
            from .r3 import fuse_regnet, to_conv_tc, to_dw2, to_pw
            # depthwise convs with half2 accumulation; not for FastViT (re-parameterized RepMixer activations are too large
            # for fp16 accumulation: 89.5 % instead of 100 % same top-1)
            model.ndw2 = 0 if "RepMixer" in names else to_dw2(model)
            model.nr3 = fuse_regnet(model)        # round 3: gconv2 (+ fused SE pool), sigmoid*x -> se_scale
            model.ntc = to_conv_tc(model)         # small-C KxK convs on the gconv2 kernel
            model.npw = to_pw(model)              # narrow 1x1 convs -> cublasLt with fused bias/relu (runtime size rule)
        except Exception as e:  # noqa: BLE001
            print(f"[clf_fast] fused convs unavailable ({type(e).__name__}: {str(e)[:100]})")
    if not friendly and not nfused and HACC in ("1", "auto") and kind in ("cnn", "dwcnn", "hybrid"):
        big = hacc_copy(model)
        if HACC == "1":
            model = big
    f = Fast(model, gacc=gacc, cl=CL, big=big, trace=TRACE and not friendly)
    f.fused = fused
    f.nfused = nfused
    return f

"""(exp 009 copy of exp 006 fastbb/cnn.py @ 744b4d6; uses the dilation-aware seg_fast/cudnn_conv)
General fast inference for CNN-style backbones (timm / torchvision ResNet, ConvNeXt V1/V2, EfficientNet, MobileNet,
RegNet, Swin's linears, ...): one call on an eval model, no per-size compilation.

    from fastbb.cnn import optimize
    model = optimize(timm.create_model("resnet50", pretrained=True), spec="hacc")
    y = model(x)            # x fp32/fp16 NCHW (any size); runs fp16 channels-last inside, returns what the model returns

What it does (each step optional via spec flags, "-" separated: "fp16" < "fold" < "hacc" < "gacc"):
- fold : eval BatchNorm folded into the preceding conv (found by tracing one tiny forward: BN input == conv output tensor;
         timm BatchNormAct2d keeps its activation), then fp16 weights + channels-last
- hacc : dense convs (groups 1) as fp16-accumulate cuDNN convs (precompiled engines only; fastbb/cudnn_conv.py)
- gacc : nn.Linear (ConvNeXt MLPs, Swin, heads) as fp16-accumulate cuBLAS GEMMs
Default spec "fold-hacc-gacc". Activations stay as the model computes them (fp16).
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cudnn_conv


def _fold_bn(model, sample):
    """Fold eval BatchNorm2d into the conv whose output it directly consumes. Returns the number folded."""
    convs, pairs = {}, []
    hooks = []

    def conv_hook(m, inp, out):
        convs[(out.data_ptr(), tuple(out.shape))] = (m, out)  # keep out alive: no pointer reuse while tracing

    def bn_hook(m, inp, out):
        x = inp[0]
        c = convs.get((x.data_ptr(), tuple(x.shape)))
        if c is not None and c[1] is x:
            pairs.append((c[0], m))

    for m in model.modules():
        if type(m) is nn.Conv2d:
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_var is not None:  # incl. mmengine SyncBatchNorm
            hooks.append(m.register_forward_hook(bn_hook))
    with torch.inference_mode():
        model(sample)
    for h in hooks:
        h.remove()
    seen = set()
    n = 0
    for c, bn in pairs:
        if id(bn) in seen or id(c) in seen:
            continue
        seen.update((id(bn), id(c)))
        with torch.no_grad():
            s = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps) if bn.affine else \
                1 / torch.sqrt(bn.running_var.float() + bn.eps)
            b = (bn.bias.float() if bn.affine else 0) - bn.running_mean.float() * s
            w = c.weight.float() * s.view(-1, 1, 1, 1)
            cb = c.bias.float() * s + b if c.bias is not None else b
            c.weight.data = w.to(c.weight.dtype)
            c.bias = nn.Parameter(cb.to(c.weight.dtype))
        bn._folded = True
        act = getattr(bn, "act", None)
        drop = getattr(bn, "drop", None)
        if act is not None or drop is not None:  # timm BatchNormAct2d: keep drop + act
            bn.forward = (lambda self, x: self.act(self.drop(x))).__get__(bn)
        else:
            bn.forward = (lambda self, x: x).__get__(bn)
        n += 1
    return n


HACC_MIN_MACS = float(__import__("os").environ.get("SEG_HACC_MIN", "3e8"))  # per conv call; below this the fp32-accumulate torch conv (cuDNN, no frontend overhead) is faster


def _sized_forward(self, x):
    if getattr(self, "_cxk", None) is not None:      # max + segx ops (set by cnn_fuse.fuse_epilogues)
        from .cnn_fuse import cx_conv
        return cx_conv(self, x)
    n, _, h, w = x.shape
    sh, sw = self.stride
    macs = n * (h // sh) * (w // sw) * self.weight.numel()
    if macs < HACC_MIN_MACS:
        return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
    return cudnn_conv.conv_half_acc(x, self.weight, self.bias, list(self.padding), list(self.stride), dilation=self.dilation)


@contextlib.contextmanager
def _gacc(on):
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = on
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


class FastCNN(nn.Module):
    def __init__(self, model, flags):
        super().__init__()
        self.model, self.flags = model, flags
        self.traced = [None]     # max path (exp 009 launch-overhead pass): seg_fast.trace.Traced(model), set by the caller

    def forward(self, x, *a, **kw):
        x = x.to(torch.float16)
        x = x.contiguous() if "nchw" in self.flags else x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", enabled=False), _gacc("gacc" in self.flags):
            if self.traced[0] is not None and not a and not kw:
                return self.traced[0](x)
            return self.model(x, *a, **kw)

    def __getattr__(self, name):  # pass-through (forward_features, feature_info, ...)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def _dw_nchw_forward(self, x):
    """depthwise conv in NCHW: cuDNN's channels-last fp16 depthwise kernels are 4-5x slower on sm_120 (dilated 3x3,
    2048 ch at 128x256: 13 ms NHWC vs 2.7 ms NCHW); output goes back to channels-last."""
    y = F.conv2d(nchw(x), self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    return y.contiguous(memory_format=torch.channels_last)


def _dw_kernel_forward(self, x):
    from .ops import dwconv
    return dwconv(x, self)


_NCHW = [None, None]


def nchw(x):
    """x.contiguous(), with a one-entry cache keyed on the tensor object (ASPP-style branches convert the same input once;
    the cached source reference keeps its memory from being reused while cached)."""
    if _NCHW[0] is x:
        return _NCHW[1]
    y = x.contiguous()
    _NCHW[0], _NCHW[1] = (x, y) if y is not x else (None, None)
    return y


def optimize(model, spec="fold-hacc-gacc", sample_hw=(64, 64)):
    flags = set(spec.split("-"))
    model = model.cuda().eval()
    info = {}
    if "fold" in flags:
        info["folded_bn"] = _fold_bn(model, torch.randn(1, 3, *sample_hw, device="cuda"))
    model.half()
    for m in model.modules():
        if "nchw" in flags:                                             # plain NCHW fp16 (e.g. depthwise-heavy nets)
            break
        if isinstance(m, nn.Conv2d) and m.weight.dim() == 4:
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
    info["dw_kernel"] = info["dw_nchw"] = 0
    for m in model.modules():
        if type(m) is nn.Conv2d and m.groups > 1 and m.groups == m.in_channels == m.out_channels \
                and m.padding_mode == "zeros" and not isinstance(m.padding, str):
            if "dwk" in flags and m.in_channels % 2 == 0:              # max: prebuilt NHWC depthwise kernel
                m.forward = _dw_kernel_forward.__get__(m)
                info["dw_kernel"] += 1
            elif max(m.dilation) > 1 and "nchw" not in flags:                                   # plain PyTorch: NCHW for dilated depthwise
                m.weight.data = m.weight.data.contiguous()
                m.forward = _dw_nchw_forward.__get__(m)
                info["dw_nchw"] += 1
    if "hacc" in flags:
        info["hacc_convs"] = cudnn_conv.patch_convs(model)
        for m in model.modules():  # small convs: cuDNN-frontend call overhead > the 2x MMA rate -> torch conv
            if isinstance(m, nn.Conv2d) and m.groups == 1 and "forward" in m.__dict__:
                m.forward = _sized_forward.__get__(m)
    f = FastCNN(model, flags)
    f.info = info
    return f

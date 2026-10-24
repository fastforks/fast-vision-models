"""Conv epilogue fusion for the max CNN path (after seg_fast.cnn.optimize folded BN and patched the convs):
conv (fp16-accumulate cuDNN or torch, no bias) -> one prebuilt kernel `bias_act_h` doing bias (+ residual) (+ ReLU) in place.
Replaces 3-4 memory passes per conv (bias add, ReLU, residual add, ReLU) by one. Patched module types:
- mmcv ConvModule with order (conv, norm, act), folded (or no) norm, ReLU or no activation, no explicit padding layer
- mmseg ResNet BasicBlock / Bottleneck (also used by HRNet branches) without plugins, folded norms
Anything else keeps its forward. Any input size; no per-size work beyond the conv plans of seg_fast.cudnn_conv.
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cudnn_conv
from .cnn import HACC_MIN_MACS
from .ops import cx, kern


def cx_prepare(conv):
    """max + segx ops: choose this conv's C++ path once (shape-independent): "cutlass" = conv_fused; "cutlass_pad" = output
    channels zero-padded to a multiple of 8 (e.g. 150-class heads; the extra outputs are sliced off); dilation via
    conv_fused_d; "torch" = aten conv (+ segx.bias_act_) for C % 8 != 0 (RGB stems) or grouped convs. No cuDNN-frontend path: it is invisible to a
    trace and costs ~15 us CPU per call."""
    if not (type(conv) is nn.Conv2d and conv.padding_mode == "zeros" and not isinstance(conv.padding, str)
            and conv.weight.dtype == torch.half):
        return None
    if conv.groups == 1 and conv.in_channels % 8 == 0:
        if conv.out_channels % 8 == 0:
            conv._cxk = "cutlass"
        else:
            k8 = (conv.out_channels + 7) // 8 * 8
            w = conv.weight.detach()
            wp = torch.zeros((k8,) + tuple(w.shape[1:]), dtype=w.dtype, device=w.device)
            wp[: w.shape[0]] = w
            bp = torch.zeros(k8, dtype=w.dtype, device=w.device)
            if conv.bias is not None:
                bp[: w.shape[0]] = conv.bias.detach()
            conv.register_buffer("_wpad", wp.contiguous(memory_format=torch.channels_last), persistent=False)
            conv.register_buffer("_bpad", bp, persistent=False)
            conv._cxk = "cutlass_pad"
    else:
        conv._cxk = "torch"
    return conv._cxk


def cx_conv(conv, x, relu=False, res=None):
    k = conv._cxk
    if k == "cutlass":
        if conv.dilation[0] == 1 and conv.dilation[1] == 1:
            return torch.ops.segx.conv_fused(x, conv.weight, conv.bias, res, list(conv.padding), list(conv.stride),
                                             1 if relu else 0)
        return torch.ops.segx.conv_fused_d(x, conv.weight, conv.bias, res, list(conv.padding), list(conv.stride),
                                           list(conv.dilation), 1 if relu else 0)
    if k == "cutlass_pad" and res is None:
        y = torch.ops.segx.conv_fused_d(x, conv._wpad, conv._bpad, None, list(conv.padding), list(conv.stride),
                                        list(conv.dilation), 1 if relu else 0)
        return y[:, : conv.out_channels]
    x = x.contiguous(memory_format=torch.channels_last)
    y = F.conv2d(x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
    y = y.contiguous(memory_format=torch.channels_last)
    if conv.bias is None and res is None and not relu:
        return y
    return torch.ops.segx.bias_act_(y, conv.bias, None if res is None else res.contiguous(memory_format=torch.channels_last),
                                    bool(relu))


def bias_act_(y, bias=None, res=None, relu=False):
    """in place y = act(y + bias (+ res)); y NHWC-contiguous fp16 [N, C, H, W] (channels-last)."""
    if cx():
        return torch.ops.segx.bias_act_(y, bias, res, bool(relu))
    C = y.shape[1]
    ok = (y.dtype == torch.float16 and C % 2 == 0 and y.is_contiguous(memory_format=torch.channels_last)
          and (res is None or (res.shape == y.shape and res.dtype == y.dtype
                               and res.is_contiguous(memory_format=torch.channels_last))))
    if not ok:
        if bias is not None:
            y.add_(bias.view(1, -1, 1, 1))
        if res is not None:
            y.add_(res)
        return y.relu_() if relu else y
    n2 = y.numel() // 2
    kern().launch("bias_act_h", ((min((n2 + 255) // 256, 65535 * 8), 1, 1), (256, 1, 1)), y,
                  bias if bias is not None else 0, res if res is not None else 0, n2, C // 2, int(relu))
    return y


def conv_e(conv, x, relu=False, res=None):
    """conv without bias -> fused bias / residual / ReLU epilogue."""
    w = conv.weight
    if conv.groups > 1 and conv.groups == conv.in_channels == conv.out_channels and conv.in_channels % 2 == 0:  # depthwise: prebuilt NHWC kernel incl. bias
        from .ops import dwconv
        y = dwconv(x, conv)
        return bias_act_(y, None, res, relu) if (relu or res is not None) else y
    if getattr(conv, "_cxk", None) is not None:     # max + segx: one CUTLASS launch, conv + bias (+ res) (+ ReLU), from C++
        return cx_conv(conv, x, relu, res)
    x = x.contiguous(memory_format=torch.channels_last)
    n, _, h, wd = x.shape
    macs = n * (h // conv.stride[0]) * (wd // conv.stride[1]) * w.numel()
    if getattr(conv, "_hacc_ok", False) and macs >= HACC_MIN_MACS:
        y = cudnn_conv.conv_half_acc(x, w, None, list(conv.padding), list(conv.stride), dilation=conv.dilation)
    else:
        y = F.conv2d(x, w, None, conv.stride, conv.padding, conv.dilation, conv.groups)
        y = y.contiguous(memory_format=torch.channels_last)
    return bias_act_(y, conv.bias, res, relu)


def _cx_forward(self, x):
    return cx_conv(self, x)


def _plain_conv(m):
    return type(m) is nn.Conv2d and m.padding_mode == "zeros" and not isinstance(m.padding, str)


def _folded(norm):
    return norm is None or getattr(norm, "_folded", False)


def _convmodule_forward(self, x, activate=True, norm=True):
    return conv_e(self.conv, x, relu=activate and self._relu)


def _basic_forward(self, x):
    identity = x if self.downsample is None else self.downsample(x)
    out = conv_e(self.conv1, x, relu=True)
    return conv_e(self.conv2, out, relu=True, res=identity.contiguous(memory_format=torch.channels_last))


def _bottleneck_forward(self, x):
    identity = x if self.downsample is None else self.downsample(x)
    out = conv_e(self.conv1, x, relu=True)
    out = conv_e(self.conv2, out, relu=True)
    return conv_e(self.conv3, out, relu=True, res=identity.contiguous(memory_format=torch.channels_last))


def _utils_basic_forward(self, x):
    """mmseg.models.utils BasicBlock (DDRNet / PIDNet): conv2 + residual + optional ReLU in one epilogue."""
    identity = x if not self.downsample else self.downsample(x)
    out = self.conv1(x)
    return conv_e(self.conv2.conv, out, relu=self._relu_out, res=identity.contiguous(memory_format=torch.channels_last))


def _utils_bottleneck_forward(self, x):
    identity = x if not self.downsample else self.downsample(x)
    out = self.conv2(self.conv1(x))
    return conv_e(self.conv3.conv, out, relu=self._relu_out, res=identity.contiguous(memory_format=torch.channels_last))


def _hrmodule_forward(self, x):
    """mmseg HRModule with fused exchange units (max + segx ops): per output branch i, the j < i downsample chains end in a
    conv whose epilogue adds the running sum (residual input), the j > i 1x1 convs are followed by one upsample + add
    kernel; the final ReLU goes into the last of these ops. Upstream: conv, BN, upsample, resize, add per term + ReLU."""
    if self.num_branches == 1:
        return [self.branches[0](x[0])]
    for i in range(self.num_branches):
        x[i] = self.branches[i](x[i])
    nb = self.num_branches
    out = []
    for i in range(len(self.fuse_layers)):
        downs, ups = list(range(i)), list(range(i + 1, nb))
        y, fresh = x[i], False
        for j in downs:
            h = x[j]
            seq = self.fuse_layers[i][j]
            for k, sub in enumerate(seq):
                if k < len(seq) - 1:
                    h = conv_e(sub[0], h, relu=True)
                else:
                    h = conv_e(sub[0], h, relu=(j == downs[-1] and not ups), res=y)
            y, fresh = h, True
        for j in ups:
            z = conv_e(self.fuse_layers[i][j][0], x[j])
            op = torch.ops.segx.upsample_add_ if fresh else torch.ops.segx.upsample_add
            y, fresh = op(y, z, float(2 ** (j - i)), j == ups[-1]), True
        out.append(y)
    return out


def _upsample_forward(self, x):
    """mmseg Upsample without a fixed size: the same op through scale_factor (output floor(t * s), index scale 1 / s, as
    upstream's size = int(t * s)), so the size is derived at run time (C++ op) instead of frozen into a trace."""
    return F.interpolate(x, None, self.scale_factor, self.mode, self.align_corners)


def _hr_ok(m):
    nb = m.num_branches
    if nb == 1:
        return True
    for i in range(len(m.fuse_layers)):
        for j in range(nb):
            f = m.fuse_layers[i][j]
            if j > i:
                if not (len(f) == 3 and _plain_conv(f[0]) and _folded(f[1]) and type(f[2]).__name__ == "Upsample"
                        and f[2].mode == "bilinear" and not f[2].align_corners and f[2].scale_factor == 2 ** (j - i)):
                    return False
            elif j < i:
                for k, sub in enumerate(f):
                    if not (_plain_conv(sub[0]) and _folded(sub[1]) and len(sub) == (3 if k < len(f) - 1 else 2)):
                        return False
    return True


def _seq_forward(self, x):
    return conv_e(self[0], x, relu=True)


def fuse_epilogues(model):
    use_cx = cx()
    for m in model.modules():                       # which convs may take the fp16-accumulate path (cnn.optimize hacc)
        if _plain_conv(m):
            m._hacc_ok = m.groups == 1 and "forward" in m.__dict__
            if use_cx and m._hacc_ok and cx_prepare(m):
                m.forward = types.MethodType(_cx_forward, m)
    n = {"convmodule": 0, "basic": 0, "bottleneck": 0, "seq": 0, "utils_block": 0, "hrmodule": 0}
    for m in model.modules():
        cname = type(m).__name__
        mod = type(m).__module__
        if type(m) is nn.Sequential and len(m) == 3 and _plain_conv(m[0]) and _folded(m[1]) \
                and isinstance(m[1], nn.modules.batchnorm._BatchNorm) and type(m[2]) is nn.ReLU:
            m.forward = types.MethodType(_seq_forward, m)                    # conv-BN-ReLU stacks (SegNet, stems)
            n["seq"] += 1
        elif cname == "Upsample" and mod.startswith("mmseg.models.utils") and not m.size and use_cx \
                and float(m.scale_factor).is_integer():
            m.forward = types.MethodType(_upsample_forward, m)   # upstream: int(t * scale) sizes (frozen by a trace)
        elif cname == "HRModule" and mod.startswith("mmseg") and use_cx and _hr_ok(m):
            m.forward = types.MethodType(_hrmodule_forward, m)
            n["hrmodule"] += 1
        elif cname == "ConvModule" and mod.startswith("mmcv"):
            act = getattr(m, "activate", None) if m.with_activation else None
            if tuple(m.order) != ("conv", "norm", "act") or m.with_explicit_padding or not _plain_conv(m.conv):
                continue
            if m.with_norm and not _folded(m.norm):
                continue
            if act is not None and type(act) is not nn.ReLU:
                continue
            m._relu = act is not None
            m.forward = types.MethodType(_convmodule_forward, m)
            n["convmodule"] += 1
        elif mod.startswith("mmseg.models.backbones.resnet") and cname in ("BasicBlock", "Bottleneck"):
            if getattr(m, "with_plugins", False) or getattr(m, "with_dcn", False):
                continue
            convs = [m.conv1, m.conv2] + ([m.conv3] if cname == "Bottleneck" else [])
            norms = [m.norm1, m.norm2] + ([m.norm3] if cname == "Bottleneck" else [])
            if not all(_plain_conv(c) for c in convs) or not all(_folded(x) for x in norms):
                continue
            m.forward = types.MethodType(_basic_forward if cname == "BasicBlock" else _bottleneck_forward, m)
            n["basic" if cname == "BasicBlock" else "bottleneck"] += 1
        elif mod.startswith("mmseg.models.utils.basic_block") and cname in ("BasicBlock", "Bottleneck"):
            last = m.conv2 if cname == "BasicBlock" else m.conv3
            act = getattr(m, "act", None)
            if act is not None and type(act) is not nn.ReLU:
                continue
            if type(last).__name__ != "ConvModule" or last.with_activation or (last.with_norm and not _folded(last.norm)) \
                    or tuple(last.order) != ("conv", "norm", "act") or last.with_explicit_padding or not _plain_conv(last.conv):
                continue
            m._relu_out = act is not None
            m.forward = types.MethodType(_utils_basic_forward if cname == "BasicBlock" else _utils_bottleneck_forward, m)
            n["utils_block"] += 1
    return n

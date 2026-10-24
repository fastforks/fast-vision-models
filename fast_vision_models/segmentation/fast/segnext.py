"""SegNeXt (mmseg MSCAN) block-level fast path (round 2), applied on top of the generic CNN optimizer (seg_fast/cnn.py).

MSCAAttention upstream: u = x.clone(); attn = conv0 (dw 5x5); three branches of depthwise strip convs (1x7 -> 7x1,
1x11 -> 11x1, 1x21 -> 21x1); attn + a0 + a1 + a2 (three adds); conv3 (1x1); attn * u.
MSCASpatialAttention: shortcut = x.clone(); ...; + shortcut.  MSCABlock: layer_scale * attn(...) / mlp(...) + x.

uf (plain PyTorch): no clones (inputs are not modified in place), branch sums accumulated in place, block residuals as
addcmul (norm1 computed once).  max: + the strip convs as two prebuilt kernels (seg_kernels.cu strip3_h): the three 1xk
convs in one pass over attn (register-blocked along W), the three kx1 convs + the branch sum + attn in one pass; the Mlp
dwconv + GELU in one kernel.
Not used (bisected on val-500): folding norm1 / norm2 into the 1x1 convs (SEG_SN_BNFOLD=1) and the layer scales into the
fp16 weights -> paired agreement 99.87 -> 99.67 % (fp16-accumulate convs + large folded biases / tiny folded weights).
"""
import os
import types

import torch

from .ops import cx, kern


def _taps(conv):
    """depthwise [C, 1, kh, kw] (one of kh, kw == 1) -> fp16 [k, C] tap-major."""
    w = conv.weight.detach()
    return w.reshape(w.shape[0], -1).t().contiguous().half()


def _attn_forward_uf(self, x):
    attn = self.conv0(x)
    s = self.conv0_2(self.conv0_1(attn))
    s += self.conv1_2(self.conv1_1(attn))
    s += self.conv2_2(self.conv2_1(attn))
    s += attn
    return self.conv3(s) * x


def _attn_forward_max(self, x):
    attn = self.conv0(x)
    N, C, H, W = attn.shape
    if not attn.is_contiguous(memory_format=torch.channels_last) or C % 2 or attn.dtype != torch.float16:
        return _attn_forward_uf(self, x)
    h, v = self._h, self._v
    if cx():                                     # C++ ops (traceable, ~3x less CPU per launch)
        y = torch.ops.segx.strip3_split(attn, *h["w"], *h["bo"], 0)
        s = torch.ops.segx.strip3_sum(y[0], y[1], y[2], *v["w"], *v["bo"], attn, 1)
        return self.conv3(s) * x
    y = [torch.empty_like(attn) for _ in range(3)]
    grid, block = kern().linear(N * W * ((H + 7) // 8) * (C // 2))
    kern().launch_packed("strip3_h", grid, block, attn, 0, 0, *h["w"], *h["b"], 0, *y, N, H, W, C, *h["k"], 0, 0)
    s = torch.empty_like(attn)
    kern().launch_packed("strip3_h", grid, block, *y, *v["w"], *v["b"], attn, s, 0, 0, N, H, W, C, *v["k"], 1, 1)
    return self.conv3(s) * x


def _spatial_forward(self, x):
    """without the shortcut (added by the block: norm1 is folded into proj_1, its output never materialized)."""
    return self.proj_2(self.spatial_gating_unit(self.activation(self.proj_1(x))))


def _mlp_forward_max(self, x):
    from .ops import dwconv
    x = self.fc1(x)
    if x.is_contiguous(memory_format=torch.channels_last) and x.dtype == torch.float16 and x.shape[1] % 2 == 0:
        return self.fc2(dwconv(x, self.dwconv, act=1))   # dwconv + exact GELU in one kernel
    return self.fc2(self.act(self.dwconv(x)))


def _block_forward(self, x, H, W):
    """x + ls1 * (proj_2(...) + norm1(x)) + ...  with norm1 = s * x + t folded: proj_1 takes x (norm1 folded into its
    weights), t into proj_2's bias -> x * (1 + ls1 * s) + ls1 * proj_2'(...); norm2 folded into fc1."""
    B, N, C = x.shape
    x = x.permute(0, 2, 1).view(B, C, H, W)
    if not self._bnfold:        # default: norms kept (folding them cost agreement 99.87 -> 99.67 %, fp16 accumulation)
        n1 = self.norm1(x)
        x = torch.addcmul(x, self.attn(n1).add_(n1), self._l1h)
        return torch.addcmul(x, self.mlp(self.norm2(x)), self._l2h).view(B, C, N).permute(0, 2, 1)
    a = self.attn(x)
    if self._fused and a.is_contiguous(memory_format=torch.channels_last) and x.is_contiguous(memory_format=torch.channels_last):
        y = torch.empty_like(a)
        n2 = a.numel() // 2
        grid, block = kern().linear(n2)
        kern().launch_packed("scale_add_h", grid, block, a, x, self._k1, self._l1, y, n2, C // 2)
        m = self.mlp(y)
        x = torch.empty_like(m)
        kern().launch_packed("scale_add_h", grid, block, m, y, self._one, self._l2, x, n2, C // 2)
    else:
        x = torch.addcmul(x, torch.addcmul(a, x, self._s1h), self._l1h)
        x = torch.addcmul(x, self.mlp(x), self._l2h)
    return x.view(B, C, N).permute(0, 2, 1)


def _bn_st(bn):
    s = bn.weight.detach().float() / torch.sqrt(bn.running_var.float() + bn.eps)
    return s, bn.bias.detach().float() - bn.running_mean.float() * s


def _fold_bn_into_1x1(bn, conv):
    """conv(bn(x)) = conv'(x) for a 1x1 conv (exact algebra in fp32)."""
    s, t = _bn_st(bn)
    with torch.no_grad():
        w = conv.weight.float()
        b = (conv.bias.float() if conv.bias is not None else 0) + w[:, :, 0, 0] @ t
        conv.weight.copy_((w * s.view(1, -1, 1, 1)).to(conv.weight.dtype))
        conv.bias = torch.nn.Parameter(b.to(conv.weight.dtype))


def _fold_scale(conv, s):
    with torch.no_grad():
        conv.weight.mul_(s.view(-1, 1, 1, 1).to(conv.weight.dtype))
        if conv.bias is not None:
            conv.bias.mul_(s.to(conv.bias.dtype))


def patch_segnext(model, fused):
    """call BEFORE seg_fast.cnn.optimize (weights still fp32, convs not yet replaced); returns counts."""
    n = {"msca": 0, "blocks": 0}
    use_kernel = fused and os.environ.get("SEG_STRIP", "1") == "1"
    for m in model.modules():
        name = type(m).__name__
        if name == "MSCABlock" and os.environ.get("SEG_SN_BNFOLD", "0") != "1":
            m._bnfold = False
            m._l1h, m._l2h = (v.detach().float().view(1, -1, 1, 1).cuda().half() for v in (m.layer_scale_1, m.layer_scale_2))
            m.forward = types.MethodType(_block_forward, m)
            if fused and os.environ.get("SEG_SN_GELU", "1") == "1":
                m.mlp.forward = types.MethodType(_mlp_forward_max, m.mlp)
            n["blocks"] += 1
        elif name == "MSCABlock":
            m._bnfold = True
            ls1 = m.layer_scale_1.detach().float()
            s1, t1 = _bn_st(m.norm1)
            _fold_bn_into_1x1(m.norm1, m.attn.proj_1)
            ls2 = m.layer_scale_2.detach().float()
            with torch.no_grad():                                  # shortcut's BN shift t1 -> proj_2 bias
                m.attn.proj_2.bias.add_(t1.to(m.attn.proj_2.bias.dtype))
            # layer scales (~1e-2) are NOT folded into fp16 weights (subnormal products cost accuracy: val-500
            # agreement 99.87 -> 99.66 % when folded); uf: two addcmuls, max: x * (1 + ls1 s1) + a * ls1 in one kernel
            m._s1h, m._l1h, m._l2h = (v.view(1, -1, 1, 1).cuda().half() for v in (s1, ls1, ls2))
            m._k1, m._l1 = (1 + ls1 * s1).contiguous().cuda(), ls1.contiguous().cuda()
            m._one, m._l2 = torch.ones_like(ls2).cuda(), ls2.contiguous().cuda()
            m._fused = fused
            _fold_bn_into_1x1(m.norm2, m.mlp.fc1)
            if fused and os.environ.get("SEG_SN_GELU", "1") == "1":
                m.mlp.forward = types.MethodType(_mlp_forward_max, m.mlp)
            m.forward = types.MethodType(_block_forward, m)
            n["blocks"] += 1
        elif name == "MSCASpatialAttention":
            m.forward = types.MethodType(_spatial_forward, m)
        elif name == "MSCAAttention":
            if use_kernel:
                hs = [m.conv0_1, m.conv1_1, m.conv2_1]
                vs = [m.conv0_2, m.conv1_2, m.conv2_2]
                assert all(c.kernel_size[0] == 1 and c.padding == (0, c.kernel_size[1] // 2) for c in hs)
                assert all(c.kernel_size[1] == 1 and c.padding == (c.kernel_size[0] // 2, 0) for c in vs)
                m._h = {"w": [None] * 3, "b": [None] * 3, "k": [c.kernel_size[1] for c in hs], "mods": hs}
                m._v = {"w": [None] * 3, "b": [None] * 3, "k": [c.kernel_size[0] for c in vs], "mods": vs}
                m.forward = types.MethodType(_attn_forward_max, m)
            else:
                m.forward = types.MethodType(_attn_forward_uf, m)
            n["msca"] += 1
    return n


def finalize_segnext(model):
    """after optimize (fp16 weights): build the tap-major strip weights for the kernel path."""
    for m in model.modules():
        if type(m).__name__ == "MSCAAttention" and hasattr(m, "_h"):
            for d in (m._h, m._v):
                d["w"] = [_taps(c) for c in d["mods"]]
                d["b"] = [c.bias.detach().half().contiguous() if c.bias is not None else 0 for c in d["mods"]]
                d["bo"] = [b if isinstance(b, torch.Tensor) else None for b in d["b"]]

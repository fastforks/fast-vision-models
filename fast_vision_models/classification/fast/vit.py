"""Fast block forward for timm VisionTransformer-style blocks (ViT, DeiT, DINOv2 timm port, SigLIP ViT): same math,
fp16, attention through exp 006 fastbb's prebuilt flash kernel for head_dim 64 (q/k/v read straight from the packed qkv
GEMM output, output written as [B, N, C] for the projection: no permutes / copies), SDPA otherwise; LayerScale + residual
as one addcmul. Any batch / token count, nothing compiled per size.

CLF_ATTN = fh (default: fp32 QK^T, fp16-accumulate PV) | hh | ff | sdpa
"""
import os
import types

import torch
import torch.nn.functional as F

ATTN = os.environ.get("CLF_ATTN", "sdpa")   # fastbb fh: +1.7 % b64, +0.4 ms b1 (ctypes launch) -> sdpa default
_att = [None]


def _attention():
    if _att[0] is None:
        from fastbb.attention import attention
        _att[0] = attention
    return _att[0]


def _gamma(ls):
    return getattr(ls, "gamma", None)


def block_forward(self, x, attn_mask=None):
    a = self.attn
    B, N, C = x.shape
    H = a.num_heads
    D = C // H
    qkv = F.linear(self.norm1(x), a.qkv.weight, a.qkv.bias).view(B, N, 3, H, D)
    va = getattr(self, "_clf_attn", ATTN)
    if va != "sdpa" and D == 64 and attn_mask is None:
        q, k, v = qkv.unbind(2)
        o = _attention()(q, k, v, scale=a.scale, variant=va).view(B, N, C)
    else:
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=a.scale).transpose(1, 2).reshape(B, N, C)
    if not isinstance(getattr(a, "norm", torch.nn.Identity()), torch.nn.Identity):
        o = a.norm(o)
    o = F.linear(o, a.proj.weight, a.proj.bias)
    g = _gamma(self.ls1)
    x = torch.addcmul(x, o, g) if g is not None else x + o
    m = self.mlp
    h = m.act(F.linear(self.norm2(x), m.fc1.weight, m.fc1.bias))
    if not isinstance(getattr(m, "norm", torch.nn.Identity()), torch.nn.Identity):
        h = m.norm(h)
    o = F.linear(h, m.fc2.weight, m.fc2.bias)
    g = _gamma(self.ls2)
    return x.addcmul_(o, g) if g is not None else x.add_(o)


def _lin(x, w4, b, res=None, act=0):
    """y = act(x @ w^T + b (+ res)) through the fused CUTLASS conv (1x1 conv over [1, K, M, 1], fp16 accumulate)."""
    sh = x.shape
    K = sh[-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    r = None if res is None else res.reshape(M, -1).view(1, M, 1, -1).permute(0, 3, 1, 2)
    y = torch.ops.clf.conv_fused(x2.view(1, M, 1, K).permute(0, 3, 1, 2), w4, b, r, [0, 0], [1, 1], act, -1)
    return y.permute(0, 2, 3, 1).reshape(*sh[:-1], -1)


def block_forward_fused(self, x, attn_mask=None):
    """max path: qkv / proj+residual / fc1+GELU / fc2+residual as fused CUTLASS GEMMs (LayerScale folded into proj / fc2)."""
    a = self.attn
    B, N, C = x.shape
    H = a.num_heads
    D = C // H
    qkv = _lin(self.norm1(x), self._w_qkv, a.qkv.bias).view(B, N, 3, H, D)
    va = getattr(self, "_clf_attn", ATTN)
    if va not in ("sdpa", "timm") and D == 64 and attn_mask is None:
        q, k, v = qkv.unbind(2)
        o = _attention()(q, k, v, scale=a.scale, variant=va).view(B, N, C)
    else:
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=a.scale).transpose(1, 2).reshape(B, N, C)
    x = _lin(o, self._w_proj, self._b_proj, res=x)
    h = _lin(self.norm2(x), self._w_fc1, self.mlp.fc1.bias, act=self._act)
    return _lin(h, self._w_fc2, self._b_fc2, res=x)


def _fusable(mod):
    a, m = mod.attn, mod.mlp
    if not isinstance(getattr(a, "norm", torch.nn.Identity()), torch.nn.Identity):
        return 0
    if not isinstance(getattr(m, "norm", torch.nn.Identity()), torch.nn.Identity):
        return 0
    act = type(m.act).__name__
    if act == "GELU":
        return {"none": 5, "tanh": 6}.get(getattr(m.act, "approximate", "none"), 0)
    if act == "GELUTanh":
        return 6
    return 0


def _prep_fused(mod):
    a, m = mod.attn, mod.mlp
    w4 = lambda w: w.detach().reshape(w.shape[0], w.shape[1], 1, 1).contiguous(memory_format=torch.channels_last)  # noqa: E731
    g1, g2 = _gamma(mod.ls1), _gamma(mod.ls2)
    with torch.no_grad():
        wp, bp = a.proj.weight.float(), a.proj.bias.float()
        w2, b2 = m.fc2.weight.float(), m.fc2.bias.float()
        if g1 is not None:
            wp, bp = wp * g1.float()[:, None], bp * g1.float()
        if g2 is not None:
            w2, b2 = w2 * g2.float()[:, None], b2 * g2.float()
        dt = a.qkv.weight.dtype
        mod.register_buffer("_w_qkv", w4(a.qkv.weight), persistent=False)
        mod.register_buffer("_w_proj", w4(wp.to(dt)), persistent=False)
        mod.register_buffer("_b_proj", bp.to(dt), persistent=False)
        mod.register_buffer("_w_fc1", w4(m.fc1.weight), persistent=False)
        mod.register_buffer("_w_fc2", w4(w2.to(dt)), persistent=False)
        mod.register_buffer("_b_fc2", b2.to(dt), persistent=False)


def patch_vit_blocks(model, attn=None, fused=False):
    """Replace forward of every plain timm Block (qkv attention, Mlp, no q/k norm) with block_forward. Returns count."""
    n = 0
    for mod in model.modules():
        if type(mod).__name__ != "Block" or not hasattr(mod, "attn") or not hasattr(mod.attn, "qkv"):
            continue
        a = mod.attn
        if not isinstance(getattr(a, "q_norm", torch.nn.Identity()), torch.nn.Identity):
            continue
        if not (hasattr(mod.mlp, "fc1") and hasattr(mod.mlp, "fc2")):
            continue
        mod._clf_attn = attn or ATTN
        act = _fusable(mod) if fused else 0
        if act and mod.attn.qkv.bias is not None and mod.mlp.fc1.bias is not None:
            from .fused import load
            load()
            _prep_fused(mod)
            mod._act = act
            mod.forward = types.MethodType(block_forward_fused, mod)
        else:
            mod.forward = types.MethodType(block_forward, mod)
        n += 1
    return n

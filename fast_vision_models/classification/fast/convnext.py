"""ConvNeXt V1 / V2 (timm ConvNeXtBlock, Linear-MLP variant) fast block forward.

Both paths keep the block in channels-last memory, where x.permute(0, 2, 3, 1) is a free contiguous [N, H, W, C] view:
  friendly: plain PyTorch, same ops as timm minus copies: LayerScale gamma folded into fc2 (weight transform), GRN applied as
            ONE addcmul  h * (1 + w * Nx) + b  (= timm's  h + b + w * (h * Nx)), residual add on the permuted-back view.
  max:      depthwise 7x7 via clf.dwconv, fc1 + GELU and fc2 + residual as fused CUTLASS GEMMs.
"""
import types

import torch
import torch.nn.functional as F


def _prep(blk):
    m = blk.mlp
    with torch.no_grad():
        w2, b2 = m.fc2.weight.float(), m.fc2.bias.float()
        if blk.gamma is not None:
            g = blk.gamma.float()
            w2, b2 = w2 * g[:, None], b2 * g
        dt = m.fc2.weight.dtype
        blk.register_buffer("_w2", w2.to(dt).contiguous(), persistent=False)
        blk.register_buffer("_b2", b2.to(dt), persistent=False)
        grn = getattr(m, "grn", None)
        if grn is not None:
            blk._grn_eps = grn.eps


def _grn(blk, h):
    """h: [N, H, W, C]; timm GlobalResponseNorm (channels-last) as one addcmul."""
    g = blk.mlp.grn
    gx = h.norm(p=2, dim=(1, 2), keepdim=True).float()                   # [N, 1, 1, C] (fp32 accumulation inside)
    nx = gx / (gx.mean(dim=-1, keepdim=True) + blk._grn_eps)
    s = (1 + g.weight.float() * nx).to(h.dtype)
    return torch.addcmul(g.bias.view(1, 1, 1, -1), h, s)


def _fwd_friendly(self, x):
    shortcut = x
    h = self.conv_dw(x).permute(0, 2, 3, 1)
    h = self.norm(h)
    m = self.mlp
    h = m.act(m.fc1(h))
    if hasattr(self, "_grn_eps"):
        h = _grn(self, h)
    h = F.linear(h, self._w2, self._b2)
    return shortcut + h.permute(0, 3, 1, 2)


def _fwd_max(self, x):
    from .vit import _lin
    c = self.conv_dw
    shortcut = x.contiguous(memory_format=torch.channels_last)
    h = torch.ops.clf.dwconv(shortcut, self._w_rsc, c.bias, list(c.stride), list(c.padding), 0).permute(0, 2, 3, 1)
    h = self.norm(h)
    m = self.mlp
    h = _lin(h, self._w1, m.fc1.bias, act=5)
    if hasattr(self, "_grn_eps"):
        h = _grn(self, h)
    res = shortcut.permute(0, 2, 3, 1)                                    # contiguous [N, H, W, C] view
    return _lin(h, self._w2_4, self._b2, res=res).permute(0, 3, 1, 2)


def _ok(blk):
    if type(blk).__name__ != "ConvNeXtBlock":
        return False
    m = blk.mlp
    return ( not blk.use_conv_mlp and isinstance(blk.shortcut, torch.nn.Identity)
            and hasattr(m, "fc1") and hasattr(m, "fc2") and type(m.act).__name__ == "GELU"
            and getattr(m.act, "approximate", "none") == "none" and m.fc1.bias is not None and m.fc2.bias is not None
            and isinstance(getattr(m, "norm", torch.nn.Identity()), torch.nn.Identity))


def patch_convnext(model, mode="friendly"):
    n = 0
    for blk in model.modules():
        if not _ok(blk):
            continue
        _prep(blk)
        c = blk.conv_dw
        dw_ok = (c.groups == c.in_channels == c.out_channels and c.in_channels % 8 == 0 and c.dilation == (1, 1)
                 and not isinstance(c.padding, str))
        if mode == "max" and dw_ok:
            from .fused import load
            load()
            w4 = lambda w: w.detach().reshape(w.shape[0], w.shape[1], 1, 1).contiguous(memory_format=torch.channels_last)  # noqa: E731
            blk.register_buffer("_w_rsc", c.weight.data[:, 0].permute(1, 2, 0).contiguous(), persistent=False)
            blk.register_buffer("_w1", w4(blk.mlp.fc1.weight), persistent=False)
            blk.register_buffer("_w2_4", w4(blk._w2), persistent=False)
            blk.forward = types.MethodType(_fwd_max, blk)
        else:
            blk.forward = types.MethodType(_fwd_friendly, blk)
        n += 1
    return n

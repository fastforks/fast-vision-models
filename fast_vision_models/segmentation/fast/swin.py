"""Swin layer fast path for the HF Swin backbones of Mask2Former / OneFormer (round 2).

HF SwinLayer per block: LN -> pad to /window -> roll (shifted blocks) -> window_partition (.contiguous) -> attention ->
window_reverse (.contiguous) -> roll back -> crop (.contiguous) -> + shortcut; the shift mask is rebuilt every call and
added to the relative-position bias for ALL windows ([nW, heads, T, T], 4.5x the activation size per shifted block).

Here (any input size; the index tables are tiny int tables cached per feature-map size, not compiled code):
- pad + roll + partition = ONE gather with a precomputed index (sentinel -> zero row, HF pads after the LayerNorm), and
  reverse + roll back + crop = ONE gather with the inverse index;
- window order: interior windows first, then the border windows (last window row / column) -- only those can mix shift
  regions, so the interior runs SDPA with the broadcast relative-position bias and only the border windows get
  bias + shift mask (exact: interior windows have mask 0 everywhere);
- uf (plain PyTorch, ONNX Gather): LN, cat zero row, index_select / index_select + residual add;
- max: two prebuilt kernels: ln_gather_h (LayerNorm + gather, fp32 statistics) and gather_add_ln_h (inverse gather +
  residual add + layernorm_after in one pass).
"""
import types

import torch
import torch.nn.functional as F

from .ops import kern

_GEOM = {}


class _Geom:
    pass


def _geom(H, W, ws, s, device):
    key = (H, W, ws, s, str(device))
    g = _GEOM.get(key)
    if g is not None:
        return g
    Hp, Wp = -(-H // ws) * ws, -(-W // ws) * ws
    nWh, nWw = Hp // ws, Wp // ws
    WY, WX = torch.meshgrid(torch.arange(nWh), torch.arange(nWw), indexing="ij")
    WY, WX = WY.flatten(), WX.flatten()
    if s > 0:
        border = (WY == nWh - 1) | (WX == nWw - 1)
        order = torch.cat([torch.nonzero(~border).flatten(), torch.nonzero(border).flatten()])
        n_int = int((~border).sum())
    else:
        order = torch.arange(nWh * nWw)
        n_int = nWh * nWw
    WY, WX = WY[order], WX[order]
    T = ws * ws
    iy, ix = torch.arange(ws).repeat_interleave(ws), torch.arange(ws).repeat(ws)
    Y, X = WY[:, None] * ws + iy[None], WX[:, None] * ws + ix[None]           # rolled padded-grid coordinates [nW, T]
    sy, sx = (Y + s) % Hp, (X + s) % Wp                                       # torch.roll(-s): out[i] = in[i + s]
    valid = ((sy < H) & (sx < W)).flatten()
    src = (sy * W + sx).flatten()
    N, R = H * W, nWh * nWw * T
    g = _Geom()
    g.nW, g.T, g.R, g.n_int = nWh * nWw, T, R, n_int
    g.idx = torch.where(valid, src, torch.full_like(src, N)).to(device)                  # uf: sentinel row N (zeros)
    g.idx_k = torch.where(valid, src, torch.full_like(src, -1)).to(torch.int32).to(device)  # kernel: -1 -> zeros
    inv = torch.empty(N, dtype=torch.long)
    inv[src[valid]] = torch.arange(R)[valid]
    g.inv, g.inv_k = inv.to(device), inv.to(torch.int32).to(device)
    g.mask = None
    if s > 0 and n_int < g.nW:                                                # HF get_attn_mask regions, border windows
        reg = ((Y >= Hp - ws).long() + (Y >= Hp - s).long()) * 3 + (X >= Wp - ws).long() + (X >= Wp - s).long()
        rb = reg[n_int:]
        g.mask = torch.where(rb[:, None, :] != rb[:, :, None], -100.0, 0.0).to(device)  # [nb, T(q), T(k)] fp32
    _GEOM[key] = g
    return g


def _layer_forward(self, hidden_states, input_dimensions, always_partition=False, **kwargs):
    H, W = input_dimensions
    ws, s = self._ws0, self._shift0
    B, N, C = hidden_states.shape
    if (min(H, W) <= ws and not always_partition) or B != 1:
        # small maps: HF clamps the window (and mutates the module) -> run HF, then restore the original sizes
        self.window_size, self.shift_size = ws, s
        out = self._orig_forward(hidden_states, input_dimensions, always_partition, **kwargs)
        self.window_size, self.shift_size = ws, s
        return out
    g = _geom(H, W, ws, s, hidden_states.device)
    a = self.attention
    hd = a.head_dim
    nh = C // hd
    ln1, ln2 = self.layernorm_before, self.layernorm_after
    x = hidden_states.reshape(N, C)
    if self._fused:
        win = torch.empty((g.R, C), device=x.device, dtype=x.dtype)
        kern().launch("ln_gather_h", ((min((g.R + 7) // 8, 65535 * 8), 1, 1), (256, 1, 1)), x, g.idx_k, ln1.weight,
                      ln1.bias, win, g.R, C, float(ln1.eps))
    else:
        xl = F.layer_norm(x, (C,), ln1.weight, ln1.bias, ln1.eps)
        win = torch.cat([xl, xl.new_zeros(1, C)]).index_select(0, g.idx)
    qkv = F.linear(win, a._qkv_w, a._qkv_b).view(g.nW, g.T, 3, nh, hd).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    rpb = a._rpb.get(g.T)
    if rpb is None:
        rpb = a._rpb[g.T] = a.relative_position_bias().to(x.dtype)            # [1, nh, T, T]
    ni = g.n_int
    if g.mask is None:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=rpb.expand(g.nW, -1, -1, -1), scale=a.scaling)
    else:
        o1 = F.scaled_dot_product_attention(q[:ni], k[:ni], v[:ni], attn_mask=rpb.expand(ni, -1, -1, -1), scale=a.scaling) \
            if ni else None
        bm = rpb + g.mask[:, None].to(x.dtype)
        o2 = F.scaled_dot_product_attention(q[ni:], k[ni:], v[ni:], attn_mask=bm, scale=a.scaling)
        o = torch.cat([o1, o2]) if o1 is not None else o2
    o = F.linear(o.transpose(1, 2).reshape(g.R, C), a.o_proj.weight, a.o_proj.bias)
    if self._fused:
        h = torch.empty((N, C), device=x.device, dtype=x.dtype)
        y = torch.empty_like(h)
        kern().launch("gather_add_ln_h", ((min((N + 7) // 8, 65535 * 8), 1, 1), (256, 1, 1)), o, g.inv_k, x, ln2.weight,
                      ln2.bias, h, y, N, C, float(ln2.eps))
    else:
        h = x + o.index_select(0, g.inv)
        y = F.layer_norm(h, (C,), ln2.weight, ln2.bias, ln2.eps)
    out = self.mlp(y) + h
    return out.view(B, N, C), None


def patch_swin_layers(backbone, fused):
    n = 0
    for m in backbone.modules():
        if m.__class__.__name__ == "SwinLayer" and hasattr(m.attention, "_qkv_w"):
            m._ws0, m._shift0, m._fused = int(m.window_size), int(m.shift_size), fused
            m._orig_forward = m.forward
            m.forward = types.MethodType(_layer_forward, m)
            n += 1
    return n

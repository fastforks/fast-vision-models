"""Swin Transformer (timm SwinTransformerStage / Block) fast paths.

friendly: timm's own SDPA attention path (`fused_attn = True`, plain PyTorch, exportable) instead of the manual q@k^T path.

max (`patch_swin_max`): the residual stream of a stage is kept in a window-major token order
    t = n * nW + w      (n = position inside the window, w = window index, after the cyclic shift)
so that
  - q, k, v come from three fused GEMMs as [B, N, nW*heads, D] (BMHK) without copies, and one SDPA call over
    nW*heads "heads" takes relative-position bias + shift mask as a broadcast [1, nW*heads, N, N] bias (precomputed),
  - proj + residual, fc1 + GELU, fc2 + residual are fused CUTLASS GEMMs (token-wise ops do not care about order),
  - roll + window partition / reverse become ONE gather per block (order of block i -> order of block i+1).
Indices / biases are built once per (stage, H, W) on the device (no per-size compilation). Falls back to the timm forward
when H or W is not a multiple of the window size.
"""
import types

import torch
import torch.nn.functional as F


def patch_swin_friendly(model):
    n = 0
    for m in model.modules():
        if type(m).__name__ == "WindowAttention" and hasattr(m, "fused_attn"):
            m.fused_attn = True
            n += 1
    return n


def _block_order(H, W, ws, shift, device):
    """token order of one block: idx[t] = raster index (h * W + w) of the token at stream position t."""
    wh, ww = ws
    sh, sw = shift
    nWh, nWw = H // wh, W // ww
    t = torch.arange(H * W, device=device)
    n, wi = t // (nWh * nWw), t % (nWh * nWw)
    ny, nx = n // ww, n % ww
    wy, wx = wi // nWw, wi % nWw
    r, c = wy * wh + ny, wx * ww + nx
    return ((r + sh) % H) * W + (c + sw) % W


def _bias(blk, H, W, device, dtype):
    a = blk.attn
    N = blk.window_area
    nW = (H // blk.window_size[0]) * (W // blk.window_size[1])
    rel = a._get_rel_pos_bias()[0].float()                     # [heads, N, N]
    heads = rel.shape[0]
    b = rel.unsqueeze(0).expand(nW, heads, N, N)
    if any(blk.shift_size):
        m = blk.attn_mask
        if m is None or m.shape[0] != nW:
            m = blk.get_attn_mask(torch.zeros(1, H, W, 1, device=device))
        b = b + m.float().view(nW, 1, N, N)
    Np = -(-N // 8) * 8                                         # 16-byte aligned rows for the efficient-attention kernel
    out = torch.zeros(1, nW * heads, N, Np, device=device, dtype=dtype)
    out[..., :N] = b.reshape(1, nW * heads, N, N)
    return out, nW              # padded [1, nW*heads, N, Np]; callers slice [..., :N]


def _w4(w):
    return w.detach().reshape(w.shape[0], w.shape[1], 1, 1).contiguous(memory_format=torch.channels_last)


def _prep_block(blk):
    from .vit import _lin  # noqa: F401  (loads the extension)
    a, m = blk.attn, blk.mlp
    C = a.qkv.weight.shape[1]
    W, b = a.qkv.weight, a.qkv.bias
    bufs = {"_wq": _w4(W[:C]), "_wk": _w4(W[C:2 * C]), "_wv": _w4(W[2 * C:]),
            "_bq": b.detach()[:C].contiguous(), "_bk": b.detach()[C:2 * C].contiguous(), "_bv": b.detach()[2 * C:].contiguous(),
            "_wp": _w4(a.proj.weight), "_w1": _w4(m.fc1.weight), "_w2": _w4(m.fc2.weight)}
    for k, v in bufs.items():
        blk.register_buffer(k, v, persistent=False)      # buffers: moved to the GPU with the model


def _block_fwd(blk, x, bias, nW):
    """x: [B, H*W, C] in this block's window-major order -> same order."""
    from .vit import _lin
    a, m = blk.attn, blk.mlp
    B, M, C = x.shape
    heads = a.num_heads
    D = C // heads
    N = M // nW
    h = blk.norm1(x)
    q = _lin(h, blk._wq, blk._bq).view(B, N, nW * heads, D).transpose(1, 2)
    k = _lin(h, blk._wk, blk._bk).view(B, N, nW * heads, D).transpose(1, 2)
    v = _lin(h, blk._wv, blk._bv).view(B, N, nW * heads, D).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=a.scale).transpose(1, 2).reshape(B, M, C)
    x = _lin(o, blk._wp, a.proj.bias, res=x)
    h = _lin(blk.norm2(x), blk._w1, m.fc1.bias, act=5)
    return _lin(h, blk._w2, m.fc2.bias, res=x)


def _stage_forward(self, x):
    x = self.downsample(x)
    B, H, W, C = x.shape
    blocks = list(self.blocks)
    if (H, W) != self._clf_hw:                     # other resolution: timm path
        for blk in blocks:
            x = blk(x)
        return x
    x = x.reshape(B, H * W, C).index_select(1, self._clf_first)
    for i, blk in enumerate(blocks):
        x = _block_fwd(blk, x, blk._clf_bias[..., :blk.window_area], blk._clf_nw)
        x = x.index_select(1, blk._clf_next)
    return x.view(B, H, W, C)


def _prep_stage(st):
    """window orders / permutations / attention biases for the stage's fixed input resolution, as buffers (built once at
    load on the CPU, moved with the model)."""
    blocks = list(st.blocks)
    H, W = blocks[0].input_resolution
    if any(H % b.window_size[0] or W % b.window_size[1] for b in blocks):
        return False
    orders = [_block_order(H, W, b.window_size, b.shift_size, "cpu") for b in blocks]
    invs = [torch.argsort(o) for o in orders]
    st.register_buffer("_clf_first", orders[0], persistent=False)
    for i, b in enumerate(blocks):
        nxt = invs[i][orders[i + 1]] if i < len(blocks) - 1 else invs[-1]
        b.register_buffer("_clf_next", nxt, persistent=False)
        bias, nW = _bias(b, H, W, "cpu", b.attn.qkv.weight.dtype)
        b.register_buffer("_clf_bias", bias, persistent=False)
        b._clf_nw = nW
    st._clf_hw = (H, W)
    return True


def patch_swin_max(model):
    from .fused import load
    load()
    n = 0
    for st in model.modules():
        if type(st).__name__ != "SwinTransformerStage":
            continue
        ok = all(type(b).__name__ == "SwinTransformerBlock" and type(b.mlp.act).__name__ == "GELU"
                 and getattr(b.mlp.act, "approximate", "none") == "none" and b.attn.qkv.bias is not None
                 for b in st.blocks)
        if not ok or not _prep_stage(st):
            continue
        for b in st.blocks:
            _prep_block(b)
        st.forward = types.MethodType(_stage_forward, st)
        n += 1
    return n

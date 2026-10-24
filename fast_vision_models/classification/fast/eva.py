"""EVA-02 (timm EvaBlock: separate q/k/v projections, 2-D RoPE, SwiGLU MLP with inner LayerNorm) max path:
  qkv as ONE fused CUTLASS GEMM (k has no bias -> zeros), RoPE for q and k in one kernel straight from the GEMM output
  (clfx.rope_qk), SDPA, proj + residual fused, fc1_g | fc1_x as ONE GEMM (hidden padded to a multiple of 8),
  SiLU * x + inner LayerNorm in one row kernel (clfx.swiglu_ln), fc2 + residual fused. LayerScale gammas folded.
"""
import types

import torch
import torch.nn.functional as F


def _w4(w):
    return w.detach().reshape(w.shape[0], w.shape[1], 1, 1).contiguous(memory_format=torch.channels_last)


def _prep(blk):
    a, m = blk.attn, blk.mlp
    C = a.q_proj.weight.shape[1]
    dt = a.q_proj.weight.dtype
    dev = a.q_proj.weight.device
    z = torch.zeros(C, dtype=dt, device=dev)
    wqkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).detach()
    bqkv = torch.cat([a.q_proj.bias if a.q_proj.bias is not None else z, a.k_proj.bias if a.k_proj.bias is not None else z,
                      a.v_proj.bias if a.v_proj.bias is not None else z]).detach()
    hid = m.fc1_g.weight.shape[0]
    hp = -(-hid // 8) * 8
    w1 = torch.zeros(2 * hp, C, dtype=dt, device=dev)
    b1 = torch.zeros(2 * hp, dtype=dt, device=dev)
    w1[:hid], w1[hp:hp + hid] = m.fc1_g.weight.detach(), m.fc1_x.weight.detach()
    b1[:hid], b1[hp:hp + hid] = m.fc1_g.bias.detach(), m.fc1_x.bias.detach()
    w2 = torch.zeros(C, hp, dtype=torch.float32, device=dev)
    w2[:, :hid] = m.fc2.weight.detach().float()
    b2 = m.fc2.bias.detach().float()
    wp, bp = a.proj.weight.detach().float(), a.proj.bias.detach().float()
    if blk.gamma_1 is not None:
        wp, bp = wp * blk.gamma_1.float()[:, None], bp * blk.gamma_1.float()
    if blk.gamma_2 is not None:
        w2, b2 = w2 * blk.gamma_2.float()[:, None], b2 * blk.gamma_2.float()
    lw = torch.zeros(hp, dtype=dt, device=dev)
    lb = torch.zeros(hp, dtype=dt, device=dev)
    lw[:hid], lb[:hid] = m.norm.weight.detach(), m.norm.bias.detach()
    bufs = {"_wqkv": _w4(wqkv), "_bqkv": bqkv, "_wp": _w4(wp.to(dt)), "_bp": bp.to(dt), "_w1": _w4(w1), "_b1": b1,
            "_w2": _w4(w2.to(dt)), "_b2": b2.to(dt), "_lw": lw, "_lb": lb}
    for k, v in bufs.items():
        blk.register_buffer(k, v, persistent=False)
    blk._hid, blk._eps = hid, m.norm.eps


def _fwd(self, x, rope=None, attn_mask=None, is_causal=False):
    from .vit import _lin
    a = self.attn
    B, N, C = x.shape
    H = a.num_heads
    D = C // H
    qkv = _lin(self.norm1(x), self._wqkv, self._bqkv)                        # [B, N, 3C]
    if rope is not None:
        q, k = torch.ops.clfx.rope_qk(qkv, rope, a.num_prefix_tokens, H)
    else:
        q, k = qkv[..., :C].reshape(B, N, H, D), qkv[..., C:2 * C].reshape(B, N, H, D)
    v = qkv.view(B, N, 3, H, D)[:, :, 2]
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask,
                                       is_causal=is_causal, scale=a.scale).transpose(1, 2).reshape(B, N, C)
    x = _lin(o, self._wp, self._bp, res=x)
    h = _lin(self.norm2(x), self._w1, self._b1)                              # [B, N, 2 * hp] = [gate | x]
    h = torch.ops.clfx.swiglu_ln(h, self._lw, self._lb, self._hid, self._eps)
    return _lin(h, self._w2, self._b2, res=x)


def patch_eva(model):
    from .extra import load as load_extra
    from .fused import load
    load()
    load_extra()
    n = 0
    for blk in model.modules():
        if type(blk).__name__ != "EvaBlock":
            continue
        a, m = blk.attn, blk.mlp
        ok = (a.qkv is None and type(m).__name__ == "SwiGLU" and type(m.act).__name__ == "SiLU"
              and type(m.norm).__name__ == "LayerNorm" and isinstance(a.norm, torch.nn.Identity)
              and isinstance(a.q_norm, torch.nn.Identity) and getattr(a, "gate", None) is None
              and not getattr(a, "rotate_half", False) and a.proj.bias is not None and m.fc2.bias is not None)
        if not ok:
            continue
        _prep(blk)
        blk.forward = types.MethodType(_fwd, blk)
        n += 1
    return n

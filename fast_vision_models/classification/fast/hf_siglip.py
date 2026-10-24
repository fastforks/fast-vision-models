"""HF transformers SiglipEncoderLayer (SigLIP / SigLIP 2 vision tower) max path: q|k|v as one fused CUTLASS GEMM, SDPA,
out_proj + residual, fc1 + GELU(tanh) and fc2 + residual as fused GEMMs (same math as the HF layer)."""
import types

import torch
import torch.nn.functional as F

ACTS = {"gelu_pytorch_tanh": 6, "gelu_new": 6, "gelu": 5}


def _w4(w):
    return w.detach().reshape(w.shape[0], w.shape[1], 1, 1).contiguous(memory_format=torch.channels_last)


def _fwd(self, hidden_states, attention_mask=None, **kwargs):
    from .vit import _lin
    a = self.self_attn
    x = hidden_states
    B, N, C = x.shape
    H, D = a.num_heads, a.head_dim
    qkv = _lin(self.layer_norm1(x), self._wqkv, self._bqkv).view(B, N, 3, H, D)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=a.scale).transpose(1, 2).reshape(B, N, C)
    x = _lin(o, self._wo, a.out_proj.bias, res=x)
    h = _lin(self.layer_norm2(x), self._w1, self.mlp.fc1.bias, act=self._act)
    return _lin(h, self._w2, self.mlp.fc2.bias, res=x)


def patch_hf_siglip(model):
    from .fused import load
    load()
    n = 0
    for lay in model.modules():
        if type(lay).__name__ != "SiglipEncoderLayer":
            continue
        a, m = lay.self_attn, lay.mlp
        an = type(m.activation_fn).__name__
        act = 6 if an in ("PytorchGELUTanh", "GELUTanh", "NewGELUActivation", "GELUPytorchTanh") else \
            5 if an == "GELUActivation" else ACTS.get(getattr(getattr(m, "config", None), "hidden_act", ""), 0)
        if not act or a.q_proj.bias is None:
            continue
        bufs = {"_wqkv": _w4(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])),
                "_bqkv": torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]).detach(),
                "_wo": _w4(a.out_proj.weight), "_w1": _w4(m.fc1.weight), "_w2": _w4(m.fc2.weight)}
        for k, v in bufs.items():
            lay.register_buffer(k, v, persistent=False)
        lay._act = act
        lay.forward = types.MethodType(_fwd, lay)
        n += 1
    return n

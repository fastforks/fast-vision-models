"""General fast paths for HF Mask2Former / OneFormer (Swin backbone + MSDeformAttn pixel decoder + masked-attention
transformer decoder). Any input size (padded to /32 like detectron2), nothing compiled per size.

v1:
- whole model pure fp16 (autocast off), convs channels-last
- Swin window attention: q|k|v as one GEMM, relative-position bias (+ shift mask) combined once per (layer type, padded
  size) and cached, SDPA on the windows
- semantic scores like detectron2 (mask logits bilinear to the padded input size -> crop -> sigmoid -> x class probs),
  but NHWC fp16 and row-chunked GEMMs straight into the argmax (fused upsample-argmax kernel when the output size differs)
"""
import os
import types

import torch
import torch.nn.functional as F

from .ops import fp16_accumulate, kern, up_argmax, upsample_argmax


def _swin_attn_forward(self, hidden_states, attention_mask=None, **kwargs):
    B_, N, C = hidden_states.shape
    hd = self.head_dim
    nh = C // hd
    qkv = F.linear(hidden_states, self._qkv_w, self._qkv_b).view(B_, N, 3, nh, hd)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    rpb = self._rpb.get(N)
    if rpb is None:                                                              # weights only: cached per window area
        rpb = self._rpb[N] = self.relative_position_bias().to(hidden_states.dtype)  # [1, nh, N, N]
    if attention_mask is not None:                                               # shifted windows: [nw, nh, N, N]
        nw = attention_mask.shape[0]
        bias = (rpb + attention_mask.to(rpb.dtype)[:, None])
        if B_ != nw:
            bias = bias.repeat(B_ // nw, 1, 1, 1)
    else:
        bias = rpb.expand(B_, -1, -1, -1)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scaling)
    o = o.transpose(1, 2).reshape(B_, N, C)
    return F.linear(o, self.o_proj.weight, self.o_proj.bias), None


def _msda_forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                  position_embeddings=None, reference_points=None, spatial_shapes=None, level_start_index=None,
                  output_attentions=False, spatial_shapes_list=None, **kw):
    """Fused MSDeformAttn (seg_kernels.cu msda_self_h): self-attention over the multi-level tokens, reference points
    recomputed in the kernel from the query position (no padding: valid ratios 1), softmax in fp32."""
    shapes_list = spatial_shapes_list if spatial_shapes_list is not None else [tuple(map(int, s)) for s in spatial_shapes]
    # HF Mask2Former / OneFormer pixel decoders always build all-False padding masks (zeros, valid ratios 1): no check
    # (it would cost a host sync per layer)
    if hidden_states.dtype != torch.float16:
        return self._orig_forward(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask,
                                  position_embeddings, reference_points, spatial_shapes_list=shapes_list,
                                  level_start_index=level_start_index, output_attentions=output_attentions,
                                  **({"spatial_shapes": spatial_shapes} if spatial_shapes is not None else {}))
    if position_embeddings is not None:
        hidden_states = hidden_states + position_embeddings
    B, S, C = encoder_hidden_states.shape
    NH, L, P = self.n_heads, self.n_levels, self.n_points
    value = self.value_proj(encoder_hidden_states)
    off = self.sampling_offsets(hidden_states)
    logit = self.attention_weights(hidden_states)
    key = tuple(shapes_list)
    meta = self._meta.get(key)
    if meta is None:
        st, acc = [], 0
        for h, w in shapes_list:
            st.append(acc)
            acc += h * w
        meta = self._meta[key] = (torch.tensor([v for hw in shapes_list for v in hw], dtype=torch.int32, device=value.device),
                                  torch.tensor(st, dtype=torch.int32, device=value.device))
    out = torch.empty_like(value)
    n = B * S * NH * (C // NH // 2)
    kern().launch("msda_self_h", ((min((n + 255) // 256, 65535 * 8), 1, 1), (256, 1, 1)), value, off, logit, meta[0],
                  meta[1], out, B, S, NH, C // NH, L, P)
    return self.output_proj(out), None


def _patch_msda(model):
    n = 0
    for m in model.modules():
        if m.__class__.__name__.endswith("PixelDecoderEncoderMultiscaleDeformableAttention"):
            m._orig_forward = m.forward
            m._meta = {}
            m.forward = types.MethodType(_msda_forward, m)
            n += 1
    return n


def _patch_swin(backbone):
    n = 0
    for m in backbone.modules():
        if m.__class__.__name__ == "SwinAttention":
            m._qkv_w = torch.cat([m.q_proj.weight, m.k_proj.weight, m.v_proj.weight], 0)
            bs = [m.q_proj.bias, m.k_proj.bias, m.v_proj.bias]
            m._qkv_b = torch.cat(bs, 0) if all(b is not None for b in bs) else None
            m._rpb = {}
            m.forward = types.MethodType(_swin_attn_forward, m)
            n += 1
    return n


class _TorchProxy:
    """`torch` for the HF Mask2Former / OneFormer modules: torch.as_tensor / torch.tensor of Python lists onto the GPU
    (spatial shapes, built every forward) are served from a cache, so no host-to-device copy happens inside a CUDA
    graph capture. Everything else is plain torch."""
    def __init__(self):
        self._c = {}

    def __getattr__(self, k):
        return getattr(torch, k)

    def _cached(self, fn, data, dtype=None, device=None, **kw):
        if device is None or kw or isinstance(data, torch.Tensor) or torch.device(device).type != "cuda":
            return fn(data, dtype=dtype, device=device, **kw)
        try:
            key = (fn.__name__, repr(data), dtype, str(device))
        except Exception:  # noqa: BLE001
            return fn(data, dtype=dtype, device=device)
        t = self._c.get(key)
        if t is None:
            t = self._c[key] = fn(data, dtype=dtype, device=device)
        return t

    def as_tensor(self, data, dtype=None, device=None, **kw):
        return self._cached(torch.as_tensor, data, dtype, device, **kw)

    def tensor(self, data, dtype=None, device=None, **kw):
        return self._cached(torch.tensor, data, dtype, device, **kw)


def _install_proxy(model):
    import importlib
    mod = importlib.import_module(type(model).__module__)
    if not isinstance(mod.torch, _TorchProxy):
        mod.torch = _TorchProxy()


class _FastMaskFormerBase:
    def __init__(self, model, dtype=torch.float16, gacc=False, msda=True, fused=True):
        self.fused = fused
        msda = msda and fused and os.environ.get("SEG_MSDA", "1") == "1"  # upstream-friendly version: HF's PyTorch MSDeformAttn (grid_sample), no custom kernels
        self.m = model.to(dtype).eval()
        self.dtype, self.gacc = dtype, gacc
        for mod in model.modules():
            if isinstance(mod, torch.nn.Conv2d):
                mod.to(memory_format=torch.channels_last)
        self.n_swin = _patch_swin(model)
        from .swin import patch_swin_layers                                     # round 2: gather-based window partition
        self.n_swin_layers = patch_swin_layers(model, fused) if os.environ.get("SEG_SWIN", "1") == "1" else 0
        self.n_msda = _patch_msda(model) if msda else 0
        if fused:
            _install_proxy(model)
        self.info = {"opt": "maskformer-fast-v1", "swin_attn_patched": self.n_swin, "msda_fused": self.n_msda, "swin_layers": self.n_swin_layers, "gacc": gacc,
                     "fused_up_argmax": fused}

    def _forward(self, x):
        raise NotImplementedError

    def masks_cls(self, x):
        """x [1, 3, H, W] (any size) -> mask logits [1, Q, h, w] at 1/4 of the /32-padded size, class logits, (H, W)."""
        H, W = x.shape[2:]
        ph, pw = (-H) % 32, (-W) % 32
        xp = F.pad(x, (0, pw, 0, ph)) if ph or pw else x
        with torch.autocast("cuda", enabled=False), fp16_accumulate(self.gacc):
            o = self._forward(xp.to(self.dtype).contiguous(memory_format=torch.channels_last))
        return o.masks_queries_logits, o.class_queries_logits, xp.shape[2:]

    def _scores_rows(self, masks, cls, Hp, Wp, H, W):
        """generator over row chunks of the semantic scores [rows, W, K] (fp16) at the input size."""
        probs = cls.float().softmax(-1)[0, :, :-1].to(self.dtype)                # [Q, K]
        mk = F.interpolate(masks, (Hp, Wp), mode="bilinear", align_corners=False)[0, :, :H, :W]  # [Q, H, W]
        Q = mk.shape[0]
        rows = max(1, (1 << 22) // (W * max(Q, probs.shape[1])))
        for r0 in range(0, H, rows):
            m = mk[:, r0:r0 + rows].permute(1, 2, 0).sigmoid()                    # [r, W, Q]
            yield r0, (m.reshape(-1, Q) @ probs).view(m.shape[0], W, -1)

    def logits(self, x):
        masks, cls, (Hp, Wp) = self.masks_cls(x)
        H, W = x.shape[2:]
        out = torch.empty((1, H, W, cls.shape[-1] - 1), device=x.device, dtype=self.dtype)
        for r0, s in self._scores_rows(masks, cls, Hp, Wp, H, W):
            out[0, r0:r0 + s.shape[0]] = s
        return out.permute(0, 3, 1, 2)

    def predict(self, x, out_size=None):
        if not self.fused:                                                       # plain PyTorch: scores -> (resize) -> argmax
            s = self.logits(x).permute(0, 2, 3, 1)
            return upsample_argmax(s, *(out_size or x.shape[2:]), fused=False)
        masks, cls, (Hp, Wp) = self.masks_cls(x)
        H, W = x.shape[2:]
        if out_size is None or tuple(out_size) == (H, W):
            out = torch.empty((1, H, W), device=x.device, dtype=torch.int64)
            for r0, s in self._scores_rows(masks, cls, Hp, Wp, H, W):
                out[0, r0:r0 + s.shape[0]] = s.argmax(-1)
            return out
        s = self.logits(x).permute(0, 2, 3, 1)
        return up_argmax(s, *out_size)


class FastMask2Former(_FastMaskFormerBase):
    def _forward(self, x):
        return self.m(pixel_values=x)


def _cast(o, src, dst):
    if isinstance(o, torch.Tensor):
        return o.to(dst) if o.dtype == src else o
    if isinstance(o, (list, tuple)):
        return type(o)(_cast(v, src, dst) for v in o)
    if isinstance(o, dict):
        for k in list(o.keys()):
            o[k] = _cast(o[k], src, dst)
    return o


def fp32_island(mod, dtype=torch.float16):
    """run `mod` in fp32 inside an fp16 model (inputs cast up, outputs cast back)."""
    mod.float()
    mod.register_forward_pre_hook(lambda m, a, k: (_cast(a, dtype, torch.float32), _cast(k, dtype, torch.float32)),
                                  with_kwargs=True)
    mod.register_forward_hook(lambda m, a, o: _cast(o, torch.float32, dtype))


def _oneformer_layer_forward(self, index, output, multi_stage_features, multi_stage_positional_embeddings,
                             attention_mask=None, query_embeddings=None, output_attentions=False):
    """HF OneFormerTransformerDecoderLayer.forward with `mask[torch.where(full rows)] = False` (a host sync) written as
    an equivalent masked_fill_ (CUDA-graph capturable)."""
    level_index = index % self.num_feature_levels
    attention_mask.masked_fill_((attention_mask.sum(-1) == attention_mask.shape[-1])[..., None], False)
    output, cross_attn_weights = self.cross_attn(output, multi_stage_features[level_index], memory_mask=attention_mask,
                                                 memory_key_padding_mask=None,
                                                 pos=multi_stage_positional_embeddings[level_index],
                                                 query_pos=query_embeddings)
    output, self_attn_weights = self.self_attn(output, output_mask=None, output_key_padding_mask=None,
                                               query_pos=query_embeddings)
    output = self.ffn(output)
    outputs = (output,)
    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)
    return outputs


def _mha_sdpa_forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, **kw):
    """nn.MultiheadAttention (seq-first, no masks) through SDPA: the query transformer's cross-attention runs over the
    1/4-resolution tokens (150 queries x 16k-131k keys); nn.MultiheadAttention materializes the [heads, 150, keys] fp32
    score matrix + softmax, SDPA's memory-efficient kernel does not (same fp32 math up to summation order). Plain PyTorch."""
    if key_padding_mask is not None or attn_mask is not None or self.batch_first:
        return self._orig_forward(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights,
                                  attn_mask=attn_mask, **kw)
    L, B, E = query.shape
    h = self.num_heads
    d = E // h
    w, b = self.in_proj_weight, self.in_proj_bias
    q = F.linear(query, w[:E], b[:E]).view(L, B, h, d).permute(1, 2, 0, 3)
    k = F.linear(key, w[E:2 * E], b[E:2 * E]).view(key.shape[0], B, h, d).permute(1, 2, 0, 3)
    v = F.linear(value, w[2 * E:], b[2 * E:]).view(value.shape[0], B, h, d).permute(1, 2, 0, 3)
    o = F.scaled_dot_product_attention(q, k, v)
    return F.linear(o.permute(2, 0, 1, 3).reshape(L, B, E), self.out_proj.weight, self.out_proj.bias), None


class FastOneFormer(_FastMaskFormerBase):
    """+ the task-conditioned query transformer (3 M params, runs on the 1/32 features) stays fp32: in fp16 it flips
    0.3 % of the pixels (val-500 agreement 99.7 -> 99.9 % with the island, measured on 40 images)."""
    def __init__(self, model, task_inputs, **kw):
        super().__init__(model, **kw)
        self.task = task_inputs
        qt = model.model.transformer_module.decoder.query_transformer
        fp32_island(qt, self.dtype)
        self.n_mha = 0
        if os.environ.get("SEG_QT_SDPA", "1") == "1":                           # round 2: SDPA in the fp32 island
            for m in qt.modules():
                if isinstance(m, torch.nn.MultiheadAttention) and m._qkv_same_embed_dim:
                    m._orig_forward = m.forward
                    m.forward = types.MethodType(_mha_sdpa_forward, m)
                    self.n_mha += 1
        if self.fused:
            for m in model.modules():
                if type(m).__name__ == "OneFormerTransformerDecoderLayer":
                    m.forward = types.MethodType(_oneformer_layer_forward, m)
        self.info["fp32_island"] = "query_transformer"
        self.info["qt_mha_sdpa"] = self.n_mha

    def _forward(self, x):
        return self.m(pixel_values=x, task_inputs=self.task)

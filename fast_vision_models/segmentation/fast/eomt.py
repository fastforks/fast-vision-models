"""General fast path for HF EoMT (DINOv2-style ViT encoder with 100 queries appended before the last 4 blocks, no pixel
decoder). Inference-time masked attention is off in the released checkpoints (attn_mask_probs = 0), so every block is
plain self-attention.

- pure fp16, q/k/v packed into one GEMM, fastbb flash kernel (head_dim 64; variant "fh" = fp32 QK^T, fp16 PV) on strided
  views of the packed output, LayerScale + residual as in-place addcmul (fastbb FastViT block math)
- position embeddings: the checkpoint's grid (32x32 at 512); other window sizes get the DINOv2 bicubic interpolation
  (cached per grid) -- the protocol only uses 512 windows
- residual-stream scaling: DINOv3 ViT-L's residual stream reaches |x| ~ 1.5e5 (> fp16 max 65504 -> inf/NaN in pure fp16).
  Every LayerNorm is scale-invariant, so the stream is kept divided by `res_scale` (16 for DINOv3): LayerScale and 1/s are
  folded into out_proj / fc2 weights, embeddings + queries are scaled once, LN eps -> eps / s^2: exact
- head: mask MLP on the queries, 2 upscale blocks (ConvTranspose 2x2 -> GELU -> dw 3x3 -> LN2d) channels-last
- semantic scores like upstream (mask logits bilinear to the window size -> sigmoid -> x class probs), but NHWC fp16 and
  as one GEMM; window merge + argmax in seg_fast.windows (fp16 accumulation buffer, fused upsample-argmax kernel)
"""
import math
import os

import torch
import torch.nn.functional as F

from .ops import fp16_accumulate

from fastbb.attention import attention


class FastEomt:
    def __init__(self, model, attn=os.environ.get("SEG_EOMT_ATTN", "fh"), gacc=False, dtype=torch.float16, res_scale=None, fused=True):
        self.fused = fused
        rope = getattr(model, "rope_embeddings", None)
        inv = rope.inv_freq.detach().clone().float() if rope is not None else None
        self.s = res_scale or (16.0 if rope is not None else 1.0)
        for l in model.layers:                                                     # fold LayerScale and 1/s (fp32)
            a = l.attention
            o = a.out_proj if hasattr(a, "out_proj") else a.o_proj
            f2 = l.mlp.fc2 if hasattr(l.mlp, "fc2") else l.mlp.down_proj
            for lin, ls, nm in ((o, l.layer_scale1.lambda1, "_out"), (f2, l.layer_scale2.lambda1, "_fc2s")):
                g = ls.detach().float() / self.s
                b = lin.bias.detach().float() if lin.bias is not None else torch.zeros_like(g)
                setattr(l, nm + "_w", (lin.weight.detach().float() * g[:, None]).to(dtype))
                setattr(l, nm + "_b", (b * g).to(dtype))
        if self.s != 1:                                                            # LN(x / s) with eps / s^2 == LN(x) exactly
            for ln in [mod for l in model.layers for mod in (l.norm1, l.norm2)] + [model.layernorm]:
                ln.eps = ln.eps / self.s ** 2
        self.m = model.to(dtype).eval()
        if rope is not None:                                                       # RoPE frequencies stay fp32
            rope.inv_freq = inv
        self.cfg = model.config
        self.dtype, self.attn, self.gacc = dtype, attn, gacc
        for l in model.layers:
            a = l.attention
            a._qkv_w = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0)
            bs = [p.bias if p.bias is not None else torch.zeros_like(p.weight[:, 0]) for p in (a.q_proj, a.k_proj, a.v_proj)]
            a._qkv_b = torch.cat(bs, 0)
            a._scale = a.scale if hasattr(a, "scale") else a.scaling
            mlp = l.mlp
            l._fc1, l._fc2 = (mlp.fc1, mlp.fc2) if hasattr(mlp, "fc1") else (mlp.up_proj, mlp.down_proj)
            assert not hasattr(mlp, "gate_proj"), "gated MLP not implemented"
        for mod in model.modules():
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                mod.to(memory_format=torch.channels_last)
        e = model.embeddings
        self.ps = self.cfg.patch_size if isinstance(self.cfg.patch_size, int) else self.cfg.patch_size[0]
        self.rope = getattr(model, "rope_embeddings", None)                       # DINOv3: RoPE on the patch tokens
        if self.rope is None:
            n = e.position_embeddings.weight.shape[0]
            self.g0 = int(math.isqrt(n))
            self.pos32 = e.position_embeddings.weight.detach().float()
        self._pos, self._rope = {}, {}
        self.info = {"opt": "eomt-fast", "attn": attn, "gacc": gacc, "res_scale": self.s, "fused_up_argmax": fused}

    def pos(self, h, w):
        if (h, w) not in self._pos:
            p = self.pos32
            if (h, w) != (self.g0, self.g0):
                p = F.interpolate(p.view(1, self.g0, self.g0, -1).permute(0, 3, 1, 2), (h, w), mode="bicubic",
                                  align_corners=False).permute(0, 2, 3, 1).reshape(h * w, -1)
            self._pos[(h, w)] = p.to(self.dtype)
        return self._pos[(h, w)]

    def rope_cs(self, x):
        key = tuple(x.shape[2:])
        if key not in self._rope:
            cos, sin = self.rope(x[:1].float())
            self._rope[key] = (cos.reshape(1, -1, 1, cos.shape[-1]).to(self.dtype), sin.reshape(1, -1, 1, sin.shape[-1]).to(self.dtype))
        return self._rope[key]

    def block(self, l, x, cs=None):
        B, N, C = x.shape
        a = l.attention
        H = a.num_heads
        qkv = F.linear(l.norm1(x), a._qkv_w, a._qkv_b).view(B, N, 3, H, C // H)
        if cs is not None:                                                         # rotate the patch tokens of q and k in place
            cos, sin = cs
            P = N - cos.shape[1]
            for j in (0, 1):
                t = qkv[:, P:, j]
                t1, t2 = t.chunk(2, -1)
                r = torch.cat((-t2, t1), -1)
                t.copy_(t * cos + r * sin)
        q, k, v = qkv.unbind(2)
        if self.attn == "sdpa":
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               scale=a._scale).transpose(1, 2).reshape(B, N, C)
        else:
            o = attention(q, k, v, scale=a._scale, variant=self.attn).view(B, N, C)
        x = x + F.linear(o, l._out_w, l._out_b)                                    # LayerScale / s folded in
        h = F.gelu(F.linear(l.norm2(x), l._fc1.weight, l._fc1.bias))
        return x.add_(F.linear(h, l._fc2s_w, l._fc2s_b))

    def heads(self, x):
        """x [B, 3, H, W] (window batch) -> mask logits [B, Q, H/4, W/4], class logits [B, Q, K+1]."""
        m, cfg = self.m, self.cfg
        e = m.embeddings
        B, _, H, W = x.shape
        h, w = H // self.ps, W // self.ps
        pe = e.patch_embeddings if self.rope is not None else e.patch_embeddings.projection
        t = pe(x.to(self.dtype).contiguous(memory_format=torch.channels_last))
        t = t.permute(0, 2, 3, 1).reshape(B, h * w, -1)
        cs = None
        if self.rope is None:
            t = t + self.pos(h, w)
        else:
            cs = self.rope_cs(x)
        t = torch.cat([e.cls_token.expand(B, -1, -1), e.register_tokens.expand(B, -1, -1), t], 1)
        if self.s != 1:
            t = t * (1 / self.s)
        L = cfg.num_hidden_layers
        for i, l in enumerate(m.layers):
            if i == L - cfg.num_blocks:
                qw = m.query.weight if self.s == 1 else m.query.weight * (1 / self.s)
                t = torch.cat([qw[None].expand(B, -1, -1), t], 1)
            t = self.block(l, t, cs)
        t = m.layernorm(t)
        Q = cfg.num_queries
        q = t[:, :Q]
        npre = 1 + cfg.num_register_tokens
        cls = m.class_predictor(q)
        p = t[:, Q + npre:].reshape(B, h, w, -1).permute(0, 3, 1, 2)  # channels-last view
        for blk in m.upscale_block.block:
            p = blk.conv2(blk.activation(blk.conv1(p)))
            p = F.layer_norm(p.permute(0, 2, 3, 1), p.shape[1:2], blk.layernorm2d.weight, blk.layernorm2d.bias,
                             blk.layernorm2d.eps)                                     # NHWC
            p = p.permute(0, 3, 1, 2)
        qm = m.mask_head(q)                                                           # [B, Q, C]
        Bp, C, h4, w4 = p.shape
        masks = torch.bmm(p.permute(0, 2, 3, 1).reshape(B, h4 * w4, C), qm.transpose(1, 2))  # [B, h4*w4, Q]
        return masks.view(B, h4, w4, Q), cls

    def scores_nhwc(self, x):
        """-> semantic scores [B, H, W, K] fp16 at the window size (upstream semantic_inference order)."""
        with torch.autocast("cuda", enabled=False), fp16_accumulate(self.gacc):
            masks, cls = self.heads(x)
            H, W = x.shape[2:]
            mk = F.interpolate(masks.permute(0, 3, 1, 2), (H, W), mode="bilinear", align_corners=False)
            mk = mk.permute(0, 2, 3, 1).sigmoid_()                                    # [B, H, W, Q] (channels-last)
            probs = cls.float().softmax(-1)[..., :-1].to(self.dtype)                  # [B, Q, K]
            B, Q = masks.shape[0], masks.shape[3]
            return torch.bmm(mk.reshape(B, H * W, Q), probs).view(B, H, W, -1)

    def logits(self, x):
        return self.scores_nhwc(x).permute(0, 3, 1, 2)

    def predict(self, x, out_size=None):
        from .windows import eomt_predict
        return eomt_predict(self.scores_nhwc, x, 512, out_size, fused=self.fused)

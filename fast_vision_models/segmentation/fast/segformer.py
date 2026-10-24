"""General fast path for HF SegformerForSemanticSegmentation (MiT-B0..B5 + all-MLP head). Any input size, nothing compiled
or tuned per size; same weights and math as upstream, only precision / layout / algebraic re-association change:

- pure fp16 weights + activations, tokens kept as NHWC ([B, H, W, C] == [B, N, C], so conv <-> token switches are views)
- k and v projections merged into one GEMM, SDPA (flash) attention on strided views, no transposes/copies
- sequence reduction conv (k = s = sr) and the Mix-FFN depthwise 3x3 on channels-last views (cuDNN NHWC)
- decode head re-associated: linear_fuse(cat(up(proj_i(f_i)))) == sum_i up((W_i proj_i) f_i): the per-stage 768-ch
  projection and the 3072->768 fuse are folded into one C_i->768 GEMM run at the stage's own resolution, then upsampled
  and summed (bilinear upsampling commutes with per-pixel affine maps); BN folded into those weights
- classifier on NHWC, then fused bilinear-upsample + argmax kernel (no full-resolution 150-channel scores)
- opt-in fp16-accumulate GEMMs (`gacc`)
"""
import torch
import torch.nn.functional as F

import os
import types

from .cnn_fuse import cx_conv
from .ops import dwconv, fast_interpolate, fp16_accumulate, upsample_argmax


def _lin(x, w4, b, res=None):
    """[B, N, C] @ w^T + b (+ res) as a 1x1 CUTLASS conv on the channels-last view [B, C, N, 1] (fp16 accumulate, bias and
    residual in the epilogue, ~5 us host time vs ~11 us for cuBLAS addmm + a separate add)."""
    r = None if res is None else res.unsqueeze(2).permute(0, 3, 1, 2)
    y = torch.ops.segx.conv_fused(x.unsqueeze(2).permute(0, 3, 1, 2), w4, b, r, [0, 0], [1, 1], 0)
    return y.permute(0, 2, 3, 1).squeeze(2)


class FastSegformer:
    def __init__(self, model, gacc=False, dtype=torch.float16, fused=True):
        self.m = model.eval().requires_grad_(False)   # inference only (a trace inserts weights as constants)
        self.dtype, self.gacc, self.fused = dtype, gacc, fused
        self.dwk = os.environ.get("SEG_SF_DWK", "1") == "1"         # max: prebuilt dwconv + GELU kernel (launch-overhead pass)
        self.stages = self._stages(model)
        model.to(dtype)
        for mod in model.modules():
            if isinstance(mod, torch.nn.Conv2d):
                mod.to(memory_format=torch.channels_last)
        for st in self.stages:
            for b in st.blocks:
                a = b.attention
                a._kv_w = torch.cat([a.k_proj.weight, a.v_proj.weight], 0)
                a._kv_b = torch.cat([a.k_proj.bias, a.v_proj.bias], 0)
        self._fold_head(model.decode_head)
        self.cx_convs = self.cx_linear_blocks = 0
        if fused and self.dwk:                     # max: patch embeddings / sequence reductions (C % 8 == 0) on the CUTLASS op
            from .cnn_fuse import cx_prepare
            from .ops import cx
            if cx():
                for st in self.stages:
                    convs = [st.patch_embeddings.proj] + [b.attention.sequence_reduction.sequence_reduction
                                                         for b in st.blocks if b.attention.sequence_reduction_ratio > 1]
                    for c in convs:
                        if c.in_channels % 8 == 0 and cx_prepare(c) == "cutlass":
                            c.forward = types.MethodType(lambda self, t: cx_conv(self, t), c)
                            self.cx_convs += 1
                    for b in st.blocks:                     # linears as 1x1 CUTLASS convs (all dims % 8 == 0 in MiT)
                        a, mm = b.attention, b.mlp
                        ws = {"q": a.q_proj.weight, "kv": a._kv_w, "o": a.o_proj.weight, "fc1": mm.fc1.weight,
                              "fc2": mm.fc2.weight}
                        if os.environ.get("SEG_SF_LIN", "1") == "1" and all(w.shape[0] % 8 == 0 and w.shape[1] % 8 == 0
                                                                            for w in ws.values()):
                            b._cxw = {k: w.detach()[:, :, None, None].contiguous(memory_format=torch.channels_last) for k, w in ws.items()}
                            b._cxw.update(nh=a.num_attention_heads, hd=a.head_dim)
                            self.cx_linear_blocks += 1
        self.info = {"opt": "segformer-fast" if fused else "segformer-uf", "gacc": gacc, "fused_up_argmax": fused,
                     "dwconv_gelu_kernel": fused and self.dwk, "cutlass_convs": self.cx_convs,
                     "cutlass_linear_blocks": self.cx_linear_blocks}
        self.tracer = None
        if fused:                                   # max: C++ executor for the whole net (seg_fast/trace.py)
            from .trace import ENABLED, Traced
            if ENABLED:
                self.tracer = Traced(self._scores, name="segformer")

    @property
    def trace_info(self):
        return self.tracer.info if self.tracer is not None else {"trace": "off"}

    @staticmethod
    def _stages(model):
        return list(model.segformer.stages)

    def _fold_head(self, h):
        C = h.config.decoder_hidden_size
        n = len(h.linear_projections)
        Wf = h.linear_fuse.weight.float().view(C, n * C)          # input = cat(reversed stages)
        bn = h.batch_norm
        sc = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
        sh = bn.bias.float() - bn.running_mean.float() * sc
        self.head_w, bias = [], torch.zeros(C, device=Wf.device)
        for i, lp in enumerate(h.linear_projections):
            j = n - 1 - i                                           # position of stage i in the cat
            Wi = Wf[:, j * C:(j + 1) * C]
            W = sc[:, None] * (Wi @ lp.proj.weight.float())        # [C, C_i]
            bias += sc * (Wi @ lp.proj.bias.float())
            self.head_w.append(W.to(self.dtype))
        self.head_b = (bias + sh).to(self.dtype)
        self.cls_w = h.classifier.weight.view(h.classifier.out_channels, -1).to(self.dtype)
        self.cls_b = h.classifier.bias.to(self.dtype)

    # ------------------------------------------------------------------ encoder
    def _block(self, b, x, H, W):
        Bn, N, C = x.shape
        a = b.attention
        hd = a.head_dim
        nh = C // hd
        y = b.layernorm_before(x)
        q = F.linear(y, a.q_proj.weight, a.q_proj.bias).view(Bn, N, nh, hd).transpose(1, 2)
        if a.sequence_reduction_ratio > 1:
            r = a.sequence_reduction
            yc = y.view(Bn, H, W, C).permute(0, 3, 1, 2)            # channels-last view
            yr = r.sequence_reduction(yc)
            yr = r.layer_norm(yr.permute(0, 2, 3, 1).reshape(Bn, -1, C))
        else:
            yr = y
        M = yr.shape[1]
        kv = F.linear(yr, a._kv_w, a._kv_b).view(Bn, M, 2, nh, hd)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, scale=a.scaling)
        o = o.transpose(1, 2).reshape(Bn, N, C)
        x = x + F.linear(o, a.o_proj.weight, a.o_proj.bias)
        m = b.mlp
        y = F.linear(b.layernorm_after(x), m.fc1.weight, m.fc1.bias)
        Ch = y.shape[-1]
        if self.fused and self.dwk and Ch % 2 == 0:              # max: depthwise 3x3 + bias + exact GELU in one kernel
            y = dwconv(y.view(Bn, H, W, Ch).permute(0, 3, 1, 2), m.dwconv.dwconv, act=1).permute(0, 2, 3, 1).reshape(Bn, N, Ch)
        else:
            y = m.dwconv.dwconv(y.view(Bn, H, W, Ch).permute(0, 3, 1, 2))
            y = F.gelu(y.permute(0, 2, 3, 1).reshape(Bn, N, Ch))
        return x.add_(F.linear(y, m.fc2.weight, m.fc2.bias))

    def _block_max(self, b, x, H, W):
        """max + segx ops: every linear as a CUTLASS 1x1 conv over the token map (residual adds fused into o_proj / fc2),
        shapes from constants (unflatten / flatten) instead of traced size arithmetic."""
        a, c, m = b.attention, b._cxw, b.mlp
        nh, hd = c["nh"], c["hd"]
        y = b.layernorm_before(x)
        q = _lin(y, c["q"], a.q_proj.bias).unflatten(-1, (nh, hd)).transpose(1, 2)
        if a.sequence_reduction_ratio > 1:
            r = a.sequence_reduction
            yr = r.sequence_reduction(y.unflatten(1, (H, W)).permute(0, 3, 1, 2))
            yr = r.layer_norm(yr.permute(0, 2, 3, 1).flatten(1, 2))
        else:
            yr = y
        kv = _lin(yr, c["kv"], a._kv_b).unflatten(-1, (2, nh, hd))
        o = F.scaled_dot_product_attention(q, kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2), scale=a.scaling)
        x = _lin(o.transpose(1, 2).flatten(2), c["o"], a.o_proj.bias, res=x)
        y = _lin(b.layernorm_after(x), c["fc1"], m.fc1.bias)
        y = dwconv(y.unflatten(1, (H, W)).permute(0, 3, 1, 2), m.dwconv.dwconv, act=1).permute(0, 2, 3, 1).flatten(1, 2)
        return _lin(y, c["fc2"], m.fc2.bias, res=x)

    def features(self, x):
        feats = []
        x = x.to(self.dtype).contiguous(memory_format=torch.channels_last)
        for st in self.stages:
            pe = st.patch_embeddings
            e = pe.proj(x)                                          # NHWC
            Bn, C, H, W = e.shape
            t = pe.layer_norm(e.permute(0, 2, 3, 1).reshape(Bn, H * W, C))
            for b in st.blocks:
                t = self._block_max(b, t, H, W) if hasattr(b, "_cxw") else self._block(b, t, H, W)
            t = st.layer_norm(t)
            feats.append(t.view(Bn, H, W, C))
            x = t.view(Bn, H, W, C).permute(0, 3, 1, 2)            # channels-last view for the next patch embedding
        return feats

    # ------------------------------------------------------------------ head
    def _scores(self, x):
        feats = self.features(x)
        B, H4, W4, _ = feats[0].shape
        acc = None
        for k, (f, Wt) in enumerate(zip(feats, self.head_w)):
            y = F.linear(f, Wt)                                     # [B, h, w, C] at the stage resolution
            if k > 0:                                               # stages 1-3 are coarser than stage 0
                y = F.interpolate(y.permute(0, 3, 1, 2), (H4, W4), mode="bilinear", align_corners=False)
                y = y.permute(0, 2, 3, 1)
            acc = y.contiguous() if acc is None else acc.add_(y)
        acc = F.relu_(acc.add_(self.head_b))
        return F.linear(acc, self.cls_w, self.cls_b)                # [B, H/4, W/4, classes]

    def scores_nhwc(self, x):
        with torch.autocast("cuda", enabled=False), fp16_accumulate(self.gacc), fast_interpolate(self.fused and self.dwk):
            if self.tracer is not None:
                return self.tracer(x)
            return self._scores(x)

    def logits(self, x):
        return self.scores_nhwc(x).permute(0, 3, 1, 2)

    def predict(self, x, out_size=None):
        s = self.scores_nhwc(x)
        H, W = out_size or x.shape[2:]
        return upsample_argmax(s, H, W, fused=self.fused)

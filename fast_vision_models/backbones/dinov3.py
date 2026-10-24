"""DINOv3 (ViT-S/B/L/H+ and ConvNeXt T/S/B/L) as standalone fastbb feature extractors, loading the HF-format checkpoints
(facebook/dinov3-*-pretrain-lvd1689m, safetensors; gated licence: never redistribute the weights).

    from fast_vision_models.backbones.dinov3 import dinov3_vit, dinov3_convnext
    m = dinov3_vit("/path/dinov3-vitl16-pretrain-lvd1689m", spec="hh-g1")      # fp16, CUDA
    out = m.forward_features(img)    # {x_norm_clstoken, x_norm_regtokens, x_norm_patchtokens} like DINOv2
    y = m(img)                       # last_hidden_state [B, 1 + 4 + HW, C] (= HF DINOv3ViTModel(...).last_hidden_state)
    f = dinov3_convnext("/path/dinov3-convnext-base-...", spec="fold-gacc-fused-trace")   # fastbb.cnn.optimize on a timm ConvNeXt

img: [B, 3, H, W] ImageNet-normalized, H, W multiples of 16 (any size, nothing compiled per size).
DINOv3 specifics (all handled): patch 16; cls + 4 register tokens in front of the patches; NO absolute position embedding, RoPE
(theta 100, centered [-1, 1] coords, rotate_half, applied to q and k of patch tokens only; prefix tokens are not rotated);
head_dim 64 for all sizes (flash64 kernel applies); LayerNorm eps 1e-5; LayerScale; k_proj without bias; ViT-H+ uses a gated SiLU MLP.
Spec "<attn>-g<n>" as fastbb.install_vit (sdpa = friendly / exportable: plain PyTorch incl. plain-torch RoPE; fh / hh = max: flash64
kernel + fused in-place RoPE kernel fastbb.rope).
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROPE_PLAIN = os.environ.get("FASTBB_ROPE", "fused") == "torch"


def _ln(n, x):
    """LayerNorm; in an ONNX export the statistics run in fp32 (TensorRT fp16 INormalization overflows on the large DINOv3 activations)."""
    if torch.onnx.is_in_onnx_export() and x.dtype == torch.float16:
        return F.layer_norm(x.float(), n.normalized_shape, n.weight.float(), n.bias.float(), n.eps).half()
    return n(x)


class _Block(nn.Module):
    def __init__(self, d, ffn, gated, act):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d, eps=1e-5), nn.LayerNorm(d, eps=1e-5)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ls1, self.ls2 = nn.Parameter(torch.empty(d)), nn.Parameter(torch.empty(d))
        self.gated, self.act = gated, act
        if gated:
            self.w12, self.w3 = nn.Linear(d, 2 * ffn), nn.Linear(ffn, d)      # [gate; up] fused, down
        else:
            self.fc1, self.fc2 = nn.Linear(d, ffn), nn.Linear(ffn, d)


class DinoV3ViT(nn.Module):
    def __init__(self, cfg, attn="sdpa", gacc=False, graphs=False):
        super().__init__()
        d, self.heads, self.nreg = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_register_tokens"]
        self.d, self.ps, self.theta = d, cfg["patch_size"], cfg["rope_theta"]
        self.attn, self.gacc = attn, gacc
        gated, act = cfg["use_gated_mlp"], cfg["hidden_act"]
        self.cls_token = nn.Parameter(torch.empty(1, 1, d))
        self.register_tokens = nn.Parameter(torch.empty(1, self.nreg, d))
        self.patch_embed = nn.Conv2d(3, d, self.ps, self.ps)
        self.blocks = nn.ModuleList([_Block(d, cfg["intermediate_size"], gated, act) for _ in range(cfg["num_hidden_layers"])])
        self.norm = nn.LayerNorm(d, eps=cfg["layer_norm_eps"])
        self._rope = {}
        self._graphs = None
        if graphs:
            from fastbb.graphs import graphed
            self._graphs = graphed(self._run_all)

    # ------------------------------------------------------------ rope tables (fp32, cached per grid)
    def rope_cs(self, h, w, device):
        from .rope import rope_table
        k = (h, w, str(device))
        if k not in self._rope:
            self._rope[k] = rope_table(h, w, self.d // self.heads, self.theta, device)
        return self._rope[k]

    # ------------------------------------------------------------ block
    def _act(self, h):
        return F.gelu(h)

    def block(self, b, x, cs):
        B, N, C = x.shape
        H = self.heads
        P = 1 + self.nreg
        qkv = F.linear(_ln(b.norm1, x), b.qkv.weight, b.qkv.bias).view(B, N, 3, H, C // H)
        from fastbb.vit import SMALL_WORK
        fast = self.attn != "sdpa" and C // H == 64 and B * N * C >= SMALL_WORK and qkv.dtype == torch.float16
        if fast:
            from .rope import rope_inplace
            from fastbb.attention import attention
            if _ROPE_PLAIN:   # ablation: plain-PyTorch RoPE in front of the flash kernel (FASTBB_ROPE=torch)
                from .rope import rope_torch
                o = attention(*rope_torch(qkv, cs, P).unbind(2), variant=self.attn).view(B, N, C)
            else:
                rope_inplace(qkv, cs, P, qscale=(C // H) ** -0.5)   # softmax scale folded into q (exact power of 2): safer for hh
                o = attention(*qkv.unbind(2), scale=1.0, variant=self.attn).view(B, N, C)
        else:
            from .rope import rope_torch
            q, k, v = rope_torch(qkv, cs, P).permute(2, 0, 3, 1, 4).unbind(0)
            if torch.onnx.is_in_onnx_export() and C < 1024:
                # ViT-S/B (q / k outliers up to 2e3): TensorRT's fp16 attention loses ~35 % (measured), so the exported graph does
                # QK^T + softmax in fp32 (plain ops; probabilities back to fp16 for P V)
                p = torch.softmax((q.float() * (C // H) ** -0.5) @ k.float().transpose(-1, -2), -1)
                o = p.to(v.dtype) @ v
            elif torch.onnx.is_in_onnx_export():   # softmax scale folded into q before the matmul (exact 2^-3)
                o = F.scaled_dot_product_attention(q * (C // H) ** -0.5, k, v, scale=1.0)
            else:
                o = F.scaled_dot_product_attention(q, k, v)
            o = o.transpose(1, 2).reshape(B, N, C)
        if self.gacc == "chunk":
            from fastbb.gemm import linear
            x = linear(o, b.proj.weight, b.proj.bias, res=x, gamma=b.ls1)
            if b.gated:
                x12 = linear(_ln(b.norm2, x), b.w12.weight, b.w12.bias)
                x1, x2 = x12.chunk(2, dim=-1)
                return linear(F.silu(x1).mul_(x2), b.w3.weight, b.w3.bias, res=x, gamma=b.ls2)
            h = linear(_ln(b.norm2, x), b.fc1.weight, b.fc1.bias, act="gelu")
            return linear(h, b.fc2.weight, b.fc2.bias, res=x, gamma=b.ls2)
        exp = torch.onnx.is_in_onnx_export()      # ONNX addcmul promotes to fp32: plain ops for the export
        o = F.linear(o, b.proj.weight, b.proj.bias)
        x = x + o * b.ls1 if exp else torch.addcmul(x, o, b.ls1)
        y = _ln(b.norm2, x)
        if b.gated:
            x1, x2 = F.linear(y, b.w12.weight, b.w12.bias).chunk(2, dim=-1)
            m = F.linear(F.silu(x1) * x2, b.w3.weight, b.w3.bias)
        else:
            m = F.linear(F.gelu(F.linear(y, b.fc1.weight, b.fc1.bias)), b.fc2.weight, b.fc2.bias)
        return x + m * b.ls2 if exp else torch.addcmul(x, m, b.ls2)

    # ------------------------------------------------------------ API
    def tokens(self, x):
        t = self.patch_embed(x.to(self.cls_token.dtype)).flatten(2).transpose(1, 2)
        B = t.shape[0]
        return torch.cat([self.cls_token.expand(B, -1, -1), self.register_tokens.expand(B, -1, -1), t], 1)

    def _run_all(self, x, layers=None):
        """-> (list of un-normed tokens at `layers`, last un-normed)"""
        h, w = x.shape[-2] // self.ps, x.shape[-1] // self.ps
        cs = self.rope_cs(h, w, x.device)
        old = torch.backends.cuda.matmul.allow_fp16_accumulation
        if x.is_cuda:
            torch.backends.cuda.matmul.allow_fp16_accumulation = self.gacc is True
        try:
            t = self.tokens(x)
            outs = []
            for i, b in enumerate(self.blocks):
                t = self.block(b, t, cs)
                if layers is not None and i in layers:
                    outs.append(t)
        finally:
            if x.is_cuda:
                torch.backends.cuda.matmul.allow_fp16_accumulation = old
        return outs if layers is not None else t

    def forward(self, x):
        """last_hidden_state [B, 1 + nreg + HW, C] (normed) -- plain tensor in / out, ONNX-exportable for the sdpa specs."""
        t = self._graphs(x) if self._graphs is not None else self._run_all(x)
        return _ln(self.norm, t)

    def forward_features(self, x):
        n = self.forward(x)
        r = 1 + self.nreg
        return {"x_norm_clstoken": n[:, 0], "x_norm_regtokens": n[:, 1:r], "x_norm_patchtokens": n[:, r:]}

    def get_intermediate_layers(self, x, n=1, reshape=False, return_class_token=False, norm=True):
        nb = len(self.blocks)
        layers = set(range(nb - n, nb)) if isinstance(n, int) else set(n)
        outs = self._run_all(x, layers)
        if norm:
            outs = [self.norm(o) for o in outs]
        r = 1 + self.nreg
        cls = [o[:, 0] for o in outs]
        outs = [o[:, r:] for o in outs]
        if reshape:
            B, _, hh, ww = x.shape
            outs = [o.reshape(B, hh // self.ps, ww // self.ps, -1).permute(0, 3, 1, 2).contiguous() for o in outs]
        return tuple(zip(outs, cls)) if return_class_token else tuple(outs)


def _load_vit_sd(path, cfg, dtype, S=1.0):
    from safetensors.torch import load_file
    sd = load_file(os.path.join(path, "model.safetensors"), device="cpu")
    out = {"cls_token": sd["embeddings.cls_token"], "register_tokens": sd["embeddings.register_tokens"],
           "patch_embed.weight": sd["embeddings.patch_embeddings.weight"], "patch_embed.bias": sd["embeddings.patch_embeddings.bias"],
           "norm.weight": sd["norm.weight"], "norm.bias": sd["norm.bias"]}
    d = cfg["hidden_size"]
    for i in range(cfg["num_hidden_layers"]):
        p, q = f"layer.{i}.", f"blocks.{i}."
        a = p + "attention."
        out[q + "qkv.weight"] = torch.cat([sd[a + "q_proj.weight"], sd[a + "k_proj.weight"], sd[a + "v_proj.weight"]], 0)
        kb = sd[a + "k_proj.bias"] if a + "k_proj.bias" in sd else torch.zeros(d)
        out[q + "qkv.bias"] = torch.cat([sd[a + "q_proj.bias"], kb, sd[a + "v_proj.bias"]], 0)
        out[q + "proj.weight"], out[q + "proj.bias"] = sd[a + "o_proj.weight"], sd[a + "o_proj.bias"]
        for n in ("norm1", "norm2"):
            out[q + n + ".weight"], out[q + n + ".bias"] = sd[p + n + ".weight"], sd[p + n + ".bias"]
        out[q + "ls1"], out[q + "ls2"] = sd[p + "layer_scale1.lambda1"] / S, sd[p + "layer_scale2.lambda1"] / S
        m = p + "mlp."
        if cfg["use_gated_mlp"]:
            out[q + "w12.weight"] = torch.cat([sd[m + "gate_proj.weight"], sd[m + "up_proj.weight"]], 0)
            out[q + "w12.bias"] = torch.cat([sd[m + "gate_proj.bias"], sd[m + "up_proj.bias"]], 0)
            out[q + "w3.weight"], out[q + "w3.bias"] = sd[m + "down_proj.weight"], sd[m + "down_proj.bias"]
        else:
            out[q + "fc1.weight"], out[q + "fc1.bias"] = sd[m + "up_proj.weight"], sd[m + "up_proj.bias"]
            out[q + "fc2.weight"], out[q + "fc2.bias"] = sd[m + "down_proj.weight"], sd[m + "down_proj.bias"]
    if S != 1.0:   # residual stream x -> x / S (exact for a power of two): LayerNorm is scale invariant, branches enter through ls1/ls2
        for k in ("cls_token", "register_tokens", "patch_embed.weight", "patch_embed.bias"):
            out[k] = out[k] / S
    return {k: v.to(dtype) for k, v in out.items()}


def dinov3_vit(path, spec="sdpa-g0", device="cuda", dtype=torch.float16, stream_scale="auto"):
    """Load an HF-format DINOv3 ViT checkpoint directory into the fastbb implementation (fp16 on the CPU before the move).
    fp16 note: the residual stream of ViT-L/16 and H+/16 reaches 1-1.6e5 (> fp16 max 65504; HF fp16 / autocast give NaN, B: 1.5e4,
    S: 1.7e3). stream_scale="auto" (8 for hidden >= 768, else 1) divides the whole stream by that power of two through the
    embedding and LayerScale weights: mathematically identical (LayerNorm is scale invariant), no extra op at run time."""
    from fastbb import parse_vit_spec
    cfg = json.load(open(os.path.join(path, "config.json")))
    assert cfg["model_type"] == "dinov3_vit"
    graphs = spec.endswith("-graph")
    attn, gacc = parse_vit_spec(spec[:-6] if graphs else spec)
    with torch.device("meta"):
        m = DinoV3ViT(cfg, attn=attn, gacc=gacc)
    S = float(8 if cfg["hidden_size"] >= 768 else 1) if stream_scale == "auto" else float(stream_scale)
    sd = {k: v.to(device) for k, v in _load_vit_sd(path, cfg, dtype, S).items()}
    m.load_state_dict(sd, strict=True, assign=True)
    m.eval().requires_grad_(False)
    if S != 1.0:   # LayerNorm(x / S) == LayerNorm(x) only with eps / S^2
        for mod in [m.norm] + [n for b in m.blocks for n in (b.norm1, b.norm2)]:
            mod.eps = mod.eps / S ** 2
    m.stream_scale = S
    if graphs:
        from fastbb.graphs import graphed
        m._graphs = graphed(m._run_all)
    return m


# ------------------------------------------------------------------------------------------------ ConvNeXt
_TIMM = {(96, 9): "convnext_tiny", (96, 27): "convnext_small", (128, 27): "convnext_base", (192, 27): "convnext_large"}


class DinoV3ConvNeXt(nn.Module):
    """timm ConvNeXt (optionally fastbb.cnn.optimize'd) + the DINOv3 final LayerNorm over [pooled; tokens]."""

    def __init__(self, body, norm):
        super().__init__()
        self.body, self.norm = body, norm

    def forward(self, x):
        f = self.body.forward_features(x)
        B, C = f.shape[:2]
        tok = f.flatten(2).transpose(1, 2)
        t = torch.cat([tok.mean(1, keepdim=True), tok], 1)
        return self.norm(t)         # [B, 1 + HW/1024.., C]: index 0 = pooled


def dinov3_convnext(path, spec="fold-gacc-fused-trace", device="cuda", dtype=torch.float16):
    """HF DINOv3 ConvNeXt checkpoint -> timm convnext_{tiny,small,base,large} (num_classes=0) + final LN. spec "fp32" / "fp16" /
    "amp" = timm eager (upstream-like), "fold" ... = fastbb.cnn.optimize specs. Output = last_hidden_state like HF."""
    import timm
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(path, "config.json")))
    name = _TIMM[(cfg["hidden_sizes"][0], cfg["depths"][2])]
    body = timm.create_model(name, pretrained=False, num_classes=0)
    sd = load_file(os.path.join(path, "model.safetensors"), device="cpu")
    new = {}
    for k, v in sd.items():
        if k.startswith("layer_norm."):
            continue
        k = k.replace("stages.0.downsample_layers.", "stem.")
        k = k.replace("downsample_layers.", "downsample.")
        k = k.replace(".layers.", ".blocks.").replace("depthwise_conv", "conv_dw").replace("layer_norm", "norm")
        k = k.replace("pointwise_conv1", "mlp.fc1").replace("pointwise_conv2", "mlp.fc2")
        new[k] = v
    miss = body.load_state_dict(new, strict=False)
    assert not miss.unexpected_keys and all(k.startswith("head.") for k in miss.missing_keys), miss
    norm = nn.LayerNorm(cfg["hidden_sizes"][-1], eps=cfg["layer_norm_eps"])
    norm.weight.data, norm.bias.data = sd["layer_norm.weight"], sd["layer_norm.bias"]
    body.eval()
    norm.eval()
    if spec in ("fp32", "amp"):
        pass
    elif spec == "fp16":
        body, norm = body.half(), norm.half()
        return DinoV3ConvNeXt(body, norm).to(device).eval()
    else:
        from fastbb.cnn import optimize
        return optimize(DinoV3ConvNeXt(body, norm), spec)     # whole net (incl. final LN) goes through fold / fused / trace / graph
    return DinoV3ConvNeXt(body, norm).to(device).eval()

"""Feature backbones: DINOv2 (official), DINOv3 (ViT + ConvNeXt, HF checkpoints) and timm CNNs.

    from fast_vision_models import backbones
    b = backbones.build("dinov2-vitl14", "cuda-kernels")     # -> Backbone(fn, mean, std, multiple)
    tokens = b.fn(x)                                 # x: ImageNet-normalized fp32 [B, 3, H, W], H, W multiples of b.multiple

Tiers: "upstream" = official model code in fp32; "portable" = fastbb plain-PyTorch path (fp16, SDPA, packed qkv; CNN: BN /
LayerNorm fold, channels-last) - exportable; "cuda-kernels" = fastbb custom kernels (flash64 attention, fused GEMM epilogues,
fused RoPE; CNN: CUTLASS fused convs + TorchScript trace). The ViT kernels come from fastbb (github.com/fastforks/fastbb).
Outputs: DINOv2 / DINOv3 ViT = normalized patch tokens [B, HW, C]; DINOv3 ConvNeXt = last_hidden_state; timm CNNs = logits.
Weights: DINOv2 from dl.fbaipublicfiles.com (fetched, see weights.py); DINOv3 from the gated HF repos
facebook/dinov3-*-pretrain-lvd1689m (accept the DINOv3 License on HF; FVM_DINOV3_DIR = local dir of the snapshots);
timm from the HF hub.
"""

import os
from dataclasses import dataclass

import torch

from ..weights import fetch

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

DINOV2 = {"vits14": "vit_small", "vitb14": "vit_base", "vitl14": "vit_large", "vitg14": "vit_giant2"}
DINOV3 = ["vits16", "vitb16", "vitl16", "vith16plus", "convnext-tiny", "convnext-small", "convnext-base", "convnext-large"]
TIMM = ["resnet18.a1_in1k", "resnet50.a1_in1k", "resnet101.a1h_in1k", "regnety_032.ra_in1k", "efficientnet_b0.ra_in1k",
        "mobilenetv2_100.ra_in1k", "convnext_tiny.fb_in22k_ft_in1k"]
SPECS = {  # tier -> fastbb spec
    "dinov2": {"portable": "sdpa-g1", "cuda-kernels": "hh-g1"},
    "dinov3-vit": {"portable": "sdpa-g1", "cuda-kernels": "hh-g1-graph"},  # -graph: CUDA graph per input shape (as measured)
    "dinov3-convnext": {"portable": "fold", "cuda-kernels": "fold-gacc-fused-trace"},
    "timm": {"portable": "fold", "cuda-kernels": "fold-gacc-fused-trace"},
}


@dataclass
class Backbone:
    fn: object
    mean: tuple
    std: tuple
    multiple: int
    module: object = None
    info: dict = None


def names():
    return [f"dinov2-{a}" for a in DINOV2] + [f"dinov3-{m}" for m in DINOV3] + [f"timm-{m}" for m in TIMM]


def _dinov2(arch, tier, spec):
    w = fetch(f"https://dl.fbaipublicfiles.com/dinov2/dinov2_{arch}/dinov2_{arch}_pretrain.pth")
    if tier == "upstream":
        repo = os.environ.get("FVM_DINOV2_REPO")  # local clone of facebookresearch/dinov2, else torch.hub
        src = {"source": "local"} if repo else {}
        m = torch.hub.load(repo or "facebookresearch/dinov2", f"dinov2_{arch}", pretrained=False, **src)
        m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True), strict=True)
        m = m.cuda().eval()
        return Backbone(lambda x: m.forward_features(x)["x_norm_patchtokens"], IMNET_MEAN, IMNET_STD, 14, m)
    from fastbb.models.dinov2 import dinov2

    m = dinov2(arch, w, spec=spec or SPECS["dinov2"][tier])
    return Backbone(lambda x: m.forward_features(x)["x_norm_patchtokens"], IMNET_MEAN, IMNET_STD, 14, m)


def _dinov3_path(model):
    repo = f"dinov3-{model}-pretrain-lvd1689m"
    root = os.environ.get("FVM_DINOV3_DIR")
    if root and os.path.isdir(os.path.join(root, repo)):
        return os.path.join(root, repo)
    from huggingface_hub import snapshot_download

    return snapshot_download(f"facebook/{repo}")


def _dinov3(model, tier, spec):
    p = _dinov3_path(model)
    vit = model.startswith("vit")
    if tier == "upstream":
        from transformers import DINOv3ConvNextModel, DINOv3ViTModel

        cls = DINOv3ViTModel if vit else DINOv3ConvNextModel
        m = cls.from_pretrained(p, dtype=torch.float32, **({"attn_implementation": "sdpa"} if vit else {})).cuda().eval()
        if vit:
            return Backbone(lambda x: m(pixel_values=x).last_hidden_state[:, 5:], IMNET_MEAN, IMNET_STD, 16, m)
        return Backbone(lambda x: m(pixel_values=x).last_hidden_state, IMNET_MEAN, IMNET_STD, 32, m)
    from .dinov3 import dinov3_convnext, dinov3_vit

    if vit:
        if spec is None and tier in ("max", "cuda-kernels") and model in ("vits16", "vitb16"):
            spec = "fh-g1-graph"   # ViT-S/B q / k outliers overflow the fp16-accumulate hh attention (NaN): fp32-acc softmax
        m = dinov3_vit(p, spec or SPECS["dinov3-vit"][tier])
        return Backbone(lambda x: m(x)[:, 5:], IMNET_MEAN, IMNET_STD, 16, m)
    m = dinov3_convnext(p, spec or SPECS["dinov3-convnext"][tier])
    return Backbone(m, IMNET_MEAN, IMNET_STD, 32, m)


def _timm(model, tier, spec):
    import timm

    m = timm.create_model(model, pretrained=True).eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    if tier == "upstream":
        m = m.cuda()
        return Backbone(m, tuple(cfg["mean"]), tuple(cfg["std"]), 32, m)
    from fastbb.cnn import optimize

    f = optimize(m, spec=spec or SPECS["timm"][tier])
    return Backbone(f, tuple(cfg["mean"]), tuple(cfg["std"]), 32, m)


def build(name, tier="cuda-kernels", spec=None):
    """name from names(); tier upstream | portable | cuda-kernels; spec overrides the fastbb spec of the tier."""
    if tier not in ("upstream", "portable", "max", "cuda-kernels"):
        raise ValueError(tier)
    tier = "cuda-kernels" if tier == "max" else tier  # old name
    fam, _, model = name.partition("-")
    if fam == "dinov2":
        return _dinov2(model, tier, spec)
    if fam == "dinov3":
        return _dinov3(model, tier, spec)
    if fam == "timm":
        return _timm(model, tier, spec)
    raise KeyError(name)

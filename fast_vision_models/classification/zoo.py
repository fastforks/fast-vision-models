"""ImageNet-1k classification model registry.

    from fast_vision_models.classification import zoo
    m = zoo.load("resnet50", impl="max")   # -> Loaded(model, transform, input_size, spec, info)
    logits = m.model(x)                              # x: [B, 3, H, W] float32, normalized (m.transform output)

impl ("upstream" / "portable" / "max" are aliases of base / friendly / opt): "base" = upstream fp32 eager (torchvision / timm / HF defaults, no cudnn.benchmark),
      "friendly" = upstream-friendly general path (plain PyTorch, exportable changes only),
      "opt"  = max general fast path from classification/fast (any batch / size, nothing compiled or tuned per size).
Every model maps a preprocessed fp32 NCHW batch to 1000 ImageNet logits (index order = sorted WordNet ids).
"""
import os
from dataclasses import dataclass, field

import torch

ALIASES = {"upstream": "base", "portable": "friendly", "max": "opt"}


@dataclass
class Spec:
    name: str
    year: int
    src: str            # tv | timm | siglip2 | dinov2
    ident: str          # torchvision builder:weights | timm name | HF repo
    family: str         # cnn | dwcnn | vit | swin | hybrid | clip
    paper: str = ""
    licence: str = ""
    pop: int = 99       # popularity rank (order of optimization)
    extra: dict = field(default_factory=dict)


SPECS = [
    # --- classics (torchvision weights) ---
    Spec("alexnet", 2012, "tv", "alexnet:IMAGENET1K_V1", "cnn", "AlexNet", "BSD-3 (torchvision)", 12),
    Spec("vgg16", 2014, "tv", "vgg16:IMAGENET1K_V1", "cnn", "VGG-16", "BSD-3 (torchvision)", 6),
    Spec("inception_v3", 2015, "tv", "inception_v3:IMAGENET1K_V1", "cnn", "Inception-v3 (GoogLeNet family)", "BSD-3 (torchvision)", 11),
    Spec("resnet18", 2015, "tv", "resnet18:IMAGENET1K_V1", "cnn", "ResNet-18", "BSD-3 (torchvision)", 3),
    Spec("resnet50", 2015, "tv", "resnet50:IMAGENET1K_V2", "cnn", "ResNet-50 (V2 recipe weights)", "BSD-3 (torchvision)", 1),
    Spec("resnet152", 2015, "tv", "resnet152:IMAGENET1K_V2", "cnn", "ResNet-152 (V2 recipe weights)", "BSD-3 (torchvision)", 13),
    Spec("densenet121", 2016, "tv", "densenet121:IMAGENET1K_V1", "cnn", "DenseNet-121", "BSD-3 (torchvision)", 10),
    Spec("squeezenet1_1", 2016, "tv", "squeezenet1_1:IMAGENET1K_V1", "cnn", "SqueezeNet 1.1", "BSD-3 (torchvision)", 21),
    Spec("mobilenet_v1", 2017, "timm", "mobilenetv1_100.ra4_e3600_r224_in1k", "dwcnn", "MobileNet V1 (timm ra4 recipe)", "Apache-2.0 (timm)", 22),
    Spec("mobilenet_v2", 2018, "tv", "mobilenet_v2:IMAGENET1K_V2", "dwcnn", "MobileNet V2", "BSD-3 (torchvision)", 4),
    Spec("shufflenet_v2", 2018, "tv", "shufflenet_v2_x1_0:IMAGENET1K_V1", "dwcnn", "ShuffleNet V2 x1.0", "BSD-3 (torchvision)", 20),
    Spec("mobilenet_v3_l", 2019, "tv", "mobilenet_v3_large:IMAGENET1K_V2", "dwcnn", "MobileNet V3 Large", "BSD-3 (torchvision)", 9),
    Spec("efficientnet_b0", 2019, "tv", "efficientnet_b0:IMAGENET1K_V1", "dwcnn", "EfficientNet-B0", "BSD-3 (torchvision)", 5),
    Spec("efficientnet_b4", 2019, "tv", "efficientnet_b4:IMAGENET1K_V1", "dwcnn", "EfficientNet-B4", "BSD-3 (torchvision)", 15),
    Spec("regnety_16gf", 2020, "tv", "regnet_y_16gf:IMAGENET1K_V2", "cnn", "RegNetY-16GF", "BSD-3 (torchvision)", 19),
    Spec("efficientnet_v2_s", 2021, "tv", "efficientnet_v2_s:IMAGENET1K_V1", "dwcnn", "EfficientNetV2-S", "BSD-3 (torchvision)", 14),
    # --- transformer era (timm) ---
    Spec("vit_b16", 2020, "timm", "vit_base_patch16_224.augreg2_in21k_ft_in1k", "vit", "ViT-B/16 (AugReg2, in21k->in1k)", "Apache-2.0", 2),
    Spec("deit_b", 2021, "timm", "deit_base_patch16_224.fb_in1k", "vit", "DeiT-B (in1k only)", "Apache-2.0", 16),
    Spec("swin_b", 2021, "timm", "swin_base_patch4_window7_224.ms_in22k_ft_in1k", "swin", "Swin-B (in22k->in1k)", "MIT", 8),
    Spec("convnext_v2_b", 2023, "timm", "convnextv2_base.fcmae_ft_in22k_in1k", "dwcnn", "ConvNeXt V2-B (FCMAE, in22k->in1k)", "CC BY-NC 4.0 (weights)", 7),
    Spec("eva02_l", 2023, "timm", "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", "vit", "EVA-02-L/14 448 (MIM m38m, in22k->in1k)", "MIT", 18),
    Spec("fastvit_sa24", 2023, "timm", "fastvit_sa24.apple_dist_in1k", "hybrid", "FastViT-SA24 (distilled)", "Apple sample-code licence (code + weights)", 24),
    Spec("efficientvit_b2", 2023, "timm", "efficientvit_b2.r288_in1k", "hybrid", "EfficientViT-B2 (MIT Han lab, 288)", "Apache-2.0", 25),
    Spec("mobilenet_v4_m", 2024, "timm", "mobilenetv4_conv_medium.e500_r256_in1k", "dwcnn", "MobileNetV4-Conv-M", "Apache-2.0", 23),
    # --- foundation-model classifiers ---
    Spec("dinov2_l_lin", 2023, "dinov2", "vit_large_patch14_dinov2.lvd142m", "vit", "DINOv2 ViT-L/14 + official linear head", "Apache-2.0", 17),
    Spec("siglip2_so400m_zs", 2025, "siglip2", "google/siglip2-so400m-patch14-384", "clip", "SigLIP 2 So400m/14 384 zero-shot", "Apache-2.0", 17),
    Spec("siglip_so400m_ft", 2024, "timm", "vit_so400m_patch14_siglip_378.webli_ft_in1k", "vit", "SigLIP So400m/14 378 fine-tuned in1k", "Apache-2.0", 26),
]
BY_NAME = {s.name: s for s in SPECS}


@dataclass
class Loaded:
    model: torch.nn.Module
    transform: object
    input_size: tuple
    spec: Spec
    info: dict


# ---------------------------------------------------------------- builders
def _tv(spec):
    import torchvision
    fn, wname = spec.ident.split(":")
    weights = torchvision.models.get_model_weights(fn)[wname]
    model = torchvision.models.get_model(fn, weights=weights)
    tf = weights.transforms()
    return model, tf, (3, tf.crop_size[0], tf.crop_size[0]), {"weights": f"{fn}.{wname}", "acc@1_ref": weights.meta["_metrics"]["ImageNet-1K"]["acc@1"]}


def _timm(spec):
    import timm
    model = timm.create_model(spec.ident, pretrained=True)
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    return model, tf, tuple(cfg["input_size"]), {"weights": spec.ident, "data_cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}}


class DinoLinear(torch.nn.Module):
    """DINOv2 ViT-L/14 backbone (timm port of the official weights) + official linear head (dinov2_vitl14_linear_head.pth,
    torch.hub `dinov2_vitl14_lc`, layers=1): logits = W [cls ; mean(patch tokens)] + b."""

    def __init__(self, vit, head):
        super().__init__()
        self.vit, self.head = vit, head

    def forward(self, x):
        t = self.vit.forward_features(x)                    # [B, 1 + N, C] after final norm
        return self.head(torch.cat([t[:, 0], t[:, 1:].mean(1)], dim=1))


def _dinov2(spec):
    import timm
    import torchvision.transforms as T
    vit = timm.create_model(spec.ident, pretrained=True, num_classes=0, img_size=224)
    url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth"
    sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    head = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0])
    head.load_state_dict(sd)
    tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return DinoLinear(vit, head), tf, (3, 224, 224), {"weights": spec.ident + " + " + url}


class SigLIPZeroShot(torch.nn.Module):
    """SigLIP 2 image tower + projection; logits = scaled cosine similarity to 1000 precomputed class text embeddings
    (prompt 'this is a photo of a {}.', open_clip IMAGENET_CLASSNAMES). Text embeddings are computed at load (startup)."""

    def __init__(self, vision, text_emb, scale, bias):
        super().__init__()
        self.vision = vision
        self.register_buffer("text_emb", text_emb)       # [1000, D] normalized
        self.scale, self.bias = scale, bias

    def forward(self, x):
        f = self.vision(pixel_values=x).pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        return (f @ self.text_emb.to(f.dtype).t()) * self.scale + self.bias


def _siglip2(spec, device):
    import torchvision.transforms as T
    from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
    from transformers import AutoModel, AutoTokenizer
    m = AutoModel.from_pretrained(spec.ident, dtype=torch.float32).eval().to(device)
    tok = AutoTokenizer.from_pretrained(spec.ident)
    texts = [f"this is a photo of a {c}." for c in IMAGENET_CLASSNAMES]
    embs = []
    with torch.inference_mode():
        for i in range(0, len(texts), 250):
            t = tok(texts[i:i + 250], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
            e = m.get_text_features(**t)
            e = e.pooler_output if hasattr(e, "pooler_output") else e
            embs.append(e / e.norm(dim=-1, keepdim=True))
    temb = torch.cat(embs)
    zs = SigLIPZeroShot(m.vision_model, temb, float(m.logit_scale.exp()), float(m.logit_bias))
    del m.text_model
    size = 384
    tf = T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return zs, tf, (3, size, size), {"weights": spec.ident, "prompt": "this is a photo of a {}."}


def load(name, impl="base", device="cuda"):
    impl = ALIASES.get(impl, impl)
    spec = BY_NAME[name]
    if spec.src == "tv":
        model, tf, size, info = _tv(spec)
    elif spec.src == "timm":
        model, tf, size, info = _timm(spec)
    elif spec.src == "dinov2":
        model, tf, size, info = _dinov2(spec)
    elif spec.src == "siglip2":
        model, tf, size, info = _siglip2(spec, device)
    else:
        raise ValueError(spec.src)
    model = model.eval()
    info["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    if impl in ("opt", "friendly"):
        from . import fast as clf_fast
        mode = "friendly" if impl == "friendly" else "max"
        if os.environ.get("CLF_BACKEND", "").startswith("fastbb"):
            model = clf_fast.optimize(model.to(device), spec, mode)
        else:  # fold + fp16 on the CPU, then move: no fp32 copy in the CUDA caching allocator
            model = clf_fast.optimize(model, spec, mode).to(device)
    elif impl == "base":
        model = model.to(device)
    else:
        raise ValueError(impl)
    return Loaded(model, tf, size, spec, info)

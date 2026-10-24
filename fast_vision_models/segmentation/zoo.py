"""Segmentation model registry: build(name, impl) -> object with .logits(x), .norm, .protocol, .info (see common.py).

impl names: "upstream" (= "base"), "portable" (= "uf"), "cuda-kernels" (= "opt", old name "max").
impl = "base": upstream code + upstream precision (fp32 unless the upstream demo uses AMP), default kernels.
impl = "opt" : "max" general fast path from segmentation/fast (any size, nothing compiled / tuned per size; prebuilt custom kernels,
               fp16-accumulate cuDNN / cuBLAS where accuracy holds).
impl = "uf"  : upstream-friendly version of the same path: only plain PyTorch ops (ONNX / TensorRT exportable): fp16,
               layout, algebraic folding, SDPA; no custom kernels, no cuDNN-frontend convs, no fp16-accumulate flags.
"""
import os

import torch

from ..weights import fetch
from .common import IMNET

ALIASES = {"upstream": "base", "portable": "uf", "cuda-kernels": "opt", "max": "opt"}

REG = {}
GACC = os.environ.get("SEG_GACC", "1") == "1"  # opt: fp16-accumulate GEMMs in the transformer paths (val-500 agreement >= 99.85 %)


def register(name):
    def deco(f):
        REG[name] = f
        return f
    return deco


GRAPH = os.environ.get("SEG_GRAPH", "0") == "1"   # opt: CUDA-graph replay of predict per input shape (seg_fast.ops.GraphCache)
# Round 2: default OFF (graphs are captured per input shape = size-specialized, not "general"); SEG_GRAPH=1 = the separate
# "max + graphs" column. SEG_GRAPH_TRIM / SEG_GRAPH_WARM / SEG_GRAPH_MAXPIX: GraphCache memory options (see ops.py).
GRAPH_OPTS = dict(trim=os.environ.get("SEG_GRAPH_TRIM", "1") == "1", warm=os.environ.get("SEG_GRAPH_WARM", "0") == "1",
                  max_pixels=int(os.environ["SEG_GRAPH_MAXPIX"]) if os.environ.get("SEG_GRAPH_MAXPIX") else None)
NO_GRAPH = set()  # (SegNeXt's Ham head draws its random NMF bases on the GPU in the max version -> capturable)


class Wrap:
    def __init__(self, fn, norm, protocol, info, module=None, predict=None):
        self.logits, self.norm, self.protocol, self.info, self.module = fn, norm, protocol, info, module
        self.predict = predict  # optional fused path: predict(x, out_size) -> int map [1, H, W]


# ------------------------------------------------------------------ SegFormer (HF transformers, nvidia weights)
def _segformer(repo, short, impl):
    from transformers import SegformerForSemanticSegmentation
    m = SegformerForSemanticSegmentation.from_pretrained(repo).cuda().eval()
    proto = {"resize": ("mmseg", (2048, short)), "infer": "whole", "pad": 1}
    info = {"repo": repo, "attn": m.config._attn_implementation}
    if impl == "base":
        return Wrap(lambda x: m(pixel_values=x).logits, IMNET, proto, info, m)
    from .fast.segformer import FastSegformer
    f = FastSegformer(m, fused=impl == "opt", gacc=GACC and impl == "opt")
    return Wrap(f.logits, IMNET, proto, info | f.info, m, f.predict)


@register("segformer-b0")
def segformer_b0(impl):
    return _segformer("nvidia/segformer-b0-finetuned-ade-512-512", 512, impl)


@register("segformer-b5")
def segformer_b5(impl):
    return _segformer("nvidia/segformer-b5-finetuned-ade-640-640", 640, impl)


def build(name, impl):
    impl = ALIASES.get(impl, impl)
    w = REG[name](impl)
    if impl == "opt" and GRAPH and w.predict is not None and name not in NO_GRAPH:
        from .fast.ops import GraphCache
        w.predict = GraphCache(w.predict, pre_replay=getattr(getattr(w.predict, "__self__", None), "pre_replay", ()),
                               **GRAPH_OPTS)
        w.info = dict(w.info, cuda_graph=True, graph_opts=GRAPH_OPTS)
    return w


# ------------------------------------------------------------------ mask classification (Mask2Former / OneFormer / EoMT)
def mask_scores(mask_logits, class_logits, size=None):
    """detectron2 / EoMT semantic_inference: mask logits upsampled to `size` (bilinear) -> sigmoid, class softmax without
    the void class -> einsum -> [B, classes, H, W]."""
    if size is not None and tuple(mask_logits.shape[-2:]) != tuple(size):
        mask_logits = torch.nn.functional.interpolate(mask_logits, size, mode="bilinear", align_corners=False)
    return torch.einsum("bqc,bqhw->bchw", class_logits.softmax(-1)[..., :-1], mask_logits.sigmoid())


def _hub(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_files_only=True)


@register("mask2former-swinl")
def mask2former_swinl(impl):
    from transformers import Mask2FormerForUniversalSegmentation
    repo = "facebook/mask2former-swin-large-ade-semantic"
    m = Mask2FormerForUniversalSegmentation.from_pretrained(repo).cuda().eval()
    proto = {"resize": ("short", 640, 2560), "infer": "whole", "pad": 32}
    info = {"repo": repo}
    if impl == "base":
        def fn(x):
            o = m(pixel_values=x)
            return mask_scores(o.masks_queries_logits, o.class_queries_logits, x.shape[2:])
        return Wrap(fn, IMNET, proto, info, m)
    from .fast.maskformer import FastMask2Former
    f = FastMask2Former(m, fused=impl == "opt", gacc=GACC and impl == "opt")
    return Wrap(f.logits, IMNET, proto, info | f.info, m, f.predict)


@register("oneformer-swinl")
def oneformer_swinl(impl):
    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
    repo = "shi-labs/oneformer_ade20k_swin_large"
    m = OneFormerForUniversalSegmentation.from_pretrained(repo).cuda().eval()
    proc = OneFormerProcessor.from_pretrained(repo)
    import numpy as np
    task = proc(images=[np.zeros((64, 64, 3), np.uint8)], task_inputs=["semantic"], return_tensors="pt")["task_inputs"].cuda()
    proto = {"resize": ("short", 640, 2560), "infer": "whole", "pad": 32}
    info = {"repo": repo, "task": "semantic"}
    if impl == "base":
        def fn(x):
            o = m(pixel_values=x, task_inputs=task)
            return mask_scores(o.masks_queries_logits, o.class_queries_logits, x.shape[2:])
        return Wrap(fn, IMNET, proto, info, m)
    from .fast.maskformer import FastOneFormer
    f = FastOneFormer(m, task, fused=impl == "opt", gacc=GACC and impl == "opt")
    return Wrap(f.logits, IMNET, proto, info | f.info, m, f.predict)


def _eomt(repo, cls_name, impl):
    import transformers
    m = getattr(transformers, cls_name).from_pretrained(repo).cuda().eval()
    proto = {"resize": ("short", 512, 10 ** 9), "infer": ("eomt", 512), "pad": 1}
    info = {"repo": repo}
    if impl == "base":
        def fn(x):
            o = m(pixel_values=x)
            return mask_scores(o.masks_queries_logits, o.class_queries_logits, x.shape[2:])
        return Wrap(fn, IMNET, proto, info, m)
    from .fast.eomt import FastEomt
    f = FastEomt(m, **({"gacc": GACC} if impl == "opt" else {"attn": "sdpa", "fused": False}))
    return Wrap(f.logits, IMNET, proto, info | f.info, m, getattr(f, "predict", None))


@register("eomt-l-dinov2")
def eomt_l_dinov2(impl):
    return _eomt("tue-mps/ade20k_semantic_eomt_large_512", "EomtForUniversalSegmentation", impl)


@register("eomt-l-dinov3")
def eomt_l_dinov3(impl):
    return _eomt("tue-mps/eomt-dinov3-ade-semantic-large-512", "EomtDinov3ForUniversalSegmentation", impl)


# ------------------------------------------------------------------ DINOv2 ViT-L/14 + linear (BN) head, official weights
DINOV2_W = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
DINOV2_HEAD = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_ade20k_linear_head.pth"


def center_pad(x, p):
    h, w = x.shape[2:]
    ph, pw = (-h) % p, (-w) % p
    return torch.nn.functional.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


@register("dinov2-l-linear")
def dinov2_l_linear(impl):
    wpath, hpath = fetch(DINOV2_W), fetch(DINOV2_HEAD)
    sd = torch.load(hpath, map_location="cuda", weights_only=False)  # official mmcv checkpoint (meta needs full unpickling)
    sd = sd.get("state_dict", sd)
    proto = {"resize": ("mmseg", (99999999, 512)), "infer": ("slide", 512, 341), "pad": 1}
    info = {"backbone": DINOV2_W, "head": DINOV2_HEAD}
    if impl == "base":
        repo = os.environ.get("FVM_DINOV2_REPO")  # local clone of facebookresearch/dinov2, else torch.hub (@7764ea0 tested)
        src = {"source": "local"} if repo else {}
        vit = torch.hub.load(repo or "facebookresearch/dinov2", "dinov2_vitl14", pretrained=False, **src)
        vit.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
        vit = vit.cuda().eval()
        bn = torch.nn.BatchNorm2d(1024).cuda().eval()
        bn.load_state_dict({k.split("bn.")[1]: v for k, v in sd.items() if ".bn." in k})
        cls_w, cls_b = sd["decode_head.conv_seg.weight"], sd["decode_head.conv_seg.bias"]

        def fn(x):
            f = vit.get_intermediate_layers(center_pad(x, 14), n=[20, 21, 22, 23], reshape=True)[3]
            return torch.nn.functional.conv2d(bn(f), cls_w, cls_b)
        info["commit"] = "dinov2@7764ea0"
        return Wrap(fn, IMNET, proto, info, vit)
    from .fast.dinov2_linear import FastDinov2Linear
    f = FastDinov2Linear(wpath, sd, **(({"spec": "fh-g1"} if GACC else {}) if impl == "opt" else {"spec": "sdpa-g0", "fused": False}))
    return Wrap(f.logits, IMNET, proto, info | f.info, None, getattr(f, "predict", None))


# ------------------------------------------------------------------ MMSegmentation classics (official mmseg checkpoints)
MMSEG = {  # name -> (config, official checkpoint URL)
    "fcn-r101": ("fcn/fcn_r101-d8_4xb4-160k_ade20k-512x512.py", 
                 "https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r101-d8_512x512_160k_ade20k/fcn_r101-d8_512x512_160k_ade20k_20200615_105816-fd192bd5.pth"),
    "pspnet-r101": ("pspnet/pspnet_r101-d8_4xb4-160k_ade20k-512x512.py", 
                 "https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x512_160k_ade20k/pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth"),
    "deeplabv3-r101": ("deeplabv3/deeplabv3_r101-d8_4xb4-160k_ade20k-512x512.py", 
                 "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_160k_ade20k/deeplabv3_r101-d8_512x512_160k_ade20k_20200615_105816-b1f72b3b.pth"),
    "deeplabv3plus-r101": ("deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py", 
                 "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k/deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38ed86bb.pth"),
    "hrnet-w48": ("hrnet/fcn_hr48_4xb4-160k_ade20k-512x512.py", 
                 "https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x512_160k_ade20k/fcn_hr48_512x512_160k_ade20k_20200614_214407-a52fc02c.pth"),
    "segnext-t": ("segnext/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py", 
                 "https://download.openmmlab.com/mmsegmentation/v0.5/segnext/segnext_mscan-t_1x16_512x512_adamw_160k_ade20k/segnext_mscan-t_1x16_512x512_adamw_160k_ade20k_20230210_140244-05bd8466.pth"),
    "unet-s5": ("unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py", 
                 "https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth"),
    "pidnet-l": ("pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py", 
                 "https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes/pidnet-l_2xb6-120k_1024x1024-cityscapes_20230303_114514-0783ca6b.pth"),
    "ddrnet-23": ("ddrnet/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.py", 
                 "https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230425_162633-81601db0.pth"),
}
CITYSCAPES_ONLY = {"unet-s5", "pidnet-l", "ddrnet-23"}


def _mm_load(model, path):
    """Official mmseg checkpoint -> model with torch.load(weights_only=True) (allowlisting the training-state globals)."""
    import codecs

    import numpy as np
    from mmengine.logging.history_buffer import HistoryBuffer
    from mmengine.runner.checkpoint import load_state_dict

    mc = np._core.multiarray
    ok = [HistoryBuffer, (mc._reconstruct, "numpy.core.multiarray._reconstruct"), np.ndarray, np.dtype, codecs.encode,
          (mc.scalar, "numpy.core.multiarray.scalar"), np.dtypes.Float64DType, np.dtypes.Int64DType, np.dtypes.Float32DType]
    with torch.serialization.safe_globals(ok):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    load_state_dict(model, ck.get("state_dict", ck), strict=True)


class MMSegNet(torch.nn.Module):
    """backbone (+neck) + decode_head.forward -> low-resolution logits (what encode_decode computes before its resize)."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        f = self.m.extract_feat(x)
        out = self.m.decode_head.forward(f)
        return out[0] if isinstance(out, (list, tuple)) else out


def _mmseg(name, impl):
    import warnings
    warnings.filterwarnings("ignore")
    import mmseg
    from mmseg.apis import init_model
    cfg, ck = MMSEG[name]
    m = init_model(os.path.join(os.path.dirname(mmseg.__file__), ".mim", "configs", cfg), None, device="cuda")
    _mm_load(m, fetch(ck))
    m = m.eval()
    align = bool(getattr(m.decode_head, "align_corners", False))
    proto = {"resize": ("mmseg", (2048, 512)), "infer": "whole", "pad": 1, "align": align}
    if name in CITYSCAPES_ONLY:
        proto["resize"] = ("mmseg", (2048, 1024))
    info = {"config": cfg, "ckpt": ck, "mmseg": "1.2.2", "test_cfg": str(m.test_cfg)}
    pre = m.data_preprocessor
    norm = (tuple(pre.mean.flatten().tolist()), tuple(pre.std.flatten().tolist()))
    net = MMSegNet(m).eval()
    if impl == "base":
        return Wrap(net, norm, proto, info, m)
    from .fast.mmseg_fast import FastMMSeg
    uf_spec = os.environ.get("SEG_SN_UF_SPEC", "fold-nchw") if name == "segnext-t" else "fold"  # SegNeXt uf: NCHW (r2)
    f = FastMMSeg(net, align=align, **({} if impl == "opt" else {"spec": uf_spec, "fused": False}))
    return Wrap(f.logits, norm, proto, info | f.info, m, f.predict)


for _n in MMSEG:
    register(_n)(lambda impl, _n=_n: _mmseg(_n, impl))


# ------------------------------------------------------------------ SegNet (own implementation of the paper, random init)
@register("segnet")
def segnet(impl):
    from .fast.segnet import build
    m = build(150)
    proto = {"resize": ("mmseg", (2048, 512)), "infer": "whole", "pad": 32}
    info = {"weights": "random init (no public ADE20K / general weights), latency only"}
    if impl == "base":
        return Wrap(m, IMNET, proto, info, m)
    from .fast.mmseg_fast import FastMMSeg
    f = FastMMSeg(m, **({} if impl == "opt" else {"spec": "fold", "fused": False}))
    return Wrap(f.logits, IMNET, proto, info | f.info, m, f.predict)

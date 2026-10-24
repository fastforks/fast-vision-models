# fast-vision-models

Faster PyTorch inference for 27 image classifiers, 17 semantic segmentation models and 19 feature backbones.
The models stay the same (same official weights and outputs). Only the implementation changes.

Each model comes in four tiers:

| tier | what it is | exportable to ONNX / TensorRT |
|---|---|---|
| **upstream (fp32)** | the official model code (timm, torchvision, Hugging Face transformers, mmsegmentation, DINOv2) with PyTorch defaults | yes |
| **upstream + flags** | the same unmodified code with the simple switches listed below | yes |
| **portable** | our plain-PyTorch implementation: BatchNorm / LayerScale folding, fp16 weights and activations, channels-last, SDPA attention, packed QKV and fused linear algebra | yes |
| **cuda-kernels** | portable plus custom CUDA kernels: CUTLASS fused conv / GEMM epilogues, depthwise, pooling and norm kernels, flash attention for head_dim 64 ([fastbb](https://github.com/fastforks/fastbb)), fused RoPE, TorchScript tracing | no |

All tiers work for any batch size and input size. Nothing is compiled or autotuned per size, except the cuDNN autotuning
of the flags row and the CUDA graphs of the DINOv3 ViT cuda-kernels tier (one graph per input shape).

## Quick start

```bash
pip install -e ".[classification]"          # or [segmentation], [backbones]; add [cuda-kernels] for the custom kernels
export CUTLASS_DIR=/path/to/cutlass         # cuda-kernels tier only: CUTLASS headers (v4.x), the kernels build on first use
```

```python
import torch
from fast_vision_models.classification import zoo
m = zoo.load("resnet50", "cuda-kernels")                     # tiers: upstream | portable | cuda-kernels
logits = m.model(torch.randn(8, 3, 224, 224, device="cuda"))

from fast_vision_models.segmentation import zoo as seg
from fast_vision_models.segmentation.common import classmap
w = seg.build("segformer-b0", "portable")
labels = classmap(w, x)                             # x: normalized [1, 3, H, W] fp32 (w.norm), labels: [1, H, W]

from fast_vision_models import backbones
b = backbones.build("dinov2-vitl14", "cuda-kernels")
tokens = b.fn(x)                                    # [B, HW/14², 1024] normalized patch tokens
```

Requirements: an NVIDIA GPU with fp16 tensor cores (sm_80 or newer), PyTorch 2.8 or newer, and CUDA_HOME pointing to an
nvcc that matches the CUDA version of your PyTorch build (cuda-kernels tier). Weights are downloaded from the official sources on
first use. `FVM_WEIGHTS` (directories, `:`-separated) points at checkpoints you already have.

| family | models | names |
|---|---|---|
| classification | 27 | `zoo.SPECS` (torchvision, timm, DINOv2 + linear head, SigLIP / SigLIP 2) |
| segmentation | 17 | `zoo.REG` (mmsegmentation CNNs, SegFormer, Mask2Former, OneFormer, EoMT, DINOv2 linear, SegNet) |
| backbones | 19 | `backbones.names()` (DINOv2, DINOv3 ViT + ConvNeXt, timm CNNs) |

## Results

RTX 5060 Ti 16 GB (sm_120), PyTorch 2.14.1+cu130, CUDA 13.0, cuDNN 9.24, timm 1.0.30, transformers 5.19.0, mmsegmentation 1.2.2.

**Flags row.** This is the unmodified upstream model with `torch.inference_mode`, `cudnn.benchmark`, TF32, fp16 weights
and inputs (`model.half()`), and channels-last. Classifiers and ViT backbones also get
`torch.backends.cuda.matmul.allow_fp16_accumulation`, which the portable tier uses as well. Where `model.half()` breaks
the model (NaN, or less than 90 % agreement), the row uses fp16 autocast instead: efficientnet_b4, segnext-t,
eomt-l-dinov3 and the DINOv3 ViT-L. The "flags precision" column shows which one applied. `torch.compile` is not part of
any tier.

**How to read the tables.**
- Times are GPU latency in ms: the median of 30 runs (classification b1), 10 (b64, backbones) or 20 (segmentation),
  after a warm-up.
- Peak memory is the PyTorch allocator peak, allocated / reserved, in MB, weights included.
- The speed-up columns divide the upstream and flags times by the cuda-kernels time.
- Quality compares each tier's output with upstream fp32:
  - classification: top-1 agreement in % on 256 ImageNet val images;
  - segmentation: pixel agreement in % on 12 ADE20K val images, or on Cityscapes-sized inputs for unet-s5, pidnet-l
    and ddrnet-23;
  - backbones: relative L2 error of the output features.
- ¹ marks rows where the portable tier is faster than cuda-kernels. For those models, use portable.
- ² marks cuda-kernels paths that this package fixed after the cited measurements, so the cuda-kernels ms and MB columns come from the new
  `bench/bench.py` runs:
  - fastvit_sa24: the half2-accumulate depthwise kernel gave only 89.5 % top-1 agreement on the re-parameterized RepMixer
    blocks, so FastViT now uses the fp32-accumulate depthwise kernel.
  - DINOv3 ViT-S/16 and ViT-B/16: the fp16-accumulate `hh` attention overflowed (NaN), so these two now use the
    fp32-softmax `fh` variant.

**Sources.** The speed, memory and flags / portable quality numbers come from the algo-only measurements of 2026-10-12:
- results/2026-10-12_classification_algo_only.md
- results/2026-10-12_segmentation_algo_only.md
- results/2026-10-12_backbones_algo_only.md

In those measurements the portable column ran with `cudnn.benchmark` switched on. In this package, portable and cuda-kernels run as
shipped: they set their own fp16-accumulate flags and leave `cudnn.benchmark` off.

The cuda-kernels-tier quality and all startup numbers are new runs of `bench/bench.py` with this package:
results/2026-10-23_bench.jsonl (measured 2026-10-23/24 with the code of the first commit; its `commit` field is "n/a" because the repository did not exist yet). The earlier zoo tables, with full-subset top-1 / mIoU and TensorRT numbers, are in results/
as well.

### Classification

| model | setting | upstream (fp32) ms | upstream + flags ms | portable ms | cuda-kernels ms | cuda-kernels vs flags | cuda-kernels vs upstream | peak MB alloc / res: upstream · flags · portable · cuda-kernels | flags precision | quality: flags / portable / cuda-kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| alexnet | b1 | 0.89 | 0.598 | 0.635 | 0.42 | 1.42× | 2.12× | 270 / 278 · 154 / 170 · 150 / 152 · 155 / 162 | half | 99.6 / 99.6 / 99.2 |
| alexnet | b64 | 9.52 | 3.33 | 3.45 | 1.99 | 1.67× | 4.78× | 465 / 630 · 350 / 394 · 256 / 278 · 298 / 304 | half | 99.6 / 99.6 / 99.2 |
| vgg16 | b1 | 3.45 | 1.69 | 1.71 | 1.31 | 1.29× | 2.63× | 611 / 622 · 333 / 396 · 310 / 396 · 323 / 340 | half | 100.0 / 100.0 / 99.6 |
| vgg16 | b64 | 143 | 67.2 | 67.4 | 28.8 | 2.33× | 4.97× | 3767 / 3776 · 2666 / 2750 · 1233 / 1572 · 1149 / 1158 | half | 100.0 / 100.0 / 99.6 |
| inception_v3 | b1 | 6.47 | 5.93 | 3.96 | 1.22 | 4.86× | 5.30× | 155 / 172 · 97 / 122 · 84 / 120 · 89 / 108 | half | 99.6 / 100.0 / 99.6 |
| inception_v3 | b64 | 94.8 | 45 | 44.6 | 19.6 | 2.30× | 4.84× | 1315 / 1940 · 909 / 1160 · 554 / 812 · 464 / 676 | half | 99.6 / 100.0 / 99.6 |
| resnet18 | b1 | 1.5 | 1.28 | 0.92 | 0.28 | 4.57× | 5.36× | 93 / 118 · 64 / 106 · 58 / 104 · 62 / 80 | half | 99.6 / 100.0 / 100.0 |
| resnet18 | b64 | 23.4 | 9.62 | 9.65 | 4.59 | 2.10× | 5.10× | 538 / 560 · 612 / 684 · 330 / 346 · 298 / 302 | half | 99.6 / 100.0 / 100.0 |
| resnet50 | b1 | 4.04 | 3.18 | 2.28 | 0.66 | 4.82× | 6.12× | 147 / 174 · 91 / 112 · 86 / 114 · 91 / 108 | half | 100.0 / 100.0 / 100.0 |
| resnet50 | b64 | 82.8 | 30.9 | 30.8 | 12.8 | 2.41× | 6.47× | 837 / 1254 · 639 / 710 · 430 / 468 · 422 / 526 | half | 100.0 / 100.0 / 100.0 |
| resnet152 | b1 | 10.3 | 9.28 | 6.42 | 1.68 | 5.52× | 6.13× | 279 / 300 · 157 / 166 · 152 / 166 · 160 / 180 | half | 99.6 / 99.6 / 99.6 |
| resnet152 | b64 | 184 | 66.7 | 66.8 | 29.3 | 2.28× | 6.28× | 970 / 1378 · 705 / 782 · 496 / 540 · 490 / 598 | half | 99.6 / 99.6 / 99.6 |
| densenet121 | b1 | 8.87 | 8.02 | 7.21 | 1.15 | 6.97× | 7.71× | 77 / 82 · 57 / 68 · 52 / 70 · 58 / 72 | half | 100.0 / 100.0 / 100.0 |
| densenet121 | b64 | 97.1 | 35.1 | 35.1 | 16.5 | 2.13× | 5.88× | 919 / 2104 · 606 / 750 · 361 / 690 · 402 / 734 | half | 100.0 / 100.0 / 100.0 |
| squeezenet1_1 | b1 | 1.23 | 1.11 | 1.23 | 0.18 | 6.17× | 6.83× | 11 / 28 · 7 / 26 · 6 / 26 · 8 / 28 | half | 99.6 / 99.6 / 99.6 |
| squeezenet1_1 | b64 | 14.8 | 6.64 | 6.78 | 1.95 | 3.41× | 7.59× | 377 / 400 · 347 / 410 · 272 / 280 · 133 / 170 | half | 99.6 / 99.6 / 99.6 |
| mobilenet_v1 | b1 | 1.65 | 1.71 | 1.16 | 0.2 | 8.55× | 8.25× | 58 / 82 · 50 / 60 · 43 / 58 · 45 / 62 | half | 99.2 / 100.0 / 100.0 |
| mobilenet_v1 | b64 | 25.3 | 9.07 | 9 | 3.2 | 2.83× | 7.91× | 706 / 816 · 721 / 864 · 243 / 312 · 209 / 228 | half | 99.2 / 100.0 / 100.0 |
| mobilenet_v2 | b1 | 2.73 | 3.02 | 2.16 | 0.28 | 10.79× | 9.75× | 56 / 62 · 54 / 70 · 42 / 58 · 44 / 60 | half | 100.0 / 98.4 / 98.4 |
| mobilenet_v2 | b64 | 28.5 | 10.9 | 10.6 | 3.9 | 2.79× | 7.31× | 752 / 1012 · 1003 / 1158 · 279 / 366 · 208 / 286 | half | 100.0 / 98.4 / 98.4 |
| shufflenet_v2 | b1 | 3.74 | 3.75 | 2.73 | 0.61 | 6.15× | 6.13× | 44 / 58 · 39 / 44 · 38 / 44 · 45 / 62 | half | 98.8 / 98.8 / 99.6 |
| shufflenet_v2 | b64 | 9.06 | 5.57 | 5.5 | 1.88 | 2.96× | 4.82× | 258 / 276 · 209 / 260 · 174 / 196 · 167 / 192 | half | 98.8 / 98.8 / 99.6 |
| mobilenet_v3_l | b1 | 3.42 | 3.56 | 2.79 | 0.54 | 6.59× | 6.33× | 61 / 70 · 52 / 108 · 45 / 108 · 47 / 62 | half | 99.6 / 99.6 / 100.0 |
| mobilenet_v3_l | b64 | 20.5 | 7.2 | 6.6 | 3.76 | 1.91× | 5.45× | 564 / 730 · 698 / 840 · 220 / 284 · 186 / 238 | half | 99.6 / 99.6 / 100.0 |
| efficientnet_b0 | b1 | 4.75 | 4.55 | 3.76 | 0.71 | 6.41× | 6.69× | 63 / 70 · 57 / 74 · 47 / 62 · 53 / 64 | half | 100.0 / 100.0 / 100.0 |
| efficientnet_b0 | b64 | 35.9 | 13.3 | 13 | 7.74 | 1.72× | 4.64× | 759 / 1020 · 1007 / 1162 · 282 / 366 · 297 / 380 | half | 100.0 / 100.0 / 100.0 |
| efficientnet_b4 | b1 | 9.04 | 10.6 | 7.51 | 1.83 | 5.79× | 4.94× | 156 / 186 · 171 / 242 · 88 / 242 · 110 / 152 | autocast-fp16 | 100.0 / 100.0 / 99.6 |
| efficientnet_b4 | b64 | 305 | 121 | 120 | 66.9 | 1.81× | 4.56× | 2999 / 4070 · 4286 / 4900 · 1022 / 1300 · 1003 / 1278 | autocast-fp16 | 100.0 / 100.0 / 99.6 |
| regnety_16gf | b1 | 11.5 | 7.52 | 6.37 | 1.7 | 4.42× | 6.76× | 393 / 438 · 220 / 244 · 212 / 220 · 212 / 236 | half | 99.6 / 99.6 / 99.2 |
| regnety_16gf | b64 | 233 | 102 | 101 | 41 | 2.49× | 5.68× | 2416 / 3218 · 1924 / 2104 · 1195 / 1274 · 744 / 770 | half | 99.6 / 99.6 / 99.2 |
| efficientnet_v2_s | b1 | 11.9 | 10.4 | 7.58 | 1.75 | 5.94× | 6.80× | 136 / 152 · 89 / 152 · 80 / 152 · 89 / 96 | half | 100.0 / 100.0 / 100.0 |
| efficientnet_v2_s | b64 | 243 | 91.1 | 89.8 | 49 | 1.86× | 4.96× | 1445 / 2000 · 1082 / 1268 · 560 / 796 · 539 / 772 | half | 100.0 / 100.0 / 100.0 |
| vit_b16 | b1 | 6.13 | 2.43 | 2.25 | 1.49 | 1.63× | 4.11× | 370 / 418 · 208 / 216 · 208 / 216 · 382 / 400 | half | 100.0 / 100.0 / 100.0 |
| vit_b16 | b64 | 233 | 45.3 | 45.5 | 38.8 | 1.17× | 6.01× | 875 / 1044 · 443 / 518 · 536 / 556 · 573 / 634 | half | 100.0 / 100.0 / 100.0 |
| deit_b | b1 | 6.13 | 2.36 | 2.22 | 1.5 | 1.57× | 4.09× | 370 / 418 · 208 / 216 · 208 / 216 · 382 / 400 | half | 100.0 / 100.0 / 100.0 |
| deit_b | b64 | 232 | 45.3 | 45.4 | 38.9 | 1.16× | 5.96× | 875 / 1044 · 443 / 518 · 536 / 556 · 573 / 634 | half | 100.0 / 100.0 / 100.0 |
| swin_b | b1 | 11.5 | 10.4 | 8.22 | 3.67 | 2.83× | 3.13× | 396 / 410 · 212 / 238 · 212 / 238 · 401 / 428 | half | 100.0 / 100.0 / 100.0 |
| swin_b | b64 | 281 | 89.9 | 72.2 | 52.7 | 1.71× | 5.33× | 1670 / 2542 · 843 / 1274 · 865 / 1306 · 763 / 936 | half | 100.0 / 100.0 / 100.0 |
| convnext_v2_b | b1 | 8.44 | 7.5 | 7.84 | 4.42 | 1.70× | 1.91× | 394 / 420 · 220 / 244 · 299 / 318 · 479 / 534 | half | 100.0 / 100.0 / 100.0 |
| convnext_v2_b | b64 | 323 | 107 | 77.6 | 69 | 1.55× | 4.68× | 1910 / 2316 · 957 / 1254 · 877 / 1102 · 973 / 1186 | half | 100.0 / 100.0 / 100.0 |
| fastvit_sa24 | b1 | 10.2 | 10.2 | 4.98 | 1.219 ² | 8.37× | 8.37× | 131 / 156 · 87 / 150 · 78 / 150 · 82 / 94 | half | 100.0 / 100.0 / 100.0 |
| fastvit_sa24 | b64 | 168 | 72.1 | 48.1 | 22.82 ² | 3.16× | 7.36× | 1219 / 1492 · 961 / 1326 · 433 / 556 · 315 / 428 | half | 100.0 / 100.0 / 100.0 |
| efficientvit_b2 | b1 | 8.4 | 10.3 | 9.22 | 0.91 | 11.32× | 9.23× | 144 / 156 · 152 / 174 · 85 / 104 · 89 / 106 | half | 99.6 / 100.0 / 100.0 |
| efficientvit_b2 | b64 | 125 | 55.5 | 55.1 | 20.2 | 2.75× | 6.19× | 1313 / 1686 · 1691 / 1898 · 476 / 624 · 357 / 504 | half | 99.6 / 100.0 / 100.0 |
| mobilenet_v4_m | b1 | 4.63 | 4.64 | 2.86 | 0.45 | 10.31× | 10.29× | 80 / 104 · 62 / 86 · 53 / 84 · 57 / 70 | half | 100.0 / 99.6 / 99.6 |
| mobilenet_v4_m | b64 | 38.1 | 12.3 | 11.8 | 5.79 | 2.12× | 6.58× | 758 / 932 · 763 / 932 · 251 / 378 · 260 / 276 | half | 100.0 / 99.6 / 99.6 |
| eva02_l | b1 | 84.5 | 17.7 | 17.7 | 14 | 1.26× | 6.04× | 1257 / 1392 · 645 / 672 · 647 / 692 · 1262 / 1388 | half | 99.6 / 99.6 / 100.0 |
| eva02_l | b64 | 5333 | 1355 | 1356 | 853 | 1.59× | 6.25× | 5140 / 6934 · 2570 / 3468 · 2719 / 3598 · 2605 / 3278 | half | 99.6 / 99.6 / 100.0 |
| dinov2_l_lin | b1 | 21.9 | 5.11 | 4.86 | 4.8 | 1.06× | 4.56× | 1210 / 1232 · 621 / 658 · 624 / 658 · 1208 / 1258 | half | 100.0 / 99.2 / 99.6 |
| dinov2_l_lin | b64 | 1049 | 198 | 190 | 168 | 1.18× | 6.24× | 2040 / 2320 · 1019 / 1174 · 1152 / 1240 · 1472 / 1674 | half | 100.0 / 99.2 / 99.6 |
| siglip2_so400m_zs | b1 | 77.7 | 13.2 | 13.4 | 13.2 | 1.00× | 5.89× | 1710 / 5870 · 881 / 916 · 902 / 992 · 1125 / 5870 | half | 99.6 / 99.6 / 99.2 |
| siglip2_so400m_zs | b64 | 4571 | 807 | 808 | 711 | 1.14× | 6.43× | 4169 / 5090 · 2095 / 2450 · 2219 / 2650 · 1869 / 2346 | half | 99.6 / 99.6 / 99.2 |
| siglip_so400m_ft | b1 | 77.1 | 12.8 | 12.9 | 13.1 ¹ | 0.98× | 5.89× | 1731 / 1822 · 880 / 908 · 888 / 906 · 1700 / 1752 | half | 99.6 / 99.6 / 99.6 |
| siglip_so400m_ft | b64 | 4571 | 798 | 798 | 711 | 1.12× | 6.43× | 4185 / 4926 · 2092 / 2436 · 2506 / 2540 · 2439 / 2808 | half | 99.6 / 99.6 / 99.6 |

### Semantic segmentation

segnet uses random weights: it is a latency demo, so its agreement numbers only show the fp16 noise of an untrained net.

| model | setting | upstream (fp32) ms | upstream + flags ms | portable ms | cuda-kernels ms | cuda-kernels vs flags | cuda-kernels vs upstream | peak MB alloc / res: upstream · flags · portable · cuda-kernels | flags precision | quality: flags / portable / cuda-kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| fcn-r101 | 512x512 | 42 | 18.7 | 17.3 | 10.9 | 1.72× | 3.85× | 491 / 826 · 296 / 430 · 223 / 332 · 205 / 408 | half | 100.0 / 100.0 / 100.0 |
| fcn-r101 | 1024x2048 | 374 | 160 | 158 | 82.1 | 1.95× | 4.56× | 1760 / 4112 · 1394 / 1830 · 790 / 1500 · 695 / 1048 | half | 100.0 / 100.0 / 100.0 |
| segnet | 512x512 | 31.6 | 16 | 13.5 | 10.3 | 1.55× | 3.07× | 550 / 774 · 367 / 672 · 253 / 354 · 252 / 430 | half | 97.2 / 97.2 / 95.9 |
| segnet | 1024x2048 | 261 | 144 | 121 | 88.1 | 1.63× | 2.96× | 3606 / 5754 · 3619 / 4594 · 1571 / 2354 · 1693 / 2536 | half | 97.2 / 97.2 / 95.9 |
| pspnet-r101 | 512x512 | 41.5 | 20.4 | 19 | 12.8 | 1.59× | 3.24× | 554 / 874 · 293 / 452 · 224 / 364 · 224 / 412 | half | 100.0 / 100.0 / 100.0 |
| pspnet-r101 | 1024x2048 | 365 | 191 | 190 | 116 | 1.65× | 3.15× | 2103 / 4730 · 1396 / 1840 · 843 / 1596 · 853 / 1116 | half | 100.0 / 100.0 / 100.0 |
| deeplabv3-r101 | 512x512 | 49.8 | 22 | 20.7 | 12.7 | 1.73× | 3.92× | 554 / 898 · 331 / 436 · 260 / 380 · 253 / 482 | half | 100.0 / 100.0 / 100.0 |
| deeplabv3-r101 | 1024x2048 | 429 | 184 | 183 | 95 | 1.94× | 4.52× | 1767 / 4184 · 1428 / 1792 · 827 / 1540 · 826 / 1090 | half | 100.0 / 100.0 / 100.0 |
| deeplabv3plus-r101 | 512x512 | 46.9 | 30.2 | 19.9 | 12.5 | 2.42× | 3.75× | 558 / 848 · 293 / 472 · 272 / 396 · 254 / 362 | half | 100.0 / 100.0 / 100.0 |
| deeplabv3plus-r101 | 1024x2048 | 417 | 211 | 181 | 101 | 2.09× | 4.13× | 2764 / 5278 · 2569 / 2672 · 1276 / 2188 · 1194 / 1634 | half | 100.0 / 100.0 / 100.0 |
| hrnet-w48 | 512x512 | 32.5 | 52.7 | 28.1 | 22.5 | 2.34× | 1.44× | 448 / 802 · 296 / 414 · 217 / 316 · 181 / 368 | half | 100.0 / 100.0 / 100.0 |
| hrnet-w48 | 1024x2048 | 193 | 84.5 | 82.8 | 52.3 | 1.62× | 3.69× | 1782 / 4518 · 2087 / 2412 · 809 / 1466 · 557 / 952 | half | 100.0 / 100.0 / 100.0 |
| segformer-b0 | 512x512 | 6.8 | 9.28 | 8.06 | 3.1 | 2.99× | 2.19× | 281 / 456 · 208 / 332 · 130 / 176 · 69 / 92 | half | 100.0 / 100.0 / 100.0 |
| segformer-b0 | 1024x2048 | 71.7 | 34.3 | 28 | 16.3 | 2.10× | 4.40× | 1914 / 3322 · 1354 / 1884 · 722 / 998 · 259 / 386 | half | 100.0 / 100.0 / 100.0 |
| segformer-b5 | 512x512 | 39.7 | 43.2 | 31.5 | 17.1 | 2.53× | 2.32× | 1051 / 1140 · 1241 / 1278 · 307 / 410 · 291 / 518 | half | 100.0 / 100.0 / 100.0 |
| segformer-b5 | 1024x2048 | 437 | 135 | 108 | 92.3 | 1.46× | 4.73× | 5848 / 6654 · 8524 / 8586 · 898 / 1504 · 727 / 1030 | half | 100.0 / 100.0 / 100.0 |
| segnext-t | 512x512 | 9.2 | 16.4 | 12.2 | 9 | 1.82× | 1.02× | 205 / 286 · 211 / 270 · 127 / 168 · 74 / 96 | autocast-fp16 | 99.9 / 99.9 / 99.9 |
| segnext-t | 1024x2048 | 62.1 | 68.7 | 39.1 | 26.4 | 2.60× | 2.35× | 1310 / 1904 · 1319 / 1558 · 752 / 830 · 255 / 324 | autocast-fp16 | 99.9 / 99.9 / 99.9 |
| mask2former-swinl | 512x512 | 79.3 | 42.3 | 39.2 | 21.3 | 1.99× | 3.72× | 1242 / 1812 · 774 / 1196 · 792 / 1016 · 708 / 932 | half | 99.9 / 99.9 / 99.9 |
| mask2former-swinl | 1024x2048 | 626 | 279 | 256 | 154 | 1.81× | 4.06× | 3924 / 8538 · 2919 / 5464 · 1834 / 3360 · 1612 / 2508 | half | 99.9 / 99.9 / 99.9 |
| oneformer-swinl | 512x512 | 95.8 | 48.3 | 46.6 | 27.7 | 1.74× | 3.46× | 1779 / 2222 · 925 / 1422 · 884 / 1168 · 784 / 1086 | half | 99.8 / 99.9 / 99.9 |
| oneformer-swinl | 1024x2048 | 742 | 335 | 322 | 221 | 1.52× | 3.36× | 7987 / 11840 · 4036 / 7870 · 2431 / 4568 · 2129 / 3722 | half | 99.8 / 99.9 / 99.9 |
| dinov2-l-linear | 512x512 | 108 | 28.2 | 26.4 | 19.3 | 1.46× | 5.60× | 1649 / 1756 · 1075 / 1196 · 711 / 774 · 707 / 778 | half | 100.0 / 99.9 / 99.9 |
| dinov2-l-linear | 1024x2048 | 1966 | 497 | 489 | 358 | 1.39× | 5.49× | 3794 / 4156 · 3212 / 3344 · 2203 / 2354 · 2211 / 2390 | half | 100.0 / 99.9 / 99.9 |
| eomt-l-dinov2 | 512x512 | 88.1 | 25.9 | 22.8 | 16.3 | 1.59× | 5.40× | 1697 / 2142 · 1096 / 1390 · 1168 / 1286 · 1165 / 1708 | half | 100.0 / 100.0 / 100.0 |
| eomt-l-dinov2 | 1024x2048 | 688 | 205 | 188 | 142 | 1.44× | 4.85× | 4914 / 8368 · 4295 / 6336 · 2297 / 3664 · 2298 / 3958 | half | 100.0 / 100.0 / 100.0 |
| eomt-l-dinov3 | 512x512 | 91.7 | 35.9 | 25 | 18.5 | 1.94× | 4.96× | 1693 / 2142 · 1698 / 2248 · 1160 / 1290 · 1157 / 1704 | autocast-fp16 | 100.0 / 100.0 / 100.0 |
| eomt-l-dinov3 | 1024x2048 | 736 | 299 | 213 | 167 | 1.79× | 4.41× | 4910 / 8366 · 4917 / 8932 · 2289 / 3710 · 2290 / 3954 | autocast-fp16 | 100.0 / 100.0 / 100.0 |
| unet-s5 | 512x512 | 34.4 | 15.1 | 14.4 | 9.3 | 1.62× | 3.70× | 744 / 1010 · 478 / 530 · 317 / 390 · 314 / 430 | half | 99.9 / 99.9 / 99.8 |
| unet-s5 | 1024x2048 | 277 | 128 | 125 | 73 | 1.75× | 3.79× | 5178 / 6648 · 3379 / 3402 · 2098 / 2624 · 2121 / 2670 | half | 99.9 / 99.9 / 99.8 |
| ddrnet-23 | 512x512 | 4.8 | 4.8 | 3.37 | 4.6 ¹ | 1.04× | 1.04× | 124 / 180 · 80 / 124 · 60 / 84 · 56 / 108 | half | 99.9 / 99.9 / 99.9 |
| ddrnet-23 | 1024x2048 | 27 | 12.5 | 12.2 | 7 | 1.79× | 3.86× | 441 / 622 · 305 / 344 · 163 / 286 · 168 / 212 | half | 99.9 / 99.9 / 99.9 |
| pidnet-l | 512x512 | 7.1 | 6.81 | 5.08 | 7 ¹ | 0.97× | 1.01× | 189 / 246 · 115 / 144 · 100 / 130 · 96 / 172 | half | 99.9 / 99.9 / 99.9 |
| pidnet-l | 1024x2048 | 46.4 | 20.7 | 20.3 | 12.5 | 1.66× | 3.71× | 506 / 688 · 339 / 376 · 236 / 352 · 244 / 308 | half | 99.9 / 99.9 / 99.9 |

### Backbones

DINOv2 cuda-kernels uses the fastbb `hh` flash attention. The DINOv3 ViT cuda-kernels tier adds a CUDA graph per input shape. Its first
call at a new size includes the capture, which the "new size" column shows. ConvNeXt and timm CNNs use the fused CUTLASS
convolutions plus a TorchScript trace.

| model | setting | upstream (fp32) ms | upstream + flags ms | portable ms | cuda-kernels ms | cuda-kernels vs flags | cuda-kernels vs upstream | peak MB alloc / res: upstream · flags · portable · cuda-kernels | flags precision | quality: flags / portable / cuda-kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| dinov2-vits14 | 896x1344 b1 | 131.26 | 23.73 | 23.46 | 16.66 | 1.42× | 7.88× | 295 / 408 · 155 / 190 · 172 / 190 · 239 / 276 | half | 0.0045 / 0.0044 / 0.0063 |
| dinov2-vitb14 | 896x1344 b1 | 301.93 | 52.47 | 51.81 | 39.75 | 1.32× | 7.60× | 670 / 898 · 346 / 436 · 385 / 434 · 481 / 548 | half | 0.0077 / 0.0079 / 0.0116 |
| dinov2-vitl14 | 896x1344 b1 | 882.87 | 146.54 | 144.93 | 113.64 | 1.29× | 7.77× | 1546 / 1718 · 796 / 998 · 849 / 948 · 897 / 1010 | half | 0.0066 / 0.0067 / 0.0073 |
| dinov2-vitg14 | 896x1344 b1 | 2880.74 | 444.39 | 439.50 | 364.43 | 1.22× | 7.90× | 5047 / 5362 · 2514 / 2936 · 2584 / 2780 · 2742 / 2986 | half | 0.0036 / 0.0133 / 0.0037 |
| dinov3-vits16 | 512px b1 | 11.24 | 4.53 | 4.32 | 1.856 ² | 2.44× | 6.06× | 174 / 184 · 104 / 144 · 105 / 114 · 123 / 150 | half | 0.0031 / 0.0030 / 0.0032 |
| dinov3-vits16 | 1024px b1 | 65.93 | 14.54 | 15.36 | 10.09 ² | 1.44× | 6.53× | 234 / 298 · 143 / 178 · 153 / 176 · 215 / 290 | half | 0.0034 / 0.0033 / 0.0032 |
| dinov3-vitb16 | 512px b1 | 25.57 | 6.01 | 6.25 | 4.405 ² | 1.36× | 5.80× | 439 / 508 · 250 / 434 · 248 / 274 · 262 / 290 | half | 0.0033 / 0.0034 / 0.0044 |
| dinov3-vitb16 | 1024px b1 | 164.18 | 32.31 | 33.49 | 24.3 ² | 1.33× | 6.76× | 560 / 700 · 323 / 480 · 344 / 378 · 388 / 494 | half | 0.0042 / 0.0041 / 0.0044 |
| dinov3-vitl16 | 512px b1 | 82.42 | 21.23 | 17.62 | 12.60 | 1.68× | 6.54× | 1281 / 1322 · 1248 / 1864 · 669 / 716 · 821 / 904 | autocast-fp16 | 0.0036 / 0.0037 / 0.0045 |
| dinov3-vitl16 | 1024px b1 | 488.95 | 112.90 | 96.60 | 66.38 | 1.70× | 7.37× | 1442 / 1608 · 1375 / 2074 · 796 / 868 · 893 / 1126 | autocast-fp16 | 0.0038 / 0.0040 / 0.0045 |
| dinov3-vith16plus | 512px b1 | 207.42 | 40.72 | 44.14 | 36.07 | 1.13× | 5.75× | 3463 / 3598 · 1733 / 1808 · 1757 / 1848 · 1885 / 2060 | half | 0.0040 / 0.0039 / 0.0042 |
| dinov3-vith16plus | 1024px b1 | 1084.81 | 207.14 | 216.50 | 164.69 | 1.26× | 6.59× | 3723 / 3972 · 1880 / 2036 · 1975 / 2146 · 1980 / 2128 | half | 0.0040 / 0.0040 / 0.0042 |
| dinov3-convnext-tiny | 512px b1 | 6.56 | 2.91 | 2.91 | 2.29 | 1.27× | 2.86× | 229 / 310 · 134 / 180 · 137 / 164 · 161 / 194 | half | 0.0022 / 0.0022 / 0.0052 |
| dinov3-convnext-tiny | 1024px b1 | 26.41 | 10.90 | 10.89 | 8.73 | 1.25× | 3.03× | 394 / 526 · 222 / 286 · 244 / 318 · 255 / 302 | half | 0.0024 / 0.0024 / 0.0052 |
| dinov3-convnext-small | 512px b1 | 11.79 | 4.90 | 5.08 | 3.73 | 1.31× | 3.16× | 312 / 392 · 180 / 262 · 180 / 206 · 203 / 236 | half | 0.0015 / 0.0015 / 0.0032 |
| dinov3-convnext-small | 1024px b1 | 46.77 | 17.54 | 17.55 | 13.55 | 1.29× | 3.45× | 477 / 608 · 268 / 356 · 286 / 358 · 298 / 344 | half | 0.0017 / 0.0017 / 0.0032 |
| dinov3-convnext-base | 512px b1 | 18.01 | 6.89 | 6.89 | 4.72 | 1.46× | 3.82× | 477 / 562 · 255 / 304 · 264 / 290 · 285 / 316 | half | 0.0015 / 0.0015 / 0.0034 |
| dinov3-convnext-base | 1024px b1 | 76.54 | 25.62 | 25.65 | 19.37 | 1.32× | 3.95× | 697 / 850 · 371 / 462 · 403 / 496 · 407 / 490 | half | 0.0016 / 0.0016 / 0.0034 |
| dinov3-convnext-large | 512px b1 | 38.90 | 12.30 | 12.31 | 9.03 | 1.36× | 4.31× | 928 / 1090 · 485 / 548 · 493 / 538 · 517 / 600 | half | 0.0014 / 0.0014 / 0.0045 |
| dinov3-convnext-large | 1024px b1 | 158.38 | 49.62 | 49.46 | 35.66 | 1.39× | 4.44× | 1258 / 1522 · 658 / 764 · 702 / 848 · 695 / 816 | half | 0.0013 / 0.0013 / 0.0045 |
| timm-resnet50.a1_in1k | 1024px b1 | 24.73 | 8.86 | 8.70 | 4.28 | 2.07× | 5.78× | 385 / 698 · 233 / 276 · 210 / 254 · 231 / 306 | half | 0.0007, top-1 agree 1.000 / 0.0007, top-1 agree 1.000 / 0.0013 |
| timm-resnet50.a1_in1k | 224px b32 | 40.32 | 14.12 | 13.86 | 6.58 | 2.15× | 6.13× | 497 / 856 · 294 / 396 · 258 / 322 · 280 / 356 | half | 0.0007, top-1 agree 1.000 / 0.0009, top-1 agree 1.000 / 0.0013 |
| timm-resnet18.a1_in1k | 1024px b1 | 6.91 | 3.12 | 3.02 | 1.55 | 2.01× | 4.46× | 263 / 360 · 175 / 196 · 174 / 196 · 170 / 236 | half | 0.0003, top-1 agree 1.000 / 0.0005, top-1 agree 1.000 / 0.0006 |
| timm-resnet18.a1_in1k | 224px b32 | 11.04 | 4.71 | 4.69 | 2.42 | 1.95× | 4.56× | 339 / 428 · 218 / 284 · 218 / 246 · 202 / 258 | half | 0.0005, top-1 agree 1.000 / 0.0006, top-1 agree 1.000 / 0.0006 |
| timm-resnet101.a1h_in1k | 1024px b1 | 37.65 | 13.96 | 13.73 | 7.06 | 1.98× | 5.33× | 458 / 772 · 270 / 312 · 245 / 292 · 268 / 358 | half | 0.0038, top-1 agree 1.000 / 0.0043, top-1 agree 1.000 / 0.0050 |
| timm-resnet101.a1h_in1k | 224px b32 | 61.60 | 21.52 | 20.86 | 10.52 | 2.05× | 5.86× | 570 / 932 · 332 / 438 · 293 / 360 · 317 / 392 | half | 0.0014, top-1 agree 1.000 / 0.0009, top-1 agree 1.000 / 0.0050 |
| timm-regnety_032.ra_in1k | 1024px b1 | 40.80 | 10.83 | 10.63 | 31.61 ¹ | 0.34× | 1.29× | 331 / 522 · 223 / 284 · 223 / 252 · 325 / 548 | half | 0.0020, top-1 agree 1.000 / 0.0016, top-1 agree 1.000 / 0.0020 |
| timm-regnety_032.ra_in1k | 224px b32 | 38.71 | 16.33 | 16.06 | 46.13 ¹ | 0.35× | 0.84× | 481 / 738 · 283 / 408 · 282 / 352 · 342 / 434 | half | 0.0027, top-1 agree 1.000 / 0.0023, top-1 agree 1.000 / 0.0020 |
| timm-efficientnet_b0.ra_in1k | 1024px b1 | 10.38 | 4.59 | 3.83 | 3.14 | 1.46× | 3.31× | 339 / 494 · 209 / 250 · 173 / 218 · 166 / 260 | half | 0.0038, top-1 agree 1.000 / 0.0057, top-1 agree 1.000 / 0.0041 |
| timm-efficientnet_b0.ra_in1k | 224px b32 | 16.15 | 6.17 | 5.60 | 4.70 | 1.31× | 3.44× | 471 / 716 · 277 / 406 · 221 / 290 · 221 / 294 | half | 0.0034, top-1 agree 1.000 / 0.0052, top-1 agree 1.000 / 0.0041 |
| timm-mobilenetv2_100.ra_in1k | 1024px b1 | 8.51 | 3.28 | 2.67 | 1.96 | 1.67× | 4.34× | 333 / 486 · 205 / 246 · 169 / 202 · 174 / 266 | half | 0.0059, top-1 agree 1.000 / 0.0085, top-1 agree 1.000 / 0.0068 |
| timm-mobilenetv2_100.ra_in1k | 224px b32 | 12.90 | 4.83 | 4.25 | 3.04 | 1.59× | 4.24× | 465 / 710 · 273 / 400 · 218 / 260 · 217 / 300 | half | 0.0053, top-1 agree 1.000 / 0.0054, top-1 agree 1.000 / 0.0068 |
| timm-convnext_tiny.fb_in22k_ft_in1k | 1024px b1 | 26.67 | 10.92 | 10.93 | 8.78 | 1.24× | 3.04× | 452 / 696 · 271 / 356 · 267 / 338 · 273 / 370 | half | 0.0012, top-1 agree 1.000 / 0.0012, top-1 agree 1.000 / 0.0027 |
| timm-convnext_tiny.fb_in22k_ft_in1k | 224px b32 | 41.77 | 15.41 | 15.42 | 13.33 | 1.16× | 3.13× | 597 / 796 · 345 / 444 · 342 / 440 · 351 / 420 | half | 0.0019, top-1 agree 1.000 / 0.0019, top-1 agree 1.000 / 0.0027 |

### Startup and first call

These are new measurements with `bench/bench.py` (one process per model and tier, 2026-10-23).
- **startup**: process start to model ready. It includes `import torch`, CUDA init, the model build and loading weights
  from the local disk. Downloads and kernel compilation are excluded, because they are cached.
- **first call**: the first call at size 1 minus the median. For the flags row this is mostly the `cudnn.benchmark`
  autotuning. For cuda-kernels it is the trace and its validation. Portable and cuda-kernels run as shipped, without `cudnn.benchmark`.
- **new size**: the same cost for the first call at size 2:
  - classification: b1 → b64;
  - segmentation: 512x512 → 1024x2048;
  - DINOv2: 896x1344 → 448x672;
  - DINOv3: 512 → 1024 px;
  - timm CNNs: 1024 px b1 → 224 px b32.

The steady-state medians of these runs are in the jsonl. Other GPU jobs ran between runs on the same machine, so use the
tables above for speed.

**Classification**

| model | startup s: upstream / flags / portable / cuda-kernels | first call s: upstream / flags / portable / cuda-kernels | new size s: upstream / flags / portable / cuda-kernels |
|---|---|---|---|
| resnet50 | 2.3 / 2.2 / 2.3 / 2.4 | 0.22 / 1.76 / 0.26 / 0.22 | 0.00 / 0.78 / 0.02 / 0.00 |
| alexnet | 2.8 / 2.5 / 3.2 / 2.5 | 0.16 / 0.38 / 0.14 / 0.16 | 0.03 / 0.18 / 0.00 / 0.00 |
| vgg16 | 3.6 / 2.8 / 2.9 / 3.2 | 0.20 / 0.49 / 0.13 / 0.20 | 0.03 / 0.83 / 0.00 / 0.00 |
| inception_v3 | 2.2 / 2.2 / 2.7 / 4.5 | 0.21 / 1.38 / 0.26 / 0.38 | 0.00 / 1.26 / 0.02 / 0.00 |
| resnet18 | 1.9 / 2.0 / 2.0 / 2.0 | 0.20 / 0.47 / 0.13 / 0.14 | 0.01 / 0.35 / 0.01 / 0.01 |
| resnet152 | 2.7 / 2.8 / 2.9 / 3.3 | 0.14 / 0.76 / 0.17 / 0.41 | 0.00 / 0.66 / 0.01 / 0.00 |
| densenet121 | 1.9 / 1.9 / 2.2 / 2.3 | 0.23 / 1.78 / 0.33 / 0.50 | 0.03 / 1.66 / 0.05 / 0.00 |
| squeezenet1_1 | 1.8 / 1.8 / 1.8 / 1.8 | 0.20 / 0.71 / 0.25 / 0.16 | 0.01 / 0.43 / 0.01 / 0.03 |
| mobilenet_v1 | 2.2 / 2.1 / 2.4 / 2.2 | 0.20 / 1.10 / 0.50 / 0.21 | 0.00 / 1.64 / 0.24 / 0.01 |
| mobilenet_v2 | 1.8 / 1.9 / 2.1 / 1.9 | 0.22 / 1.26 / 0.54 / 0.27 | 0.01 / 1.49 / 0.29 / 0.01 |
| shufflenet_v2 | 1.8 / 1.8 / 2.1 / 1.9 | 0.27 / 1.75 / 0.54 / 0.21 | 0.01 / 0.59 / 0.32 / 0.01 |
| mobilenet_v3_l | 1.9 / 1.9 / 2.0 / 2.0 | 0.24 / 2.04 / 0.70 / 0.32 | 0.01 / 2.78 / 0.50 / 0.01 |
| efficientnet_b0 | 1.9 / 1.9 / 2.1 / 2.0 | 0.24 / 2.78 / 0.68 / 0.36 | 0.05 / 2.24 / 0.86 / 0.01 |
| efficientnet_b4 | 2.1 / 2.1 / 3.5 / 2.4 | 0.23 / 2.35 / 0.83 / 0.54 | 0.00 / 5.58 / 0.43 / 0.00 |
| regnety_16gf | 3.0 / 3.0 / 3.2 / 3.0 | 0.20 / 1.20 / 0.31 / 0.34 | 0.00 / 1.94 / 0.02 / 0.00 |
| efficientnet_v2_s | 2.2 / 3.8 / 2.3 / 2.4 | 0.20 / 1.89 / 0.52 / 0.57 | 0.00 / 1.97 / 0.24 / 0.00 |
| vit_b16 | 2.8 / 4.8 / 3.1 / 2.9 | 0.17 / 0.23 / 0.13 / 0.45 | 0.02 / 0.04 / 0.00 / 0.01 |
| deit_b | 3.1 / 6.0 / 3.1 / 2.9 | 0.19 / 0.22 / 0.13 / 0.43 | 0.02 / 0.04 / 0.00 / 0.01 |
| swin_b | 3.2 / 5.4 / 3.3 / 2.9 | 0.22 / 0.23 / 0.17 / 0.92 | 0.00 / 0.01 / 0.00 / 0.00 |
| convnext_v2_b | 3.1 / 3.3 / 3.3 / 3.0 | 0.42 / 1.00 / 0.34 / 0.92 | 0.01 / 2.23 / 0.12 / 0.01 |
| eva02_l | 5.6 / 5.9 / 5.8 / 5.0 | 0.22 / 0.21 / 0.09 / 0.76 | 0.02 / 0.26 / 0.00 / 0.00 |
| efficientvit_b2 | 2.4 / 2.4 / 2.5 / 2.9 | 0.24 / 1.59 / 0.71 / 0.45 | 0.01 / 2.58 / 0.34 / 0.00 |
| mobilenet_v4_m | 2.3 / 2.2 / 2.4 / 2.6 | 0.23 / 1.54 / 0.61 / 0.30 | 0.00 / 1.63 / 0.35 / 0.01 |
| dinov2_l_lin | 5.5 / 6.0 / 6.0 / 6.3 | 0.20 / 0.22 / 0.09 / 0.70 | 0.01 / 0.05 / 0.00 / 0.00 |
| siglip2_so400m_zs | 13.8 / 18.5 / 12.6 / 17.6 | 0.09 / 0.24 / 0.12 / 1.10 | 0.03 / 0.27 / 0.00 / 0.00 |
| siglip_so400m_ft | 7.6 / 7.2 / 8.6 / 9.5 | 0.18 / 0.21 / 0.10 / 0.86 | 0.01 / 0.27 / 0.00 / 0.00 |
| fastvit_sa24 | 2.7 / 2.4 / 3.2 / 3.7 | 0.26 / 1.87 / 0.57 / 0.47 | 0.00 / 2.71 / 0.33 / 0.00 |

**Segmentation**

| model | startup s: upstream / flags / portable / cuda-kernels | first call s: upstream / flags / portable / cuda-kernels | new size s: upstream / flags / portable / cuda-kernels |
|---|---|---|---|
| segformer-b0 | 3.6 / 3.8 / 3.7 / 3.7 | 0.32 / 1.37 / 0.38 / 0.22 | 0.00 / 0.53 / 0.01 / 0.00 |
| fcn-r101 | 4.8 / 5.3 / 6.2 / 5.3 | 0.24 / 1.20 / 0.21 / 0.12 | 0.01 / 1.64 / 0.00 / 0.00 |
| segnet | 1.3 / 1.2 / 1.7 / 1.9 | 0.32 / 1.22 / 0.00 / 0.04 | 0.03 / 1.15 / 0.00 / 0.00 |
| pspnet-r101 | 4.3 / 4.8 / 4.7 / 5.2 | 0.25 / 1.24 / 0.18 / 0.10 | 0.01 / 1.46 / 0.00 / 0.00 |
| deeplabv3-r101 | 5.6 / 5.0 / 5.0 / 6.4 | 0.23 / 1.30 / 0.16 / 0.14 | 0.01 / 2.10 / 0.00 / 0.00 |
| deeplabv3plus-r101 | 4.1 / 4.8 / 5.8 / 4.8 | 0.23 / 2.87 / 0.17 / 0.10 | 0.01 / 8.62 / 0.00 / 0.00 |
| hrnet-w48 | 3.8 / 4.9 / 4.5 / 4.4 | 0.21 / 1.02 / 0.15 / 0.17 | 0.00 / 0.83 / 0.01 / 0.00 |
| segformer-b5 | 3.8 / 5.4 / 4.6 / 4.5 | 0.30 / 1.00 / 0.35 / 0.48 | 0.00 / 0.87 / 0.00 / 0.00 |
| segnext-t | 3.2 / 3.4 / 3.8 / 3.8 | 0.40 / 1.99 / 1.83 / 0.15 | 0.00 / 2.03 / 0.25 / 0.00 |
| mask2former-swinl | 3.7 / 5.5 / 4.8 / 4.8 | 0.87 / 0.93 / 0.56 / 2.01 | 0.00 / 0.21 / 0.01 / 0.01 |
| oneformer-swinl | 4.5 / 7.1 / 5.8 / 5.9 | 0.49 / 0.88 / 0.62 / 0.66 | 0.00 / 0.20 / 0.01 / 0.01 |
| dinov2-l-linear | 4.5 / 5.8 / 2.2 / 2.1 | 0.25 / 0.73 / 0.23 / 0.28 | 0.00 / 0.00 / 0.04 / 0.04 |
| eomt-l-dinov2 | 3.6 / 5.0 / 5.1 / 5.1 | 0.41 / 0.70 / 0.36 / 0.28 | 0.00 / 1.47 / 0.01 / 0.12 |
| eomt-l-dinov3 | 3.6 / 5.1 / 5.0 / 5.2 | 0.38 / 0.70 / 0.36 / 0.29 | 0.00 / 1.46 / 0.00 / 0.12 |
| unet-s5 | 3.5 / 4.3 / 3.8 / 3.9 | 0.19 / 1.24 / 0.11 / 0.06 | 0.00 / 1.55 / 0.00 / 0.00 |
| ddrnet-23 | 3.5 / 3.9 / 4.0 / 4.0 | 0.22 / 1.07 / 0.23 / 0.08 | 0.01 / 0.72 / 0.01 / 0.00 |
| pidnet-l | 3.6 / 4.2 / 4.2 / 4.0 | 0.26 / 1.28 / 0.19 / 0.10 | 0.01 / 0.90 / 0.01 / 0.00 |

**Backbones**

| model | startup s: upstream / flags / portable / cuda-kernels | first call s: upstream / flags / portable / cuda-kernels | new size s: upstream / flags / portable / cuda-kernels |
|---|---|---|---|
| dinov2-vits14 | 1.4 / 1.3 / 1.2 / 1.0 | 0.14 / 0.17 / 0.16 / 0.16 | 0.00 / 0.01 / 0.00 / 0.00 |
| timm-resnet18.a1_in1k | 2.2 / 2.3 / 2.4 / 2.3 | 0.21 / 0.91 / 0.22 / 0.12 | 0.00 / 0.48 / 0.00 / 0.03 |
| dinov2-vitb14 | 2.4 / 2.6 / 1.5 / 1.3 | 0.15 / 0.18 / 0.13 / 0.13 | 0.00 / 0.01 / 0.00 / 0.00 |
| dinov2-vitl14 | 5.9 / 4.2 / 2.2 / 1.2 | 0.13 / 0.24 / 0.12 / 0.13 | 0.00 / 0.01 / 0.00 / 0.00 |
| dinov2-vitg14 | 18.7 / 20.1 / 5.2 / 4.2 | 0.13 / 0.23 / 0.12 / 0.32 | 0.00 / 0.01 / 0.00 / 0.00 |
| dinov3-vitl16 | 3.0 / 4.3 / 2.3 / 2.1 | 0.20 / 0.21 / 0.24 / 0.25 | 0.02 / 0.02 / 0.00 / 0.07 |
| dinov3-vith16plus | 3.4 / 6.1 / 2.7 / 3.4 | 0.19 / 0.20 / 0.23 / 0.27 | 0.03 / 0.03 / 0.00 / 0.17 |
| dinov3-convnext-tiny | 2.8 / 3.4 / 2.6 / 2.6 | 0.31 / 0.87 / 0.41 / 0.18 | 0.00 / 0.49 / 0.00 / 0.06 |
| dinov3-convnext-small | 2.8 / 3.4 / 2.5 / 2.9 | 0.30 / 0.89 / 0.31 / 0.19 | 0.00 / 0.47 / 0.00 / 0.11 |
| dinov3-convnext-base | 3.0 / 3.9 / 2.9 / 3.3 | 0.30 / 0.92 / 0.33 / 0.21 | 0.00 / 0.60 / 0.00 / 0.13 |
| dinov3-convnext-large | 3.2 / 3.8 / 3.8 / 4.4 | 0.32 / 1.42 / 0.23 / 0.14 | 0.00 / 0.89 / 0.00 / 0.15 |
| timm-resnet50.a1_in1k | 2.6 / 2.6 / 2.4 / 2.7 | 0.22 / 1.01 / 0.23 / 0.12 | 0.01 / 0.60 / 0.01 / 0.05 |
| timm-resnet101.a1h_in1k | 2.5 / 2.7 / 2.6 / 2.9 | 0.19 / 0.91 / 0.20 / 0.12 | 0.02 / 0.60 / 0.01 / 0.07 |
| timm-regnety_032.ra_in1k | 2.7 / 2.5 / 2.4 / 2.8 | 0.28 / 1.31 / 0.32 / 0.49 | 0.00 / 1.02 / 0.05 / 0.20 |
| timm-efficientnet_b0.ra_in1k | 2.9 / 2.2 / 2.3 / 2.4 | 0.31 / 2.15 / 1.80 / 0.35 | 0.09 / 1.86 / 1.64 / 0.07 |
| timm-mobilenetv2_100.ra_in1k | 2.4 / 2.2 / 2.2 / 2.4 | 0.24 / 1.40 / 0.47 / 0.11 | 0.01 / 1.13 / 0.25 / 0.04 |
| timm-convnext_tiny.fb_in22k_ft_in1k | 2.4 / 2.5 / 2.3 / 2.6 | 0.30 / 0.95 / 0.35 / 0.16 | 0.00 / 0.62 / 0.13 / 0.08 |
| dinov3-vits16 | 2.8 / 3.1 / 1.1 / 1.1 | 0.20 / 0.19 / 0.27 / 0.26 | 0.00 / 0.01 / 0.00 / 0.02 |
| dinov3-vitb16 | 2.9 / 3.1 / 1.1 / 1.2 | 0.20 / 0.20 / 0.24 / 0.24 | 0.00 / 0.02 / 0.00 / 0.03 |

## Benchmark

```bash
python bench/bench.py --family cls --model resnet50 --tier flags --out results/my_runs.jsonl
python bench/bench.py --family seg --model mask2former-swinl --tier cuda-kernels
python bench/bench.py --family bb --model dinov3-vitl16 --tier portable
```

One model and tier per process, so startup is measured honestly. Quality data: `FVM_IMAGENET` (ImageNet val class
folders), `FVM_ADE20K` (ADEChallengeData2016).

## Layout

```
fast_vision_models/
  classification/   zoo.py (model registry, tiers) + fast/ (optimizer, CUDA sources in fast/csrc)
  segmentation/     zoo.py, common.py (protocols, ADE20K helpers) + fast/ (per-architecture fast paths, CUDA sources)
  backbones/        __init__.py (DINOv2, DINOv3, timm), dinov3.py (DINOv3 re-implementation), rope.py + csrc/rope.cu
  _build.py         JIT build + cache of the CUDA extensions (~/.cache/fast-vision-models or $FVM_CACHE)
  weights.py        checkpoint lookup / download
bench/bench.py      unified benchmark (speed, memory, startup, quality)
results/            measured numbers (jsonl + the cited result tables)
docs/LICENSES.md    code and weights licence per model
```

## Licences

The code is Apache-2.0. The weights keep their own licences, and some are **non-commercial**: SegFormer (NVIDIA),
ConvNeXt V2 (CC BY-NC 4.0) and probably Mask2Former. Others have custom terms: FastViT (Apple) and DINOv3 / EoMT-DINOv3
(DINOv3 License). The datasets behind most checkpoints (ImageNet, ADE20K, Cityscapes) are non-commercial. See
[docs/LICENSES.md](docs/LICENSES.md). No weights are included in this repository.

# Licences

*Not legal advice.* The code in this repository is Apache-2.0 (LICENSE, NOTICE). It contains no model weights, no
checkpoints, no datasets and no prebuilt binaries. Each checkpoint is downloaded from its official source when it is first
used, and its own licence applies. That licence decides what you may do with the model's outputs and with the weights.
Checked 2026-10-10 from the model cards and repositories. Rows marked * were not re-checked online.

What the code does with upstream material: it re-implements the forward passes on top of the upstream libraries, which it
loads at run time and does not copy. Those libraries are timm (Apache-2.0), torchvision (BSD-3-Clause), Hugging Face
transformers (Apache-2.0), mmsegmentation 1.2.2 (Apache-2.0) and DINOv2 via torch.hub (Apache-2.0). None of these
licences forbids publishing modified or derived code. `backbones/dinov3.py` is an independent implementation written
against the transformers port of DINOv3 and contains no code from facebookresearch/dinov3 (see the DINOv3 row).

## Classification (`fast_vision_models.classification`)

| models | code we build on | weights licence | commercial use of the weights |
|---|---|---|---|
| alexnet, vgg16, inception_v3, resnet18/50/152, densenet121, squeezenet1_1, mobilenet_v2, shufflenet_v2, mobilenet_v3_l, efficientnet_b0/b4/v2_s, regnety_16gf (torchvision weights) | torchvision BSD-3-Clause | not stated separately; the torchvision docs say the weights may carry the dataset's terms | unclear (ImageNet-1k) |
| mobilenet_v1, mobilenet_v4_m, efficientvit_b2 (timm) | timm Apache-2.0 | Apache-2.0 (HF cards) | yes, by licence (ImageNet data risk) |
| vit_b16, deit_b | timm Apache-2.0 | Apache-2.0 | yes, by licence (ImageNet data risk) |
| swin_b | timm Apache-2.0 (Swin: MIT) | MIT | yes, by licence (ImageNet data risk) |
| **convnext_v2_b** | timm Apache-2.0 (ConvNeXt V2: MIT) | **CC BY-NC 4.0** | **no** |
| eva02_l | timm Apache-2.0 (EVA: MIT) | MIT | yes, by licence (dataset risks) |
| **fastvit_sa24** | timm Apache-2.0 (timm's own implementation) | **Apple sample-code licence** ("other") | check Apple's terms |
| dinov2_l_lin | timm Apache-2.0, DINOv2 Apache-2.0 | Apache-2.0 | yes, by licence (ImageNet head) |
| siglip_so400m_ft, siglip2_so400m_zs | timm / transformers Apache-2.0 | Apache-2.0 | yes, by licence |

## Segmentation (`fast_vision_models.segmentation`)

| models | code we build on | weights licence | commercial use of the weights |
|---|---|---|---|
| fcn-r101, pspnet-r101, deeplabv3-r101, deeplabv3plus-r101, hrnet-w48, segnext-t (ADE20K); unet-s5, pidnet-l, ddrnet-23 (Cityscapes) | mmsegmentation Apache-2.0 | Apache-2.0 (OpenMMLab model zoo)* | yes, by licence (ADE20K / Cityscapes data are non-commercial) |
| **segformer-b0, segformer-b5** | transformers Apache-2.0 (the original NVIDIA code is not used) | **NVIDIA Source Code License: non-commercial** | **no** |
| **mask2former-swinl** | transformers Apache-2.0 (Mask2Former: MIT) | "other" on the HF card* | treat as non-commercial until checked |
| oneformer-swinl | transformers Apache-2.0 (OneFormer: MIT) | MIT* | yes, by licence (dataset risks) |
| dinov2-l-linear | DINOv2 Apache-2.0 | Apache-2.0 | yes, by licence |
| eomt-l-dinov2 | transformers Apache-2.0 (EoMT: MIT) | MIT* | yes, by licence (dataset risks) |
| **eomt-l-dinov3** | transformers Apache-2.0 (EoMT: MIT) | MIT card*, built on DINOv3 weights | DINOv3 License terms apply (see below) |
| segnet | own implementation, random weights | none | latency demo only |

## Backbones (`fast_vision_models.backbones`)

| models | code we build on | weights licence | commercial use of the weights |
|---|---|---|---|
| dinov2-vits14/vitb14/vitl14/vitg14 | DINOv2 Apache-2.0, fastbb Apache-2.0 | Apache-2.0 | yes, by licence |
| **dinov3-vits16/vitb16/vitl16/vith16plus, dinov3-convnext-tiny/small/base/large** | transformers Apache-2.0 for the upstream tier; own re-implementation (`backbones/dinov3.py`) for portable / max | **DINOv3 License** (custom, gated on HF) | yes, with conditions: field-of-use exclusions (military, weapons, nuclear, espionage), redistribution only under the agreement, acknowledgement in publications. The weights are gated and are never redistributed here. |
| timm-resnet18/50/101, regnety_032, efficientnet_b0, mobilenetv2_100, convnext_tiny | timm Apache-2.0 | Apache-2.0 (timm) | yes, by licence (ImageNet data risk) |

## Datasets used only for the quality checks (not included)

ImageNet-1k (ILSVRC 2012) is limited to non-commercial research and education. ADE20K (MIT CSAIL terms) is research use.
Cityscapes is non-commercial. Most of the weights above were trained on one of these datasets. That is a training-data
risk, not a licence verdict.

## Excluded from this repository

- BERT / CLIP-text fast paths (exp 006): they need fastbb kernel changes that are not yet published.
- TensorRT engines and plugins, every prebuilt `.so` / `.fatbin` / `.cubin` (the CUDA sources build on first use) and all
  weights.

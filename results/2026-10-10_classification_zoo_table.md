# 008 zoo — upstream / upstream-friendly / max (general) / TensorRT 11 — 2026-10-10 (after round 3: max rows of the 19 CNN /
# hybrid models re-measured at experiment commit 4cbc373 (dw models: a94f3f0); round 2 at 19de102, round 1 at 4a8f11a)
GPU: NVIDIA GeForce RTX 5060 Ti 16 GB (sm_120); torch 2.14.1+cu130, cuDNN 9.24, timm 1.0.30, torchvision 0.29.1,
transformers 5.19.0; TensorRT 11.4.0.106 (exp 003 .venv-trt). CUTLASS v4.7.1 (exp 002 third_party) for clf_fast/csrc.

Impls: base = upstream fp32 eager (no cudnn.benchmark). friendly = upstream-friendly (plain PyTorch, ONNX/TRT-exportable:
BN fold, fp16, channels-last, SDPA, fp16-acc cuBLAS flag; timm Swin SDPA path; ConvNeXt GRN as one addcmul + gamma fold).
max = clf_fast general path: CUTLASS fused conv/GEMM (bias + residual + act, fp16 accumulate, fixed tile rule), stem C3->8 +
fixed-channel tiles, depthwise / pool / affine kernels, concat-free buffers (DenseNet, Inception, SqueezeNet), fused-GEMM
transformer blocks (ViT/DeiT/DINOv2/SigLIP/HF SigLIP 2, Swin window-major, ConvNeXt, EVA-02 rope/SwiGLU kernels),
round 2 (results/2026-10-10_round2.md): direct stem conv from the fp32 input, expand+dw kernel (Cin <= 64), ShuffleNet
layout folding, grouped-conv kernel, SE channel padding, dw-multiplier / GELU / LayerScale fusion, EfficientViT tensor-core
linear attention; round 3 (results/2026-10-10_round3.md): fused stem conv + max pool, merged / split Fire expand by size,
cublasLt 1x1 GEMMs with fused bias/ReLU for narrow convs, rewritten grouped-conv kernel (also small-C dense KxK convs), SE
sigmoid-scale kernel, depthwise kernel with row-mixed half2/fp32 accumulation,
TorchScript trace (fusers off) validated at the first call. No CUDA graphs, nothing compiled or tuned per input size.
TRT = friendly model -> ONNX (dynamo, opset 18, dynamic batch 1..64) -> strongly typed fp16 engine, opt level 3,
one profile (1 / 64 / 64), shared timing cache (bench/trt.py). SigLIP 2 zero-shot (HF) not exported.

Commands: speed  LOCK=gpulock bench/speed_all.sh results/2026-10-10_speed.jsonl "<impl>" [groups]   (rows: latest per model x impl)
          acc    LOCK=gpulock-eval bench/eval_run.sh "<models>" "<impls>" in1k --limit 1000   (fixed 1000-image subset, every 50th)
          TRT    bench/trt_all.sh "<models>"  (export: bench/trt.py export, CPU)  -> results/2026-10-10_trt.jsonl
          table  python bench/table.py
Protocol: random input at native resolution; b1 median of 30 after 10 warm-ups; b64 median of 10 after 3 (img/s);
peak allocated/reserved GB of the b64 phase (TRT: execution-context device memory / process GPU-memory delta);
startup = process launch -> model ready (TRT: cold = launch -> first b1 output incl. deserialize); first call = first b1
call - median (max: includes the one-time trace + validation); new size = first b64 call - median.
top-1 on the 1k subset; (same %) = identical top-1 vs upstream on the same images. The user decided: no 10k / 50k eval.

| model | year | params M | top-1 base | top-1 fr / max / TRT (same %) | b1 ms base / fr / max / TRT | b64 img/s base / fr / max / TRT | b64 peak GB a/r base / fr / max; TRT ctx/proc | startup s base / fr / max; TRT cold | first call s base / fr / max | new size s base / fr / max | TRT build s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| alexnet | 2012 | 61.1 | 56.2 | 56.2 / 56.3 / 56.2 (99.5 / 99.6 / 99.6) | 0.89 / 0.65 / 0.42 / 0.50 | 6726 / 18444 / 32143 / 34897 | 0.45/0.62 / 0.28/0.29 / 0.29/0.30; 0.05/0.21 | 2.2 / 2.3 / 2.5; 2.2 | 0.17 / 0.14 / 0.15 | 0.029 / 0.001 / 0.000 | 11.7 |
| vgg16 | 2014 | 138.36 | 71.2 | 71.3 / 71.1 / 71.1 (99.6 / 99.6 / 99.7) | 3.45 / 1.73 / 1.31 / 1.33 | 448 / 946 / 2222 / 2330 | 3.68/3.69 / 1.24/1.53 / 1.12/1.13; 0.79/1.11 | 2.8 / 2.9 / 3.0; 2.3 | 0.17 / 0.11 / 0.17 | 0.030 / 0.000 / -0.001 | 25.3 |
| inception_v3 | 2015 | 27.16 | 77.2 | 77.2 / 77.2 / 77.3 (99.9 / 99.9 / 99.9) | 6.47 / 3.99 / 1.22 / 1.33 | 675 / 1385 / 3270 / 3910 | 1.28/1.89 / 0.57/0.82 / 0.45/0.66; 0.36/0.52 | 2.1 / 2.2 / 2.4; 1.1 | 0.21 / 0.27 / 0.36 | 0.004 / 0.019 / 0.000 | 23.8 |
| resnet18 | 2015 | 11.69 | 69.5 | 69.4 / 69.3 / 69.4 (99.9 / 99.6 / 99.7) | 1.50 / 0.96 / 0.28 / 0.57 | 2732 / 6290 / 13955 / 15898 | 0.53/0.55 / 0.35/0.36 / 0.29/0.29; 0.12/0.19 | 1.9 / 2.0 / 2.0; 1.4 | 0.21 / 0.14 / 0.14 | 0.009 / 0.015 / 0.012 | 15.1 |
| resnet50 | 2015 | 25.56 | 80.2 | 80.2 / 80.2 / 80.2 (99.9 / 100.0 / 100.0) | 4.04 / 2.32 / 0.66 / 1.20 | 773 / 2034 / 5013 / 5534 | 0.82/1.22 / 0.45/0.48 / 0.41/0.51; 0.26/0.37 | 6.9 / 2.2 / 2.3; 1.1 | 0.32 / 0.23 / 0.16 | 0.004 / 0.016 / 0.001 | 22.7 |
| resnet152 | 2015 | 60.19 | 81.3 | 81.4 / 81.5 / 81.4 (99.8 / 99.8 / 99.9) | 10.26 / 6.45 / 1.68 / 4.41 | 348 / 944 / 2187 / 2412 | 0.95/1.35 / 0.52/0.55 / 0.48/0.58; 0.26/0.43 | 2.4 / 2.6 / 2.8; 2.1 | 0.14 / 0.14 / 0.28 | 0.002 / 0.007 / 0.000 | 59.7 |
| densenet121 | 2016 | 7.98 | 73.3 | 73.3 / 73.3 / 73.3 (99.9 / 99.7 / 99.7) | 8.87 / 7.26 / 1.15 / 1.45 | 659 / 1772 / 3877 / 4391 | 0.90/2.05 / 0.38/0.61 / 0.39/0.72; 0.22/0.29 | 2.0 / 2.0 / 2.3; 1.1 | 0.24 / 0.33 / 0.47 | 0.033 / 0.048 / 0.000 | 35.4 |
| squeezenet1_1 | 2016 | 1.24 | 59.5 | 59.6 / 59.5 / 59.6 (99.6 / 99.4 / 99.6) | 1.23 / 1.23 / 0.18 / 0.18 | 4329 / 9313 / 32879 / 31704 | 0.37/0.39 / 0.27/0.27 / 0.13/0.17; 0.14/0.21 | 1.8 / 1.9 / 1.9; 1.1 | 0.20 / 0.25 / 0.16 | 0.013 / 0.015 / 0.028 | 8.7 |
| mobilenet_v1 | 2017 | 4.23 | 74.1 | 74.0 / 74.2 / 73.8 (99.8 / 99.4 / 99.7) | 1.65 / 2.00 / 0.20 / 0.23 | 2527 / 6969 / 20013 / 18234 | 0.69/0.80 / 0.32/0.43 / 0.20/0.22; 0.21/0.28 | 2.1 / 2.3 / 2.2; 1.1 | 0.22 / 0.64 / 0.21 | 0.003 / 0.256 / 0.013 | 8.4 |
| mobilenet_v2 | 2018 | 3.5 | 70.6 | 70.6 / 70.5 / 70.6 (100.0 / 99.9 / 100.0) | 2.73 / 2.21 / 0.28 / 0.34 | 2244 / 5927 / 16413 / 15252 | 0.73/0.99 / 0.30/0.39 / 0.20/0.28; 0.20/0.28 | 1.9 / 2.0 / 2.1; 2.1 | 0.23 / 0.52 / 0.28 | 0.006 / 0.253 / 0.011 | 43.8 |
| shufflenet_v2 | 2018 | 2.28 | 68.6 | 68.6 / 68.6 / 68.6 (99.9 / 99.8 / 99.8) | 3.74 / 2.85 / 0.61 / 0.31 | 7067 / 10929 / 34038 / 33656 | 0.25/0.27 / 0.20/0.22 / 0.16/0.19; 0.06/0.13 | 1.9 / 2.0 / 1.9; 1.1 | 0.29 / 0.53 / 0.19 | 0.006 / 0.308 / 0.013 | 13.5 |
| mobilenet_v3_l | 2019 | 5.48 | 74.1 | 74.2 / 74.2 / 74.4 (99.6 / 99.4 / 99.6) | 3.42 / 2.90 / 0.54 / 0.40 | 3120 / 9558 / 17036 / 16630 | 0.55/0.71 / 0.25/0.32 / 0.18/0.23; 0.19/0.26 | 2.0 / 2.1 / 2.1; 2.1 | 0.26 / 0.94 / 0.32 | 0.015 / 0.514 / 0.012 | 63.2 |
| efficientnet_b0 | 2019 | 5.29 | 77.0 | 76.9 / 77.0 / 77.0 (99.7 / 99.8 / 99.8) | 4.75 / 4.03 / 0.71 / 0.64 | 1783 / 4758 / 8272 / 6185 | 0.74/1.00 / 0.31/0.39 / 0.29/0.37; 0.38/0.46 | 2.4 / 2.5 / 2.1; 2.0 | 0.28 / 0.76 / 0.36 | 0.060 / 0.522 / 0.007 | 55.9 |
| efficientnet_b4 | 2019 | 19.34 | 82.2 | 82.1 / 82.2 / 82.0 (99.9 / 99.8 / 99.8) | 9.04 / 8.20 / 1.83 / 3.18 | 210 / 530 / 956 / 691 | 2.93/3.97 / 1.03/1.30 / 0.98/1.25; 1.65/1.85 | 2.1 / 2.3 / 2.5; 1.1 | 0.25 / 0.86 / 0.66 | 0.003 / 0.452 / 0.000 | 40.3 |
| regnety_16gf | 2020 | 83.59 | 82.8 | 82.7 / 82.7 / 82.8 (99.9 / 99.9 / 100.0) | 11.48 / 6.57 / 1.70 / 4.38 | 275 / 620 / 1562 / 1647 | 2.36/3.14 / 1.20/1.28 / 0.73/0.75; 0.48/0.69 | 3.2 / 2.9 / 3.0; 1.2 | 0.22 / 0.29 / 0.34 | 0.004 / 0.021 / -0.000 | 40.3 |
| efficientnet_v2_s | 2021 | 21.46 | 83.1 | 83.1 / 83.0 / 83.0 (100.0 / 99.9 / 99.9) | 11.86 / 8.32 / 1.75 / 2.89 | 263 / 711 / 1307 / 1065 | 1.41/1.95 / 0.58/0.81 / 0.53/0.75; 0.53/0.72 | 2.6 / 2.3 / 2.6; 1.1 | 0.28 / 0.55 / 0.66 | 0.004 / 0.252 / 0.000 | 32.8 |
| vit_b16 | 2020 | 86.57 | 84.2 | 84.3 / 84.2 / 84.3 (99.8 / 99.6 / 99.8) | 6.13 / 2.37 / 1.49 / 2.11 | 275 / 1415 / 1649 / 1511 | 0.85/1.02 / 0.56/0.57 / 0.56/0.62; 0.13/0.37 | 3.1 / 2.9 / 2.8; 2.4 | 0.18 / 0.13 / 0.52 | 0.021 / -0.000 / 0.010 | 56.1 |
| deit_b | 2021 | 86.57 | 81.1 | 81.1 / 81.1 / 81.1 (100.0 / 100.0 / 100.0) | 6.13 / 2.44 / 1.50 / 2.09 | 276 / 1416 / 1647 / 1519 | 0.85/1.02 / 0.56/0.57 / 0.56/0.62; 0.13/0.37 | 3.1 / 3.6 / 2.8; 1.2 | 0.18 / 0.16 / 0.53 | 0.021 / -0.000 / 0.010 | 4.3 |
| swin_b | 2021 | 87.77 | 84.7 | 84.8 / 84.7 / 84.7 (99.9 / 100.0 / 100.0) | 11.50 / 8.77 / 3.67 / 2.54 | 228 / 886 / 1215 / 1383 | 1.63/2.48 / 0.88/1.31 / 0.75/0.91; 0.38/0.62 | 3.8 / 2.8 / 2.9; 2.4 | 0.23 / 0.17 / 0.88 | 0.004 / 0.000 / -0.001 | 70.6 |
| convnext_v2_b | 2023 | 88.72 | 86.0 | 86.0 / 86.0 / 85.4 (100.0 / 100.0 / 99.4) | 8.44 / 7.75 / 4.42 / 3.36 | 198 / 823 / 927 / 737 | 1.87/2.26 / 0.89/1.11 / 0.95/1.16; 0.53/0.76 | 2.9 / 2.8 / 3.0; 2.2 | 0.43 / 0.34 / 0.90 | 0.006 / 0.123 / 0.012 | 49.5 |
| eva02_l | 2023 | 305.08 | 88.7 | 88.7 / 88.7 / 88.8 (100.0 / 100.0 / 99.9) | 84.47 / 17.78 / 13.99 / 15.08 | 12 / 47 / 75 / 71 | 5.02/6.77 / 2.69/3.55 / 2.54/3.20; 1.93/2.70 | 5.7 / 5.0 / 4.9; 1.5 | 0.22 / 0.09 / 0.76 | 0.029 / -0.002 / 0.003 | 21.9 |
| fastvit_sa24 | 2023 | 21.55 | 80.5 | 80.5 / 77.4 / 80.5 (99.9 / 85.5 / 99.9) | 10.20 / 5.46 / 0.98 / 1.55 | 380 / 1309 / 3122 / 2693 | 1.19/1.46 / 0.46/0.57 / 0.34/0.45; 0.27/0.39 | 2.5 / 3.0 / 3.2; 1.2 | 0.28 / 0.60 / 0.47 | 0.003 / 0.359 / 0.000 | 53.4 |
| efficientvit_b2 | 2023 | 24.33 | 82.3 | 82.3 / 82.1 / 82.2 (99.8 / 99.6 / 99.6) | 8.40 / 9.39 / 0.91 / 1.35 | 511 / 1132 / 3170 / 2194 | 1.28/1.65 / 0.50/0.64 / 0.35/0.49; 0.30/0.43 | 2.2 / 2.3 / 2.5; 1.2 | 0.25 / 0.72 / 0.45 | 0.006 / 0.342 / 0.000 | 34.7 |
| mobilenet_v4_m | 2024 | 9.72 | 78.1 | 78.1 / 78.2 / 78.1 (100.0 / 99.9 / 100.0) | 4.63 / 2.96 / 0.45 / 0.68 | 1678 / 4574 / 11060 / 10975 | 0.74/0.91 / 0.28/0.39 / 0.25/0.27; 0.15/0.23 | 2.2 / 2.4 / 2.4; 1.1 | 0.24 / 0.65 / 0.30 | 0.003 / 0.377 / 0.008 | 17.1 |
| dinov2_l_lin | 2023 | 305.28 | 86.4 | 86.4 / 86.4 / 86.4 (100.0 / 100.0 / 100.0) | 21.86 / 4.87 / 4.80 / 6.04 | 61 / 335 / 382 / 345 | 1.99/2.27 / 1.16/1.24 / 1.44/1.63; 0.27/0.92 | 4.9 / 4.9 / 5.4; 1.5 | 0.21 / 0.09 / 0.70 | 0.014 / -0.005 / -0.001 | 12.1 |
| siglip2_so400m_zs | 2025 | 428.23 | 69.1 | 69.1 / 69.1 / - (99.6 / 99.7 / -) | 77.72 / 13.38 / 13.16 / - | 14 / 79 / 90 / - | 4.07/4.97 / 2.20/2.62 / 1.83/2.29; - | 12.2 / 10.7 / 10.6; - | 0.09 / 0.12 / 0.80 | 0.016 / -0.001 / -0.006 | - |
| siglip_so400m_ft | 2024 | 429.38 | 88.9 | 88.9 / 88.9 / 88.9 (100.0 / 100.0 / 100.0) | 77.11 / 12.96 / 13.12 / 12.30 | 14 / 80 / 90 / 90 | 4.09/4.81 / 2.48/2.52 / 2.38/2.74; 0.77/1.72 | 6.0 / 5.9 / 6.9; 1.8 | 0.20 / 0.09 / 0.82 | 0.023 / -0.006 / -0.009 | 39.9 |

Notes
- SigLIP 2 zero-shot top-1 (69.1, single prompt 'this is a photo of a {}.', original-case class names) is below the published
  ~84: a 100-image CPU check gave +5..10 points for lowercased prompts / prompt ensembles (HF: lowercase text). Paired
  comparison unaffected; not tuned (performance focus).
- CUDA graphs: not part of max (size-specialized). bench/speed.py still records b1_graph_ms (one graph per shape) in the
  jsonl as a separate reference column.
- EVA-02-L b64 row uses batch 64; EVA/SigLIP/DINOv2 inputs 448/378/384/224 (native).

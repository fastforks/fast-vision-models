# 009 segmentation — algo-only: A = upstream + one-liner switches vs B = upstream-friendly (uf) (2026-10-12)
date 2026-10-12; GPU NVIDIA GeForce RTX 5060 Ti; torch 2.14.1+cu130; bench script bench/algo_only.py @ 3bbb13d (+ bench/algo_only_all.sh, algo_only_table.py)
command: bench/algo_only_all.sh runs/algo_only/seg.jsonl (gpulock groups; raw rows runs/algo_only/seg.jsonl); table: python bench/algo_only_table.py
Switches for A and B (global, same process): cudnn.benchmark, TF32, channels_last, pure fp16 (A: base model .half() + fp16 channels_last input; uf is fp16 channels_last by construction).
uf uses NO fp16-accumulate flag, so A does not either. B = seg_models.build(name, 'uf') unchanged (fp32 input). Pipeline = seg_common.classmap (logits -> upsample -> argmax or model.predict).
Sizes 512x512 and 1024x2048, random input seed 0; median of 20 runs (10 if > 0.2 s, 5 if > 1 s) after 4 warm-ups. Memory = max allocated / reserved MB of that size's timing phase incl. weights (reserved: cache of earlier phases of the process is kept only partly, empty_cache before each size).
Accuracy = pixel agreement (%) of the class map vs upstream fp32 (no switches) on 12 ADE20K val images (every 41st) at the model's protocol resize; Cityscapes nets (unet-s5, ddrnet-23, pidnet-l) on the same images resized to 512x1024; segnet has random weights.
Context columns: ms from results/2026-10-10_round2_table.md (up = upstream fp32, max = graph-free max); MB from results/2026-10-10_bench.jsonl (records chosen as bench/summary_r2.py does; the 'up' record is the latest base run).

| model | setting | A ms | B ms | A÷B | A MB alloc / res | B MB alloc / res | A precision | acc A / B | up fp32 ms | max ms | up fp32 MB alloc / res | max MB alloc / res | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fcn-r101 | 512x512 | 18.7 | 17.3 | 1.08 | 296 / 430 | 223 / 332 | half | 100.0 / 100.0 | 42 | 10.9 | 491 / 826 | 205 / 408 | [cheap] BN fold |
| fcn-r101 | 1024x2048 | 160 | 158 | 1.01 | 1394 / 1830 | 790 / 1500 | half | 100.0 / 100.0 | 374 | 82.1 | 1760 / 4112 | 695 / 1048 | [cheap] BN fold |
| segnet | 512x512 | 16 | 13.5 | 1.19 | 367 / 672 | 253 / 354 | half | 97.2 / 97.2 | 31.6 | 10.3 | 550 / 774 | 252 / 430 | [cheap] BN fold |
| segnet | 1024x2048 | 144 | 121 | 1.19 | 3619 / 4594 | 1571 / 2354 | half | 97.2 / 97.2 | 261 | 88.1 | 3606 / 5754 | 1693 / 2536 | [cheap] BN fold |
| pspnet-r101 | 512x512 | 20.4 | 19 | 1.07 | 293 / 452 | 224 / 364 | half | 100.0 / 100.0 | 41.5 | 12.8 | 554 / 874 | 224 / 412 | [cheap] BN fold |
| pspnet-r101 | 1024x2048 | 191 | 190 | 1.01 | 1396 / 1840 | 843 / 1596 | half | 100.0 / 100.0 | 365 | 116 | 2103 / 4730 | 853 / 1116 | [cheap] BN fold |
| deeplabv3-r101 | 512x512 | 22 | 20.7 | 1.06 | 331 / 436 | 260 / 380 | half | 100.0 / 100.0 | 49.8 | 12.7 | 554 / 898 | 253 / 482 | [cheap] BN fold |
| deeplabv3-r101 | 1024x2048 | 184 | 183 | 1.01 | 1428 / 1792 | 827 / 1540 | half | 100.0 / 100.0 | 429 | 95 | 1767 / 4184 | 826 / 1090 | [cheap] BN fold |
| deeplabv3plus-r101 | 512x512 | 30.2 | 19.9 | 1.51 | 293 / 472 | 272 / 396 | half | 100.0 / 100.0 | 46.9 | 12.5 | 558 / 848 | 254 / 362 | [cheap] BN fold |
| deeplabv3plus-r101 | 1024x2048 | 211 | 181 | 1.16 | 2569 / 2672 | 1276 / 2188 | half | 100.0 / 100.0 | 417 | 101 | 2764 / 5278 | 1194 / 1634 | [cheap] BN fold |
| hrnet-w48 | 512x512 | 52.7 | 28.1 | 1.87 | 296 / 414 | 217 / 316 | half | 100.0 / 100.0 | 32.5 | 22.5 | 448 / 802 | 181 / 368 | [cheap] BN fold |
| hrnet-w48 | 1024x2048 | 84.5 | 82.8 | 1.02 | 2087 / 2412 | 809 / 1466 | half | 100.0 / 100.0 | 193 | 52.3 | 1782 / 4518 | 557 / 952 | [cheap] BN fold |
| segformer-b0 | 512x512 | 9.28 | 8.06 | 1.15 | 208 / 332 | 130 / 176 | half | 100.0 / 100.0 | 6.8 | 3.1 | 281 / 456 | 69 / 92 | [cheap] SDPA, BN fold |
| segformer-b0 | 1024x2048 | 34.3 | 28 | 1.23 | 1354 / 1884 | 722 / 998 | half | 100.0 / 100.0 | 71.7 | 16.3 | 1914 / 3322 | 259 / 386 | [cheap] SDPA, BN fold |
| segformer-b5 | 512x512 | 43.2 | 31.5 | 1.37 | 1241 / 1278 | 307 / 410 | half | 100.0 / 100.0 | 39.7 | 17.1 | 1051 / 1140 | 291 / 518 | [cheap] SDPA, BN fold |
| segformer-b5 | 1024x2048 | 135 | 108 | 1.25 | 8524 / 8586 | 898 / 1504 | half | 100.0 / 100.0 | 437 | 92.3 | 5848 / 6654 | 727 / 1030 | [cheap] SDPA, BN fold |
| segnext-t | 512x512 | 16.4 | 12.2 | 1.35 | 211 / 270 | 127 / 168 | autocast-fp16 | 99.9 / 99.9 | 9.2 | 9 | 205 / 286 | 74 / 96 | [cheap] BN fold (5); MSCA strip-conv block rewrite (13 blocks); NMF fp32 island |
| segnext-t | 1024x2048 | 68.7 | 39.1 | 1.76 | 1319 / 1558 | 752 / 830 | autocast-fp16 | 99.9 / 99.9 | 62.1 | 26.4 | 1310 / 1904 | 255 / 324 | [cheap] BN fold (5); MSCA strip-conv block rewrite (13 blocks); NMF fp32 island |
| mask2former-swinl | 512x512 | 42.3 | 39.2 | 1.08 | 774 / 1196 | 792 / 1016 | half | 99.9 / 99.9 | 79.3 | 21.3 | 1242 / 1812 | 708 / 932 | [cheap] Swin attention patch (24 layers) + SDPA, fp16 channels_last; no custom kernels |
| mask2former-swinl | 1024x2048 | 279 | 256 | 1.09 | 2919 / 5464 | 1834 / 3360 | half | 99.9 / 99.9 | 626 | 154 | 3924 / 8538 | 1612 / 2508 | [cheap] Swin attention patch (24 layers) + SDPA, fp16 channels_last; no custom kernels |
| oneformer-swinl | 512x512 | 48.3 | 46.6 | 1.04 | 925 / 1422 | 884 / 1168 | half | 99.8 / 99.9 | 95.8 | 27.7 | 1779 / 2222 | 784 / 1086 | [cheap] Swin attention patch (24 layers) + SDPA, fp32 query-transformer SDPA island |
| oneformer-swinl | 1024x2048 | 335 | 322 | 1.04 | 4036 / 7870 | 2431 / 4568 | half | 99.8 / 99.9 | 742 | 221 | 7987 / 11840 | 2129 / 3722 | [cheap] Swin attention patch (24 layers) + SDPA, fp32 query-transformer SDPA island |
| dinov2-l-linear | 512x512 | 28.2 | 26.4 | 1.07 | 1075 / 1196 | 711 / 774 | half | 100.0 / 99.9 | 108 | 19.3 | 1649 / 1756 | 707 / 778 | [cheap] SDPA ViT (fastbb standalone, sdpa-g0), BN folded into classifier |
| dinov2-l-linear | 1024x2048 | 497 | 489 | 1.02 | 3212 / 3344 | 2203 / 2354 | half | 100.0 / 99.9 | 1966 | 358 | 3794 / 4156 | 2211 / 2390 | [cheap] SDPA ViT (fastbb standalone, sdpa-g0), BN folded into classifier |
| eomt-l-dinov2 | 512x512 | 25.9 | 22.8 | 1.13 | 1096 / 1390 | 1168 / 1286 | half | 100.0 / 100.0 | 88.1 | 16.3 | 1697 / 2142 | 1165 / 1708 | [cheap] SDPA |
| eomt-l-dinov2 | 1024x2048 | 205 | 188 | 1.09 | 4295 / 6336 | 2297 / 3664 | half | 100.0 / 100.0 | 688 | 142 | 4914 / 8368 | 2298 / 3958 | [cheap] SDPA |
| eomt-l-dinov3 | 512x512 | 35.9 | 25 | 1.43 | 1698 / 2248 | 1160 / 1290 | autocast-fp16 | 100.0 / 100.0 | 91.7 | 18.5 | 1693 / 2142 | 1157 / 1704 | [cheap] SDPA |
| eomt-l-dinov3 | 1024x2048 | 299 | 213 | 1.41 | 4917 / 8932 | 2289 / 3710 | autocast-fp16 | 100.0 / 100.0 | 736 | 167 | 4910 / 8366 | 2290 / 3954 | [cheap] SDPA |
| unet-s5 | 512x512 | 15.1 | 14.4 | 1.05 | 478 / 530 | 317 / 390 | half | 99.9 / 99.9 | 34.4 | 9.3 | 744 / 1010 | 314 / 430 | [cheap] BN fold |
| unet-s5 | 1024x2048 | 128 | 125 | 1.02 | 3379 / 3402 | 2098 / 2624 | half | 99.9 / 99.9 | 277 | 73 | 5178 / 6648 | 2121 / 2670 | [cheap] BN fold |
| ddrnet-23 | 512x512 | 4.8 | 3.37 | 1.42 | 80 / 124 | 60 / 84 | half | 99.9 / 99.9 | 4.8 | 4.6 | 124 / 180 | 56 / 108 | [cheap] BN fold |
| ddrnet-23 | 1024x2048 | 12.5 | 12.2 | 1.03 | 305 / 344 | 163 / 286 | half | 99.9 / 99.9 | 27 | 7 | 441 / 622 | 168 / 212 | [cheap] BN fold |
| pidnet-l | 512x512 | 6.81 | 5.08 | 1.34 | 115 / 144 | 100 / 130 | half | 99.9 / 99.9 | 7.1 | 7 | 189 / 246 | 96 / 172 | [cheap] BN fold |
| pidnet-l | 1024x2048 | 20.7 | 20.3 | 1.02 | 339 / 376 | 236 / 352 | half | 99.9 / 99.9 | 46.4 | 12.5 | 506 / 688 | 244 / 308 | [cheap] BN fold |

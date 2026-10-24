# 008 classification — algo-only: A = upstream + one-liner switches vs B = upstream-friendly (2026-10-12)
date 2026-10-12; GPU NVIDIA GeForce RTX 5060 Ti; torch 2.14.1+cu130; bench script bench/algo_only.py @ 433d3a5 (+ bench/algo_only_all.sh, algo_only_table.py)
command: bench/algo_only_all.sh runs/algo_only/clf_v2.jsonl (gpulock groups; raw rows runs/algo_only/clf_v2.jsonl); table: python bench/algo_only_table.py
Switches for A and B (global, same process): cudnn.benchmark, TF32, allow_fp16_accumulation, hard fp16 (A: model.half() + fp16 channels_last input), channels_last.
B = clf_models.load(name, 'friendly') unchanged (fp32 input -> fp32 logits; input cast counted in B). b1 median of 30 after 12 warm-ups; b64 (halved on OOM) median of 10 (5 if > 1 s) after 4 warm-ups.
Memory = max allocated / reserved MB of that phase (weights included). Accuracy = top-1 agreement vs plain fp32 upstream on 256 real ImageNet val images (every 195th).
Rerun 2026-10-12 with the fixed script (A model freed before B is built; every group started only with 1-min load average < 2). All rows measured, no corrections applied; raw rows results/2026-10-12_algo_only_raw.jsonl.
Context columns copied from results/2026-10-10_zoo_table.md (ms; up fp32 = base, max = opt) and results/2026-10-10_speed.jsonl (peak MB, latest line per model x impl; '–' if not recorded at that batch); b64 ms = 64000 / img/s.

| model | setting | A ms | B ms | A÷B | A MB alloc / res | B MB alloc / res | A precision | acc A / B | up fp32 ms | max ms | up fp32 MB alloc / res | max MB alloc / res | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alexnet | b1 | 0.598 | 0.635 | 0.94 | 154 / 170 | 150 / 152 | half | 99.6 / 99.6 | 0.89 | 0.42 | 270 / 278 | 155 / 162 | [cheap] BN fold (fx graph) |
| alexnet | b64 | 3.33 | 3.45 | 0.96 | 350 / 394 | 256 / 278 | half | 99.6 / 99.6 | 9.52 | 1.99 | 465 / 630 | 298 / 304 | [cheap] BN fold (fx graph) |
| vgg16 | b1 | 1.69 | 1.71 | 0.99 | 333 / 396 | 310 / 396 | half | 100.0 / 100.0 | 3.45 | 1.31 | 611 / 622 | 323 / 340 | [cheap] BN fold (fx graph) |
| vgg16 | b64 | 67.2 | 67.4 | 1.00 | 2666 / 2750 | 1233 / 1572 | half | 100.0 / 100.0 | 143 | 28.8 | 3767 / 3776 | 1149 / 1158 | [cheap] BN fold (fx graph) |
| inception_v3 | b1 | 5.93 | 3.96 | 1.50 | 97 / 122 | 84 / 120 | half | 99.6 / 100.0 | 6.47 | 1.22 | 155 / 172 | 89 / 108 | [cheap] BN fold (fx graph) |
| inception_v3 | b64 | 45 | 44.6 | 1.01 | 909 / 1160 | 554 / 812 | half | 99.6 / 100.0 | 94.8 | 19.6 | 1315 / 1940 | 464 / 676 | [cheap] BN fold (fx graph) |
| resnet18 | b1 | 1.28 | 0.92 | 1.39 | 64 / 106 | 58 / 104 | half | 99.6 / 100.0 | 1.5 | 0.28 | 93 / 118 | 62 / 80 | [cheap] BN fold (fx graph) |
| resnet18 | b64 | 9.62 | 9.65 | 1.00 | 612 / 684 | 330 / 346 | half | 99.6 / 100.0 | 23.4 | 4.59 | 538 / 560 | 298 / 302 | [cheap] BN fold (fx graph) |
| resnet50 | b1 | 3.18 | 2.28 | 1.40 | 91 / 112 | 86 / 114 | half | 100.0 / 100.0 | 4.04 | 0.66 | 147 / 174 | 91 / 108 | [cheap] BN fold (fx graph) |
| resnet50 | b64 | 30.9 | 30.8 | 1.00 | 639 / 710 | 430 / 468 | half | 100.0 / 100.0 | 82.8 | 12.8 | 837 / 1254 | 422 / 526 | [cheap] BN fold (fx graph) |
| resnet152 | b1 | 9.28 | 6.42 | 1.45 | 157 / 166 | 152 / 166 | half | 99.6 / 99.6 | 10.3 | 1.68 | 279 / 300 | 160 / 180 | [cheap] BN fold (fx graph) |
| resnet152 | b64 | 66.7 | 66.8 | 1.00 | 705 / 782 | 496 / 540 | half | 99.6 / 99.6 | 184 | 29.3 | 970 / 1378 | 490 / 598 | [cheap] BN fold (fx graph) |
| densenet121 | b1 | 8.02 | 7.21 | 1.11 | 57 / 68 | 52 / 70 | half | 100.0 / 100.0 | 8.87 | 1.15 | 77 / 82 | 58 / 72 | [cheap] BN fold (fx graph) |
| densenet121 | b64 | 35.1 | 35.1 | 1.00 | 606 / 750 | 361 / 690 | half | 100.0 / 100.0 | 97.1 | 16.5 | 919 / 2104 | 402 / 734 | [cheap] BN fold (fx graph) |
| squeezenet1_1 | b1 | 1.11 | 1.23 | 0.91 | 7 / 26 | 6 / 26 | half | 99.6 / 99.6 | 1.23 | 0.18 | 11 / 28 | 8 / 28 | [cheap] BN fold (fx graph) |
| squeezenet1_1 | b64 | 6.64 | 6.78 | 0.98 | 347 / 410 | 272 / 280 | half | 99.6 / 99.6 | 14.8 | 1.95 | 377 / 400 | 133 / 170 | [cheap] BN fold (fx graph) |
| mobilenet_v1 | b1 | 1.71 | 1.16 | 1.48 | 50 / 60 | 43 / 58 | half | 99.2 / 100.0 | 1.65 | 0.2 | 58 / 82 | 45 / 62 | [cheap] BN fold (fx graph) |
| mobilenet_v1 | b64 | 9.07 | 9 | 1.01 | 721 / 864 | 243 / 312 | half | 99.2 / 100.0 | 25.3 | 3.2 | 706 / 816 | 209 / 228 | [cheap] BN fold (fx graph) |
| mobilenet_v2 | b1 | 3.02 | 2.16 | 1.40 | 54 / 70 | 42 / 58 | half | 100.0 / 98.4 | 2.73 | 0.28 | 56 / 62 | 44 / 60 | [cheap] BN fold (fx graph) |
| mobilenet_v2 | b64 | 10.9 | 10.6 | 1.03 | 1003 / 1158 | 279 / 366 | half | 100.0 / 98.4 | 28.5 | 3.9 | 752 / 1012 | 208 / 286 | [cheap] BN fold (fx graph) |
| shufflenet_v2 | b1 | 3.75 | 2.73 | 1.37 | 39 / 44 | 38 / 44 | half | 98.8 / 98.8 | 3.74 | 0.61 | 44 / 58 | 45 / 62 | [cheap] BN fold (fx graph) |
| shufflenet_v2 | b64 | 5.57 | 5.5 | 1.01 | 209 / 260 | 174 / 196 | half | 98.8 / 98.8 | 9.06 | 1.88 | 258 / 276 | 167 / 192 | [cheap] BN fold (fx graph) |
| mobilenet_v3_l | b1 | 3.56 | 2.79 | 1.28 | 52 / 108 | 45 / 108 | half | 99.6 / 99.6 | 3.42 | 0.54 | 61 / 70 | 47 / 62 | [cheap] BN fold (fx graph) |
| mobilenet_v3_l | b64 | 7.2 | 6.6 | 1.09 | 698 / 840 | 220 / 284 | half | 99.6 / 99.6 | 20.5 | 3.76 | 564 / 730 | 186 / 238 | [cheap] BN fold (fx graph) |
| efficientnet_b0 | b1 | 4.55 | 3.76 | 1.21 | 57 / 74 | 47 / 62 | half | 100.0 / 100.0 | 4.75 | 0.71 | 63 / 70 | 53 / 64 | [cheap] BN fold (fx graph) |
| efficientnet_b0 | b64 | 13.3 | 13 | 1.02 | 1007 / 1162 | 282 / 366 | half | 100.0 / 100.0 | 35.9 | 7.74 | 759 / 1020 | 297 / 380 | [cheap] BN fold (fx graph) |
| efficientnet_b4 | b1 | 10.6 | 7.51 | 1.42 | 171 / 242 | 88 / 242 | autocast-fp16 | 100.0 / 100.0 | 9.04 | 1.83 | 156 / 186 | 110 / 152 | [cheap] BN fold (fx graph) |
| efficientnet_b4 | b64 | 121 | 120 | 1.01 | 4286 / 4900 | 1022 / 1300 | autocast-fp16 | 100.0 / 100.0 | 305 | 66.9 | 2999 / 4070 | 1003 / 1278 | [cheap] BN fold (fx graph) |
| regnety_16gf | b1 | 7.52 | 6.37 | 1.18 | 220 / 244 | 212 / 220 | half | 99.6 / 99.6 | 11.5 | 1.7 | 393 / 438 | 212 / 236 | [cheap] BN fold (fx graph) |
| regnety_16gf | b64 | 102 | 101 | 1.00 | 1924 / 2104 | 1195 / 1274 | half | 99.6 / 99.6 | 233 | 41 | 2416 / 3218 | 744 / 770 | [cheap] BN fold (fx graph) |
| efficientnet_v2_s | b1 | 10.4 | 7.58 | 1.38 | 89 / 152 | 80 / 152 | half | 100.0 / 100.0 | 11.9 | 1.75 | 136 / 152 | 89 / 96 | [cheap] BN fold (fx graph) |
| efficientnet_v2_s | b64 | 91.1 | 89.8 | 1.01 | 1082 / 1268 | 560 / 796 | half | 100.0 / 100.0 | 243 | 49 | 1445 / 2000 | 539 / 772 | [cheap] BN fold (fx graph) |
| vit_b16 | b1 | 2.43 | 2.25 | 1.08 | 208 / 216 | 208 / 216 | half | 100.0 / 100.0 | 6.13 | 1.49 | 370 / 418 | 382 / 400 | [cheap] SDPA attention, LayerScale+residual addcmul |
| vit_b16 | b64 | 45.3 | 45.5 | 1.00 | 443 / 518 | 536 / 556 | half | 100.0 / 100.0 | 233 | 38.8 | 875 / 1044 | 573 / 634 | [cheap] SDPA attention, LayerScale+residual addcmul |
| deit_b | b1 | 2.36 | 2.22 | 1.07 | 208 / 216 | 208 / 216 | half | 100.0 / 100.0 | 6.13 | 1.5 | 370 / 418 | 382 / 400 | [cheap] SDPA attention, LayerScale+residual addcmul |
| deit_b | b64 | 45.3 | 45.4 | 1.00 | 443 / 518 | 536 / 556 | half | 100.0 / 100.0 | 232 | 38.9 | 875 / 1044 | 573 / 634 | [cheap] SDPA attention, LayerScale+residual addcmul |
| swin_b | b1 | 10.4 | 8.22 | 1.27 | 212 / 238 | 212 / 238 | half | 100.0 / 100.0 | 11.5 | 3.67 | 396 / 410 | 401 / 428 | [cheap] timm SDPA window attention |
| swin_b | b64 | 89.9 | 72.2 | 1.24 | 843 / 1274 | 865 / 1306 | half | 100.0 / 100.0 | 281 | 52.7 | 1670 / 2542 | 763 / 936 | [cheap] timm SDPA window attention |
| convnext_v2_b | b1 | 7.5 | 7.84 | 0.96 | 220 / 244 | 299 / 318 | half | 100.0 / 100.0 | 8.44 | 4.42 | 394 / 420 | 479 / 534 | [cheap] gamma fold into fc2, GRN as one addcmul |
| convnext_v2_b | b64 | 107 | 77.6 | 1.38 | 957 / 1254 | 877 / 1102 | half | 100.0 / 100.0 | 323 | 69 | 1910 / 2316 | 973 / 1186 | [cheap] gamma fold into fc2, GRN as one addcmul |
| fastvit_sa24 | b1 | 10.2 | 4.98 | 2.05 | 87 / 150 | 78 / 150 | half | 100.0 / 100.0 | 10.2 | 0.98 | 131 / 156 | 82 / 94 | [cheap] BN fold (fx graph); reparam |
| fastvit_sa24 | b64 | 72.1 | 48.1 | 1.50 | 961 / 1326 | 433 / 556 | half | 100.0 / 100.0 | 168 | 20.5 | 1219 / 1492 | 347 / 460 | [cheap] BN fold (fx graph); reparam |
| efficientvit_b2 | b1 | 10.3 | 9.22 | 1.11 | 152 / 174 | 85 / 104 | half | 99.6 / 100.0 | 8.4 | 0.91 | 144 / 156 | 89 / 106 | [cheap] BN fold (fx graph); reparam |
| efficientvit_b2 | b64 | 55.5 | 55.1 | 1.01 | 1691 / 1898 | 476 / 624 | half | 99.6 / 100.0 | 125 | 20.2 | 1313 / 1686 | 357 / 504 | [cheap] BN fold (fx graph); reparam |
| mobilenet_v4_m | b1 | 4.64 | 2.86 | 1.62 | 62 / 86 | 53 / 84 | half | 100.0 / 99.6 | 4.63 | 0.45 | 80 / 104 | 57 / 70 | [cheap] BN fold (fx graph) |
| mobilenet_v4_m | b64 | 12.3 | 11.8 | 1.04 | 763 / 932 | 251 / 378 | half | 100.0 / 99.6 | 38.1 | 5.79 | 758 / 932 | 260 / 276 | [cheap] BN fold (fx graph) |
| eva02_l | b1 | 17.7 | 17.7 | 1.00 | 645 / 672 | 647 / 692 | half | 99.6 / 99.6 | 84.5 | 14 | 1257 / 1392 | 1262 / 1388 | [cheap] SDPA attention, LayerScale+residual addcmul |
| eva02_l | b64 | 1355 | 1356 | 1.00 | 2570 / 3468 | 2719 / 3598 | half | 99.6 / 99.6 | 5333 | 853 | 5140 / 6934 | 2605 / 3278 | [cheap] SDPA attention, LayerScale+residual addcmul |
| dinov2_l_lin | b1 | 5.11 | 4.86 | 1.05 | 621 / 658 | 624 / 658 | half | 100.0 / 99.2 | 21.9 | 4.8 | 1210 / 1232 | 1208 / 1258 | [cheap] SDPA attention, LayerScale+residual addcmul |
| dinov2_l_lin | b64 | 198 | 190 | 1.04 | 1019 / 1174 | 1152 / 1240 | half | 100.0 / 99.2 | 1049 | 168 | 2040 / 2320 | 1472 / 1674 | [cheap] SDPA attention, LayerScale+residual addcmul |
| siglip2_so400m_zs | b1 | 13.2 | 13.4 | 0.99 | 881 / 916 | 902 / 992 | half | 99.6 / 99.6 | 77.7 | 13.2 | 1710 / 5870 | 1125 / 5870 | [cheap] BN-free; HF encoder as is (SDPA) |
| siglip2_so400m_zs | b64 | 807 | 808 | 1.00 | 2095 / 2450 | 2219 / 2650 | half | 99.6 / 99.6 | 4571 | 711 | 4169 / 5090 | 1869 / 2346 | [cheap] BN-free; HF encoder as is (SDPA) |
| siglip_so400m_ft | b1 | 12.8 | 12.9 | 0.99 | 880 / 908 | 888 / 906 | half | 99.6 / 99.6 | 77.1 | 13.1 | 1731 / 1822 | 1700 / 1752 | [cheap] SDPA attention, LayerScale+residual addcmul |
| siglip_so400m_ft | b64 | 798 | 798 | 1.00 | 2092 / 2436 | 2506 / 2540 | half | 99.6 / 99.6 | 4571 | 711 | 4185 / 4926 | 2439 / 2808 | [cheap] SDPA attention, LayerScale+residual addcmul |

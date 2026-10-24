"""Window protocols for the optimized paths: same window placement as bench/seg_common (EoMT windows, mmseg slide), but
NHWC fp16 score accumulation and the fused upsample-argmax kernel (no fp32 [C, H, W] buffers)."""
import torch
import torch.nn.functional as F

from .ops import upsample_argmax


def eomt_starts(L, crop):
    if L <= crop:
        return [0]
    n = -(-L // crop)
    ov = n * crop - L
    opc = ov / (n - 1) if ov > 0 else 0
    return [int(j * (crop - opc)) for j in range(n)]


def slide_starts(L, crop, stride):
    g = max(L - crop + stride - 1, 0) // stride + 1
    return [max(min(i * stride + crop, L) - crop, 0) for i in range(g)]


def merge_predict(scores_fn, x, wins, crop, out_size, batch=8, fused=True):
    """scores_fn(windows [b, 3, crop, crop]) -> NHWC fp16 scores at crop size; windows = list of (y, x) starts."""
    H, W = x.shape[2:]
    acc, cnt = None, torch.zeros((1, H, W, 1), device=x.device, dtype=torch.float16)
    for s in range(0, len(wins), batch):
        ws = wins[s:s + batch]
        cs = torch.cat([x[:, :, y:y + crop, xx:xx + crop] for y, xx in ws], 0)
        sc = scores_fn(cs)
        if sc.shape[1:3] != (cs.shape[2], cs.shape[3]):
            sc = F.interpolate(sc.permute(0, 3, 1, 2), cs.shape[2:], mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        if acc is None:
            if len(wins) == 1:
                return upsample_argmax(sc.contiguous(), *(out_size or (H, W)), fused=fused)
            acc = torch.zeros((1, H, W, sc.shape[3]), device=x.device, dtype=torch.float16)
        for k, (y, xx) in enumerate(ws):
            acc[:, y:y + crop, xx:xx + crop].add_(sc[k:k + 1])
            cnt[:, y:y + crop, xx:xx + crop] += 1
    acc.div_(cnt)
    return upsample_argmax(acc, *(out_size or (H, W)), fused=fused)


def eomt_predict(scores_fn, x, crop, out_size=None, fused=True):
    H, W = x.shape[2:]
    wins = [(y, xx) for y in eomt_starts(H, crop) for xx in eomt_starts(W, crop)]
    return merge_predict(scores_fn, x, wins, crop, out_size, fused=fused)


def slide_predict(scores_fn, x, crop, stride, out_size=None, fused=True):
    H, W = x.shape[2:]
    wins = [(y, xx) for y in slide_starts(H, crop, stride) for xx in slide_starts(W, crop, stride)]
    return merge_predict(scores_fn, x, wins, crop, out_size, fused=fused)

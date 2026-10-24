"""Shared segmentation pieces: ADE20K val loading, single-scale protocols, mIoU, timing helpers.

Model interface (zoo.py): an entry builds an object with
    .logits(x)   x = normalized float32 [1, 3, H, W] on the GPU (any H, W) -> class scores [1, C, h, w] (any res, any float dtype)
    .norm        (mean, std) in 0..255 RGB
    .protocol    dict: resize = ("mmseg", (long, short)) | ("short", s, max_long); infer = "whole" | ("slide", crop, stride);
                 pad = divisor the input must be padded to (whole mode) or 1
Latency (bench/bench.py) = logits at the input size -> bilinear to input size -> argmax (int map on the GPU).
"""
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ADE = os.environ.get("FVM_ADE20K", "data/ADEChallengeData2016")  # ADE20K release dir (images/, annotations/)
NUM_CLASSES = 150
IMNET = ((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))


def ade_val(subset="500"):
    names = sorted(os.listdir(os.path.join(ADE, "images", "validation")))
    if subset == "500":
        names = names[::4]
    elif subset not in ("all", "2000"):
        names = names[: int(subset)]
    return [(os.path.join(ADE, "images", "validation", n),
             os.path.join(ADE, "annotations", "validation", n.replace(".jpg", ".png"))) for n in names]


def load_pair(ip, ap):
    img = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(ap, cv2.IMREAD_UNCHANGED).astype(np.int64) - 1  # reduce_zero_label: 0 -> -1 (ignore)
    gt[gt < 0] = 255
    return img, gt


def target_size(h, w, resize):
    kind = resize[0]
    if kind == "mmseg":  # mmcv rescale keep_ratio to fit (long, short)
        L, S = resize[1]
        s = min(L / max(h, w), S / min(h, w))
    elif kind == "short":  # detectron2 ResizeShortestEdge(s, max_size)
        S, M = resize[1], resize[2]
        s = S / min(h, w)
        if max(h, w) * s > M:
            s = M / max(h, w)
    else:
        raise ValueError(kind)
    return int(h * s + 0.5), int(w * s + 0.5)


def preprocess(img, resize, norm, device="cuda"):
    h, w = img.shape[:2]
    th, tw = target_size(h, w, resize)
    r = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(r).to(device).permute(2, 0, 1)[None].float()
    m = torch.tensor(norm[0], device=device).view(1, 3, 1, 1)
    s = torch.tensor(norm[1], device=device).view(1, 3, 1, 1)
    return (x - m) / s


def pad_to(x, d):
    if d <= 1:
        return x, (x.shape[2], x.shape[3])
    h, w = x.shape[2:]
    ph, pw = (-h) % d, (-w) % d
    return (F.pad(x, (0, pw, 0, ph)) if ph or pw else x), (h, w)


def run_scores(model, x, out_size=None):
    """Scores [1, C, H, W] (fp32) at out_size (default: the size of x) following model.protocol['infer'].
    whole: the (cropped) scores are resized straight to out_size (mmseg postprocess); slide: accumulated at x's size."""
    inf = model.protocol.get("infer", "whole")
    H, W = x.shape[2:]
    if inf == "whole":
        H, W = out_size or (H, W)
        xp, (h, w) = pad_to(x, model.protocol.get("pad", 1))
        s = model.logits(xp).float()
        if xp.shape[2:] != x.shape[2:]:  # crop the padded region away (scores are at a fraction of the padded size)
            fh, fw = s.shape[2] / xp.shape[2], s.shape[3] / xp.shape[3]
            s = s[:, :, : round(h * fh), : round(w * fw)]
        return F.interpolate(s, (H, W), mode="bilinear", align_corners=model.protocol.get("align", False))
    if inf[0] == "eomt":  # EoMT window_imgs_semantic, generalized to 2-D; all windows in one batch (as upstream)
        crop = inf[1]
        ys, xs = eomt_starts(H, crop), eomt_starts(W, crop)
        wins = [(y, x) for y in ys for x in xs]
        cs = torch.cat([x[:, :, y:y + crop, xx:xx + crop] for y, xx in wins], 0)
        s = model.logits(cs).float()
        s = F.interpolate(s, (crop, crop), mode="bilinear", align_corners=False) if s.shape[2:] != (crop, crop) else s
        out = torch.zeros((1, s.shape[1], H, W), device=x.device)
        cnt = torch.zeros((1, 1, H, W), device=x.device)
        for k, (y, xx) in enumerate(wins):
            out[:, :, y:y + crop, xx:xx + crop] += s[k:k + 1]
            cnt[:, :, y:y + crop, xx:xx + crop] += 1
        out = out / cnt
        return out if out_size is None else F.interpolate(out, out_size, mode="bilinear", align_corners=False)
    _, crop, stride = inf  # mmseg slide_inference
    out, cnt = None, torch.zeros((1, 1, H, W), device=x.device)
    hg = max(H - crop + stride - 1, 0) // stride + 1
    wg = max(W - crop + stride - 1, 0) // stride + 1
    for i in range(hg):
        for j in range(wg):
            y1, x1 = i * stride, j * stride
            y2, x2 = min(y1 + crop, H), min(x1 + crop, W)
            y1, x1 = max(y2 - crop, 0), max(x2 - crop, 0)
            c = x[:, :, y1:y2, x1:x2]
            cp, (h, w) = pad_to(c, model.protocol.get("pad", 1))
            s = model.logits(cp).float()
            s = F.interpolate(s, cp.shape[2:], mode="bilinear", align_corners=False)[:, :, :h, :w]
            if out is None:
                out = torch.zeros((1, s.shape[1], H, W), device=x.device)
            out[:, :, y1:y2, x1:x2] += s
            cnt[:, :, y1:y2, x1:x2] += 1
    out = out / cnt
    return out if out_size is None else F.interpolate(out, out_size, mode="bilinear", align_corners=False)


def eomt_starts(L, crop):
    """EoMT window starts along one axis: ceil(L / crop) windows, overlap spread evenly (int() of the float start)."""
    if L <= crop:
        return [0]
    n = -(-L // crop)
    ov = n * crop - L
    opc = ov / (n - 1) if ov > 0 else 0
    return [int(j * (crop - opc)) for j in range(n)]


def classmap(model, x, out_size=None):
    """int class map [1, H, W] at out_size (default x's size): the model's fused path (model.predict) if it has one."""
    if getattr(model, "predict", None) is not None:
        return model.predict(x, out_size)
    return run_scores(model, x, out_size).argmax(1)


@torch.inference_mode()
def predict(model, img):
    x = preprocess(img, model.protocol["resize"], model.norm)
    return classmap(model, x, img.shape[:2])[0]


class MIoU:
    def __init__(self, n=NUM_CLASSES, device="cuda"):
        self.n = n
        self.hist = torch.zeros((n, n), dtype=torch.int64, device=device)

    def add(self, pred, gt):
        gt = torch.as_tensor(gt, device=pred.device)
        m = gt != 255
        idx = gt[m] * self.n + pred[m].long()
        self.hist += torch.bincount(idx, minlength=self.n * self.n).view(self.n, self.n)

    def result(self):
        h = self.hist.double()
        inter = h.diag()
        union = h.sum(0) + h.sum(1) - inter
        valid = h.sum(1) > 0
        iou = inter / union.clamp(min=1)
        return {"mIoU": float(iou[valid].mean() * 100), "aAcc": float(inter.sum() / h.sum() * 100)}


def git_rev(path):
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()

"""DINOv2 ViT-L/14 + linear BN head (official ADE20K head) on the fastbb standalone DINOv2 (exp 006, official weights):
fp16 ViT with the fastbb flash kernel, BN folded into the 1x1 classifier (one GEMM on the last layer's normed patch
tokens, NHWC scores), windows batched (mmseg slide placement, seg_fast.windows), fused upsample-argmax.
Center padding to a multiple of 14 as in the dinov2 segmentation hub models."""
import os

import torch
import torch.nn.functional as F



class FastDinov2Linear:
    def __init__(self, weights, head_sd, spec=os.environ.get("SEG_DINOV2_SPEC", "fh-g0"), fused=True):
        self.fused = fused
        from fastbb.models.dinov2 import dinov2
        self.vit = dinov2("vitl14", weights, spec=spec)
        g = {k.split("bn.")[1]: v.float() for k, v in head_sd.items() if ".bn." in k}
        sc = g["weight"] / torch.sqrt(g["running_var"] + 1e-5)
        sh = g["bias"] - g["running_mean"] * sc
        W = head_sd["decode_head.conv_seg.weight"].float().flatten(1)
        self.w = (W * sc[None]).half().cuda()
        self.b = (head_sd["decode_head.conv_seg.bias"].float() + W @ sh).half().cuda()
        self.info = {"opt": "dinov2-linear-fast", "spec": spec, "fused_up_argmax": fused}

    def scores_nhwc(self, x):
        B, _, H, W = x.shape
        ph, pw = (-H) % 14, (-W) % 14
        xp = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        t = self.vit.get_intermediate_layers(xp, n=[23])[0]                 # [B, hw, 1024] normed fp16
        h, w = xp.shape[2] // 14, xp.shape[3] // 14
        return F.linear(t, self.w, self.b).view(B, h, w, -1)

    def logits(self, x):
        return self.scores_nhwc(x).permute(0, 3, 1, 2)

    def predict(self, x, out_size=None):
        from .windows import slide_predict
        return slide_predict(self.scores_nhwc, x, 512, 341, out_size, fused=self.fused)

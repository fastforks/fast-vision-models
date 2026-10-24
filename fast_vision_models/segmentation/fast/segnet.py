"""SegNet (Badrinarayanan et al. 2015/2017), own PyTorch implementation of the paper architecture: VGG16-BN encoder
(13 conv3x3-BN-ReLU, 5 max-pools that keep their indices) and a mirrored decoder (max-unpool with the encoder indices,
conv3x3-BN-ReLU stacks, final conv3x3 to the classes). No trained weights for ADE20K exist -> random init, latency only.

FastSegNet: general fp16 channels-last path: BN folded into the convs, ReLU in place, fp16-accumulate cuDNN convs
(seg_fast/cudnn_conv, precompiled engines only), pooling indices kept as int64 (PyTorch max_unpool2d needs them).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CFG_ENC = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]


def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SegNet(nn.Module):
    def __init__(self, num_classes=150):
        super().__init__()
        self.enc = nn.ModuleList()
        c = 3
        for stage in CFG_ENC:
            layers = []
            for o in stage:
                layers.append(cbr(c, o))
                c = o
            self.enc.append(nn.Sequential(*layers))
        self.dec = nn.ModuleList()
        dec_cfg = [[512, 512, 512], [512, 512, 256], [256, 256, 128], [128, 64], [64]]
        for k, stage in enumerate(dec_cfg):
            layers = []
            for o in stage:
                layers.append(cbr(c, o))
                c = o
            self.dec.append(nn.Sequential(*layers))
        self.cls = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, x):
        idx, sizes = [], []
        for st in self.enc:
            x = st(x)
            sizes.append(x.shape[2:])
            x, i = F.max_pool2d(x, 2, 2, return_indices=True)
            idx.append(i)
        for st in self.dec:
            x = F.max_unpool2d(x, idx.pop(), 2, 2, output_size=sizes.pop())
            x = st(x)
        return self.cls(x)


def build(num_classes=150, seed=0):
    torch.manual_seed(seed)
    m = SegNet(num_classes)
    for mod in m.modules():  # non-trivial BN statistics so folding is exercised
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.1, 0.1)
            mod.running_var.uniform_(0.5, 1.5)
    return m.cuda().eval()

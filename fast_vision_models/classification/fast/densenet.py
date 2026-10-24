"""DenseNet (torchvision _DenseBlock) without the O(L^2) concatenations (plain PyTorch, upstream-friendly):
one [N, C_total, H, W] buffer per block; layer i reads the channel slice [:, :C_i] and writes its growth_rate new channels
into [:, C_i:C_i + k]. Same math as torch.cat(features, 1) at every layer.
"""
import types

import torch


def _copy(dst, src):
    # aten op (not Tensor.copy_ / setitem): fx records it as an impure call_function, so dead-code elimination keeps it
    return torch.ops.aten.copy_.default(dst, src)


def _block_forward(self, x):
    layers = list(self.values())
    k = layers[0].conv2.out_channels
    N, C0, H, W = x.shape
    ct = C0 + k * len(layers)
    buf = torch.empty((N, ct, H, W), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
    _copy(buf[:, :C0], x)
    c = C0
    for lay in layers:
        h = lay.relu1(lay.norm1(buf[:, :c]))
        h = lay.relu2(lay.norm2(lay.conv1(h)))
        _copy(buf[:, c:c + k], lay.conv2(h))
        c += k
    return buf


def patch_densenet(model):
    n = 0
    for m in model.modules():
        if type(m).__name__ == "_DenseBlock":
            m.forward = types.MethodType(_block_forward, m)
            n += 1
    return n

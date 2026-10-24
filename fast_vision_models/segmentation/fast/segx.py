"""torch.ops.segx: C++ ops of the max path (seg_fast/csrc), compiled once per source hash into the user cache
(problem sizes are runtime arguments: nothing is compiled per input size).

- segx_conv.cu: CUTLASS implicit-GEMM conv with fused bias / residual / activation, fp16 accumulation (copy of exp 008's
  clf_fast/csrc/fused_conv.cu; fixed tile rule inside the op) -> conv_fused, conv_fused_into, maxpool, avgpool, ...
- segx_ops.cu: the seg_kernels.cu kernels (bias_act_, dwconv, upsample_bilinear / upsample_like, up_argmax, scale_add).
"""
import os


from ..._build import jit_load

HERE = os.path.dirname(os.path.abspath(__file__))
SRCS = [os.path.join(HERE, "csrc", "segx_conv.cu"), os.path.join(HERE, "csrc", "segx_ops.cu")]
DEPS = [os.path.join(HERE, "seg_kernels.cu")]
_loaded = [False]


def load():
    if _loaded[0]:
        return True
    jit_load("segx", SRCS, deps=DEPS)
    _loaded[0] = True
    return True

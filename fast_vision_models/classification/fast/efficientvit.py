"""EfficientViT (MIT) LiteMLA max path: qkv 1x1 (CUTLASS, into a concat buffer) -> dw5x5 + grouped 1x1 in one kernel
(clfmb.mbconv_into, block-diagonal project, writes the second half of the buffer) -> ReLU linear attention kernel
(clfmb.linattn, fp32) -> proj 1x1 (+ BN folded, + residual of the EfficientVitBlock). Replaces the eager leaf module
inside the fx graph (the timm module has data-dependent Python control flow)."""
import os

import torch
import torch.nn as nn

V2 = os.environ.get("CLF_LMLA", "2") == "2"


class FastLiteMLA(nn.Module):
    def __init__(self, m):
        super().__init__()
        assert len(m.aggreg) == 1 and m.dim == 32 and isinstance(m.kernel_func, nn.ReLU)
        for sub in (m.qkv.norm, m.qkv.act, m.proj.act):
            assert isinstance(sub, nn.Identity), sub
        dt = m.qkv.conv.weight.dtype
        self.register_buffer("w_qkv", m.qkv.conv.weight.detach().contiguous(memory_format=torch.channels_last))
        self.register_buffer("b_qkv", m.qkv.conv.bias.detach() if m.qkv.conv.bias is not None else None)
        dw, gp = m.aggreg[0]
        C = dw.weight.shape[0]
        assert gp.groups * 32 == C and dw.weight.shape[0] % 32 == 0
        self.C, self.k = C, dw.kernel_size[0]
        self.register_buffer("wd", dw.weight.detach()[:, 0].permute(1, 2, 0).contiguous())
        self.register_buffer("bd", dw.bias.detach() if dw.bias is not None else torch.zeros(C, dtype=dt, device=dw.weight.device))
        self.register_buffer("w2g", gp.weight.detach().reshape(C // 32, 32, 32).contiguous())
        self.register_buffer("w2c", gp.weight.detach().contiguous(memory_format=torch.channels_last))   # [C, 32, 1, 1]
        self.groups = gp.groups
        self.has_bd = dw.bias is not None
        self.register_buffer("b2g", gp.bias.detach() if gp.bias is not None else None)
        self.register_buffer("colmap", torch.tensor([C + 8 * j for j in range(C // 8)], dtype=torch.int32,
                                                    device=dw.weight.device))
        pw, bn = m.proj.conv, m.proj.norm
        w, b = pw.weight.float(), (pw.bias.float() if pw.bias is not None else torch.zeros(pw.weight.shape[0], device=pw.weight.device))
        if isinstance(bn, nn.BatchNorm2d):
            sc = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
            w, b = w * sc.view(-1, 1, 1, 1), (b - bn.running_mean.float()) * sc + bn.bias.float()
        self.register_buffer("w_p", w.to(dt).contiguous(memory_format=torch.channels_last))
        self.register_buffer("b_p", b.to(dt))
        self.eps = float(m.eps)

    def forward(self, x, res=None):
        if V2:   # qkv conv -> dwconv -> grouped 1x1 (clfmb.gconv) -> tensor-core linear attention over both tensors
            qkv = torch.ops.clf.conv_fused(x, self.w_qkv, self.b_qkv, None, [0, 0], [1, 1], 0)
            p = self.k // 2
            d = torch.ops.clf.dwconv(qkv, self.wd, self.bd if self.has_bd else None, [1, 1], [p, p], 0)
            agg = torch.ops.clfmb.gconv(d, self.w2c, self.b2g, [1, 1], [0, 0], self.groups, 0)
            a = torch.ops.clfmb.linattn2(qkv, agg, self.eps)
            return torch.ops.clf.conv_fused(a, self.w_p, self.b_p, res, [0, 0], [1, 1], 0)
        n, _, h, w = x.shape
        buf = torch.empty((n, 2 * self.C, h, w), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
        torch.ops.clf.conv_fused_into(x, self.w_qkv, self.b_qkv, None, [0, 0], [1, 1], 0, buf[:, :self.C])
        p = self.k // 2
        torch.ops.clfmb.mbconv_into(buf, None, None, self.wd, self.bd, self.w2g, self.b2g, [1, 1], [p, p], [0, 0, 0], 0,
                                    self.C, buf, self.colmap, None, None)
        a = torch.ops.clfmb.linattn(buf, 32, self.eps)
        return torch.ops.clf.conv_fused(a, self.w_p, self.b_p, res, [0, 0], [1, 1], 0)


def patch_litemla(gm):
    """swap LiteMLA leaf modules in an fx GraphModule for FastLiteMLA; fold the following residual add."""
    import operator
    from .mbconv import load
    load()
    mods = dict(gm.named_modules())
    n = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_module" or type(mods.get(node.target)).__name__ != "LiteMLA":
            continue
        gm.add_submodule(node.target, FastLiteMLA(mods[node.target]))
        x = node.args[0]
        us = list(node.users)
        if len(us) == 1 and us[0].op == "call_function" and us[0].target in (operator.add, torch.add) and \
                set(us[0].args) == {node, x} and not us[0].kwargs:
            add = us[0]
            node.args = (x, x)
            add.replace_all_uses_with(node)
            gm.graph.erase_node(add)
        n += 1
    gm.graph.lint()
    gm.recompile()
    return n

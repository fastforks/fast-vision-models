"""Fused conv (+bias +residual +activation) for the max path (csrc/fused_conv.cu, CUTLASS implicit GEMM, fp16 accumulate).

The extension is compiled once per source hash into the user cache (fast_vision_models/_build.py) (no per-size compilation; the problem size is a
runtime argument) and registered as torch.ops.clf.conv_fused. `fuse_graph(gm)` rewrites an fx GraphModule (after BN fold):
    conv -> [add(other)] -> [act]     ==>   clf.conv_fused(x, w, b, other, pad, stride, act)
for Conv2d with groups 1, dilation 1, zero padding, C % 8 == 0, K % 8 == 0 (others stay torch convs).
At small B*H*W (< CLF_FUSED_PIX, default 0 = always) the plain torch conv path is used instead (the caller keeps both graphs).
"""
import operator
import os

import torch

from ..._build import jit_load
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "csrc", "fused_conv.cu")
_loaded = [False]

ACTS = {nn.ReLU: 1, nn.ReLU6: 2, nn.SiLU: 3, nn.Hardswish: 4, F.relu: 1, F.relu6: 2, F.silu: 3, F.hardswish: 4,
        torch.relu: 1}


def load():
    if _loaded[0]:
        return True
    jit_load("clf_fused", [SRC])
    _loaded[0] = True
    return True


def _act_of(node, mods):
    if node.op == "call_module":
        m = mods[node.target]
        if type(m) is nn.GELU:
            return 5 if m.approximate == "none" else 6
        return ACTS.get(type(m))
    if node.op == "call_function":
        return ACTS.get(node.target)
    if node.op == "call_method" and node.target in ("relu", "relu_"):
        return 1
    return None


def _ok_conv(m, stem_ok=False):
    return (type(m) is nn.Conv2d and m.groups == 1 and m.dilation == (1, 1) and m.padding_mode == "zeros"
            and not isinstance(m.padding, str) and (m.in_channels % 8 == 0 or (stem_ok and m.in_channels < 8))
            and m.out_channels % 8 == 0 and m.weight.dtype == torch.half)


STEM_K = (16, 24, 32, 40, 48, 64)


def direct_stem(gm):
    """If the graph input feeds exactly one Conv2d with C <= 4, R*S <= 25, K in STEM_K: replace it (+ its activation) by
    clfmb.stem_direct, which reads the raw fp32 NCHW input (no conversion kernel). Sets gm.raw_input. Returns True if done."""
    if os.environ.get("CLF_STEM", "1") != "1":
        return False
    mods = dict(gm.named_modules())
    ph = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if len(ph) != 1 or len(ph[0].users) != 1:
        return False
    u = next(iter(ph[0].users))
    c = mods.get(u.target) if u.op == "call_module" else None
    if not (type(c) is nn.Conv2d and c.groups == 1 and c.dilation == (1, 1) and c.padding_mode == "zeros"
            and not isinstance(c.padding, str) and c.in_channels <= 4 and c.kernel_size[0] * c.kernel_size[1] <= 25
            and c.out_channels in STEM_K and c.weight.dtype == torch.half):
        return False
    from .mbconv import load as lm
    lm()
    c.register_buffer("w_t", c.weight.data.permute(2, 3, 1, 0).contiguous())
    last, act = u, 0
    if len(u.users) == 1:
        a = _act_of(next(iter(u.users)), mods)
        if a is not None:
            act, last = a, next(iter(u.users))
    g = gm.graph
    with g.inserting_before(last.next):
        w = g.get_attr(u.target + ".w_t")
        b = g.get_attr(u.target + ".bias") if c.bias is not None else None
        new = g.call_function(torch.ops.clfmb.stem_direct, (ph[0], w, b, list(c.stride), list(c.padding), act))
    last.replace_all_uses_with(new)
    new.args = (ph[0], w, b, list(c.stride), list(c.padding), act)
    g.eliminate_dead_code()
    return True


def pad_stem(gm):
    """If the graph input feeds exactly one Conv2d with C < 8: zero-pad its weight to C = 8 (input then comes from
    clf.input_nhwc8). Returns True if done."""
    mods = dict(gm.named_modules())
    ph = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if len(ph) != 1 or len(ph[0].users) != 1:
        return False
    u = next(iter(ph[0].users))
    if u.op != "call_module" or not _ok_conv(mods.get(u.target), stem_ok=True) or mods[u.target].in_channels >= 8:
        return False
    c = mods[u.target]
    w = torch.zeros(c.out_channels, 8, *c.kernel_size, dtype=c.weight.dtype, device=c.weight.device)
    w[:, :c.in_channels] = c.weight.data
    c.weight = nn.Parameter(w, requires_grad=False)
    c.in_channels = 8
    return True


def replace_maxpool(gm):
    mods = dict(gm.named_modules())
    n = 0
    for node in list(gm.graph.nodes):
        m = mods.get(node.target) if node.op == "call_module" else None
        if type(m) is nn.MaxPool2d and m.dilation in (1, (1, 1)) and not m.return_indices:
            pair = lambda v: list(v) if isinstance(v, (tuple, list)) else [v, v]  # noqa: E731
            with gm.graph.inserting_before(node):
                new = gm.graph.call_function(torch.ops.clf.maxpool, (node.args[0], pair(m.kernel_size),
                                             pair(m.stride if m.stride is not None else m.kernel_size), pair(m.padding),
                                             bool(m.ceil_mode)))
            node.replace_all_uses_with(new)
            gm.graph.erase_node(node)
            n += 1
    return n


ADDS = (operator.add, torch.add, operator.iadd)


def fold_layer_scale(gm):
    """Conv2d -> mul(per-channel constant gamma)  ==>  Conv2d with gamma folded into weight and bias (timm LayerScale2d;
    lets the following residual add fuse into the conv epilogue). Dropout calls (eval: identity) are dropped first."""
    mods = dict(gm.named_modules())
    g = gm.graph
    for node in list(g.nodes):
        if node.op == "call_module" and isinstance(mods.get(node.target), nn.Dropout):
            node.replace_all_uses_with(node.args[0])
            g.erase_node(node)
    n = 0
    for node in list(g.nodes):
        m = mods.get(node.target) if node.op == "call_module" else None
        if type(m) is not nn.Conv2d or len(node.users) != 1:
            continue
        u = next(iter(node.users))
        if not (u.op == "call_function" and u.target in (operator.mul, torch.mul) and len(u.args) == 2 and not u.kwargs):
            continue
        other = u.args[1] if u.args[0] is node else u.args[0]
        if not (isinstance(other, torch.fx.Node) and other.op == "get_attr"):
            continue
        gam = gm.get_parameter(other.target) if other.target in dict(gm.named_parameters()) else None
        if gam is None or gam.numel() != m.out_channels or sum(1 for _ in [n_ for n_ in g.nodes if n_.op == "call_module" and n_.target == node.target]) != 1:
            continue
        with torch.no_grad():
            gv = gam.detach().reshape(-1).float()
            m.weight.data = (m.weight.float() * gv.view(-1, 1, 1, 1)).to(m.weight.dtype)
            if m.bias is not None:
                m.bias.data = (m.bias.float() * gv).to(m.bias.dtype)
        u.replace_all_uses_with(node)
        g.erase_node(u)
        n += 1
    g.eliminate_dead_code()
    return n


def fuse_graph(gm):
    """In-place fx rewrite of conv -> [add(other)] -> [act] chains; returns the number of fused convs."""
    load()
    gm.nls = fold_layer_scale(gm)
    gm.npad = pad_conv_pairs(gm)
    gm.raw_input = direct_stem(gm)
    gm.pad8 = False if gm.raw_input else pad_stem(gm)
    gm.nmaxpool = 0
    mods = dict(gm.named_modules())
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if node.op != "call_module" or not _ok_conv(mods.get(node.target)):
            continue
        conv = mods[node.target]
        conv.weight.data = conv.weight.data.contiguous(memory_format=torch.channels_last)
        last, res, act = node, None, 0
        users = list(last.users)
        if len(users) == 1 and users[0].op == "call_function" and users[0].target in ADDS and len(users[0].args) == 2 \
                and not users[0].kwargs:
            u = users[0]
            other = u.args[1] if u.args[0] is last else u.args[0]
            if isinstance(other, torch.fx.Node) and other is not last:
                res, last = other, u
                users = list(last.users)
        if len(users) == 1:
            a = _act_of(users[0], mods)
            if a is not None:
                act, last = a, users[0]
        with g.inserting_before(last.next):
            w = g.get_attr(node.target + ".weight")
            b = g.get_attr(node.target + ".bias") if conv.bias is not None else None
            new = g.call_function(torch.ops.clf.conv_fused,
                                  (node.args[0], w, b, res, list(conv.padding), list(conv.stride), act))
        last.replace_all_uses_with(new)
        new.args = (node.args[0], w, b, res, list(conv.padding), list(conv.stride), act)
        n += 1
    g.eliminate_dead_code()
    g.lint()
    gm.recompile()
    return n


def _ok_dw(m):
    return (type(m) is nn.Conv2d and m.groups == m.in_channels == m.out_channels and m.in_channels % 8 == 0
            and m.dilation == (1, 1) and m.padding_mode == "zeros" and not isinstance(m.padding, str)
            and m.weight.dtype == torch.half)


def fuse_dw(gm):
    """depthwise Conv2d [-> act]  ==>  clf.dwconv (weight stored as [R, S, C] buffer)."""
    mods = dict(gm.named_modules())
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if node.op != "call_module" or not _ok_dw(mods.get(node.target)):
            continue
        conv = mods[node.target]
        conv.register_buffer("w_rsc", conv.weight.data[:, 0].permute(1, 2, 0).contiguous())
        last, act = node, 0
        users = list(node.users)
        if len(users) == 1:
            a = _act_of(users[0], mods)
            if a is not None:
                act, last = a, users[0]
        with g.inserting_before(last.next):
            w = g.get_attr(node.target + ".w_rsc")
            b = g.get_attr(node.target + ".bias") if conv.bias is not None else None
            new = g.call_function(torch.ops.clf.dwconv, (node.args[0], w, b, list(conv.stride), list(conv.padding), act))
        last.replace_all_uses_with(new)
        new.args = (node.args[0], w, b, list(conv.stride), list(conv.padding), act)
        n += 1
    g.eliminate_dead_code()
    return n


def replace_avgpool(gm):
    """nn.AvgPool2d / F.avg_pool2d (constant geometry, no ceil_mode / divisor_override) -> clf.avgpool (C % 8 == 0 checked
    at run time by the op)."""
    mods = dict(gm.named_modules())
    pair = lambda v: list(v) if isinstance(v, (tuple, list)) else [v, v]  # noqa: E731
    n = 0
    for node in list(gm.graph.nodes):
        geo = None
        if node.op == "call_module" and type(mods.get(node.target)) is nn.AvgPool2d:
            m = mods[node.target]
            if not m.ceil_mode and m.divisor_override is None:
                geo = (m.kernel_size, m.stride if m.stride is not None else m.kernel_size, m.padding, m.count_include_pad)
        elif node.op == "call_function" and node.target is F.avg_pool2d:
            names = ["input", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"]
            a = dict(zip(names, node.args))
            a.update(node.kwargs)
            if not a.get("ceil_mode", False) and a.get("divisor_override") is None and \
                    all(not isinstance(a.get(k), torch.fx.Node) for k in ("kernel_size", "stride", "padding")):
                st = a.get("stride") or a["kernel_size"]
                geo = (a["kernel_size"], st, a.get("padding", 0), a.get("count_include_pad", True))
        if geo is None:
            continue
        with gm.graph.inserting_before(node):
            new = gm.graph.call_function(torch.ops.clf.avgpool, (node.args[0], pair(geo[0]), pair(geo[1]), pair(geo[2]),
                                                                 bool(geo[3])))
        node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        n += 1
    return n


def fuse_all(gm):
    n = fuse_graph(gm)
    if os.environ.get("CLF_DW", "1") == "1":
        n += fuse_dw(gm)
    replace_bn_act(gm)
    fuse_into(gm)
    from . import r3
    gm.nmergecat = r3.merge_cat_convs(gm)
    cat_into(gm)
    k = replace_maxpool(gm) + replace_avgpool(gm)
    gm.nstempool = r3.fuse_stem_pool(gm)
    gm.graph.lint()
    gm.recompile()
    return n, k


def replace_bn_act(gm):
    """remaining (unfoldable, e.g. pre-activation) eval BatchNorm2d [-> ReLU] -> clf.affine_act (one pass, reads channel
    slices of a wider buffer without a copy)."""
    mods = dict(gm.named_modules())
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        m = mods.get(node.target) if node.op == "call_module" else None
        if not isinstance(m, nn.BatchNorm2d) or type(m) is not nn.BatchNorm2d or m.num_features % 8:
            continue
        with torch.no_grad():
            sc = (m.weight.float() if m.affine else 1) / torch.sqrt(m.running_var.float() + m.eps)
            sh = (m.bias.float() if m.affine else 0) - m.running_mean.float() * sc
        m.register_buffer("_sc", sc.contiguous(), persistent=False)
        m.register_buffer("_sh", sh.contiguous(), persistent=False)
        last, act = node, 0
        users = list(node.users)
        if len(users) == 1 and _act_of(users[0], mods) == 1:
            last, act = users[0], 1
        with g.inserting_before(last.next):
            s_ = g.get_attr(node.target + "._sc")
            t_ = g.get_attr(node.target + "._sh")
            new = g.call_function(torch.ops.clf.affine_act, (node.args[0], s_, t_, act))
        last.replace_all_uses_with(new)
        n += 1
    g.eliminate_dead_code()
    return n


def fuse_into(gm):
    """aten.copy_(buf[idx], clf.conv_fused(...))  ->  clf.conv_fused_into(..., out=buf[idx])  (DenseNet / concat buffers)."""
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target is not torch.ops.aten.copy_.default:
            continue
        view, val = node.args[:2]
        if not (isinstance(val, torch.fx.Node) and val.target is torch.ops.clf.conv_fused and len(val.users) == 1):
            continue
        with g.inserting_before(node):
            g.call_function(torch.ops.clf.conv_fused_into.default, tuple(val.args[:7]) + (view,))
        node.replace_all_uses_with(view)
        g.erase_node(node)
        g.erase_node(val)
        n += 1
    return n


def _cat_buffer(x, w, pad, stride, ctot):
    """channels-last fp16 buffer [N, ctot, P, Q] for the outputs of convs that read x (shape arithmetic is traced)."""
    n, _, h, wd = x.shape
    p = (h + 2 * pad[0] - w.shape[2]) // stride[0] + 1
    q = (wd + 2 * pad[1] - w.shape[3]) // stride[1] + 1
    return torch.empty((n, ctot, p, q), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)


def cat_into(gm):
    """torch.cat([conv_fused | cat of conv_fused, ...], dim=1)  ->  one buffer, every conv writes its channel slice
    (clf.conv_fused_into): no concatenation copies (SqueezeNet Fire, Inception A / C / E)."""
    g = gm.graph
    n = 0

    def parts(node):
        if node.op == "call_function" and node.target is torch.cat and len(node.args) >= 1 and \
                (node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)) == 1:
            out = []
            for a in node.args[0]:
                if not isinstance(a, torch.fx.Node):
                    return None
                if a.target is torch.cat and len(a.users) == 1:
                    sub = parts(a)
                    if sub is None:
                        return None
                    out += sub
                elif a.target is torch.ops.clf.conv_fused and len(a.users) == 1:
                    out.append(a)
                else:
                    return None
            return out
        return None

    for node in list(g.nodes):
        if node.op != "call_function" or node.target is not torch.cat:
            continue
        if any(u.target is torch.cat for u in node.users):   # handled as part of the outer cat
            continue
        ps = parts(node)
        if not ps or len(ps) < 2:
            continue
        ks = []
        for c in ps:
            wn = c.args[1]
            w = gm.get_parameter(wn.target) if wn.op == "get_attr" else None
            if w is None or w.shape[0] % 8:
                ks = None
                break
            ks.append(w.shape[0])
        if ks is None:
            continue
        first = min(ps, key=lambda c: list(g.nodes).index(c))
        with g.inserting_before(first):
            buf = g.call_function(_cat_buffer, (first.args[0], first.args[1], first.args[4], first.args[5], sum(ks)))
        off = 0
        for c, k in zip(ps, ks):
            with g.inserting_before(c):
                view = g.call_function(operator.getitem, (buf, (slice(None), slice(off, off + k))))
                g.call_function(torch.ops.clf.conv_fused_into.default, tuple(c.args[:7]) + (view,))
            off += k
        node.replace_all_uses_with(buf)
        g.eliminate_dead_code()
        n += 1
    return n


def pad_conv_pairs(gm):
    """Conv2d A (out % 8 != 0) -> [act] -> Conv2d B (in = A.out, only consumer): zero-pad A's output / B's input channels
    to a multiple of 8 (SE squeeze/excite 1x1 convs of RegNet / EfficientNet), so both run on the CUTLASS path instead of
    cuDNN with layout transforms. Exact: padded channels are act(0 + 0) = 0 for relu/silu/gelu/hardswish and get zero weights."""
    mods = dict(gm.named_modules())
    n = 0
    for node in list(gm.graph.nodes):
        a = mods.get(node.target) if node.op == "call_module" else None
        if type(a) is not nn.Conv2d or a.groups != 1 or a.out_channels % 8 == 0 or len(node.users) != 1:
            continue
        u = next(iter(node.users))
        if _act_of(u, mods) is not None and len(u.users) == 1:
            u = next(iter(u.users))
        b = mods.get(u.target) if u.op == "call_module" else None
        if type(b) is not nn.Conv2d or b.groups != 1 or b.in_channels != a.out_channels:
            continue
        if sum(1 for x in gm.graph.nodes if x.op == "call_module" and x.target in (node.target, u.target)) != 2:
            continue
        c = a.out_channels
        cp = (c + 7) // 8 * 8
        with torch.no_grad():
            wa = torch.zeros((cp,) + tuple(a.weight.shape[1:]), dtype=a.weight.dtype, device=a.weight.device)
            wa[:c] = a.weight
            a.weight = nn.Parameter(wa, requires_grad=False)
            if a.bias is not None:
                ba = torch.zeros(cp, dtype=a.bias.dtype, device=a.bias.device)
                ba[:c] = a.bias
                a.bias = nn.Parameter(ba, requires_grad=False)
            wb = torch.zeros((b.out_channels, cp) + tuple(b.weight.shape[2:]), dtype=b.weight.dtype, device=b.weight.device)
            wb[:, :c] = b.weight
            b.weight = nn.Parameter(wb, requires_grad=False)
        a.out_channels = cp
        b.in_channels = cp
        n += 1
    return n

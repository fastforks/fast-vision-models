"""Round-3 kernels (csrc/r3.cu = extension `clfr3`) and fx passes for the max path.

- stem_pool: direct stem conv + act + max pool in one kernel (`fuse_stem_pool`, after fused.fuse_all)
Compiled once per source hash into the user cache (problem size = runtime arguments).
"""
import os

import torch

from ..._build import jit_load

from .fused import HERE

SRC = os.path.join(HERE, "csrc", "r3.cu")
_loaded = [False]
MERGE_M = int(os.environ.get("CLF_MERGE_M", "8192"))   # clfr3.cat2: merged conv up to this many output pixels


def load():
    if _loaded[0]:
        return True
    from .fused import load as _lf

    _lf()
    jit_load("clf_r3", [SRC], extra_cflags=["-lineinfo"], extra_ldflags=["-lcublasLt"])
    _loaded[0] = True
    return True


def fuse_stem_pool(gm):
    """clfmb.stem_direct -> clf.maxpool (square, only consumer)  ==>  clfr3.stem_pool. Returns 1 if done."""
    if os.environ.get("CLF_STEMPOOL", "1") != "1":
        return 0
    from .mbconv import load as lm
    lm()
    g = gm.graph
    for node in g.nodes:
        if node.op != "call_function" or node.target is not torch.ops.clfmb.stem_direct or len(node.users) != 1:
            continue
        u = next(iter(node.users))
        if u.op != "call_function" or u.target is not torch.ops.clf.maxpool:
            return 0
        _, k, s, p, ceil = u.args
        if k[0] != k[1] or s[0] != s[1] or p[0] != p[1]:
            return 0
        load()
        x, w, b, st, pad, act = node.args
        with g.inserting_before(u):
            new = g.call_function(torch.ops.clfr3.stem_pool, (x, w, b, st, pad, act, k[0], s[0], p[0], bool(ceil)))
        u.replace_all_uses_with(new)
        g.erase_node(u)
        g.erase_node(node)
        g.lint()
        gm.recompile()
        return 1
    return 0


def _param(gm, n):
    if n is None:
        return None
    try:
        return gm.get_parameter(n.target)
    except AttributeError:
        return gm.get_buffer(n.target)


def merge_cat_convs(gm):
    """torch.cat([conv_fused(x, ...), conv_fused(x, ...), ...], 1) where every conv reads the same x with the same stride,
    act, no residual and square odd kernels with 'same' padding (k // 2)  ==>  ONE conv_fused with the smaller kernels
    zero-embedded at the centre of the largest (SqueezeNet Fire: expand 1x1 + expand 3x3 -> one 3x3 conv, K = 128).
    More MACs for the small kernels, but one launch, one read of x and full-width tiles. Run before fused.cat_into."""
    if os.environ.get("CLF_MERGECAT", "1") != "1":
        return 0
    import torch.nn as nn
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target is not torch.cat:
            continue
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        ps = node.args[0]
        if dim != 1 or len(ps) < 2 or not all(isinstance(p, torch.fx.Node) and p.target is torch.ops.clf.conv_fused
                                              and len(p.users) == 1 for p in ps):
            continue
        x0 = ps[0].args[0]
        ws = [_param(gm, p.args[1]) for p in ps]
        ok = all(p.args[0] is x0 and p.args[3] is None and list(p.args[5]) == list(ps[0].args[5]) and p.args[6] == ps[0].args[6]
                 for p in ps)
        ok = ok and all(w.shape[2] == w.shape[3] and w.shape[2] % 2 == 1 and list(p.args[4]) == [w.shape[2] // 2] * 2
                        for w, p in zip(ws, ps))
        if not ok or list(ps[0].args[5]) != [1, 1]:
            continue
        load()
        kk = max(w.shape[2] for w in ws)
        ktot = sum(w.shape[0] for w in ws)
        C = ws[0].shape[1]
        wm = torch.zeros(ktot, C, kk, kk, dtype=ws[0].dtype, device=ws[0].device)
        bm = torch.zeros(ktot, dtype=ws[0].dtype, device=ws[0].device)
        off = 0
        for w, p in zip(ws, ps):
            o = (kk - w.shape[2]) // 2
            wm[off:off + w.shape[0], :, o:o + w.shape[2], o:o + w.shape[2]] = w
            b = _param(gm, p.args[2])
            if b is not None:
                bm[off:off + w.shape[0]] = b
            off += w.shape[0]
        name = f"_r3_merged{n}"
        mod = nn.Module()
        mod.register_buffer("weight", wm.contiguous(memory_format=torch.channels_last))
        mod.register_buffer("bias", bm)
        gm.add_submodule(name, mod)
        two = len(ps) == 2 and ws[0].shape[2] == 1 and ws[1].shape[2] == kk
        with g.inserting_before(node):
            wn = g.get_attr(name + ".weight")
            bn = g.get_attr(name + ".bias")
            if two:   # runtime choice merged / separate by problem size (clfr3.cat2)
                new = g.call_function(torch.ops.clfr3.cat2, (x0, wn, bn, ps[0].args[1], ps[0].args[2], ps[1].args[1],
                                                             ps[1].args[2], ps[0].args[6], MERGE_M))
            else:
                new = g.call_function(torch.ops.clf.conv_fused, (x0, wn, bn, None, [kk // 2] * 2, list(ps[0].args[5]),
                                                                 ps[0].args[6]))
        node.replace_all_uses_with(new)
        g.erase_node(node)
        for p in ps:
            if not p.users:
                g.erase_node(p)
        n += 1
    g.eliminate_dead_code()
    g.lint()
    gm.recompile()
    return n


def to_pw(gm):
    """clf.conv_fused / conv_fused_into 1x1 stride-1 convs (act none / relu, no residual) -> clfr3.pw / pw_into: cublasLt
    GEMM with fused bias + relu for narrow GEMMs above a size threshold, CUTLASS otherwise (decided in C++ per call)."""
    if os.environ.get("CLF_PW", "1") != "1":
        return 0
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target not in (torch.ops.clf.conv_fused, torch.ops.clf.conv_fused.default,
                                                              torch.ops.clf.conv_fused_into.default):
            continue
        a = node.args
        w = _param(gm, a[1]) if isinstance(a[1], torch.fx.Node) and a[1].op == "get_attr" else None
        if w is None or w.shape[2:] != (1, 1) or list(a[4]) != [0, 0] or list(a[5]) != [1, 1] or a[3] is not None \
                or a[6] not in (0, 1) or min(w.shape[0], w.shape[1]) > 256:
            continue
        load()
        with g.inserting_before(node):
            if node.target is torch.ops.clf.conv_fused_into.default:
                new = g.call_function(torch.ops.clfr3.pw_into.default, (a[0], a[1], a[2], None, a[6], a[7]))
            else:
                new = g.call_function(torch.ops.clfr3.pw, (a[0], a[1], a[2], None, a[6]))
        node.replace_all_uses_with(new)
        g.erase_node(node)
        n += 1
    g.lint()
    gm.recompile()
    return n


def fuse_regnet(gm):
    """clfmb.gconv -> clfr3.gconv2 (round-3 kernel); gconv -> AdaptiveAvgPool2d(1) + mul(sigmoid(.), gconv) (SE) ->
    gconv2_pool (fused pool sums); sigmoid(s) * x -> clfr3.se_scale. Returns (n_gconv, n_pool, n_se)."""
    import operator
    import torch.nn as nn
    if os.environ.get("CLF_GC2", "1") != "1":
        return 0, 0, 0
    from .mbconv import load as lm
    lm()
    load()
    mods = dict(gm.named_modules())
    g = gm.graph
    ng = npool = nse = 0

    def is_sig(n):
        return (n.op == "call_module" and isinstance(mods.get(n.target), nn.Sigmoid)) or \
            (n.op == "call_function" and n.target in (torch.sigmoid, torch.nn.functional.sigmoid))

    def is_gap(n):
        if n.op == "call_module" and isinstance(mods.get(n.target), nn.AdaptiveAvgPool2d):
            os_ = mods[n.target].output_size
            return os_ in (1, (1, 1))
        return False
    for node in list(g.nodes):        # SE scale first (pattern needs the sigmoid node)
        if node.op == "call_function" and node.target in (operator.mul, torch.mul) and len(node.args) == 2:
            a, b = node.args
            s_, x_ = (a, b) if isinstance(a, torch.fx.Node) and is_sig(a) else (b, a) if isinstance(b, torch.fx.Node) and is_sig(b) else (None, None)
            if s_ is None or len(s_.users) != 1 or not isinstance(x_, torch.fx.Node):
                continue
            with g.inserting_before(node):
                new = g.call_function(torch.ops.clfr3.se_scale, (x_, s_.args[0]))
            node.replace_all_uses_with(new)
            g.erase_node(node)
            g.erase_node(s_)
            nse += 1
    for node in list(g.nodes):
        if node.op != "call_function" or node.target is not torch.ops.clfmb.gconv:
            continue
        args = node.args
        gaps = [u for u in node.users if is_gap(u)]
        with g.inserting_before(node):
            if len(gaps) == 1 and os.environ.get("CLF_GC2_POOL", "0") == "1":   # fused SE pool: no gain at b64 (A/B)
                t = g.call_function(torch.ops.clfr3.gconv2_pool, tuple(args))
                y = g.call_function(operator.getitem, (t, 0))
                mean = g.call_function(operator.getitem, (t, 1))
                gaps[0].replace_all_uses_with(mean)
                g.erase_node(gaps[0])
                npool += 1
            else:
                y = g.call_function(torch.ops.clfr3.gconv2, tuple(args))
        node.replace_all_uses_with(y)
        g.erase_node(node)
        ng += 1
    g.eliminate_dead_code()
    g.lint()
    gm.recompile()
    return ng, npool, nse


def to_conv_tc(gm, max_c=None):
    """clf.conv_fused / conv_fused_into with a KxK kernel (K > 1), no residual, act <= 4, C % 16 == 0, C <= TC_MAX_C (48),
    K % 16 == 0  ->  clfr3.conv_tc / conv_tc_into (gconv2 kernel: 1.1-2x faster than CUTLASS for small C, convtc_test.py)."""
    if os.environ.get("CLF_TC", "1") != "1":
        return 0
    max_c = max_c or int(os.environ.get("CLF_TC_MAX_C", "48"))
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if node.op != "call_function" or node.target not in (torch.ops.clf.conv_fused, torch.ops.clf.conv_fused.default,
                                                              torch.ops.clf.conv_fused_into.default):
            continue
        a = node.args
        w = _param(gm, a[1]) if isinstance(a[1], torch.fx.Node) and a[1].op == "get_attr" else None
        if w is None or w.shape[2:] == (1, 1) or a[3] is not None or a[6] > 4 or w.shape[1] % 16 or w.shape[1] > max_c \
                or w.shape[0] % 16:
            continue
        load()
        with g.inserting_before(node):
            if node.target is torch.ops.clf.conv_fused_into.default:
                new = g.call_function(torch.ops.clfr3.conv_tc_into.default, (a[0], a[1], a[2], list(a[5]), list(a[4]), a[6], a[7]))
            else:
                new = g.call_function(torch.ops.clfr3.conv_tc, (a[0], a[1], a[2], list(a[5]), list(a[4]), a[6]))
        node.replace_all_uses_with(new)
        g.erase_node(node)
        n += 1
    g.lint()
    gm.recompile()
    return n


def to_dw2(gm):
    """clf.dwconv -> clfr3.dwconv2 (half2 accumulation)."""
    if os.environ.get("CLF_DW2", "1") != "1":
        return 0
    maxk = int(os.environ.get("CLF_DW2_MAXK", "7"))
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if node.op == "call_function" and node.target in (torch.ops.clf.dwconv, torch.ops.clf.dwconv.default):
            w = node.args[1]
            if w.op == "get_attr":
                t = gm
                for part in w.target.split("."):
                    t = getattr(t, part)
                if max(t.shape[:2]) > maxk:   # weight [R, S, C]
                    continue
            load()
            node.target = torch.ops.clfr3.dwconv2
            n += 1
    gm.recompile()
    return n

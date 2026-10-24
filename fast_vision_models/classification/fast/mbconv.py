"""Fused inverted-residual blocks for the max path (csrc/mbconv.cu): one kernel per MBConv block, expanded and depthwise
tensors stay in shared memory. Compiled once per source hash (problem size = runtime arguments).

`fuse_mbconv(gm)` runs after fused.fuse_all and rewrites
    [conv_fused 1x1 (expand, act1)] -> dwconv (act2) -> conv_fused 1x1 (project, [+res], act3)
into clf_mb.mbconv(...) when every intermediate has a single consumer (SE blocks do not match: their project conv reads a
mul, not the dwconv). Weights are zero-padded (Cin -> 16, expanded channels -> 32) into buffers of a new submodule.
"""
import os

import torch

from ..._build import jit_load
import torch.nn as nn

from .fused import HERE

SRC = os.path.join(HERE, "csrc", "mbconv.cu")
_loaded = [False]
KC = 32


def load():
    if _loaded[0]:
        return True
    from .fused import load as _lf

    _lf()
    jit_load("clf_mb", [SRC], extra_cflags=["-lineinfo"])
    _loaded[0] = True
    return True


def _rup(v, m):
    return (v + m - 1) // m * m


def _attr(gm, node):
    if node is None:
        return None
    assert node.op == "get_attr", node
    mod, _, name = node.target.rpartition(".")
    return getattr(gm.get_submodule(mod) if mod else gm, name)


def _is_pw(gm, node):
    """clf.conv_fused 1x1 / stride 1 / pad 0"""
    if not (isinstance(node, torch.fx.Node) and node.op == "call_function" and node.target is torch.ops.clf.conv_fused):
        return False
    w = _attr(gm, node.args[1])
    return w.shape[2:] == (1, 1) and list(node.args[4]) == [0, 0] and list(node.args[5]) == [1, 1]


class _MB(nn.Module):
    pass


def fuse_mbconv(gm):
    load()
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if not _is_pw(gm, node):
            continue
        d = node.args[0]
        if not (isinstance(d, torch.fx.Node) and d.op == "call_function" and d.target is torch.ops.clf.dwconv
                and len(d.users) == 1):
            continue
        e = d.args[0]
        expand = _is_pw(gm, e) and len(e.users) == 1 and e.args[3] is None
        w2 = _attr(gm, node.args[1])
        b2 = _attr(gm, node.args[2])
        wd = _attr(gm, d.args[1])                  # [R, S, E]
        bd = _attr(gm, d.args[2])
        R, S, E = wd.shape
        cout = w2.shape[0]
        if expand:
            w1 = _attr(gm, e.args[1])
            b1 = _attr(gm, e.args[2])
            cin = w1.shape[1]
            x = e.args[0]
        else:
            cin = E
            x = e
        if cin % 8 or cout % 8 or E % 8 or bd is None or (expand and b1 is None):
            continue
        if not expand and cout > 2 * cin:    # dw -> wide 1x1 (FastViT fc1): the dw would be recomputed per output chunk
            continue
        ep = _rup(E, KC)
        dev, dt = w2.device, w2.dtype
        m = _MB()
        if expand:
            cinp = _rup(cin, 16)
            W1 = torch.zeros(ep, cinp, device=dev, dtype=dt)
            W1[:E, :cin] = w1.reshape(E, cin)
            B1 = torch.zeros(ep, device=dev, dtype=dt)
            B1[:E] = b1
            m.register_buffer("w1", W1)
            m.register_buffer("b1", B1)
        WD = torch.zeros(R, S, ep, device=dev, dtype=dt)
        WD[:, :, :E] = wd
        BD = torch.zeros(ep, device=dev, dtype=dt)
        BD[:E] = bd
        W2 = torch.zeros(cout, ep, device=dev, dtype=dt)
        W2[:, :E] = w2.reshape(cout, E)
        m.register_buffer("wd", WD)
        m.register_buffer("bd", BD)
        m.register_buffer("w2", W2)
        if b2 is not None:
            m.register_buffer("b2", b2.detach().clone())
        name = f"mb{n}"
        gm.add_submodule(name, m)
        with g.inserting_before(node):
            ga = lambda k: g.get_attr(f"{name}.{k}")  # noqa: E731
            args = (x, ga("w1") if expand else None, ga("b1") if expand else None, ga("wd"), ga("bd"), ga("w2"),
                    ga("b2") if b2 is not None else None, node.args[3], list(d.args[3]), list(d.args[4]),
                    [int(e.args[6]) if expand else 0, int(d.args[5]), int(node.args[6])], 0)
            new = g.call_function(torch.ops.clfmb.mbconv, args)
        node.replace_all_uses_with(new)
        g.erase_node(node)
        g.erase_node(d)
        if expand:
            g.erase_node(e)
        n += 1
    g.eliminate_dead_code()
    g.lint()
    gm.recompile()
    return n


def fuse_gconv(gm):
    """grouped (not depthwise) Conv2d [-> act] with equal in/out group width % 16 == 0  ==>  clfmb.gconv"""
    from .fused import _act_of
    load()
    mods = dict(gm.named_modules())
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        m = mods.get(node.target) if node.op == "call_module" else None
        if not (type(m) is nn.Conv2d and 1 < m.groups < m.in_channels and m.in_channels == m.out_channels
                and (m.in_channels // m.groups) % 16 == 0 and m.dilation == (1, 1) and m.padding_mode == "zeros"
                and not isinstance(m.padding, str) and m.weight.dtype == torch.half):
            continue
        m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
        last, act = node, 0
        us = list(node.users)
        if len(us) == 1:
            a = _act_of(us[0], mods)
            if a is not None:
                act, last = a, us[0]
        with g.inserting_before(last.next):
            w = g.get_attr(node.target + ".weight")
            b = g.get_attr(node.target + ".bias") if m.bias is not None else None
            new = g.call_function(torch.ops.clfmb.gconv, (node.args[0], w, b, list(m.stride), list(m.padding), m.groups, act))
        last.replace_all_uses_with(new)
        new.args = (node.args[0], w, b, list(m.stride), list(m.padding), m.groups, act)
        n += 1
    g.eliminate_dead_code()
    g.lint()
    gm.recompile()
    return n


def fuse_dwffn(gm):
    """dwconv (act2) -> conv_fused 1x1 (fc1, act1) -> conv_fused 1x1 (fc2, [+res], act3)  ==>  clfmb.mbconv_dwffn
    (FastViT ConvMlp / RepMixer-style blocks: hidden tensor stays in shared memory)."""
    load()
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        if not _is_pw(gm, node):
            continue
        p1 = node.args[0]
        if not (_is_pw(gm, p1) and len(p1.users) == 1 and p1.args[3] is None):
            continue
        d = p1.args[0]
        if not (isinstance(d, torch.fx.Node) and d.op == "call_function" and d.target is torch.ops.clf.dwconv
                and len(d.users) == 1):
            continue
        wd, bd = _attr(gm, d.args[1]), _attr(gm, d.args[2])
        w1, b1 = _attr(gm, p1.args[1]), _attr(gm, p1.args[2])
        w2, b2 = _attr(gm, node.args[1]), _attr(gm, node.args[2])
        R, S, C = wd.shape
        E, cout = w1.shape[0], w2.shape[0]
        if C % 8 or cout % 8 or E % 8 or b1 is None or C > 128 or cout > 128:   # smem plan: Xs + 2 weight chunks
            continue
        cinp, ep = _rup(C, 16), _rup(E, KC)
        dev, dt = w2.device, w2.dtype
        m = _MB()
        WD = torch.zeros(R, S, cinp, device=dev, dtype=dt); WD[:, :, :C] = wd
        BD = torch.zeros(cinp, device=dev, dtype=dt)
        if bd is not None:
            BD[:C] = bd
        W1 = torch.zeros(ep, cinp, device=dev, dtype=dt); W1[:E, :C] = w1.reshape(E, C)
        B1 = torch.zeros(ep, device=dev, dtype=dt); B1[:E] = b1
        W2 = torch.zeros(cout, ep, device=dev, dtype=dt); W2[:, :E] = w2.reshape(cout, E)
        for k, v in (("wd", WD), ("bd", BD), ("w1", W1), ("b1", B1), ("w2", W2)):
            m.register_buffer(k, v)
        if b2 is not None:
            m.register_buffer("b2", b2.detach().clone())
        name = f"ffn{n}"
        gm.add_submodule(name, m)
        with g.inserting_before(node):
            ga = lambda k: g.get_attr(f"{name}.{k}")  # noqa: E731
            args = (d.args[0], ga("wd"), ga("bd"), ga("w1"), ga("b1"), ga("w2"), ga("b2") if b2 is not None else None,
                    node.args[3], list(d.args[3]), list(d.args[4]), [int(p1.args[6]), int(d.args[5]), int(node.args[6])], 0)
            new = g.call_function(torch.ops.clfmb.mbconv_dwffn, args)
        node.replace_all_uses_with(new)
        g.erase_node(node)
        g.erase_node(p1)
        g.erase_node(d)
        n += 1
    g.eliminate_dead_code()
    g.lint()
    gm.recompile()
    return n


def fuse_dwmult(gm):
    """Conv2d with groups == C_in and C_out = mult * C_in (mult > 1) [-> act]  ==>  clfmb.dwmult (FastViT patch embed)"""
    from .fused import _act_of
    load()
    mods = dict(gm.named_modules())
    g = gm.graph
    n = 0
    for node in list(g.nodes):
        m = mods.get(node.target) if node.op == "call_module" else None
        if not (type(m) is nn.Conv2d and m.groups == m.in_channels and m.out_channels % m.in_channels == 0
                and m.out_channels > m.in_channels and m.out_channels % 8 == 0 and m.dilation == (1, 1)
                and m.padding_mode == "zeros" and not isinstance(m.padding, str) and m.weight.dtype == torch.half):
            continue
        m.register_buffer("w_rsk", m.weight.data[:, 0].permute(1, 2, 0).contiguous())
        last, act = node, 0
        us = list(node.users)
        if len(us) == 1:
            a = _act_of(us[0], mods)
            if a is not None:
                act, last = a, us[0]
        with g.inserting_before(last.next):
            w = g.get_attr(node.target + ".w_rsk")
            b = g.get_attr(node.target + ".bias") if m.bias is not None else None
            new = g.call_function(torch.ops.clfmb.dwmult, (node.args[0], w, b, list(m.stride), list(m.padding), act))
        last.replace_all_uses_with(new)
        new.args = (node.args[0], w, b, list(m.stride), list(m.padding), act)
        n += 1
    g.eliminate_dead_code()
    g.lint()
    gm.recompile()
    return n


def fuse_xdw(gm, max_cin=64):
    """conv_fused 1x1 (expand, act1, no res) -> dwconv (act2)  ==>  clfmb.xdw (expanded tensor only in smem; the project
    conv stays a CUTLASS conv with fused residual). Skipped for wide inputs (Cin > max_cin: halo tile too large)."""
    load()
    g = gm.graph
    n = 0
    for d in list(g.nodes):
        if not (d.op == "call_function" and d.target is torch.ops.clf.dwconv):
            continue
        e = d.args[0]
        if not (_is_pw(gm, e) and len(e.users) == 1 and e.args[3] is None):
            continue
        w1, b1 = _attr(gm, e.args[1]), _attr(gm, e.args[2])
        wd, bd = _attr(gm, d.args[1]), _attr(gm, d.args[2])
        R, S, E = wd.shape
        cin = w1.shape[1]
        if cin % 8 or E % 8 or cin > max_cin or b1 is None or bd is None:
            continue
        if int(e.args[6]) in (3, 5, 6) or int(d.args[5]) in (3, 5, 6):   # silu / gelu: fp32 epilogues, measured slower
            continue
        cinp, ep = _rup(cin, 16), _rup(E, KC)
        dev, dt = w1.device, w1.dtype
        m = _MB()
        W1 = torch.zeros(ep, cinp, device=dev, dtype=dt); W1[:E, :cin] = w1.reshape(E, cin)
        B1 = torch.zeros(ep, device=dev, dtype=dt); B1[:E] = b1
        WD = torch.zeros(R, S, ep, device=dev, dtype=dt); WD[:, :, :E] = wd
        BD = torch.zeros(ep, device=dev, dtype=dt); BD[:E] = bd
        for k, v in (("w1", W1), ("b1", B1), ("wd", WD), ("bd", BD)):
            m.register_buffer(k, v)
        name = f"xdw{n}"
        gm.add_submodule(name, m)
        with g.inserting_before(d):
            ga = lambda k: g.get_attr(f"{name}.{k}")  # noqa: E731
            new = g.call_function(torch.ops.clfmb.xdw, (e.args[0], ga("w1"), ga("b1"), ga("wd"), ga("bd"), E,
                                                         list(d.args[3]), list(d.args[4]), [int(e.args[6]), int(d.args[5])], 0))
        d.replace_all_uses_with(new)
        g.erase_node(d)
        g.erase_node(e)
        n += 1
    g.eliminate_dead_code()
    g.lint()
    gm.recompile()
    return n

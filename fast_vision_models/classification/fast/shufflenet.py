"""ShuffleNet V2 max path: one fused kernel per unit (clfmb.mbconv_into), channel split + shuffle absorbed into the
physical channel layout and the weights.

Every unit writes a fresh buffer with the physical layout [P1 | P2 | B2 | B1] (each part Q = ceil8(C/4) channels, zero
padded): P1 + P2 = the logical channels the NEXT unit passes through (its x1), B2 + B1 = the ones its branch reads.
  b2 (branch output) rows -> P2 | B2 = one contiguous 2Q slice (weights permuted, pad rows zero)
  x1 (passthrough)       -> P1, B1 by a gather inside the same kernel (stride-1 units)
  b1 (downsample branch) -> P1, B1 via an output column map
Every physical channel is tracked as a logical channel index (or -1 = zero pad); the first conv of each consumer
(branch 1x1, downsample dw, conv5) is built with permuted input columns, so nothing is ever shuffled at run time.
"""
import os

import torch
import torch.nn as nn


def _fold(conv, bn):
    sc = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
    w = conv.weight.double() * sc.view(-1, 1, 1, 1)
    b = bn.bias.double() - bn.running_mean.double() * sc
    if conv.bias is not None:
        b = b + conv.bias.double() * sc
    return w, b


def _r(v, m):
    return (v + m - 1) // m * m


def _pw(w, b, rows, cols, rows_pad, cols_pad):
    """dense 1x1 weight [rows_pad, cols_pad] from logical w [O, I]: rows[i] / cols[j] = logical index or -1."""
    W = torch.zeros(rows_pad, cols_pad, dtype=torch.float64)
    B = torch.zeros(rows_pad, dtype=torch.float64)
    w = w.reshape(w.shape[0], w.shape[1])
    r, c = torch.tensor(rows), torch.tensor(cols)
    ri, ci = (r >= 0).nonzero().flatten(), (c >= 0).nonzero().flatten()
    W[ri[:, None], ci[None, :]] = w[r[ri][:, None], c[ci][None, :]]
    B[ri] = b[r[ri]]
    return W, B


def _dw(w, b, chans, pad):
    """depthwise weight [R, S, pad] + bias from logical w [C, 1, R, S]; chans[i] = logical channel or -1."""
    R, S = w.shape[2:]
    WD = torch.zeros(R, S, pad, dtype=torch.float64)
    BD = torch.zeros(pad, dtype=torch.float64)
    c = torch.tensor(chans)
    ci = (c >= 0).nonzero().flatten()
    WD[:, :, ci] = w[c[ci], 0].permute(1, 2, 0)
    BD[ci] = b[c[ci]]
    return WD, BD


class _Unit(nn.Module):
    pass


class FastShuffleNetV2(nn.Module):
    """big=False: one fused clfmb kernel per unit (best at small batch); big=True: same layout / weights, standard kernels
    per unit (CUTLASS 1x1 reading the channel slice in place, dwconv, CUTLASS 1x1 into the output slice, gather)."""

    def __init__(self, tv, big=False):
        super().__init__()
        self.raw_input = True          # stem reads the fp32 NCHW input directly (clfmb.stem_direct)
        self.big = big
        dt = torch.float16
        c1, bn1 = tv.conv1[0], tv.conv1[1]
        w, b = _fold(c1, bn1)
        ws = torch.zeros(w.shape[0], 8, 3, 3, dtype=torch.float64)
        ws[:, :3] = w
        self.register_buffer("stem_w", ws.to(dt).contiguous(memory_format=torch.channels_last))
        self.register_buffer("stem_wt", w.to(dt).permute(2, 3, 1, 0).contiguous())   # [R, S, 3, K] for stem_direct
        self.register_buffer("stem_b", b.to(dt))
        pin = list(range(w.shape[0]))          # physical -> logical channel of the current buffer
        self.units = nn.ModuleList()
        self.meta = []                          # per unit: (downsample, Q, cin_phys)
        for stage in (tv.stage2, tv.stage3, tv.stage4):
            for blk in stage:
                u = _Unit()
                b2 = blk.branch2
                w21, b21 = _fold(b2[0], b2[1])
                wd2, bd2 = _fold(b2[3], b2[4])
                w23, b23 = _fold(b2[5], b2[6])
                bf = w23.shape[0]
                h = bf // 2
                Q = _r(h, 8)
                cphys = len(pin)
                ds = blk.stride > 1
                ep = _r(bf, 32)
                # branch2 output rows: P2 slots (b2[i], i < h) then B2 slots (b2[h + i])
                rows2 = [i if i < h else -1 for i in range(Q)] + [h + i if i < h else -1 for i in range(Q)]
                if ds:
                    cols = list(pin)                                   # branch2 reads the whole input
                    w1cols = cols
                    cin_off, cin = 0, cphys
                else:
                    half = cphys // 2
                    clog = sum(1 for p in pin if p >= 0)               # logical C of the input
                    cols = [p - clog // 2 if p >= 0 else -1 for p in pin[half:]]
                    assert all(c < 0 or c >= 0 and c < clog // 2 for c in cols)
                    w1cols = cols
                    cin_off, cin = half, cphys - half
                cinp = _r(cin, 16)
                W1, B1 = _pw(w21, b21, list(range(bf)), w1cols, ep, cinp)
                WD, BD = _dw(wd2, bd2, list(range(bf)), ep)
                W3, B3 = _pw(w23, b23, rows2, list(range(bf)), 2 * Q, ep)
                for k, v in (("w1", W1), ("b1", B1), ("wd", WD), ("bd", BD), ("w2", W3), ("b2", B3)):
                    u.register_buffer(k, v.to(dt))
                u.register_buffer("colmap", torch.tensor([Q + 8 * j for j in range(2 * Q // 8)], dtype=torch.int32))
                if ds:
                    b1m = blk.branch1
                    wd1, bd1 = _fold(b1m[0], b1m[1])
                    w12, b12 = _fold(b1m[2], b1m[3])
                    ep1 = _r(cphys, 32)
                    WD1, BD1 = _dw(wd1, bd1, list(pin), ep1)
                    rows1 = [i if i < h else -1 for i in range(Q)] + [h + i if i < h else -1 for i in range(Q)]
                    W12, B12 = _pw(w12, b12, rows1, list(pin) + [-1] * (ep1 - cphys), 2 * Q, ep1)
                    for k, v in (("wd1", WD1), ("bd1", BD1), ("w12", W12), ("b12", B12)):
                        u.register_buffer(k, v.to(dt))
                    u.register_buffer("colmap1", torch.tensor([8 * j for j in range(Q // 8)] +
                                                              [3 * Q + 8 * j for j in range(Q // 8)], dtype=torch.int32))
                else:
                    # passthrough x1 = input logical 0 .. C/2-1 -> P1 (x1[i], i < h) and B1 (x1[h + i])
                    where = {p: k for k, p in enumerate(pin) if p >= 0}
                    gsrc = [where[i] if i < h else -1 for i in range(Q)] + [where[h + i] if i < h else -1 for i in range(Q)]
                    gdst = [8 * j for j in range(Q // 8)] + [3 * Q + 8 * j for j in range(Q // 8)]
                    u.register_buffer("gsrc", torch.tensor(gsrc, dtype=torch.int32))
                    u.register_buffer("gdst", torch.tensor(gdst, dtype=torch.int32))
                if big:   # standard-kernel weights: [K, C, 1, 1] / [R, S, C] views of the same values
                    cl = torch.channels_last
                    u.register_buffer("k1", u.w1[:, :cin].reshape(ep, cin, 1, 1).contiguous(memory_format=cl))
                    u.register_buffer("k3", u.w2.reshape(2 * Q, ep, 1, 1).contiguous(memory_format=cl))
                    if ds:
                        u.register_buffer("kd1", u.wd1[:, :, :cphys].contiguous())
                        u.register_buffer("bd1c", u.bd1[:cphys].contiguous())
                        w12 = u.w12[:, :cphys]
                        u.register_buffer("k12a", w12[:Q].reshape(Q, cphys, 1, 1).contiguous(memory_format=cl))
                        u.register_buffer("k12b", w12[Q:].reshape(Q, cphys, 1, 1).contiguous(memory_format=cl))
                        u.register_buffer("b12a", u.b12[:Q].contiguous())
                        u.register_buffer("b12b", u.b12[Q:].contiguous())
                self.units.append(u)
                self.meta.append((ds, Q, cin_off, cin))
                # output physical map: P1 -> 2i, P2 -> 2i+1, B2 -> 2(h+i)+1, B1 -> 2(h+i)
                pin = ([2 * i if i < h else -1 for i in range(Q)] + [2 * i + 1 if i < h else -1 for i in range(Q)] +
                       [2 * (h + i) + 1 if i < h else -1 for i in range(Q)] + [2 * (h + i) if i < h else -1 for i in range(Q)])
        w5, b5 = _fold(tv.conv5[0], tv.conv5[1])
        W5, B5 = _pw(w5, b5, list(range(w5.shape[0])), pin, w5.shape[0], len(pin))
        self.register_buffer("w5", W5.to(dt).view(W5.shape[0], -1, 1, 1).contiguous(memory_format=torch.channels_last))
        self.register_buffer("b5", B5.to(dt))
        self.fc = nn.Linear(tv.fc.in_features, tv.fc.out_features).to(dt)
        self.fc.load_state_dict({k: v.to(dt) for k, v in tv.fc.state_dict().items()})

    def forward(self, x):
        if self.big:
            return self.forward_big(x)
        ops = torch.ops
        x = _stem(self, x)
        for u, (ds, Q, cin_off, cin) in zip(self.units, self.meta):
            n, _, hh, ww = x.shape
            if ds:
                ho, wo = (hh + 1) // 2, (ww + 1) // 2
                out = torch.empty((n, 4 * Q, ho, wo), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
                ops.clfmb.mbconv_into(x, None, None, u.wd1, u.bd1, u.w12, u.b12, [2, 2], [1, 1], [0, 0, 1], 0, cin, out,
                                      u.colmap1, None, None)
                ops.clfmb.mbconv_into(x, u.w1, u.b1, u.wd, u.bd, u.w2, u.b2, [2, 2], [1, 1], [1, 0, 1], cin_off, cin, out,
                                      u.colmap, None, None)
            else:
                out = torch.empty((n, 4 * Q, hh, ww), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
                ops.clfmb.mbconv_into(x, u.w1, u.b1, u.wd, u.bd, u.w2, u.b2, [1, 1], [1, 1], [1, 0, 1], cin_off, cin, out,
                                      u.colmap, u.gsrc, u.gdst)
            x = out
        x = ops.clf.conv_fused(x, self.w5, self.b5, None, [0, 0], [1, 1], 1)
        return self.fc(x.mean((2, 3)))


def _stem(self, x):
    if os.environ.get("CLF_STEMPOOL", "1") == "1":   # round 3: conv + relu + max pool in one kernel
        return torch.ops.clfr3.stem_pool(x, self.stem_wt, self.stem_b, [2, 2], [1, 1], 1, 3, 2, 1, False)
    x = torch.ops.clfmb.stem_direct(x, self.stem_wt, self.stem_b, [2, 2], [1, 1], 1)
    return torch.ops.clf.maxpool(x, [3, 3], [2, 2], [1, 1], False)



def _forward_big(self, x):
    ops = torch.ops
    x = _stem(self, x)
    for u, (ds, Q, cin_off, cin) in zip(self.units, self.meta):
        n, _, hh, ww = x.shape
        if ds:
            ho, wo = (hh + 1) // 2, (ww + 1) // 2
            out = torch.empty((n, 4 * Q, ho, wo), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
            d1 = ops.clf.dwconv(x, u.kd1, u.bd1c, [2, 2], [1, 1], 0)
            ops.clf.conv_fused_into(d1, u.k12a, u.b12a, None, [0, 0], [1, 1], 1, out[:, :Q])
            ops.clf.conv_fused_into(d1, u.k12b, u.b12b, None, [0, 0], [1, 1], 1, out[:, 3 * Q:])
            h = ops.clf.conv_fused(x, u.k1, u.b1, None, [0, 0], [1, 1], 1)
            h = ops.clf.dwconv(h, u.wd, u.bd, [2, 2], [1, 1], 0)
        else:
            out = torch.empty((n, 4 * Q, hh, ww), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
            h = ops.clf.conv_fused(x[:, cin_off:cin_off + cin], u.k1, u.b1, None, [0, 0], [1, 1], 1)
            h = ops.clf.dwconv(h, u.wd, u.bd, [1, 1], [1, 1], 0)
            ops.clfmb.chan_gather(x, out, u.gsrc, u.gdst)
        ops.clf.conv_fused_into(h, u.k3, u.b2, None, [0, 0], [1, 1], 1, out[:, Q:3 * Q])
        x = out
    x = ops.clf.conv_fused(x, self.w5, self.b5, None, [0, 0], [1, 1], 1)
    return self.fc(x.mean((2, 3)))


FastShuffleNetV2.forward_big = _forward_big


def build(tv):
    from .fused import load as lf
    from .mbconv import load as lm
    lf()
    lm()
    from .r3 import load as l3
    l3()
    return FastShuffleNetV2(tv.eval().cpu()), FastShuffleNetV2(tv.eval().cpu(), big=True)

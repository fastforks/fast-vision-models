// exp 008 round 2: fused inverted-residual (MBConv) block in ONE kernel, size-independent (problem size = runtime args,
// compiled once, no per-size tuning):
//     y = act3( W2 . act2( dw_RxS( act1( W1 . x + b1 ) ) + bd ) + b2 (+ res) )
// x, res, y: NHWC fp16 (channels-last NCHW); W1 [Ep][Cin_p] (optional: no expand -> the dw reads x), Wd [R][S][Ep],
// W2 [Cout][Ep]; Ep = expanded channels zero-padded to a multiple of KC (padded channels contribute exactly 0).
// The expanded and depthwise tensors live only in shared memory: one CTA = one TH x TW output tile of one image x one
// chunk of output channels; the expand GEMM is computed for the in-image part of the input halo (compact rows), the
// depthwise conv reads it from smem, the project GEMM accumulates in registers over KC-channel chunks of Ep.
// Tensor cores: mma.sync m16n8k16 fp16 accumulate (as the CUTLASS convs), ldmatrix from padded smem rows (row stride =
// 16 * odd bytes -> conflict-free). Depthwise accumulates in fp32.
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

namespace {

constexpr int KC = 32;          // expanded channels per chunk
constexpr int NWARP = 8;
constexpr int NTH = NWARP * 32;
constexpr int SKC = KC + 8;     // smem row stride (halfs) of KC-wide buffers
constexpr int NTW_MAX = 16;     // n8 tiles per warp in the project GEMM (32 fp16x2 accumulator registers)

struct Args {
    const __half* x; const __half* w1; const __half* b1; const __half* wd; const __half* bd; const __half* w2;
    const __half* b2; const __half* res; __half* y;
    const __half* xb;       // input buffer base (channel 0; x = xb + cin_off)
    const int* colmap;      // [Cout/8] physical output channel of each 8-column block (null: identity)
    const int* gsrc;        // [G] passthrough gather: y[p, gdst-block j] = x[p, gsrc[j]] (-1: zero), stride-1 blocks only
    const int* gdst;        // [G/8] physical output channel of each gathered 8-block
    int G, ldx, ldy, gproj, dwf, ldwd, dbg;   // dwf: dw first (dw on x -> smem, then FFN: expand GEMM -> project GEMM)   // gproj: block-diagonal project (groups of KC channels, Cout == Ep)        // pixel strides of x (input buffer) and y
    int N, H, W, Cin, Cin_p, Ep, Cout, R, S, sh, sw, ph, pw, P, Q;
    int TH, TW, HH, HW, tp_n, tq_n, MT, cout_ch, nip;   // nip: max compact halo rows (padded to 16)
    int act1, act2, act3;
};

__device__ __forceinline__ float actf(float v, int act) {
    switch (act) {
        case 1: return fmaxf(v, 0.f);
        case 2: return fminf(fmaxf(v, 0.f), 6.f);
        case 3: return __fdividef(v, 1.f + __expf(-v));
        case 4: return v * fminf(fmaxf(v + 3.f, 0.f), 6.f) * (1.f / 6.f);
        case 5: return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
        case 6: return 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
    return v;
}

__device__ __forceinline__ __half2 acth2(__half2 v, int act) {   // relu / relu6 / none in half2, others via fp32
    if (act == 0) return v;
    if (act == 1) return __hmax2(v, __float2half2_rn(0.f));
    if (act == 2) return __hmin2(__hmax2(v, __float2half2_rn(0.f)), __float2half2_rn(6.f));
    float2 f = __half22float2(v);
    return __floats2half2_rn(actf(f.x, act), actf(f.y, act));
}

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp16(void* dst, const void* src, bool pred) {
    unsigned d = smem_u32(dst);
    int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm2(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(unsigned* d, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned h2u(__half2 h) { return *reinterpret_cast<unsigned*>(&h); }
__device__ __forceinline__ __half2 u2h(unsigned u) { return *reinterpret_cast<__half2*>(&u); }

__global__ void __launch_bounds__(NTH) mbconv_kernel(const Args a) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const bool expand = a.w1 != nullptr;
    // ---- tile coordinates
    int t = blockIdx.x;
    const int tq = t % a.tq_n; t /= a.tq_n;
    const int tp = t % a.tp_n; const int n = t / a.tp_n;
    const int oy0 = tp * a.TH, ox0 = tq * a.TW;
    const int th = min(a.TH, a.P - oy0), tw = min(a.TW, a.Q - ox0);
    const int TP = th * tw;                                   // valid output pixels of this tile
    const int iy0 = oy0 * a.sh - a.ph, ix0 = ox0 * a.sw - a.pw;
    const int y0 = max(iy0, 0), x0 = max(ix0, 0);
    const int y1 = min(iy0 + (th - 1) * a.sh + a.R, a.H), x1 = min(ix0 + (tw - 1) * a.sw + a.S, a.W);
    const int cw = x1 - x0;
    const int NI = a.dwf ? a.MT * 16 : (y1 - y0) * cw;        // GEMM1 rows: compact in-image halo rows (dwf: tile pixels)
    const int NIp = (NI + 15) & ~15;
    const int co0 = blockIdx.y * a.cout_ch;
    const int cch = min(a.cout_ch, a.Cout - co0);             // output channels of this CTA (multiple of 8)
    const int sx = a.Cin_p + 8;
    // ---- shared memory carve
    __half* Xs = reinterpret_cast<__half*>(smem);                              // [nip][sx]  (expand only)
    __half* Ec = Xs + (expand ? a.nip * sx : 0);                                // [nip][SKC] (x2 without expand)
    __half* Dc = Ec + a.nip * SKC * (expand ? 1 : 2);                           // [MT*16][SKC]
    __half* W1s = Dc + a.MT * 16 * SKC;                                         // [2][KC][sx] (expand only)
    __half* W2s = W1s + (expand ? 2 * KC * sx : 0);                             // [2][cout_ch][SKC]
    const int nchunk = a.Ep / KC;
    const long long img = (long long)n * a.H * a.W;

    auto load_w = [&](int c, int buf) {
        if (expand) {
            const __half* src = a.w1 + (long long)c * KC * a.Cin_p;
            __half* dst = W1s + buf * KC * sx;
            const int cpr = a.Cin_p / 8;
            for (int i = tid; i < KC * cpr; i += NTH) {
                int r = i / cpr, k = i % cpr;
                cp16(dst + r * sx + k * 8, src + (long long)r * a.Cin_p + k * 8, true);
            }
        } else {   // input chunk (compact halo rows) -> Ec[buf]
            __half* dst = Ec + buf * a.nip * SKC;
            for (int i = tid; i < NI * (KC / 8); i += NTH) {
                int r = i / (KC / 8), k = i % (KC / 8);
                int gy = y0 + r / cw, gx = x0 + r % cw;
                int ch = c * KC + k * 8;
                bool ok = ch < a.Cin;
                const __half* src = a.x + (img + (long long)gy * a.W + gx) * a.ldx + (ok ? ch : 0);
                cp16(dst + r * SKC + k * 8, src, ok);
            }
        }
        const __half* src2 = a.gproj ? a.w2 + (long long)c * KC * KC : a.w2 + (long long)co0 * a.Ep + c * KC;
        const int ld2 = a.gproj ? KC : a.Ep;
        __half* dst2 = W2s + buf * a.cout_ch * SKC;
        for (int i = tid; i < (a.gproj ? KC : cch) * (KC / 8); i += NTH) {
            int r = i / (KC / 8), k = i % (KC / 8);
            cp16(dst2 + r * SKC + k * 8, src2 + (long long)r * ld2 + k * 8, true);
        }
    };

    // ---- dw first: Xs[px][Cin_p] = act2(dw(x) + bd) for the tile pixels (from global / L2), zero rows / pad channels
    if (a.dwf) {
        const int cpr = a.Cin_p / 8;
        for (int i = tid; i < NIp * cpr; i += NTH) {
            int px = i / cpr, cg = i % cpr;
            __align__(16) __half o[8];
            if (px < TP && cg * 8 < a.Cin) {
                float sacc[8];
                uint4 bv = __ldg(reinterpret_cast<const uint4*>(a.bd + cg * 8));
                const __half* hb = reinterpret_cast<const __half*>(&bv);
#pragma unroll
                for (int k = 0; k < 8; ++k) sacc[k] = __half2float(hb[k]);
                int oy = oy0 + px / tw, ox = ox0 + px % tw;
                for (int r = 0; r < a.R; ++r) {
                    int gy = oy * a.sh - a.ph + r;
                    if (gy < 0 || gy >= a.H) continue;
                    for (int q = 0; q < a.S; ++q) {
                        int gx = ox * a.sw - a.pw + q;
                        if (gx < 0 || gx >= a.W) continue;
                        uint4 v = __ldg(reinterpret_cast<const uint4*>(a.x + (img + (long long)gy * a.W + gx) * a.ldx + cg * 8));
                        uint4 w = __ldg(reinterpret_cast<const uint4*>(a.wd + (long long)(r * a.S + q) * a.ldwd + cg * 8));
                        const __half2* hv = reinterpret_cast<const __half2*>(&v);
                        const __half2* hw = reinterpret_cast<const __half2*>(&w);
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            float2 fv = __half22float2(hv[k]), fw = __half22float2(hw[k]);
                            sacc[2 * k] += fv.x * fw.x; sacc[2 * k + 1] += fv.y * fw.y;
                        }
                    }
                }
#pragma unroll
                for (int k = 0; k < 8; ++k) o[k] = __float2half(actf(sacc[k], a.act2));
            } else {
#pragma unroll
                for (int k = 0; k < 8; ++k) o[k] = __float2half(0.f);
            }
            *reinterpret_cast<uint4*>(Xs + px * sx + cg * 8) = *reinterpret_cast<const uint4*>(o);
        }
    } else if (expand) {
        const int cpr = a.Cin_p / 8;
        for (int i = tid; i < NIp * cpr; i += NTH) {
            int r = i / cpr, k = i % cpr;
            bool ok = r < NI && k * 8 < a.Cin;
            int rr = ok ? r : 0;
            int gy = y0 + rr / cw, gx = x0 + rr % cw;
            const __half* src = a.x + (img + (long long)gy * a.W + gx) * a.ldx + (ok ? k * 8 : 0);
            cp16(Xs + r * sx + k * 8, src, ok);
        }
    }
    load_w(0, 0);
    cp_commit();

    // project GEMM accumulators: warp -> m-tile (warp % MT), n-group (warp / MT) of ntw n8 tiles
    const int mt2 = warp % a.MT, ng = warp / a.MT, ngroups = NWARP / a.MT;
    const int ntiles = cch / 8;
    const int ntw = (ntiles + ngroups - 1) / ngroups;
    const int nt0 = ng * ntw;
    unsigned acc[NTW_MAX][2];
#pragma unroll
    for (int j = 0; j < NTW_MAX; ++j) acc[j][0] = acc[j][1] = 0u;

    const int g = lane >> 2, tq4 = lane & 3;
    for (int c = 0; c < nchunk; ++c) {
        const int buf = c & 1;
        if (c + 1 < nchunk) { load_w(c + 1, buf ^ 1); cp_commit(); cp_wait<1>(); } else { cp_wait<0>(); }
        __syncthreads();
        const __half* E;                       // expanded chunk [rows][SKC]
        if (expand) {
            // ---- expand GEMM: Ec[NI][KC] = act1(Xs . W1c^T + b1)
            const __half* W1c = W1s + buf * KC * sx;
            for (int mt = warp; mt < NIp / 16; mt += NWARP) {
                unsigned e[4][2] = {{0u, 0u}, {0u, 0u}, {0u, 0u}, {0u, 0u}};
                for (int k = 0; k < a.Cin_p; k += 16) {
                    unsigned af[4];
                    ldsm4(af[0], af[1], af[2], af[3], Xs + (mt * 16 + (lane & 15)) * sx + k + (lane >> 4) * 8);
#pragma unroll
                    for (int nb = 0; nb < 2; ++nb) {
                        unsigned b0, b1, b2, b3;
                        ldsm4(b0, b1, b2, b3, W1c + (nb * 16 + (lane >> 4) * 8 + (lane & 7)) * sx + k + ((lane >> 3) & 1) * 8);
                        mma16816(e[nb * 2], af, b0, b1);
                        mma16816(e[nb * 2 + 1], af, b2, b3);
                    }
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int col = j * 8 + tq4 * 2;
                    __half2 bb = *reinterpret_cast<const __half2*>(a.b1 + c * KC + col);
                    float2 fb = __half22float2(bb);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        int r = mt * 16 + g + h * 8;
                        float2 v = __half22float2(u2h(e[j][h]));
                        __half2 o = __floats2half2_rn(actf(v.x + fb.x, a.act1), actf(v.y + fb.y, a.act1));
                        if (r < NI) *reinterpret_cast<__half2*>(Ec + r * SKC + col) = o;
                    }
                }
            }
            __syncthreads();
            E = Ec;
        } else {
            E = Ec + buf * a.nip * SKC;
        }
        const __half* Ad = a.dwf ? Ec : Dc;    // A operand of the project GEMM
        if (!a.dwf) {
        // ---- depthwise RxS (+ bd, act2): Dc[px][KC], fp32 accumulation; rows >= TP are zero
        for (int i = tid; i < a.MT * 16 * (KC / 8); i += NTH) {
            int px = i / (KC / 8), cg = i % (KC / 8);
            float s[8];
            int ch = c * KC + cg * 8;
            __align__(16) __half o[8];
            if (px < TP) {
                uint4 bv = __ldg(reinterpret_cast<const uint4*>(a.bd + ch));
                const __half* hb = reinterpret_cast<const __half*>(&bv);
#pragma unroll
                for (int k = 0; k < 8; ++k) s[k] = __half2float(hb[k]);
                int oy = oy0 + px / tw, ox = ox0 + px % tw;
                for (int r = 0; r < a.R; ++r) {
                    int gy = oy * a.sh - a.ph + r;
                    if (gy < 0 || gy >= a.H) continue;
                    for (int q = 0; q < a.S; ++q) {
                        int gx = ox * a.sw - a.pw + q;
                        if (gx < 0 || gx >= a.W) continue;
                        uint4 v = *reinterpret_cast<const uint4*>(E + ((gy - y0) * cw + (gx - x0)) * SKC + cg * 8);
                        uint4 w = __ldg(reinterpret_cast<const uint4*>(a.wd + (long long)(r * a.S + q) * a.Ep + ch));
                        const __half2* hv = reinterpret_cast<const __half2*>(&v);
                        const __half2* hw = reinterpret_cast<const __half2*>(&w);
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            float2 fv = __half22float2(hv[k]), fw = __half22float2(hw[k]);
                            s[2 * k] += fv.x * fw.x; s[2 * k + 1] += fv.y * fw.y;
                        }
                    }
                }
#pragma unroll
                for (int k = 0; k < 8; ++k) o[k] = __float2half(actf(s[k], a.act2));
            } else {
#pragma unroll
                for (int k = 0; k < 8; ++k) o[k] = __float2half(0.f);
            }
            *reinterpret_cast<uint4*>(Dc + px * SKC + cg * 8) = *reinterpret_cast<const uint4*>(o);
        }
        __syncthreads();
        }
        if (a.gproj) {   // ---- grouped project: chunk c -> output channels [c*KC, c*KC + KC), written directly
            const __half* W2c = W2s + buf * a.cout_ch * SKC;
            for (int tl = warp; tl < a.MT * (KC / 8); tl += NWARP) {
                int mt = tl / (KC / 8), nt = tl % (KC / 8);
                unsigned d[2] = {0u, 0u};
#pragma unroll
                for (int kk = 0; kk < KC; kk += 16) {
                    unsigned af[4], b0, b1;
                    ldsm4(af[0], af[1], af[2], af[3], Dc + (mt * 16 + (lane & 15)) * SKC + kk + (lane >> 4) * 8);
                    ldsm2(b0, b1, W2c + (nt * 8 + (lane & 7)) * SKC + kk + ((lane >> 3) & 1) * 8);
                    mma16816(d, af, b0, b1);
                }
                int co = c * KC + nt * 8 + tq4 * 2;
                int cph = a.colmap ? a.colmap[co >> 3] + (co & 7) : co;
                float2 fb = a.b2 ? __half22float2(*reinterpret_cast<const __half2*>(a.b2 + co)) : make_float2(0.f, 0.f);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    int px = mt * 16 + g + h * 8;
                    if (px >= TP) continue;
                    int oy = oy0 + px / tw, ox = ox0 + px % tw;
                    long long pix = ((long long)n * a.P + oy) * a.Q + ox;
                    float2 v = __half22float2(u2h(d[h]));
                    *reinterpret_cast<__half2*>(a.y + pix * a.ldy + cph) =
                        __floats2half2_rn(actf(v.x + fb.x, a.act3), actf(v.y + fb.y, a.act3));
                }
            }
        } else
        // ---- project GEMM: acc[TP][cch] += Dc . W2c^T
        {
            const __half* W2c = W2s + buf * a.cout_ch * SKC;
#pragma unroll
            for (int kk = 0; kk < KC; kk += 16) {
                unsigned af[4];
                ldsm4(af[0], af[1], af[2], af[3], Ad + (mt2 * 16 + (lane & 15)) * SKC + kk + (lane >> 4) * 8);
#pragma unroll
                for (int j = 0; j < NTW_MAX; ++j) {
                    if (j < ntw && nt0 + j < ntiles) {
                        unsigned b0, b1;
                        ldsm2(b0, b1, W2c + ((nt0 + j) * 8 + (lane & 7)) * SKC + kk + ((lane >> 3) & 1) * 8);
                        mma16816(acc[j], af, b0, b1);
                    }
                }
            }
        }
        __syncthreads();
    }
    // ---- epilogue: + b2 (+ res), act3 -> y
#pragma unroll
    for (int j = 0; j < NTW_MAX; ++j) {
        if (a.gproj) break;
        if (j < ntw && nt0 + j < ntiles) {
            int co = co0 + (nt0 + j) * 8 + tq4 * 2;
            int cph = a.colmap ? a.colmap[co >> 3] + (co & 7) : co;   // physical output channel
            float2 fb = a.b2 ? __half22float2(*reinterpret_cast<const __half2*>(a.b2 + co)) : make_float2(0.f, 0.f);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int px = mt2 * 16 + g + h * 8;
                if (px >= TP) continue;
                int oy = oy0 + px / tw, ox = ox0 + px % tw;
                long long pix = ((long long)n * a.P + oy) * a.Q + ox;
                float2 v = __half22float2(u2h(acc[j][h]));
                v.x += fb.x; v.y += fb.y;
                if (a.res) {
                    float2 r = __half22float2(*reinterpret_cast<const __half2*>(a.res + pix * a.Cout + co));
                    v.x += r.x; v.y += r.y;
                }
                *reinterpret_cast<__half2*>(a.y + pix * a.ldy + cph) = __floats2half2_rn(actf(v.x, a.act3), actf(v.y, a.act3));
            }
        }
    }
    // ---- passthrough gather (ShuffleNet identity half; stride 1: output pixel == input pixel)
    if (a.G && blockIdx.y == 0) {
        for (int i = tid; i < TP * (a.G / 8); i += NTH) {
            int px = i / (a.G / 8), b = i % (a.G / 8);
            int oy = oy0 + px / tw, ox = ox0 + px % tw;
            const __half* xr = a.xb + (img + (long long)oy * a.W + ox) * a.ldx;
            __align__(16) __half o[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) { int sc = a.gsrc[b * 8 + k]; o[k] = sc >= 0 ? xr[sc] : __float2half(0.f); }
            *reinterpret_cast<uint4*>(a.y + (((long long)n * a.P + oy) * a.Q + ox) * a.ldy + a.gdst[b]) =
                *reinterpret_cast<const uint4*>(o);
        }
    }
}

int smem_bytes(bool expand, int nip, int Cin_p, int MT, int cout_ch) {
    int sx = Cin_p + 8;
    long long h = (expand ? (long long)nip * sx : 0) + (long long)nip * SKC * (expand ? 1 : 2) + MT * 16 * SKC +
                  (expand ? 2 * KC * sx : 0) + 2LL * cout_ch * SKC;
    return int(h * 2);
}

const int SMEM_MAX = 99 * 1024;
const int SMEM_TOTAL = 100 * 1024;
const int SMS_ = [] { int d = 0, n = 36; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); return n; }();

}  // namespace

// x [N, Cin, H, W] channels-last fp16; w1 [Ep, Cin_p] or None; b1 [Ep]; wd [R, S, Ep]; bd [Ep]; w2 [Cout, Ep]; b2 [Cout];
// res [N, Cout, P, Q] channels-last or None; acts = [act1, act2, act3]. Tile rule: TH = ceil(P / ceil(P / 8)) (same for TW),
// shrunk until the smem plan fits; output-channel chunks of <= 128 * 8 / MT (grid.y).
static at::Tensor mb_impl(at::Tensor x, c10::optional<at::Tensor> w1, c10::optional<at::Tensor> b1, at::Tensor wd,
                  c10::optional<at::Tensor> bd, at::Tensor w2, c10::optional<at::Tensor> b2, c10::optional<at::Tensor> res,
                  std::vector<int64_t> stride, std::vector<int64_t> pad, std::vector<int64_t> acts, int64_t tile,
                  c10::optional<at::Tensor> out, c10::optional<at::Tensor> colmap, c10::optional<at::Tensor> gsrc,
                  c10::optional<at::Tensor> gdst, int64_t cin_off, int64_t cin, bool dwf = false) {
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "mbconv: fp16 CUDA x");
    Args a{};
    a.N = x.size(0); a.H = x.size(2); a.W = x.size(3);
    a.ldx = x.size(1);
    a.Cin = cin > 0 ? (int)cin : a.ldx - (int)cin_off;
    TORCH_CHECK(cin_off % 8 == 0 && cin_off + a.Cin <= a.ldx && a.ldx % 8 == 0, "mbconv: input channel slice");
    a.R = wd.size(0); a.S = wd.size(1); a.ldwd = wd.size(2);
    a.Ep = dwf ? w1->size(0) : wd.size(2);
    a.dwf = dwf;
    a.gproj = w2.dim() == 3;            // [Ep / KC, KC, KC] block-diagonal project weight
    a.Cout = a.gproj ? a.Ep : w2.size(0);
    a.sh = stride[0]; a.sw = stride[1]; a.ph = pad[0]; a.pw = pad[1];
    a.P = (a.H + 2 * a.ph - a.R) / a.sh + 1; a.Q = (a.W + 2 * a.pw - a.S) / a.sw + 1;
    bool expand = w1.has_value();
    a.Cin_p = expand ? w1->size(1) : a.Cin;
    TORCH_CHECK(a.Cin % 8 == 0 && a.Cout % 8 == 0 && a.Ep % KC == 0 && (a.gproj ? w2.size(0) * KC == a.Ep && w2.size(1) == KC && w2.size(2) == KC : w2.size(1) == a.Ep) && (!expand || a.Cin_p % 16 == 0),
                "mbconv: channel layout");
    TORCH_CHECK(expand || a.Ep >= a.Cin, "mbconv: no-expand needs Ep >= Cin");
    TORCH_CHECK(!dwf || (expand && a.ldwd == a.Cin_p), "mbconv_dwffn: wd [R, S, Cin_p], w1 [Ep, Cin_p]");
    a.xb = (const __half*)x.data_ptr();
    a.x = a.xb + cin_off;
    a.w1 = expand ? (const __half*)w1->data_ptr() : nullptr;
    a.b1 = b1 ? (const __half*)b1->data_ptr() : nullptr;
    TORCH_CHECK(!expand || a.b1, "mbconv: expand needs b1");
    a.wd = (const __half*)wd.data_ptr();
    TORCH_CHECK(bd.has_value(), "mbconv: bd");
    a.bd = (const __half*)bd->data_ptr();
    a.w2 = (const __half*)w2.data_ptr();
    a.b2 = b2 ? (const __half*)b2->data_ptr() : nullptr;
    at::Tensor y;
    if (out) {
        y = *out;
        TORCH_CHECK(y.is_contiguous(at::MemoryFormat::ChannelsLast) && y.size(0) == a.N && y.size(2) == a.P &&
                    y.size(3) == a.Q && y.size(1) % 8 == 0 && y.scalar_type() == at::kHalf, "mbconv: out buffer");
        a.ldy = y.size(1);
    } else {
        y = at::empty({a.N, a.Cout, a.P, a.Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
        a.ldy = a.Cout;
    }
    TORCH_CHECK(colmap.has_value() || a.Cout <= a.ldy, "mbconv: Cout > out channels");
    if (colmap) { TORCH_CHECK(colmap->numel() == a.Cout / 8 && colmap->scalar_type() == at::kInt, "colmap"); a.colmap = colmap->data_ptr<int>(); }
    if (gsrc) {
        TORCH_CHECK(gdst && gsrc->numel() % 8 == 0 && gdst->numel() * 8 == gsrc->numel() && a.sh == 1 && a.sw == 1 &&
                    a.P == a.H && a.Q == a.W, "mbconv: gather");
        a.gsrc = gsrc->data_ptr<int>(); a.gdst = gdst->data_ptr<int>(); a.G = gsrc->numel();
    }
    if (res) {
        *res = res->contiguous(at::MemoryFormat::ChannelsLast);
        TORCH_CHECK(res->size(1) == a.Cout && res->size(2) == a.P && res->size(3) == a.Q && res->size(0) == a.N, "mbconv: res shape");
        a.res = (const __half*)res->data_ptr();
    }
    a.y = (__half*)y.data_ptr();
    a.act1 = acts[0]; a.act2 = acts[1]; a.act3 = acts[2];
    // tile rule (fixed, shape-only)
    int tgt = tile > 0 ? (int)tile : 8;
    auto side = [&](int L) { int nt = (L + tgt - 1) / tgt; return (L + nt - 1) / nt; };
    a.TH = side(a.P); a.TW = side(a.Q);
    for (;;) {
        int TP = a.TH * a.TW;
        a.MT = TP > 64 ? 8 : TP > 32 ? 4 : TP > 16 ? 2 : 1;
        int maxch = 8 * NTW_MAX * (NWARP / a.MT);
        int nch = (a.Cout + maxch - 1) / maxch;
        a.cout_ch = a.gproj ? KC : ((a.Cout / 8 + nch - 1) / nch) * 8;
        a.HH = (a.TH - 1) * a.sh + a.R; a.HW = (a.TW - 1) * a.sw + a.S;
        a.nip = dwf ? a.MT * 16 : (a.HH * a.HW + 15) & ~15;
        if (smem_bytes(expand, a.nip, a.Cin_p, a.MT, a.cout_ch) <= SMEM_MAX || (a.TH == 1 && a.TW == 1)) break;
        if (a.TH >= a.TW) a.TH = (a.TH + 1) / 2; else a.TW = (a.TW + 1) / 2;
    }
    int smem = smem_bytes(expand, a.nip, a.Cin_p, a.MT, a.cout_ch);
    TORCH_CHECK(smem <= SMEM_MAX, "mbconv: smem plan too large");
    a.tp_n = (a.P + a.TH - 1) / a.TH; a.tq_n = (a.Q + a.TW - 1) / a.TW;
    static bool attr = false;
    if (!attr) {
        TORCH_CHECK(cudaFuncSetAttribute(mbconv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX) == cudaSuccess,
                    "mbconv: smem attr");
        attr = true;
    }
    dim3 grid(a.N * a.tp_n * a.tq_n, a.gproj ? 1 : (a.Cout + a.cout_ch - 1) / a.cout_ch);
    mbconv_kernel<<<grid, NTH, smem, at::cuda::getCurrentCUDAStream()>>>(a);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "mbconv launch");
    return y;
}

at::Tensor mbconv(at::Tensor x, c10::optional<at::Tensor> w1, c10::optional<at::Tensor> b1, at::Tensor wd,
                  c10::optional<at::Tensor> bd, at::Tensor w2, c10::optional<at::Tensor> b2, c10::optional<at::Tensor> res,
                  std::vector<int64_t> stride, std::vector<int64_t> pad, std::vector<int64_t> acts, int64_t tile) {
    return mb_impl(x, w1, b1, wd, bd, w2, b2, res, stride, pad, acts, tile, c10::nullopt, c10::nullopt, c10::nullopt,
                   c10::nullopt, 0, 0);
}

at::Tensor mbconv_dwffn(at::Tensor x, at::Tensor wd, c10::optional<at::Tensor> bd, at::Tensor w1, c10::optional<at::Tensor> b1,
                        at::Tensor w2, c10::optional<at::Tensor> b2, c10::optional<at::Tensor> res, std::vector<int64_t> stride,
                        std::vector<int64_t> pad, std::vector<int64_t> acts, int64_t tile) {
    // wd [R, S, Cin_p] over the input channels, w1 [Ep, Cin_p], w2 [Cout, Ep]
    return mb_impl(x, w1, b1, wd, bd, w2, b2, res, stride, pad, acts, tile, c10::nullopt, c10::nullopt, c10::nullopt,
                   c10::nullopt, 0, 0, true);
}

void mbconv_into(at::Tensor x, c10::optional<at::Tensor> w1, c10::optional<at::Tensor> b1, at::Tensor wd,
                 c10::optional<at::Tensor> bd, at::Tensor w2, c10::optional<at::Tensor> b2, std::vector<int64_t> stride,
                 std::vector<int64_t> pad, std::vector<int64_t> acts, int64_t cin_off, int64_t cin, at::Tensor out,
                 c10::optional<at::Tensor> colmap, c10::optional<at::Tensor> gsrc, c10::optional<at::Tensor> gdst) {
    mb_impl(x, w1, b1, wd, bd, w2, b2, c10::nullopt, stride, pad, acts, 0, out, colmap, gsrc, gdst, cin_off, cin);
}


// ReLU linear attention of EfficientViT LiteMLA (dim 32): buffer [B, H, W, heads * 96] with per-head [q | k | v];
// out[b, p, h * 32 + d] = sum_i q'[p,i] kv[i,d] / (sum_i q'[p,i] kv[i,32] + eps), kv = sum_p k'[p]^T [v_p, 1], q' k' = relu.
// One CTA per (head, image), fp32 math (as timm).
__global__ void __launch_bounds__(256) linattn_kernel(const __half* __restrict__ x, __half* __restrict__ y, int HW, int ldx,
                                                      int heads, float eps) {
    __shared__ float kv[32 * 33];
    __shared__ float ks[64][33], vs[64][33];
    const int h = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    const __half* xb = x + (long long)b * HW * ldx + h * 96;
    float acc[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    for (int p0 = 0; p0 < HW; p0 += 64) {
        int np = min(64, HW - p0);
        for (int i = tid; i < 64 * 8; i += 256) {          // 64 px x (32 k + 32 v) halfs as 8 x uint4 per px
            int p = i / 8, part = i % 8;
            __align__(16) __half hv[8];
            if (p < np) *reinterpret_cast<uint4*>(hv) = __ldg(reinterpret_cast<const uint4*>(xb + (long long)(p0 + p) * ldx + 32) + part);
            else for (int k = 0; k < 8; ++k) hv[k] = __float2half(0.f);
            for (int k = 0; k < 8; ++k) {
                int c = part * 8 + k;
                float f = __half2float(hv[k]);
                if (c < 32) ks[p][c] = fmaxf(f, 0.f); else vs[p][c - 32] = f;
            }
            if (part == 0) vs[p][32] = p < np ? 1.f : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 5; ++j) {
            int e = tid + 256 * j;
            if (e < 32 * 33) {
                int i = e / 33, d = e % 33;
                float s = 0.f;
                for (int p = 0; p < 64; ++p) s += ks[p][i] * vs[p][d];
                acc[j] += s;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 5; ++j) { int e = tid + 256 * j; if (e < 32 * 33) kv[e] = acc[j]; }
    __syncthreads();
    for (int p = tid; p < HW; p += 256) {
        const __half* q = xb + (long long)p * ldx;
        float qf[32];
#pragma unroll
        for (int c = 0; c < 32; c += 8) {
            uint4 u = __ldg(reinterpret_cast<const uint4*>(q + c));
            const __half* hq = reinterpret_cast<const __half*>(&u);
#pragma unroll
            for (int k = 0; k < 8; ++k) qf[c + k] = fmaxf(__half2float(hq[k]), 0.f);
        }
        float den = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) den += qf[i] * kv[i * 33 + 32];
        float inv = 1.f / (den + eps);
        __half* yo = y + ((long long)b * HW + p) * heads * 32 + h * 32;
#pragma unroll
        for (int d0 = 0; d0 < 32; d0 += 8) {
            __align__(16) __half o[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float s = 0.f;
#pragma unroll
                for (int i = 0; i < 32; ++i) s += qf[i] * kv[i * 33 + d0 + k];
                o[k] = __float2half(s * inv);
            }
            *reinterpret_cast<uint4*>(yo + d0) = *reinterpret_cast<const uint4*>(o);
        }
    }
}

at::Tensor linattn(at::Tensor x, int64_t dim, double eps) {
    TORCH_CHECK(dim == 32 && x.scalar_type() == at::kHalf && x.is_contiguous(at::MemoryFormat::ChannelsLast) && x.size(1) % 96 == 0,
                "linattn: dim 32, fp16 channels-last [B, heads*96, H, W]");
    int B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3), heads = C / 96;
    auto y = at::empty({B, heads * 32, H, W}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    linattn_kernel<<<dim3(heads, B), 256, 0, at::cuda::getCurrentCUDAStream()>>>((const __half*)x.data_ptr(), (__half*)y.data_ptr(),
                                                                                   H * W, C, heads, (float)eps);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "linattn launch");
    return y;
}

// grouped conv (group width GW = C/G = K/G, GW % 16 == 0), NHWC fp16, weight [K, R, S, GW], fp16 accumulation (mma.sync),
// + bias + act. Multistage cp.async implicit GEMM: CTA = 128 output pixels x BN = 16 * NT output channels of one group,
// 8 warps = 4 (M, 32 rows) x 2 (N, 8 * NT cols); k blocks of 16 channels of one tap, 2 per stage.
template <int NT, int MI>     // MI m16 tiles per warp: warps = (128 / (16 MI)) x 2
__global__ void __launch_bounds__(32 * (8 / MI) * 2) gconv_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                                                    const __half* __restrict__ bias, __half* __restrict__ y, int N, int H,
                                                    int W, int C, int K, int GW, int R, int S, int sh, int sw, int ph, int pw,
                                                    int P, int Q, int act) {
    constexpr int BM = 128, BN = 16 * NT, ST = 3, SK = 32;   // 64-B rows, 16-B chunks XOR-swizzled by (row >> 1) & 3
    constexpr int WMN = 8 / MI, NTH = 32 * WMN * 2, RPT = BM * 4 / NTH;   // warps along M, threads, A chunks per thread
    extern __shared__ __align__(16) unsigned char smem[];
    __half* As = reinterpret_cast<__half*>(smem);          // [ST][BM][SK]
    __half* Bs = As + ST * BM * SK;                          // [ST][BN][SK]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long long M = (long long)N * P * Q;
    const long long m0 = (long long)blockIdx.x * BM;
    const int k0 = blockIdx.y * BN;                          // first output channel
    const int grp = k0 / GW, cbase = grp * GW;
    const int kpb = GW / 16, nkb = R * S * kpb, nst = (nkb + 1) / 2;
    // A rows of this thread: tid/4 + j * NTH/4
    int an[RPT], ap[RPT], aq[RPT]; bool av[RPT];
#pragma unroll
    for (int j = 0; j < RPT; ++j) {
        long long m = m0 + tid / 4 + (NTH / 4) * j;
        av[j] = m < M;
        long long mm = av[j] ? m : 0;
        aq[j] = mm % Q; long long t = mm / Q; ap[j] = t % P; an[j] = t / P;
    }
    auto load = [&](int st, int buf) {
        __half* a_ = As + buf * BM * SK;
        __half* b_ = Bs + buf * BN * SK;
#pragma unroll
        for (int j = 0; j < RPT; ++j) {
            int row = tid / 4 + (NTH / 4) * j, part = tid % 4;
            int kb = st * 2 + part / 2;
            bool ok = av[j] && kb < nkb;
            const __half* src = x;
            if (ok) {
                int tap = kb / kpb, cb = kb % kpb, r = tap / S, s2 = tap % S;
                int ih = ap[j] * sh - ph + r, iw = aq[j] * sw - pw + s2;
                ok = ih >= 0 && ih < H && iw >= 0 && iw < W;
                if (ok) src = x + (((long long)an[j] * H + ih) * W + iw) * C + cbase + cb * 16 + (part & 1) * 8;
            }
            cp16(a_ + row * SK + ((part ^ ((row >> 1) & 3)) * 8), src, ok);
        }
        for (int i = tid; i < BN * 4; i += NTH) {
            int row = i / 4, part = i % 4;
            int kb = st * 2 + part / 2;
            bool ok = kb < nkb;
            const __half* src = w;
            if (ok) {
                int tap = kb / kpb, cb = kb % kpb;
                src = w + ((long long)(k0 + row) * R * S + tap) * GW + cb * 16 + (part & 1) * 8;
            }
            cp16(b_ + row * SK + ((part ^ ((row >> 1) & 3)) * 8), src, ok);
        }
    };
    const int wm = warp % WMN, wn = warp / WMN;
    unsigned acc[MI][NT][2];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = 0u;
#pragma unroll
    for (int st = 0; st < ST - 1; ++st) { if (st < nst) load(st, st); cp_commit(); }
    for (int st = 0; st < nst; ++st) {
        cp_wait<ST - 2>();
        __syncthreads();
        if (st + ST - 1 < nst) load(st + ST - 1, (st + ST - 1) % ST);
        cp_commit();
        const __half* a_ = As + (st % ST) * BM * SK;
        const __half* b_ = Bs + (st % ST) * BN * SK;
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            unsigned af[MI][4];
#pragma unroll
            for (int i = 0; i < MI; ++i)
            {
                int row = wm * MI * 16 + i * 16 + (lane & 15), ch = kk / 8 + (lane >> 4);
                ldsm4(af[i][0], af[i][1], af[i][2], af[i][3], a_ + row * SK + ((ch ^ ((row >> 1) & 3)) * 8));
            }
#pragma unroll
            for (int j = 0; j < NT; j += 2) {
                int brow = wn * NT * 8 + j * 8 + (lane >> 4) * 8 + (lane & 7), bch = kk / 8 + ((lane >> 3) & 1);
                const __half* bp = b_ + brow * SK + ((bch ^ ((brow >> 1) & 3)) * 8);
                if (j + 1 < NT) {
                    unsigned b0, b1, b2, b3;
                    ldsm4(b0, b1, b2, b3, bp);
#pragma unroll
                    for (int i = 0; i < MI; ++i) { mma16816(acc[i][j], af[i], b0, b1); mma16816(acc[i][j + 1], af[i], b2, b3); }
                } else {
                    unsigned b0, b1;
                    int r2 = wn * NT * 8 + j * 8 + (lane & 7);
                    ldsm2(b0, b1, b_ + r2 * SK + (((kk / 8 + ((lane >> 3) & 1)) ^ ((r2 >> 1) & 3)) * 8));
#pragma unroll
                    for (int i = 0; i < MI; ++i) mma16816(acc[i][j], af[i], b0, b1);
                }
            }
        }
    }
    cp_wait<0>();
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        int co = k0 + wn * NT * 8 + j * 8 + t4 * 2;
        float2 fb = bias ? __half22float2(*reinterpret_cast<const __half2*>(bias + co)) : make_float2(0.f, 0.f);
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                long long m = m0 + wm * MI * 16 + i * 16 + g + h * 8;
                if (m >= M) continue;
                float2 v = __half22float2(u2h(acc[i][j][h]));
                *reinterpret_cast<__half2*>(y + m * K + co) = __floats2half2_rn(actf(v.x + fb.x, act), actf(v.y + fb.y, act));
            }
    }
}

template <int NT, int MI>
static void gconv_launch1(dim3 grid, cudaStream_t st, const __half* x, const __half* w, const __half* b, __half* y, int N, int H,
                         int W, int C, int K, int GW, int R, int S, int sh, int sw, int ph, int pw, int P, int Q, int act) {
    int smem = 3 * (128 + 16 * NT) * 32 * 2;
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(gconv_kernel<NT, MI>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    gconv_kernel<NT, MI><<<grid, 32 * (8 / MI) * 2, smem, st>>>(x, w, b, y, N, H, W, C, K, GW, R, S, sh, sw, ph, pw, P, Q, act);
}

static const int GCONV_MI = [] { const char* v = getenv("CLF_GCONV_MI"); return v ? atoi(v) : 2; }();

template <int NT>
static void gconv_launch(dim3 grid, cudaStream_t st, const __half* x, const __half* w, const __half* b, __half* y, int N, int H,
                         int W, int C, int K, int GW, int R, int S, int sh, int sw, int ph, int pw, int P, int Q, int act) {
    if (GCONV_MI == 4) gconv_launch1<NT, 4>(grid, st, x, w, b, y, N, H, W, C, K, GW, R, S, sh, sw, ph, pw, P, Q, act);
    else gconv_launch1<NT, 2>(grid, st, x, w, b, y, N, H, W, C, K, GW, R, S, sh, sw, ph, pw, P, Q, act);
}

// x [N, C, H, W] channels-last; w [K, GW, R, S] (channels-last = [K, R, S, GW]); groups = C / GW
at::Tensor gconv(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, std::vector<int64_t> stride,
                 std::vector<int64_t> pad, int64_t groups, int64_t act) {
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    TORCH_CHECK(w.is_contiguous(at::MemoryFormat::ChannelsLast) && x.scalar_type() == at::kHalf, "gconv: layout");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3), K = w.size(0), GW = w.size(1), R = w.size(2), S = w.size(3);
    TORCH_CHECK(C == GW * groups && K % groups == 0 && K / groups == GW && GW % 16 == 0, "gconv: group shape");
    int P = (H + 2 * pad[0] - R) / stride[0] + 1, Q = (W + 2 * pad[1] - S) / stride[1] + 1;
    auto y = at::empty({N, K, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    int NT = GW % 128 == 0 ? 8 : GW == 112 ? 7 : GW % 64 == 0 ? 4 : 2;   // BN = 16 * NT divides GW
    long long M = (long long)N * P * Q;
    dim3 grid((M + 127) / 128, K / (16 * NT));
    auto st = at::cuda::getCurrentCUDAStream();
    const __half* xp = (const __half*)x.data_ptr(); const __half* wp = (const __half*)w.data_ptr();
    const __half* bp = bias ? (const __half*)bias->data_ptr() : nullptr; __half* yp = (__half*)y.data_ptr();
    switch (NT) {
        case 8: gconv_launch<8>(grid, st, xp, wp, bp, yp, N, H, W, C, K, GW, R, S, stride[0], stride[1], pad[0], pad[1], P, Q, act); break;
        case 7: gconv_launch<7>(grid, st, xp, wp, bp, yp, N, H, W, C, K, GW, R, S, stride[0], stride[1], pad[0], pad[1], P, Q, act); break;
        case 4: gconv_launch<4>(grid, st, xp, wp, bp, yp, N, H, W, C, K, GW, R, S, stride[0], stride[1], pad[0], pad[1], P, Q, act); break;
        default: gconv_launch<2>(grid, st, xp, wp, bp, yp, N, H, W, C, K, GW, R, S, stride[0], stride[1], pad[0], pad[1], P, Q, act);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gconv launch");
    return y;
}

// depthwise conv with channel multiplier (groups == C, K = C * mult; FastViT patch embeddings), NHWC fp16, w [R, S, K]
__global__ void dwmult_kernel(const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
                              __half* __restrict__ y, int N, int H, int W, int C, int K8, int mult, int P, int Q, int R, int S,
                              int sh, int sw, int ph, int pw, int act) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * P * Q * K8;
    if (i >= total) return;
    int c = i % K8; long long t = i / K8; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = b ? __half2float(b[c * 8 + k]) : 0.f;
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int ww = q * sw - pw + s2; if (ww < 0 || ww >= W) continue;
            const __half* xp = x + (((long long)n * H + h) * W + ww) * C;
            uint4 g = __ldg(reinterpret_cast<const uint4*>(w + ((long long)r * S + s2) * K8 * 8) + c);
            const __half* hg = reinterpret_cast<const __half*>(&g);
            if (mult == 2) {     // 8 outputs <- 4 consecutive inputs (8-byte load)
                uint2 u = __ldg(reinterpret_cast<const uint2*>(xp + c * 4));
                const __half* hx = reinterpret_cast<const __half*>(&u);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[k] += __half2float(hx[k >> 1]) * __half2float(hg[k]);
            } else {
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[k] += __half2float(__ldg(xp + (c * 8 + k) / mult)) * __half2float(hg[k]);
            }
        }
    }
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(actf(acc[k], act));
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

at::Tensor dwmult(at::Tensor x, at::Tensor w_rsk, c10::optional<at::Tensor> bias, std::vector<int64_t> stride,
                  std::vector<int64_t> pad, int64_t act) {
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3), R = w_rsk.size(0), S = w_rsk.size(1), K = w_rsk.size(2);
    TORCH_CHECK(K % C == 0 && K % 8 == 0 && x.scalar_type() == at::kHalf, "dwmult");
    int P = (H + 2 * pad[0] - R) / stride[0] + 1, Q = (W + 2 * pad[1] - S) / stride[1] + 1;
    auto y = at::empty({N, K, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    long long total = (long long)N * P * Q * (K / 8);
    dwmult_kernel<<<(total + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>((const __half*)x.data_ptr(),
        (const __half*)w_rsk.data_ptr(), bias ? (const __half*)bias->data_ptr() : nullptr, (__half*)y.data_ptr(), N, H, W, C,
        K / 8, K / C, P, Q, R, S, stride[0], stride[1], pad[0], pad[1], (int)act);
    return y;
}

// expand 1x1 (+b1, act1) + depthwise RxS (+bd, act2) in one kernel; the expanded tensor stays in smem, the dw output D
// [N, P, Q, E] goes to global (the project 1x1 stays a CUTLASS conv with fused residual).
// Persistent streaming CTAs: grid.y = chunk of XC expanded channels (its W1 rows, dw weights and biases stay in smem), the
// CTA loops over output tiles; the next tile's input halo is prefetched (cp.async, double buffer) while the current tile
// runs expand GEMM (mma.sync) -> smem -> dw -> global. w1 [Ep][Cin_p], b1 [Ep], wd [R][S][Ep], bd [Ep].
constexpr int XC = 32, XNT = 256;
template <int RT, int ST_>   // compile-time taps (0 = runtime)
__global__ void __launch_bounds__(XNT) xdw_kernel(const Args a) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int R = RT ? RT : a.R, S = ST_ ? ST_ : a.S;
    const int c0 = blockIdx.y * XC;
    const int sx = a.Cin_p + 8;
    constexpr int SE = XC + 8;
    __half* W1s = reinterpret_cast<__half*>(smem);   // [XC][sx]
    __half* Wds = W1s + XC * sx;                     // [R*S][XC]
    __half* B1s = Wds + R * S * XC;                  // [XC]
    __half* Bds = B1s + XC;                          // [XC]
    __half* Ec = Bds + XC;                           // [nip][SE]
    __half* Xs0 = Ec + a.nip * SE;                   // [2][nip][sx]
    const int cpr = a.Cin_p / 8;
    for (int i = tid; i < XC * cpr; i += XNT) {
        int r = i / cpr, k = i % cpr;
        cp16(W1s + r * sx + k * 8, a.w1 + (long long)(c0 + r) * a.Cin_p + k * 8, true);
    }
    for (int i = tid; i < R * S * (XC / 8); i += XNT) {
        int tap = i / (XC / 8), k = i % (XC / 8);
        cp16(Wds + tap * XC + k * 8, a.wd + (long long)tap * a.Ep + c0 + k * 8, true);
    }
    if (tid < XC / 8) cp16(B1s + tid * 8, a.b1 + c0 + tid * 8, true);
    else if (tid < XC / 4) cp16(Bds + (tid - XC / 8) * 8, a.bd + c0 + (tid - XC / 8) * 8, true);
    const int ntiles = a.N * a.tp_n * a.tq_n;
    struct Tile { int n, oy0, ox0, th, tw, y0, x0, cw, NI; };
    auto tile_of = [&](int t) {
        Tile T;
        int tq = t % a.tq_n; t /= a.tq_n;
        int tp = t % a.tp_n; T.n = t / a.tp_n;
        T.oy0 = tp * a.TH; T.ox0 = tq * a.TW;
        T.th = min(a.TH, a.P - T.oy0); T.tw = min(a.TW, a.Q - T.ox0);
        int iy0 = T.oy0 * a.sh - a.ph, ix0 = T.ox0 * a.sw - a.pw;
        T.y0 = max(iy0, 0); T.x0 = max(ix0, 0);
        int y1 = min(iy0 + (T.th - 1) * a.sh + R, a.H), x1 = min(ix0 + (T.tw - 1) * a.sw + S, a.W);
        T.cw = x1 - T.x0; T.NI = (y1 - T.y0) * T.cw;
        return T;
    };
    auto load_x = [&](const Tile& T, __half* Xs) {
        const long long img = (long long)T.n * a.H * a.W;
        int NIp = (T.NI + 15) & ~15;
        for (int i = tid; i < NIp * cpr; i += XNT) {
            int r = i / cpr, k = i % cpr;
            bool ok = r < T.NI && k * 8 < a.Cin;
            int rr = ok ? r : 0;
            int gy = T.y0 + rr / T.cw, gx = T.x0 + rr % T.cw;
            cp16(Xs + r * sx + k * 8, a.x + (img + (long long)gy * a.W + gx) * a.ldx + (ok ? k * 8 : 0), ok);
        }
    };
    int t = blockIdx.x;
    Tile cur;
    if (t < ntiles) { cur = tile_of(t); load_x(cur, Xs0); }
    cp_commit();
    const int g = lane >> 2, tq4 = lane & 3;
    for (int it = 0; t < ntiles; ++it, t += gridDim.x) {
        __half* Xs = Xs0 + (it & 1) * a.nip * sx;
        Tile nxt;
        bool more = t + (int)gridDim.x < ntiles;
        if (more) { nxt = tile_of(t + gridDim.x); if (!(a.dbg & 4)) load_x(nxt, Xs0 + ((it + 1) & 1) * a.nip * sx); }
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        // ---- expand GEMM: Ec[NI][XC] = act1(Xs . W1^T + b1)
        const int NIp = (a.dbg & 1) ? 0 : (cur.NI + 15) & ~15;
        for (int mt = warp; mt < NIp / 16; mt += XNT / 32) {
            unsigned e[4][2] = {{0u, 0u}, {0u, 0u}, {0u, 0u}, {0u, 0u}};
            for (int k = 0; k < a.Cin_p; k += 16) {
                unsigned af[4];
                ldsm4(af[0], af[1], af[2], af[3], Xs + (mt * 16 + (lane & 15)) * sx + k + (lane >> 4) * 8);
#pragma unroll
                for (int nb = 0; nb < 2; ++nb) {
                    unsigned b0, b1, b2, b3;
                    ldsm4(b0, b1, b2, b3, W1s + (nb * 16 + (lane >> 4) * 8 + (lane & 7)) * sx + k + ((lane >> 3) & 1) * 8);
                    mma16816(e[nb * 2], af, b0, b1);
                    mma16816(e[nb * 2 + 1], af, b2, b3);
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int col = j * 8 + tq4 * 2;
                __half2 hb = *reinterpret_cast<const __half2*>(B1s + col);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    int r = mt * 16 + g + h * 8;
                    *reinterpret_cast<__half2*>(Ec + r * SE + col) = acth2(__hadd2(u2h(e[j][h]), hb), a.act1);
                }
            }
        }
        __syncthreads();
        // ---- depthwise (+bd, act2) -> D (global)
        const int TP = cur.th * cur.tw;
        for (int i = tid; i < TP * (XC / 8); i += XNT) {
            int px = i / (XC / 8), cg = i % (XC / 8);
            int ch = c0 + cg * 8;
            if (ch >= a.Cout) continue;
            if (a.dbg & 2) {
                int oy = cur.oy0 + px / cur.tw, ox = cur.ox0 + px % cur.tw;
                *reinterpret_cast<uint4*>(a.y + (((long long)cur.n * a.P + oy) * a.Q + ox) * a.Cout + ch) = make_uint4(0, 0, 0, 0);
                continue;
            }
            __half2 hacc[4];   // fp16 accumulation (as the tensor-core convs): 4 HFMA2 per tap, no conversions
            {
                uint4 bv = *reinterpret_cast<const uint4*>(Bds + cg * 8);
                const __half2* hb = reinterpret_cast<const __half2*>(&bv);
#pragma unroll
                for (int k = 0; k < 4; ++k) hacc[k] = hb[k];
            }
            int oy = cur.oy0 + px / cur.tw, ox = cur.ox0 + px % cur.tw;
            int gy0 = oy * a.sh - a.ph, gx0 = ox * a.sw - a.pw;
#pragma unroll
            for (int r = 0; r < (RT ? RT : 8); ++r) {
                if (!RT && r >= R) break;
                int gy = gy0 + r;
                if (gy < 0 || gy >= a.H) continue;
#pragma unroll
                for (int q = 0; q < (ST_ ? ST_ : 8); ++q) {
                    if (!ST_ && q >= S) break;
                    int gx = gx0 + q;
                    if (gx < 0 || gx >= a.W) continue;
                    uint4 v = *reinterpret_cast<const uint4*>(Ec + ((gy - cur.y0) * cur.cw + (gx - cur.x0)) * SE + cg * 8);
                    uint4 w = *reinterpret_cast<const uint4*>(Wds + (r * S + q) * XC + cg * 8);
                    const __half2* hv = reinterpret_cast<const __half2*>(&v);
                    const __half2* hw = reinterpret_cast<const __half2*>(&w);
#pragma unroll
                    for (int k = 0; k < 4; ++k) hacc[k] = __hfma2(hv[k], hw[k], hacc[k]);
                }
            }
            __align__(16) __half2 o[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) o[k] = acth2(hacc[k], a.act2);
            *reinterpret_cast<uint4*>(a.y + (((long long)cur.n * a.P + oy) * a.Q + ox) * a.Cout + ch) =
                *reinterpret_cast<const uint4*>(o);
        }
        __syncthreads();
        cur = nxt;
    }
    cp_wait<0>();
}

// x [N, Cin, H, W] channels-last; w1 [Ep, Cin_p]; b1 [Ep]; wd [R, S, Ep]; bd [Ep]; E = real expanded channels (% 8)
at::Tensor xdw(at::Tensor x, at::Tensor w1, at::Tensor b1, at::Tensor wd, at::Tensor bd, int64_t E, std::vector<int64_t> stride,
               std::vector<int64_t> pad, std::vector<int64_t> acts, int64_t tile) {
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    Args a{};
    a.N = x.size(0); a.Cin = x.size(1); a.H = x.size(2); a.W = x.size(3); a.ldx = a.Cin;
    a.R = wd.size(0); a.S = wd.size(1); a.Ep = wd.size(2); a.Cin_p = w1.size(1); a.Cout = E;
    TORCH_CHECK(x.scalar_type() == at::kHalf && a.Cin % 8 == 0 && a.Cin_p % 16 == 0 && a.Ep % XC == 0 && E % 8 == 0 &&
                E <= a.Ep && w1.size(0) == a.Ep, "xdw: layout");
    a.sh = stride[0]; a.sw = stride[1]; a.ph = pad[0]; a.pw = pad[1];
    a.P = (a.H + 2 * a.ph - a.R) / a.sh + 1; a.Q = (a.W + 2 * a.pw - a.S) / a.sw + 1;
    a.x = (const __half*)x.data_ptr(); a.w1 = (const __half*)w1.data_ptr(); a.b1 = (const __half*)b1.data_ptr();
    a.wd = (const __half*)wd.data_ptr(); a.bd = (const __half*)bd.data_ptr();
    a.act1 = acts[0]; a.act2 = acts[1];
    { static int dbg = [] { const char* v = getenv("CLF_XDW_DBG"); return v ? atoi(v) : 0; }(); a.dbg = dbg; }
    auto y = at::empty({a.N, E, a.P, a.Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    a.y = (__half*)y.data_ptr();
    int tgt = tile > 0 ? (int)tile : 8;
    auto side = [&](int L) { int nt = (L + tgt - 1) / tgt; return (L + nt - 1) / nt; };
    a.TH = side(a.P); a.TW = side(a.Q);
    const int XBUDGET = 48 * 1024;      // aim for >= 2 CTAs (16 warps) per SM
    int smem = 0;
    for (;;) {
        a.HH = (a.TH - 1) * a.sh + a.R; a.HW = (a.TW - 1) * a.sw + a.S;
        a.nip = (a.HH * a.HW + 15) & ~15;
        smem = 2 * (XC * (a.Cin_p + 8) + a.R * a.S * XC + 2 * XC + a.nip * (XC + 8) + 2 * a.nip * (a.Cin_p + 8));
        if (smem <= XBUDGET || (a.TH * a.TW <= 16)) break;
        if (a.TH >= a.TW) a.TH = (a.TH + 1) / 2; else a.TW = (a.TW + 1) / 2;
    }
    TORCH_CHECK(smem <= SMEM_MAX, "xdw: smem");
    a.tp_n = (a.P + a.TH - 1) / a.TH; a.tq_n = (a.Q + a.TW - 1) / a.TW;
    int ntiles = a.N * a.tp_n * a.tq_n, nch = (E + XC - 1) / XC;
    int per_sm = std::max(1, std::min(4, SMEM_TOTAL / std::max(smem, 1)));
    int gx = std::max(1, std::min(ntiles, (SMS_ * per_sm + nch - 1) / nch));
    dim3 grid(gx, nch);
    auto st = at::cuda::getCurrentCUDAStream();
    auto go = [&](auto kern) {
        static bool attr = false;   // one flag per instantiation (lambda per call site is fine: attr set before first use)
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
        kern<<<grid, XNT, smem, st>>>(a);
        (void)attr;
    };
    if (a.R == 3 && a.S == 3) go(xdw_kernel<3, 3>);
    else if (a.R == 5 && a.S == 5) go(xdw_kernel<5, 5>);
    else go(xdw_kernel<0, 0>);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "xdw launch");
    return y;
}

// out[p, gdst[b] + k] = x[p, gsrc[8 b + k]] (-1: 0) for every pixel (ShuffleNet passthrough channels), x / out channels-last
__global__ void chan_gather_kernel(const __half* __restrict__ x, __half* __restrict__ y, const int* __restrict__ gsrc,
                                   const int* __restrict__ gdst, long long npix, int nb, int ldx, int ldy) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= npix * nb) return;
    int b = i % nb; long long p = i / nb;
    const __half* xr = x + p * ldx;
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { int sc = __ldg(gsrc + b * 8 + k); o[k] = sc >= 0 ? xr[sc] : __float2half(0.f); }
    *reinterpret_cast<uint4*>(y + p * ldy + __ldg(gdst + b)) = *reinterpret_cast<const uint4*>(o);
}

void chan_gather(at::Tensor x, at::Tensor out, at::Tensor gsrc, at::Tensor gdst) {
    TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast) && out.is_contiguous(at::MemoryFormat::ChannelsLast) &&
                x.size(0) == out.size(0) && x.size(2) == out.size(2) && x.size(3) == out.size(3) && gsrc.numel() == 8 * gdst.numel(),
                "chan_gather");
    long long npix = (long long)x.size(0) * x.size(2) * x.size(3);
    int nb = gdst.numel();
    chan_gather_kernel<<<(npix * nb + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const __half*)x.data_ptr(),
        (__half*)out.data_ptr(), gsrc.data_ptr<int>(), gdst.data_ptr<int>(), npix, nb, x.size(1), out.size(1));
}

__device__ __forceinline__ void ldsm4t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816f(float* d, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ReLU linear attention (EfficientViT LiteMLA, head dim 32) on tensor cores, fp32 accumulation. Heads 0..HA-1 come from
// xa [B, H, W, HA*96], heads HA.. from xb [B, H, W, HB*96] (per head [q | k | v]: the qkv conv output and the multi-scale
// aggregate, no concatenation). One CTA (4 warps) per (head, image):
//   phase A: KV[32][40] = sum_p relu(k_p)^T [v_p, 1, 0..] / HW   (mma, A = K^T via ldmatrix.trans, warps split the pixels)
//   phase B: out_p = relu(q_p) KV[:, :32] / (relu(q_p) KV[:, 32] + eps / HW)  -> y[b, p, h*32 + d]
__global__ void __launch_bounds__(128) linattn_tc(const __half* __restrict__ xa, const __half* __restrict__ xb,
                                                  __half* __restrict__ y, int HW, int HA, int HB, float eps) {
    constexpr int SK = 40, SV = 48;
    __shared__ __align__(16) __half Ks[64 * SK];
    __shared__ __align__(16) __half Vs[64 * SV];
    __shared__ __align__(16) float red[4][32 * 40];
    __shared__ __align__(16) __half KVt[40 * SK];
    const int h = blockIdx.x, b = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int ld = h < HA ? HA * 96 : HB * 96;
    const __half* xh = (h < HA ? xa + (long long)b * HW * ld + h * 96 : xb + (long long)b * HW * ld + (h - HA) * 96);
    const int g = lane >> 2, t4 = lane & 3;
    float acc[2][5][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 5; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
    for (int p0 = 0; p0 < HW; p0 += 64) {
        for (int i = tid; i < 64 * 8; i += 128) {
            int p = i / 8, part = i % 8;
            uint4 u = make_uint4(0, 0, 0, 0);
            if (p0 + p < HW) u = __ldg(reinterpret_cast<const uint4*>(xh + (long long)(p0 + p) * ld + 32) + part);
            __half2* hv = reinterpret_cast<__half2*>(&u);
            if (part < 4) {
#pragma unroll
                for (int k = 0; k < 4; ++k) hv[k] = __hmax2(hv[k], __float2half2_rn(0.f));
                *reinterpret_cast<uint4*>(Ks + p * SK + part * 8) = u;
            } else {
                *reinterpret_cast<uint4*>(Vs + p * SV + (part - 4) * 8) = u;
            }
            if (part == 0) {   // ones column (32) for valid pixels, zeros 33..39
                __align__(16) __half o[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) o[k] = __float2half(k == 0 && p0 + p < HW ? 1.f : 0.f);
                *reinterpret_cast<uint4*>(Vs + p * SV + 32) = *reinterpret_cast<const uint4*>(o);
            }
        }
        __syncthreads();
        {
            const int kp = warp * 16;     // this warp's k16 slice of the 64 pixels
            unsigned af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int p = kp + (lane & 7) + ((lane >> 4) & 1) * 8, i = mi * 16 + ((lane >> 3) & 1) * 8;
                ldsm4t(af[mi][0], af[mi][1], af[mi][2], af[mi][3], Ks + p * SK + i);
            }
#pragma unroll
            for (int nj = 0; nj < 5; nj += 2) {
                int p = kp + (lane & 7) + ((lane >> 3) & 1) * 8, d = nj * 8 + ((lane >> 4) & 1) * 8;
                unsigned b0, b1, b2, b3;
                ldsm4t(b0, b1, b2, b3, Vs + p * SV + d);     // nj = 4: second pair reads cols 40..47 (unused)
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    mma16816f(acc[mi][nj], af[mi], b0, b1);
                    if (nj + 1 < 5) mma16816f(acc[mi][nj + 1], af[mi], b2, b3);
                }
            }
        }
        __syncthreads();
    }
    // reduce the 4 warps' partial KV, scale by 1 / HW, store transposed fp16 KVt[d][i]
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int nj = 0; nj < 5; ++nj)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                int i = mi * 16 + g + (k >> 1) * 8, d = nj * 8 + t4 * 2 + (k & 1);
                red[warp][i * 40 + d] = acc[mi][nj][k];
            }
    __syncthreads();
    const float inv = 1.f / HW;
    for (int e = tid; e < 32 * 40; e += 128) {
        int i = e / 40, d = e % 40;
        float v = (red[0][e] + red[1][e] + red[2][e] + red[3][e]) * inv;
        KVt[d * SK + i] = __float2half(v);
    }
    __syncthreads();
    // phase B
    unsigned bf[5][2][2];   // B fragments of KV (k = i 0..31, n = d), constant over pixels
#pragma unroll
    for (int nj = 0; nj < 5; ++nj)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
            ldsm2(bf[nj][kk][0], bf[nj][kk][1], KVt + (nj * 8 + (lane & 7)) * SK + kk * 16 + ((lane >> 3) & 1) * 8);
    const float epsn = eps * inv;
    for (int p0 = warp * 16; p0 < HW; p0 += 64) {
        // A = relu(q) rows p0..p0+15 straight from global into fragments (row g / g+8, cols 2t.. per k16 half)
        unsigned af[2][4];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                int p = p0 + g + (r & 1) * 8, i = kk * 16 + t4 * 2 + (r >> 1) * 8;
                __half2 v = p < HW ? *reinterpret_cast<const __half2*>(xh + (long long)p * ld + i) : __float2half2_rn(0.f);
                v = __hmax2(v, __float2half2_rn(0.f));
                af[kk][r] = h2u(v);
            }
        float o[5][4];
#pragma unroll
        for (int nj = 0; nj < 5; ++nj) {
#pragma unroll
            for (int k = 0; k < 4; ++k) o[nj][k] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) mma16816f(o[nj], af[kk], bf[nj][kk][0], bf[nj][kk][1]);
        }
        float den0 = __shfl_sync(0xffffffffu, o[4][0], lane & ~3), den1 = __shfl_sync(0xffffffffu, o[4][2], lane & ~3);
        float i0 = 1.f / (den0 + epsn), i1 = 1.f / (den1 + epsn);
        int pa = p0 + g, pb = p0 + g + 8;
        __half* ya = y + ((long long)b * HW + pa) * (HA + HB) * 32 + h * 32;
        __half* yb = y + ((long long)b * HW + pb) * (HA + HB) * 32 + h * 32;
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
            int d = nj * 8 + t4 * 2;
            if (pa < HW) *reinterpret_cast<__half2*>(ya + d) = __floats2half2_rn(o[nj][0] * i0, o[nj][1] * i0);
            if (pb < HW) *reinterpret_cast<__half2*>(yb + d) = __floats2half2_rn(o[nj][2] * i1, o[nj][3] * i1);
        }
    }
}

at::Tensor linattn2(at::Tensor xa, at::Tensor xb, double eps) {
    TORCH_CHECK(xa.scalar_type() == at::kHalf && xa.is_contiguous(at::MemoryFormat::ChannelsLast) && xa.size(1) % 96 == 0 &&
                xb.is_contiguous(at::MemoryFormat::ChannelsLast) && xb.size(1) % 96 == 0 && xa.size(2) == xb.size(2) &&
                xa.size(3) == xb.size(3), "linattn2: fp16 channels-last [B, heads*96, H, W] x2");
    int B = xa.size(0), H = xa.size(2), W = xa.size(3), HA = xa.size(1) / 96, HB = xb.size(1) / 96;
    auto y = at::empty({B, (HA + HB) * 32, H, W}, xa.options().memory_format(at::MemoryFormat::ChannelsLast));
    linattn_tc<<<dim3(HA + HB, B), 128, 0, at::cuda::getCurrentCUDAStream()>>>((const __half*)xa.data_ptr(), (const __half*)xb.data_ptr(),
                                                                                (__half*)y.data_ptr(), H * W, HA, HB, (float)eps);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "linattn2 launch");
    return y;
}

// direct stem conv straight from the network input: x fp32 (or fp16) NCHW [N, C <= 4, H, W] -> y fp16 NHWC [N, K, P, Q]
// (replaces the fp32 -> NHWC8 conversion kernel + the implicit-GEMM stem). One thread = one output pixel x all K channels,
// half2 accumulation (as the tensor-core convs), weights [R*S*C][K] in smem (8-channel vector loads, broadcast).
template <int K, class T>
__global__ void __launch_bounds__(128) stem_direct_kernel(const T* __restrict__ x, const __half* __restrict__ w,
                                                          const __half* __restrict__ b, __half* __restrict__ y, int N, int C,
                                                          int H, int W, int P, int Q, int R, int S, int sh, int sw, int ph,
                                                          int pw, int act) {
    extern __shared__ __align__(16) __half ws[];
    const int nw = R * S * C * K;
    for (int i = threadIdx.x * 8; i < nw; i += blockDim.x * 8)
        *reinterpret_cast<uint4*>(ws + i) = __ldg(reinterpret_cast<const uint4*>(w + i));
    __syncthreads();
    long long pix = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (pix >= (long long)N * P * Q) return;
    int q = pix % Q; long long t = pix / Q; int p = t % P; int n = t / P;
    __half2 acc[K / 2];
#pragma unroll
    for (int k = 0; k < K / 2; ++k) acc[k] = __float2half2_rn(0.f);
    const long long HW = (long long)H * W;
    const T* xn = x + (long long)n * C * HW;
    for (int r = 0; r < R; ++r) {
        int ih = p * sh - ph + r;
        if (ih < 0 || ih >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int iw = q * sw - pw + s2;
            if (iw < 0 || iw >= W) continue;
            const __half* wt = ws + ((r * S + s2) * C) * K;
            for (int c = 0; c < C; ++c) {
                __half2 xv = __float2half2_rn((float)xn[c * HW + (long long)ih * W + iw]);
#pragma unroll
                for (int k8 = 0; k8 < K / 8; ++k8) {
                    uint4 wv = *reinterpret_cast<const uint4*>(wt + c * K + k8 * 8);
                    const __half2* hw = reinterpret_cast<const __half2*>(&wv);
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[k8 * 4 + j] = __hfma2(xv, hw[j], acc[k8 * 4 + j]);
                }
            }
        }
    }
    __half* yo = y + pix * K;
#pragma unroll
    for (int k8 = 0; k8 < K / 8; ++k8) {
        __align__(16) __half o[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 v = __half22float2(acc[k8 * 4 + j]);
            float b0 = b ? __half2float(b[k8 * 8 + 2 * j]) : 0.f, b1 = b ? __half2float(b[k8 * 8 + 2 * j + 1]) : 0.f;
            o[2 * j] = __float2half(actf(v.x + b0, act));
            o[2 * j + 1] = __float2half(actf(v.y + b1, act));
        }
        *reinterpret_cast<uint4*>(yo + k8 * 8) = *reinterpret_cast<const uint4*>(o);
    }
}

// x [N, C, H, W] fp32 / fp16 (any layout, made contiguous NCHW); w_t [R, S, C, K] fp16 contiguous; bias [K]
at::Tensor stem_direct(at::Tensor x, at::Tensor w_t, c10::optional<at::Tensor> bias, std::vector<int64_t> stride,
                       std::vector<int64_t> pad, int64_t act) {
    x = x.contiguous();
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int R = w_t.size(0), S = w_t.size(1), K = w_t.size(3);
    TORCH_CHECK(w_t.size(2) == C && C <= 4 && w_t.is_contiguous() && w_t.scalar_type() == at::kHalf, "stem_direct: weight");
    int P = (H + 2 * pad[0] - R) / stride[0] + 1, Q = (W + 2 * pad[1] - S) / stride[1] + 1;
    auto y = at::empty({N, K, P, Q}, x.options().dtype(at::kHalf).memory_format(at::MemoryFormat::ChannelsLast));
    long long total = (long long)N * P * Q;
    int smem = R * S * C * K * 2;
    dim3 grid((total + 127) / 128);
    auto st = at::cuda::getCurrentCUDAStream();
    const __half* wp = (const __half*)w_t.data_ptr(); const __half* bp = bias ? (const __half*)bias->data_ptr() : nullptr;
    __half* yp = (__half*)y.data_ptr();
#define STEM_K(KK)                                                                                                         \
    if (K == KK) {                                                                                                         \
        if (x.scalar_type() == at::kFloat)                                                                                 \
            stem_direct_kernel<KK, float><<<grid, 128, smem, st>>>(x.data_ptr<float>(), wp, bp, yp, N, C, H, W, P, Q, R, S, \
                                                                   stride[0], stride[1], pad[0], pad[1], (int)act);        \
        else                                                                                                               \
            stem_direct_kernel<KK, __half><<<grid, 128, smem, st>>>((const __half*)x.data_ptr(), wp, bp, yp, N, C, H, W, P, \
                                                                    Q, R, S, stride[0], stride[1], pad[0], pad[1], (int)act); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "stem_direct launch");                                            \
        return y;                                                                                                          \
    }
    STEM_K(16) STEM_K(24) STEM_K(32) STEM_K(40) STEM_K(48) STEM_K(64)
#undef STEM_K
    TORCH_CHECK(false, "stem_direct: K not supported");
    return y;
}

TORCH_LIBRARY(clfmb, m) {
    m.def("stem_direct(Tensor x, Tensor w_t, Tensor? bias, int[] stride, int[] pad, int act) -> Tensor");
    m.def("linattn2(Tensor xa, Tensor xb, float eps) -> Tensor");
    m.def("chan_gather(Tensor x, Tensor(a!) out, Tensor gsrc, Tensor gdst) -> ()");
    m.def("xdw(Tensor x, Tensor w1, Tensor b1, Tensor wd, Tensor bd, int E, int[] stride, int[] pad, int[] acts, int tile=0) -> Tensor");
    m.def("dwmult(Tensor x, Tensor w_rsk, Tensor? bias, int[] stride, int[] pad, int act) -> Tensor");
    m.def("gconv(Tensor x, Tensor w, Tensor? bias, int[] stride, int[] pad, int groups, int act) -> Tensor");
    m.def("linattn(Tensor x, int dim, float eps) -> Tensor");
    m.def("mbconv(Tensor x, Tensor? w1, Tensor? b1, Tensor wd, Tensor? bd, Tensor w2, Tensor? b2, Tensor? res, int[] stride, "
          "int[] pad, int[] acts, int tile=0) -> Tensor");
    m.def("mbconv_dwffn(Tensor x, Tensor wd, Tensor? bd, Tensor w1, Tensor? b1, Tensor w2, Tensor? b2, Tensor? res, int[] stride, "
          "int[] pad, int[] acts, int tile=0) -> Tensor");
    m.def("mbconv_into(Tensor x, Tensor? w1, Tensor? b1, Tensor wd, Tensor? bd, Tensor w2, Tensor? b2, int[] stride, "
          "int[] pad, int[] acts, int cin_off, int cin, Tensor(a!) out, Tensor? colmap, Tensor? gsrc, Tensor? gdst) -> ()");
}
TORCH_LIBRARY_IMPL(clfmb, CUDA, m) { m.impl("mbconv", mbconv); m.impl("mbconv_into", mbconv_into); m.impl("linattn", linattn); m.impl("gconv", gconv); m.impl("mbconv_dwffn", mbconv_dwffn); m.impl("dwmult", dwmult); m.impl("xdw", xdw); m.impl("chan_gather", chan_gather); m.impl("linattn2", linattn2); m.impl("stem_direct", stem_direct); }

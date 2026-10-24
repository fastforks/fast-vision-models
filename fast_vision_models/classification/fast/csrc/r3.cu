// exp 008 round 3 kernels (extension `clfr3`), size-independent (problem size = runtime args, compiled once).
//   stem_pool: direct stem conv (C <= 4, fp32/fp16 NCHW input) + act + max pool in one kernel (the stem output never
//              goes to global memory: SqueezeNet / ShuffleNet stem 111x111x64 = 4x the pooled bytes).
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <cublasLt.h>
#include <mutex>
#include <unordered_map>

namespace {

__device__ __forceinline__ float actf(float v, int act) {
    switch (act) {
        case 1: return fmaxf(v, 0.f);
        case 2: return fminf(fmaxf(v, 0.f), 6.f);
        case 3: return __fdividef(v, 1.f + __expf(-v));
        case 4: return v * fminf(fmaxf(v + 3.f, 0.f), 6.f) * (1.f / 6.f);
    }
    return v;
}


__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(void* dst, const void* src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(pred ? 16 : 0));
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
__device__ __forceinline__ __half2 u2h(unsigned u) { return *reinterpret_cast<__half2*>(&u); }

// ---------------------------------------------------------------------------------------------------------------------
// gconv2: grouped conv (group width GW = C/G = K/G, GW % 16 == 0), NHWC fp16, w [K, R, S, GW], fp16 accumulation, + bias
// + act (+ optional fp32 per-(image, channel) output sums for a following global average pool, e.g. SE).
// Round-3 rewrite of clfmb.gconv: CTA = 128 output pixels x BN = 16 NT channels of one group, 8 warps (4 M x 2 N),
// KB k16-blocks per stage (one tap, 16 channels each), ST-stage cp.async pipeline, no divisions in the main loop
// (per-thread tap / channel-block counters), C tile staged in smem -> 16-B coalesced stores.
template <int NT, int KB, int ST>
__global__ void __launch_bounds__(256) gconv2_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                                                     const __half* __restrict__ bias, __half* __restrict__ y,
                                                     float* __restrict__ pool, int N, int H, int W, int C, int K, int GW, int KG, int ldy,
                                                     int R, int S, int sh, int sw, int ph, int pw, int P, int Q, int act) {
    constexpr int BM = 128, BN = 16 * NT, SK = 16 * KB, CH = 2 * KB;   // smem row: SK halfs = CH 16-B chunks
    constexpr int APT = BM * CH / 256, ROWS_STEP = 256 / CH;            // A chunks per thread, row step
    constexpr int CS = BN + 8;                                          // C staging row stride (halfs)
    extern __shared__ __align__(16) unsigned char smem[];
    __half* As = reinterpret_cast<__half*>(smem);          // [ST][BM][SK]
    __half* Bs = As + ST * BM * SK;                          // [ST][BN][SK]
    auto swz = [](int row, int ch) { return KB == 4 ? (ch ^ (row & 7)) : KB == 2 ? (ch ^ ((row >> 1) & 3)) : (ch ^ ((row >> 2) & 1)); };
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long long M = (long long)N * P * Q;
    const long long m0 = (long long)blockIdx.x * BM;
    const int k0 = blockIdx.y * BN;
    const int cbase = (k0 / KG) * GW;     // grouped: KG = GW; dense: KG = K -> 0
    const int kpb = GW / 16, nkb = R * S * kpb, nst = (nkb + KB - 1) / KB;
    const int part = tid % CH, kbo = part / 2, hf = part & 1;
    // A rows of this thread: tid / CH + i * ROWS_STEP
    const __half* rowp[APT];
    int ih0[APT], iw0[APT];
#pragma unroll
    for (int i = 0; i < APT; ++i) {
        long long m = m0 + tid / CH + (long long)ROWS_STEP * i;
        bool v = m < M;
        long long mm = v ? m : 0;
        int q = mm % Q; long long t = mm / Q; int p = t % P; int n = t / P;
        ih0[i] = v ? p * sh - ph : -100000;
        iw0[i] = q * sw - pw;
        rowp[i] = x + (long long)n * H * W * C + cbase + hf * 8;
    }
    // per-thread k-block state for A (all rows of the thread share kb = st * KB + kbo)
    int a_cb = kbo % kpb, a_tap = kbo / kpb;
    int a_r = a_tap / S, a_s = a_tap % S;
    const int RS = R * S;
    auto load = [&](int st, int buf) {
        __half* a_ = As + buf * BM * SK;
        __half* b_ = Bs + buf * BN * SK;
        const int kb = st * KB + kbo;
        const bool kok = kb < nkb;
#pragma unroll
        for (int i = 0; i < APT; ++i) {
            int row = tid / CH + ROWS_STEP * i;
            int ih = ih0[i] + a_r, iw = iw0[i] + a_s;
            bool ok = kok && ih >= 0 && ih < H && iw >= 0 && iw < W;
            const __half* src = ok ? rowp[i] + ((long long)ih * W + iw) * C + a_cb * 16 : x;
            cp16(a_ + row * SK + swz(row, part) * 8, src, ok);
        }
        for (int i = tid; i < BN * CH; i += 256) {
            int row = i / CH, pt = i % CH;
            int kb2 = st * KB + pt / 2;
            bool ok = kb2 < nkb;
            const __half* src = ok ? w + (long long)(k0 + row) * RS * GW + kb2 * 16 + (pt & 1) * 8 : w;
            cp16(b_ + row * SK + swz(row, pt) * 8, src, ok);
        }
        // advance the A state by KB k-blocks
        a_cb += KB;
        while (a_cb >= kpb) { a_cb -= kpb; if (++a_s == S) { a_s = 0; ++a_r; } }
    };
    const int wm = warp & 3, wn = warp >> 2;
    unsigned acc[2][NT][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
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
        for (int kk = 0; kk < KB; ++kk) {
            unsigned af[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int row = wm * 32 + i * 16 + (lane & 15), ch = kk * 2 + (lane >> 4);
                ldsm4(af[i][0], af[i][1], af[i][2], af[i][3], a_ + row * SK + swz(row, ch) * 8);
            }
#pragma unroll
            for (int j = 0; j < NT; j += 2) {
                if (j + 1 < NT) {
                    int brow = wn * NT * 8 + j * 8 + (lane >> 4) * 8 + (lane & 7), bch = kk * 2 + ((lane >> 3) & 1);
                    unsigned b0, b1, b2, b3;
                    ldsm4(b0, b1, b2, b3, b_ + brow * SK + swz(brow, bch) * 8);
#pragma unroll
                    for (int i = 0; i < 2; ++i) { mma16816(acc[i][j], af[i], b0, b1); mma16816(acc[i][j + 1], af[i], b2, b3); }
                } else {
                    int r2 = wn * NT * 8 + j * 8 + (lane & 7), bch = kk * 2 + ((lane >> 3) & 1);
                    unsigned b0, b1;
                    ldsm2(b0, b1, b_ + r2 * SK + swz(r2, bch) * 8);
#pragma unroll
                    for (int i = 0; i < 2; ++i) mma16816(acc[i][j], af[i], b0, b1);
                }
            }
        }
    }
    cp_wait<0>();
    __syncthreads();
    // epilogue: bias + act -> smem C tile [BM][CS] -> coalesced 16-B stores (+ pool sums)
    __half* Cs = reinterpret_cast<__half*>(smem);
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        int col = wn * NT * 8 + j * 8 + t4 * 2;
        __half2 hb = bias ? *reinterpret_cast<const __half2*>(bias + k0 + col) : __float2half2_rn(0.f);
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int row = wm * 32 + i * 16 + g + h * 8;
                __half2 v = __hadd2(u2h(acc[i][j][h]), hb);
                if (act == 1) v = __hmax2(v, __float2half2_rn(0.f));
                else if (act) { float2 f = __half22float2(v); v = __floats2half2_rn(actf(f.x, act), actf(f.y, act)); }
                *reinterpret_cast<__half2*>(Cs + row * CS + col) = v;
            }
    }
    __syncthreads();
    for (int i = tid; i < BM * (BN / 8); i += 256) {
        int row = i / (BN / 8), c8 = i % (BN / 8);
        long long m = m0 + row;
        if (m < M) *reinterpret_cast<uint4*>(y + m * ldy + k0 + c8 * 8) = *reinterpret_cast<const uint4*>(Cs + row * CS + c8 * 8);
    }
    if (pool) {   // per-(image, channel) sums over the tile's rows: thread = column pair x row group
        constexpr int NCP = BN / 2, NRG = 256 / NCP;
        const int PQ = P * Q;
        int cp = tid % NCP, rg = tid / NCP;
        if (rg < NRG) {
            constexpr int RPG = (BM + NRG - 1) / NRG;
            int r0 = rg * RPG, r1 = min(r0 + RPG, BM);
            float sx = 0.f, sy = 0.f;
            int cur = -1;
            for (int r = r0; r < r1; ++r) {
                long long m = m0 + r;
                if (m >= M) break;
                int n = (int)(m / PQ);
                if (n != cur) {
                    if (cur >= 0) { atomicAdd(pool + (long long)cur * K + k0 + 2 * cp, sx); atomicAdd(pool + (long long)cur * K + k0 + 2 * cp + 1, sy); }
                    cur = n; sx = sy = 0.f;
                }
                float2 f = __half22float2(*reinterpret_cast<const __half2*>(Cs + r * CS + 2 * cp));
                sx += f.x; sy += f.y;
            }
            if (cur >= 0) { atomicAdd(pool + (long long)cur * K + k0 + 2 * cp, sx); atomicAdd(pool + (long long)cur * K + k0 + 2 * cp + 1, sy); }
        }
    }
}

template <int NT, int KB, int ST>
static void gconv2_launch(dim3 grid, cudaStream_t st, const __half* x, const __half* w, const __half* b, __half* y, float* pool,
                          int N, int H, int W, int C, int K, int GW, int KG, int ldy, int R, int S, int sh, int sw, int ph, int pw,
                          int P, int Q, int act) {
    int smem = std::max(ST * (128 + 16 * NT) * 16 * KB * 2, 128 * (16 * NT + 8) * 2);
    static int attr = 0;
    if (smem > attr) { cudaFuncSetAttribute(gconv2_kernel<NT, KB, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = smem; }
    gconv2_kernel<NT, KB, ST><<<grid, 256, smem, st>>>(x, w, b, y, pool, N, H, W, C, K, GW, KG, ldy, R, S, sh, sw, ph, pw, P, Q, act);
}

static const int GC2_CFG = [] { const char* v = getenv("CLF_GC2_CFG"); return v ? atoi(v) : -1; }();   // 0: KB2 ST3, 1: KB4 ST2, 2: KB2 ST4, 3: KB4 ST3

template <int NT>
static void gconv2_cfg(dim3 grid, cudaStream_t st, const __half* x, const __half* w, const __half* b, __half* y, float* pool,
                       int N, int H, int W, int C, int K, int GW, int KG, int ldy, int R, int S, int sh, int sw, int ph, int pw,
                       int P, int Q, int act, int cfg) {
    switch (cfg) {
        case 1: gconv2_launch<NT, 4, 2>(grid, st, x, w, b, y, pool, N, H, W, C, K, GW, KG, ldy, R, S, sh, sw, ph, pw, P, Q, act); break;
        case 2: gconv2_launch<NT, 2, 4>(grid, st, x, w, b, y, pool, N, H, W, C, K, GW, KG, ldy, R, S, sh, sw, ph, pw, P, Q, act); break;
        case 3: gconv2_launch<NT, 4, 3>(grid, st, x, w, b, y, pool, N, H, W, C, K, GW, KG, ldy, R, S, sh, sw, ph, pw, P, Q, act); break;
        default: gconv2_launch<NT, 2, 3>(grid, st, x, w, b, y, pool, N, H, W, C, K, GW, KG, ldy, R, S, sh, sw, ph, pw, P, Q, act);
    }
}

// x [N, C, H, W] channels-last; w [K, GW, R, S] channels-last (groups = C / GW; groups 1 = dense conv, C % 16 == 0);
// pool (optional) fp32 [N, K] zero-initialised by the caller; out (optional): channel slice of a wider channels-last buffer
static at::Tensor gconv2_impl(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, std::vector<int64_t> stride,
                              std::vector<int64_t> pad, int64_t groups, int64_t act, float* pool, int cfg,
                              c10::optional<at::Tensor> out = c10::nullopt) {
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    TORCH_CHECK(w.is_contiguous(at::MemoryFormat::ChannelsLast) && x.scalar_type() == at::kHalf, "gconv2: layout");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3), K = w.size(0), GW = w.size(1), R = w.size(2), S = w.size(3);
    int KG = K / groups;
    TORCH_CHECK(C == GW * groups && K % groups == 0 && GW % 16 == 0 && KG % 16 == 0, "gconv2: group shape");
    int P = (H + 2 * pad[0] - R) / stride[0] + 1, Q = (W + 2 * pad[1] - S) / stride[1] + 1;
    at::Tensor y;
    int ldy = K;
    if (out) {
        y = *out;
        ldy = y.stride(3);
        TORCH_CHECK(y.size(0) == N && y.size(1) == K && y.size(2) == P && y.size(3) == Q && y.stride(1) == 1 &&
                    y.stride(2) == (long long)Q * ldy && ldy % 8 == 0 && ((uintptr_t)y.data_ptr()) % 16 == 0, "gconv2: out slice");
    } else {
        y = at::empty({N, K, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    }
    int NT = KG % 128 == 0 ? 8 : KG % 112 == 0 ? 7 : KG % 64 == 0 ? 4 : KG % 32 == 0 ? 2 : 1;
    long long M = (long long)N * P * Q;
    dim3 grid((M + 127) / 128, K / (16 * NT));
    auto st = at::cuda::getCurrentCUDAStream();
    const __half* xp = (const __half*)x.data_ptr(); const __half* wp = (const __half*)w.data_ptr();
    const __half* bp = bias ? (const __half*)bias->data_ptr() : nullptr; __half* yp = (__half*)y.data_ptr();
    // bench/gconv2_test.py + convtc_test.py: KB2/ST3, except stride 2 with >= 4 channel blocks per tap (RegNet GW 112): KB4/ST3
    if (cfg < 0) cfg = GC2_CFG >= 0 ? GC2_CFG : (stride[0] > 1 && GW >= 64 ? 3 : 0);
#define GC2(NN) gconv2_cfg<NN>(grid, st, xp, wp, bp, yp, pool, N, H, W, C, K, GW, KG, ldy, R, S, stride[0], stride[1], pad[0], pad[1], P, Q, act, cfg)
    switch (NT) { case 8: GC2(8); break; case 7: GC2(7); break; case 4: GC2(4); break; case 2: GC2(2); break; default: GC2(1); }
#undef GC2
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gconv2 launch");
    return y;
}

// dense conv (groups 1, C % 16 == 0, K % 16 == 0) on the gconv2 kernel, optional output channel slice
at::Tensor conv_tc(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, std::vector<int64_t> stride,
                   std::vector<int64_t> pad, int64_t act, int64_t cfg) {
    return gconv2_impl(x, w, bias, stride, pad, 1, act, nullptr, (int)cfg);
}
void conv_tc_into(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, std::vector<int64_t> stride,
                  std::vector<int64_t> pad, int64_t act, at::Tensor out) {
    gconv2_impl(x, w, bias, stride, pad, 1, act, nullptr, -1, out);
}

at::Tensor gconv2(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, std::vector<int64_t> stride,
                  std::vector<int64_t> pad, int64_t groups, int64_t act, int64_t cfg) {
    return gconv2_impl(x, w, bias, stride, pad, groups, act, nullptr, (int)cfg);
}

// grouped conv + per-(image, channel) mean of its output (fused global average pool for SE): returns (y, mean [N, K, 1, 1]
// fp16 channels-last)
__global__ void sums_to_mean_k(const float* __restrict__ s, __half* __restrict__ o, int n, float inv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __float2half(s[i] * inv);
}

std::tuple<at::Tensor, at::Tensor> gconv2_pool(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias,
                                               std::vector<int64_t> stride, std::vector<int64_t> pad, int64_t groups, int64_t act) {
    int N = x.size(0), K = w.size(0);
    auto st = at::cuda::getCurrentCUDAStream();
    auto sums = at::empty({N, K}, x.options().dtype(at::kFloat));
    cudaMemsetAsync(sums.data_ptr(), 0, sizeof(float) * N * K, st);
    auto y = gconv2_impl(x, w, bias, stride, pad, groups, act, sums.data_ptr<float>(), -1);
    float inv = 1.f / (float)(y.size(2) * y.size(3));
    auto mean = at::empty({N, K, 1, 1}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    sums_to_mean_k<<<(N * K + 255) / 256, 256, 0, st>>>(sums.data_ptr<float>(), (__half*)mean.data_ptr(), N * K, inv);
    return {y, mean};
}

// SE scale: y[n, p, c] = x[n, p, c] * sigmoid(s[n, c]); x channels-last [N, C, H, W], s [N, C, 1, 1]; 8 channels / thread
__global__ void se_scale_k(const __half* __restrict__ x, const __half* __restrict__ s, __half* __restrict__ y, long long HW, int C8,
                           long long total) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = i % C8; long long n = i / C8 / HW;
    uint4 v = __ldg(reinterpret_cast<const uint4*>(x) + i);
    uint4 g = __ldg(reinterpret_cast<const uint4*>(s) + n * C8 + c);
    __half2* hv = reinterpret_cast<__half2*>(&v);
    const __half2* hg = reinterpret_cast<const __half2*>(&g);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float2 f = __half22float2(hg[k]);
        hv[k] = __hmul2(hv[k], __floats2half2_rn(1.f / (1.f + __expf(-f.x)), 1.f / (1.f + __expf(-f.y))));
    }
    reinterpret_cast<uint4*>(y)[i] = v;
}

at::Tensor se_scale(at::Tensor x, at::Tensor s) {
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    if (C % 8 || x.scalar_type() != at::kHalf || s.numel() != (long long)N * C || x.dim() != 4) return x * at::sigmoid(s);
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    s = s.reshape({N, C}).contiguous();
    auto y = at::empty_like(x, at::MemoryFormat::ChannelsLast);
    long long total = (long long)N * H * W * (C / 8);
    se_scale_k<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const __half*)x.data_ptr(),
        (const __half*)s.data_ptr(), (__half*)y.data_ptr(), (long long)H * W, C / 8, total);
    return y;
}

// ---------------------------------------------------------------------------------------------------------------------
// dwconv2: depthwise conv (groups == C == K), NHWC fp16, weight [R, S, C], + bias + act, HALF2 accumulation (as the
// tensor-core convs; round-2 lesson: fp32 conversions per element made the dw kernels issue-bound). One thread = 8
// channels x TW consecutive output columns; compile-time taps for 3x3 / 5x5 (RT = 0: runtime).
template <int TW, int RT, int MODE>   // MODE 0: half2 accumulation, 1: fp32, 2: half2 per kernel row + fp32 across rows
__global__ void __launch_bounds__(128) dwconv2_k(const __half* __restrict__ x, const __half* __restrict__ w,
                                                 const __half* __restrict__ b, __half* __restrict__ y, int N, int H, int W,
                                                 int C8, int P, int Q, int R_, int S_, int sh, int sw, int ph, int pw, int act) {
    const int R = RT ? RT : R_, S = RT ? RT : S_;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int QT = (Q + TW - 1) / TW;
    long long total = (long long)N * P * QT * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int qt = t % QT; t /= QT; int p = t % P; int n = t / P;
    int q0 = qt * TW;
    __half2 bb[4];
    if (b) { uint4 bv = __ldg(reinterpret_cast<const uint4*>(b) + c); const __half2* hb = reinterpret_cast<const __half2*>(&bv);
#pragma unroll
             for (int k = 0; k < 4; ++k) bb[k] = hb[k]; }
    else {
#pragma unroll
        for (int k = 0; k < 4; ++k) bb[k] = __float2half2_rn(0.f);
    }
    __half2 acc[TW][4];
    constexpr bool F32 = MODE != 0;
    float2 af[F32 ? TW : 1][4];
#pragma unroll
    for (int j = 0; j < TW; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) { acc[j][k] = bb[k]; if (F32) af[F32 ? j : 0][k] = __half22float2(bb[k]); }
    const int L = (TW - 1) * sw + S;
    const int iw0 = q0 * sw - pw;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        const uint4* xrow = reinterpret_cast<const uint4*>(x + ((long long)n * H + h) * W * C8 * 8) + c;
        uint4 wr[RT ? RT : 7];
#pragma unroll
        for (int s2 = 0; s2 < S; ++s2) wr[s2] = __ldg(reinterpret_cast<const uint4*>(w + ((long long)r * S + s2) * C8 * 8) + c);
        if (MODE == 2) {
#pragma unroll
            for (int j = 0; j < TW; ++j)
#pragma unroll
                for (int k = 0; k < 4; ++k) acc[j][k] = __float2half2_rn(0.f);
        }
#pragma unroll
        for (int l = 0; l < L; ++l) {
            int ww = iw0 + l; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(xrow + (long long)ww * C8);
            const __half2* hv = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int j = 0; j < TW; ++j) {
                int s2 = l - j * sw;
                if (s2 < 0 || s2 >= S) continue;
                const __half2* hg = reinterpret_cast<const __half2*>(&wr[s2]);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (MODE == 1) {
                        float2 a = __half22float2(hv[k]), b2 = __half22float2(hg[k]);
                        af[F32 ? j : 0][k].x += a.x * b2.x; af[F32 ? j : 0][k].y += a.y * b2.y;
                    } else {
                        acc[j][k] = __hfma2(hv[k], hg[k], acc[j][k]);
                    }
                }
            }
        }
        if (MODE == 2) {
#pragma unroll
            for (int j = 0; j < TW; ++j)
#pragma unroll
                for (int k = 0; k < 4; ++k) { float2 f = __half22float2(acc[j][k]); af[F32 ? j : 0][k].x += f.x; af[F32 ? j : 0][k].y += f.y; }
        }
    }
#pragma unroll
    for (int j = 0; j < TW; ++j) {
        int q = q0 + j; if (q >= Q) break;
        __align__(16) __half2 o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            __half2 v = F32 ? __floats2half2_rn(af[F32 ? j : 0][k].x, af[F32 ? j : 0][k].y) : acc[j][k];
            if (act == 1) v = __hmax2(v, __float2half2_rn(0.f));
            else if (act == 2) v = __hmin2(__hmax2(v, __float2half2_rn(0.f)), __float2half2_rn(6.f));
            else if (act) { float2 f = __half22float2(v); v = __floats2half2_rn(actf(f.x, act), actf(f.y, act)); }
            o[k] = v;
        }
        reinterpret_cast<uint4*>(y + (((long long)n * P + p) * Q + q) * C8 * 8)[c] = *reinterpret_cast<uint4*>(o);
    }
}

static const int DW2_MODE = [] { const char* v = getenv("CLF_DW2_MODE"); return v ? atoi(v) : 2; }();   // 0 half2, 1 fp32, 2 row-mixed
static const int DW2_TW = [] { const char* v = getenv("CLF_DW2_TW"); return v ? atoi(v) : 1; }();   // dw2_test.py: TW 1 best (1.3-1.6x on small maps)

at::Tensor dwconv2(at::Tensor x, at::Tensor w_rsc, c10::optional<at::Tensor> bias, std::vector<int64_t> stride,
                   std::vector<int64_t> padding, int64_t act) {
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    TORCH_CHECK(x.scalar_type() == at::kHalf && x.size(1) % 8 == 0 && w_rsc.is_contiguous(), "dwconv2 input");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3), R = w_rsc.size(0), S = w_rsc.size(1);
    TORCH_CHECK(R <= 7 && S <= 7, "dwconv2: taps <= 7");
    int P = (H + 2 * padding[0] - R) / stride[0] + 1, Q = (W + 2 * padding[1] - S) / stride[1] + 1;
    auto y = at::empty({N, C, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    auto st = at::cuda::getCurrentCUDAStream();
    const __half* xp = (const __half*)x.data_ptr(); const __half* wp = (const __half*)w_rsc.data_ptr();
    const __half* bp = bias ? (const __half*)bias->data_ptr() : nullptr; __half* yp = (__half*)y.data_ptr();
    int tw = DW2_TW;
#define DW2(TWV, RTV) { long long total = (long long)N * P * ((Q + TWV - 1) / TWV) * (C / 8);                          \
        if (DW2_MODE == 1) dwconv2_k<TWV, RTV, 1><<<(total + 127) / 128, 128, 0, st>>>(xp, wp, bp, yp, N, H, W, C / 8, P, Q, R, S, \
                                                                 stride[0], stride[1], padding[0], padding[1], (int)act);  \
        else if (DW2_MODE == 2) dwconv2_k<TWV, RTV, 2><<<(total + 127) / 128, 128, 0, st>>>(xp, wp, bp, yp, N, H, W, C / 8, P, Q, R, S, \
                                                                 stride[0], stride[1], padding[0], padding[1], (int)act);  \
        else dwconv2_k<TWV, RTV, 0><<<(total + 127) / 128, 128, 0, st>>>(xp, wp, bp, yp, N, H, W, C / 8, P, Q, R, S,   \
                                                                 stride[0], stride[1], padding[0], padding[1], (int)act); }
    int rt = (R == S && (R == 3 || R == 5 || R == 7)) ? R : 0;
    if (rt == 0) {   // other kernel shapes: round-1 kernel (fp32 accumulation; runtime taps would spill the weight array)
        static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("clf::dwconv", "")
            .typed<at::Tensor(at::Tensor, at::Tensor, c10::optional<at::Tensor>, std::vector<int64_t>, std::vector<int64_t>, int64_t)>();
        return op.call(x, w_rsc, bias, stride, padding, act);
    }
    if (tw == 4) { if (rt == 3) DW2(4, 3) else if (rt == 5) DW2(4, 5) else DW2(4, 7) }
    else if (tw == 1) { if (rt == 3) DW2(1, 3) else if (rt == 5) DW2(1, 5) else DW2(1, 7) }
    else { if (rt == 3) DW2(2, 3) else if (rt == 5) DW2(2, 5) else DW2(2, 7) }
#undef DW2
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dwconv2 launch");
    return y;
}

// One CTA = TP x TQ pooled output pixels of one image.
//  phase 0: input patch (C x IH x IW, zero padded) -> smem as fp16
//  phase 1: stem tile SH x SW (= the pool windows of the CTA) x K channels -> smem (row stride KS halfs, conflict-free);
//           items = (8-channel group, stem pixel), group-major so a warp shares its weights (smem broadcast);
//           positions outside the stem output get -inf (max-pool padding semantics)
//  phase 2: max over the pool window, 8 channels per thread, 16-B coalesced stores (NHWC)
constexpr int TP = 8, TQ = 8, NTH = 256;
static const int TC_MAX_C = [] { const char* v = getenv("CLF_TC_MAX_C"); return v ? atoi(v) : 48; }();

template <int K, int KG, int RT, int CT, int SST, int PKT, class T>   // KG channels per item; RT/CT/SST/PKT: compile-time
__global__ void __launch_bounds__(NTH) stem_pool_kernel(const T* __restrict__ x, const __half* __restrict__ w,  // taps / C /
                                                        const __half* __restrict__ b, __half* __restrict__ y, int N, int C_,  // stride /
                                                        int H, int W, int P, int Q, int R_, int S_, int sh_, int sw_, int ph,  // pool k
                                                        int pw, int act, int PK_, int PS, int PP, int P2, int Q2, int SH,      // (0: runtime)
                                                        int SW, int IH, int IW, int tq_n) {
    const int R = RT ? RT : R_, S = RT ? RT : S_, C = CT ? CT : C_, sh = SST ? SST : sh_, sw = SST ? SST : sw_;
    const int PK = PKT ? PKT : PK_;
    // K == 64: 8 chunks XOR-swizzled by pixel; else row padding with (KS / 8) odd. Both: 16-B accesses conflict-free
    constexpr int KS = K == 64 ? 64 : (K / 8) % 2 ? K + 16 : K + 8;
    auto sidx = [](int px, int c8) { return px * KS + (K == 64 ? (c8 ^ (px & 7)) : c8) * 8; };
    extern __shared__ __align__(16) unsigned char smem[];
    __half* st = reinterpret_cast<__half*>(smem);                    // [SH * SW][KS]
    __half* ws = st + SH * SW * KS;                                  // [R * S * C][K]
    __half* xs = ws + ((R * S * C * K + 7) / 8) * 8;                 // [C][IH][IW]
    const int tid = threadIdx.x;
    const int n = blockIdx.y;
    const int tp0 = (blockIdx.x / tq_n) * TP, tq0 = (blockIdx.x % tq_n) * TQ;
    const int sp0 = tp0 * PS - PP, sq0 = tq0 * PS - PP;              // stem tile origin (stem output coords)
    const int ih0 = sp0 * sh - ph, iw0 = sq0 * sw - pw;              // input patch origin
    const int nw = R * S * C * K;
    for (int i = tid * 8; i < nw; i += NTH * 8) *reinterpret_cast<uint4*>(ws + i) = __ldg(reinterpret_cast<const uint4*>(w + i));
    const long long HW = (long long)H * W;
    const T* xn = x + (long long)n * C * HW;
    {   // all loads first (independent, in flight together), then convert + store: no serialized load latency
        constexpr int MAXL = 32;
        float v[MAXL];
        const int tot = C * IH * IW;
#pragma unroll
        for (int j = 0; j < MAXL; ++j) {
            int i = tid + j * NTH;
            v[j] = 0.f;
            if (i < tot) {
                int c = i / (IH * IW), r = i - c * (IH * IW), yy = r / IW, xx = r - yy * IW;
                int ih = ih0 + yy, iw = iw0 + xx;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) v[j] = (float)__ldg(xn + c * HW + (long long)ih * W + iw);
            }
        }
#pragma unroll
        for (int j = 0; j < MAXL; ++j) {
            int i = tid + j * NTH;
            if (i < tot) xs[i] = __float2half_rn(v[j]);
        }
        for (int i = tid + MAXL * NTH; i < tot; i += NTH) {      // large patches (big stem strides): plain loop
            int c = i / (IH * IW), r = i - c * (IH * IW), yy = r / IW, xx = r - yy * IW;
            int ih = ih0 + yy, iw = iw0 + xx;
            xs[i] = __float2half_rn((ih >= 0 && ih < H && iw >= 0 && iw < W) ? (float)xn[c * HW + (long long)ih * W + iw] : 0.f);
        }
    }
    __syncthreads();
    const int npx = SH * SW;
    const __half2 ninf = __half2(__ushort_as_half(0xfc00), __ushort_as_half(0xfc00));
    constexpr int NG = K / KG;
    for (int it = tid; it < npx * NG; it += NTH) {
        int kg = it / npx, px = it % npx;
        int ty = px / SW, tx = px % SW;
        int sp = sp0 + ty, sq = sq0 + tx;
        if (sp < 0 || sp >= P || sq < 0 || sq >= Q) {
            __half2 o[4] = {ninf, ninf, ninf, ninf};
#pragma unroll
            for (int u = 0; u < KG / 8; ++u) *reinterpret_cast<uint4*>(st + sidx(px, kg * (KG / 8) + u)) = *reinterpret_cast<uint4*>(o);
            continue;
        }
        __half2 acc[KG / 2];
#pragma unroll
        for (int j = 0; j < KG / 2; ++j) acc[j] = __float2half2_rn(0.f);
        const __half* xp = xs + (ty * sh) * IW + tx * sw;
        const __half* wp = ws + kg * KG;
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int s2 = 0; s2 < S; ++s2)
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    __half2 xv = __half2half2(xp[(c * IH + r) * IW + s2]);
                    const __half* wt = wp + ((r * S + s2) * C + c) * K;
#pragma unroll
                    for (int u = 0; u < KG / 8; ++u) {
                        uint4 wv = *reinterpret_cast<const uint4*>(wt + u * 8);
                        const __half2* hw = reinterpret_cast<const __half2*>(&wv);
#pragma unroll
                        for (int j = 0; j < 4; ++j) acc[u * 4 + j] = __hfma2(xv, hw[j], acc[u * 4 + j]);
                    }
                }
#pragma unroll
        for (int u = 0; u < KG / 8; ++u) {
            __align__(16) __half o[8];
            int k0 = kg * KG + u * 8;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 v = __half22float2(acc[u * 4 + j]);
                float b0 = b ? __half2float(b[k0 + 2 * j]) : 0.f, b1 = b ? __half2float(b[k0 + 2 * j + 1]) : 0.f;
                o[2 * j] = __float2half(actf(v.x + b0, act));
                o[2 * j + 1] = __float2half(actf(v.y + b1, act));
            }
            *reinterpret_cast<uint4*>(st + sidx(px, kg * (KG / 8) + u)) = *reinterpret_cast<const uint4*>(o);
        }
    }
    __syncthreads();
    for (int it = tid; it < TP * TQ * (K / 8); it += NTH) {
        int c8 = it % (K / 8), op = it / (K / 8);
        int pp = tp0 + op / TQ, pq = tq0 + op % TQ;
        if (pp >= P2 || pq >= Q2) continue;
        int ly = (op / TQ) * PS, lx = (op % TQ) * PS;
        __half2 m[4] = {ninf, ninf, ninf, ninf};
#pragma unroll
        for (int r = 0; r < PK; ++r)
#pragma unroll
            for (int s2 = 0; s2 < PK; ++s2) {
                uint4 v = *reinterpret_cast<const uint4*>(st + sidx((ly + r) * SW + lx + s2, c8));
                const __half2* hv = reinterpret_cast<const __half2*>(&v);
#pragma unroll
                for (int j = 0; j < 4; ++j) m[j] = __hmax2(m[j], hv[j]);
            }
        reinterpret_cast<uint4*>(y + (((long long)n * P2 + pp) * Q2 + pq) * K)[c8] = *reinterpret_cast<uint4*>(m);
    }
}

// x [N, C <= 4, H, W] fp32 / fp16; w_t [R, S, C, K] fp16; bias [K]; pool: square kernel pk, stride ps, pad pp, ceil_mode
at::Tensor stem_pool(at::Tensor x, at::Tensor w_t, c10::optional<at::Tensor> bias, std::vector<int64_t> stride,
                     std::vector<int64_t> pad, int64_t act, int64_t pk, int64_t ps, int64_t pp, bool ceil) {
    x = x.contiguous();
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int R = w_t.size(0), S = w_t.size(1), K = w_t.size(3);
    TORCH_CHECK(w_t.size(2) == C && C <= 4 && w_t.is_contiguous() && w_t.scalar_type() == at::kHalf, "stem_pool: weight");
    int P = (H + 2 * pad[0] - R) / stride[0] + 1, Q = (W + 2 * pad[1] - S) / stride[1] + 1;
    auto osz = [&](int L) {
        int o = ceil ? (L + 2 * (int)pp - (int)pk + (int)ps - 1) / (int)ps + 1 : (L + 2 * (int)pp - (int)pk) / (int)ps + 1;
        if (ceil && (o - 1) * ps >= L + pp) --o;
        return o;
    };
    int P2 = osz(P), Q2 = osz(Q);
    auto y = at::empty({N, K, P2, Q2}, x.options().dtype(at::kHalf).memory_format(at::MemoryFormat::ChannelsLast));
    int SH = (TP - 1) * ps + pk, SW = (TQ - 1) * ps + pk;
    int IH = (SH - 1) * stride[0] + R, IW = (SW - 1) * stride[1] + S;
    int KS = K == 64 ? 64 : (K / 8) % 2 ? K + 16 : K + 8;
    int smem = (SH * SW * KS + ((R * S * C * K + 7) / 8) * 8 + C * IH * IW) * 2;
    smem = (smem + 15) / 16 * 16;
    int tp_n = (P2 + TP - 1) / TP, tq_n = (Q2 + TQ - 1) / TQ;
    dim3 grid(tp_n * tq_n, N);
    auto st = at::cuda::getCurrentCUDAStream();
    const __half* wp = (const __half*)w_t.data_ptr(); const __half* bp = bias ? (const __half*)bias->data_ptr() : nullptr;
    __half* yp = (__half*)y.data_ptr();
    bool fast = R == 3 && S == 3 && C == 3 && stride[0] == 2 && stride[1] == 2 && pk == 3;
#define SP_L(KK, KG, RT, CT, SST, PKT, TT, XP)                                                                           \
    {                                                                                                                     \
        auto kern = stem_pool_kernel<KK, KG, RT, CT, SST, PKT, TT>;                                                       \
        static int attr = 0;                                                                                              \
        if (smem > attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = smem; }   \
        kern<<<grid, NTH, smem, st>>>(XP, wp, bp, yp, N, C, H, W, P, Q, R, S, stride[0], stride[1], pad[0], pad[1],         \
                                      (int)act, pk, ps, pp, P2, Q2, SH, SW, IH, IW, tq_n);                                \
    }
#define SP_T(KK, KG, RT, CT, SST, PKT)                                                                                    \
    if (x.scalar_type() == at::kFloat) SP_L(KK, KG, RT, CT, SST, PKT, float, x.data_ptr<float>())                       \
    else SP_L(KK, KG, RT, CT, SST, PKT, __half, (const __half*)x.data_ptr())
#define SP_K(KK, KG)                                                                                                      \
    if (K == KK) {                                                                                                        \
        if (fast) SP_T(KK, KG, 3, 3, 2, 3) else SP_T(KK, KG, 0, 0, 0, 0)                                                  \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "stem_pool launch");                                             \
        return y;                                                                                                         \
    }
    SP_K(16, 16) SP_K(24, 8) SP_K(32, 16) SP_K(40, 8) SP_K(48, 16) SP_K(64, 16)
#undef SP_K
    TORCH_CHECK(false, "stem_pool: K not supported");
    return y;
}

// ---------------------------------------------------------------------------------------------------------------------
// pw: 1x1 / stride-1 conv as a cublasLt GEMM with fused epilogue  y = act(x . w^T + bias (+ res)),  act: 0 none, 1 relu.
// fp16 in / out / accumulate (CUBLAS_COMPUTE_16F). x [N, C, H, W] channels-last (may be a channel slice: pixel stride
// ldx), w [K, C, 1, 1], res [N, K, H, W] channels-last, out: optional channel slice of a wider buffer (pixel stride ldy).
// Column-major view: D[K, M] = op(W)[K, C] . X[C, M] (+ C[K, M]) with M = N * H * W.
// Plans (descriptors + heuristic algo, no timing) are cached per (M, K, C, ldx, ldy, ldr, epilogue).
// Small GEMMs (M * K * C < PW_MIN_MACS) or other activations go to CUTLASS clf::conv_fused (decision at run time, so a
// TorchScript trace stays shape-generic).
struct LtPlan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
    cublasLtMatmulAlgo_t algo;
    bool ok = false;
};
struct LtKey {
    long long M; int K, C, ldx, ldy, ldr, epi;
    bool operator==(const LtKey& o) const {
        return M == o.M && K == o.K && C == o.C && ldx == o.ldx && ldy == o.ldy && ldr == o.ldr && epi == o.epi;
    }
};
struct LtKeyHash {
    size_t operator()(const LtKey& k) const {
        size_t h = std::hash<long long>()(k.M);
        for (int v : {k.K, k.C, k.ldx, k.ldy, k.ldr, k.epi}) h = h * 1000003u ^ std::hash<int>()(v);
        return h;
    }
};
static std::unordered_map<LtKey, LtPlan, LtKeyHash> g_plans;
static std::mutex g_plans_mu;
static const long long PW_MIN_MACS = [] { const char* v = getenv("CLF_PW_MIN_MACS"); return v ? atoll(v) : 150000000ll; }();

static LtPlan& lt_plan(const LtKey& k, cublasLtHandle_t h) {
    std::lock_guard<std::mutex> g(g_plans_mu);
    auto it = g_plans.find(k);
    if (it != g_plans.end()) return it->second;
    LtPlan& p = g_plans[k];
    bool ok = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_16F, CUDA_R_16F) == CUBLAS_STATUS_SUCCESS;
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    ok = ok && cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)) == CUBLAS_STATUS_SUCCESS;
    cublasLtEpilogue_t ep = k.epi == 0 ? CUBLASLT_EPILOGUE_DEFAULT : k.epi == 1 ? CUBLASLT_EPILOGUE_RELU
                          : k.epi == 2 ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_RELU_BIAS;
    ok = ok && cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep)) == CUBLAS_STATUS_SUCCESS;
    if (k.epi >= 2) {
        cudaDataType_t bt = CUDA_R_16F;
        ok = ok && cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)) == CUBLAS_STATUS_SUCCESS;
    }
    ok = ok && cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16F, k.C, k.K, k.C) == CUBLAS_STATUS_SUCCESS;     // W^T as [C, K]
    ok = ok && cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16F, k.C, k.M, k.ldx) == CUBLAS_STATUS_SUCCESS;   // X^T [C, M]
    ok = ok && cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16F, k.K, k.M, k.ldr > 0 ? k.ldr : k.ldy) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatrixLayoutCreate(&p.d, CUDA_R_16F, k.K, k.M, k.ldy) == CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulPreference_t pref = nullptr;
    size_t ws = at::cuda::getCUDABlasLtWorkspaceSize();
    ok = ok && cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS;
    ok = ok && cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)) == CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulHeuristicResult_t res[16];
    int nres = 0;
    static const int pick = [] { const char* v = getenv("CLF_PW_ALGO"); return v ? atoi(v) : 0; }();   // dev: i-th heuristic
    ok = ok && cublasLtMatmulAlgoGetHeuristic(h, p.op, p.a, p.b, p.c, p.d, pref, pick + 1, res, &nres) == CUBLAS_STATUS_SUCCESS && nres > 0;
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (ok) p.algo = res[std::min(pick, nres - 1)].algo;
    p.ok = ok;
    return p;
}

static at::Tensor conv_fused_call(const at::Tensor& x, const at::Tensor& w, const c10::optional<at::Tensor>& b,
                                  const c10::optional<at::Tensor>& r, int64_t act) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("clf::conv_fused", "")
        .typed<at::Tensor(at::Tensor, at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                          std::vector<int64_t>, std::vector<int64_t>, int64_t, int64_t)>();
    std::vector<int64_t> z{0, 0}, o{1, 1};
    return op.call(x, w, b, r, z, o, act, -1);
}
static void conv_fused_into_call(const at::Tensor& x, const at::Tensor& w, const c10::optional<at::Tensor>& b,
                                 const c10::optional<at::Tensor>& r, int64_t act, const at::Tensor& out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("clf::conv_fused_into", "")
        .typed<void(at::Tensor, at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                    std::vector<int64_t>, std::vector<int64_t>, int64_t, at::Tensor)>();
    std::vector<int64_t> z{0, 0}, o{1, 1};
    op.call(x, w, b, r, z, o, act, out);
}

static bool cl_slice(const at::Tensor& t, int& ld) {     // channels-last 4-D tensor, possibly a channel slice
    long long l3 = t.stride(3);
    bool ok = t.dim() == 4 && t.stride(1) == 1 && t.stride(2) == l3 * t.size(3) && t.stride(0) == l3 * t.size(3) * t.size(2) &&
              l3 % 8 == 0 && ((uintptr_t)t.data_ptr()) % 16 == 0 && t.scalar_type() == at::kHalf;
    ld = (int)l3;
    return ok;
}

static at::Tensor pw_impl(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> res,
                          int64_t act, c10::optional<at::Tensor> out, int64_t force) {
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3), K = w.size(0);
    long long M = (long long)N * H * W;
    int ldx = 0, ldy = 0, ldr = 0;
    // rule (bench/pw_test.py sweep): cublasLt wins 1.1-2x for narrow GEMMs (min(C, K) <= 256) without residual once
    // M*K*C >= PW_MIN_MACS; with a residual (beta * C) or wide GEMMs (1232^2: worse algo with the bias epilogue) CUTLASS wins
    bool use = (act == 0 || act == 1) && C % 8 == 0 && K % 8 == 0 && cl_slice(x, ldx) &&
               (force > 0 || (force == 0 && !res && std::min(C, K) <= 256 && M * K * (long long)C >= PW_MIN_MACS));
    if (use && res) use = res->is_contiguous(at::MemoryFormat::ChannelsLast) && cl_slice(*res, ldr);
    if (use && out) use = cl_slice(*out, ldy);
    if (!use) {
        if (out) { conv_fused_into_call(x, w, bias, res, act, *out); return *out; }
        return conv_fused_call(x, w, bias, res, act);
    }
    at::Tensor y = out ? *out : at::empty({N, K, H, W}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    if (!out) ldy = K;
    int epi = (bias ? 2 : 0) + (act == 1 ? 1 : 0);
    cublasLtHandle_t h = at::cuda::getCurrentCUDABlasLtHandle();
    LtPlan& p = lt_plan(LtKey{M, K, C, ldx, ldy, res ? ldr : 0, epi}, h);
    if (!p.ok) {
        if (out) { conv_fused_into_call(x, w, bias, res, act, *out); return *out; }
        return conv_fused_call(x, w, bias, res, act);
    }
    if (bias) {
        const void* bp = bias->data_ptr();
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp));
    }
    __half one = __float2half(1.f), beta = __float2half(res ? 1.f : 0.f);
    auto st = cublasLtMatmul(h, p.op, &one, w.data_ptr(), p.a, x.data_ptr(), p.b, &beta, res ? res->data_ptr() : y.data_ptr(),
                             p.c, y.data_ptr(), p.d, &p.algo, at::cuda::getCUDABlasLtWorkspace(),
                             at::cuda::getCUDABlasLtWorkspaceSize(), at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "pw: cublasLtMatmul failed ", (int)st);
    return y;
}

at::Tensor pw(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> res, int64_t act,
              int64_t force) {
    return pw_impl(x, w, bias, res, act, c10::nullopt, force);
}
void pw_into(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> res, int64_t act,
             at::Tensor out, int64_t force) {
    pw_impl(x, w, bias, res, act, out, force);
}

// cat2: torch.cat([conv1x1(x), convKxK(x, pad K/2)], 1) of two convs reading the same x (SqueezeNet Fire expand).
// Small problems (M = N*H*W <= mmax: launch-bound, batch 1): ONE conv with the 1x1 zero-embedded at the centre of the KxK
// (wm, bm); large: two kernels writing their channel slices of one buffer (pw for the 1x1, CUTLASS for the KxK).
at::Tensor cat2(at::Tensor x, at::Tensor wm, at::Tensor bm, at::Tensor w1, c10::optional<at::Tensor> b1, at::Tensor w3,
                c10::optional<at::Tensor> b3, int64_t act, int64_t mmax) {
    long long M = (long long)x.size(0) * x.size(2) * x.size(3);
    int R = w3.size(2), C = x.size(1);
    std::vector<int64_t> pd{R / 2, R / 2}, o{1, 1};
    const bool tc = C % 16 == 0 && C <= TC_MAX_C;     // small-C KxK convs: gconv2 kernel beats CUTLASS (convtc_test.py)
    if (M <= mmax) {
        if (tc && wm.size(0) % 16 == 0) return gconv2_impl(x, wm, bm, o, pd, 1, act, nullptr, -1);
        static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("clf::conv_fused", "")
            .typed<at::Tensor(at::Tensor, at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                              std::vector<int64_t>, std::vector<int64_t>, int64_t, int64_t)>();
        return op.call(x, wm, bm, c10::nullopt, pd, o, act, -1);
    }
    int K1 = w1.size(0), K3 = w3.size(0);
    auto y = at::empty({x.size(0), K1 + K3, x.size(2), x.size(3)}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    pw_impl(x, w1, b1, c10::nullopt, act, y.narrow(1, 0, K1), 0);
    static auto opi = c10::Dispatcher::singleton().findSchemaOrThrow("clf::conv_fused_into", "")
        .typed<void(at::Tensor, at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
                    std::vector<int64_t>, std::vector<int64_t>, int64_t, at::Tensor)>();
    if (tc && K3 % 16 == 0) gconv2_impl(x, w3, b3, o, pd, 1, act, nullptr, -1, y.narrow(1, K1, K3));
    else opi.call(x, w3, b3, c10::nullopt, pd, o, act, y.narrow(1, K1, K3));
    return y;
}

}  // namespace

TORCH_LIBRARY(clfr3, m) {
    m.def("pw(Tensor x, Tensor w, Tensor? bias, Tensor? res, int act, int force=0) -> Tensor");
    m.def("pw_into(Tensor x, Tensor w, Tensor? bias, Tensor? res, int act, Tensor(a!) out, int force=0) -> ()");
    m.def("cat2(Tensor x, Tensor wm, Tensor bm, Tensor w1, Tensor? b1, Tensor w3, Tensor? b3, int act, int mmax) -> Tensor");
    m.def("gconv2(Tensor x, Tensor w, Tensor? bias, int[] stride, int[] pad, int groups, int act, int cfg=-1) -> Tensor");
    m.def("gconv2_pool(Tensor x, Tensor w, Tensor? bias, int[] stride, int[] pad, int groups, int act) -> (Tensor, Tensor)");
    m.def("conv_tc(Tensor x, Tensor w, Tensor? bias, int[] stride, int[] pad, int act, int cfg=-1) -> Tensor");
    m.def("conv_tc_into(Tensor x, Tensor w, Tensor? bias, int[] stride, int[] pad, int act, Tensor(a!) out) -> ()");
    m.def("dwconv2(Tensor x, Tensor w_rsc, Tensor? bias, int[] stride, int[] padding, int act) -> Tensor");
    m.def("se_scale(Tensor x, Tensor s) -> Tensor");
    m.def("stem_pool(Tensor x, Tensor w_t, Tensor? bias, int[] stride, int[] pad, int act, int pk, int ps, int pp, bool ceil) -> Tensor");
}
TORCH_LIBRARY_IMPL(clfr3, CUDA, m) { m.impl("stem_pool", stem_pool); m.impl("pw", pw); m.impl("pw_into", pw_into); m.impl("cat2", cat2); m.impl("gconv2", gconv2); m.impl("gconv2_pool", gconv2_pool); m.impl("se_scale", se_scale); m.impl("conv_tc", conv_tc); m.impl("conv_tc_into", conv_tc_into); m.impl("dwconv2", dwconv2); }

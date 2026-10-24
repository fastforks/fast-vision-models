// exp 008: extra fused elementwise / row ops for transformer classifiers (namespace clfx; built by clf_fast/extra.py).
//  rope_qk(qkv, emb, npt, H)   : qkv [B, N, 3*H*D] fp16 (fused GEMM output) -> q, k [B, N, H, D] with rotary embedding
//                                (interleaved pairs, timm apply_rot_embed_cat half=False; first npt tokens unrotated)
//  swiglu_ln(gx, w, b, hid, eps): gx [M, 2*Hp] = [gate | x] (each padded to Hp >= hid) -> LN(silu(gate) * x) over the first
//                                hid channels, [M, Hp] fp16 with zero padding (timm SwiGLU with norm, EVA-02)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void rope_qk_k(const __half* __restrict__ qkv, const float* __restrict__ emb, __half* __restrict__ q,
                          __half* __restrict__ k, int N, int H, int D, int npt, long long total_pairs) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;   // one (token, head, pair) per thread, q and k
    if (i >= total_pairs) return;
    int D2 = D / 2;
    int j = i % D2; long long t = i / D2; int h = t % H; long long tok = t / H;   // tok = b * N + n
    int n = tok % N;
    long long in_q = tok * 3 * H * D + h * D + 2 * j, in_k = in_q + H * D;
    long long o = (tok * H + h) * D + 2 * j;
    __half2 qv = *reinterpret_cast<const __half2*>(qkv + in_q);
    __half2 kv = *reinterpret_cast<const __half2*>(qkv + in_k);
    if (n < npt) {
        *reinterpret_cast<__half2*>(q + o) = qv;
        *reinterpret_cast<__half2*>(k + o) = kv;
        return;
    }
    const float* e = emb + (long long)(n - npt) * 2 * D;
    float s0 = e[2 * j], s1 = e[2 * j + 1], c0 = e[D + 2 * j], c1 = e[D + 2 * j + 1];
    float2 a = __half22float2(qv), b = __half22float2(kv);
    *reinterpret_cast<__half2*>(q + o) = __floats2half2_rn(a.x * c0 - a.y * s0, a.y * c1 + a.x * s1);
    *reinterpret_cast<__half2*>(k + o) = __floats2half2_rn(b.x * c0 - b.y * s0, b.y * c1 + b.x * s1);
}

std::vector<at::Tensor> rope_qk(at::Tensor qkv, at::Tensor emb, int64_t npt, int64_t H) {
    TORCH_CHECK(qkv.is_contiguous() && qkv.scalar_type() == at::kHalf, "rope_qk: contiguous fp16 qkv");
    int B = qkv.size(0), N = qkv.size(1), D = qkv.size(2) / 3 / H;
    auto e = emb.to(at::kFloat).contiguous();
    TORCH_CHECK(e.size(0) == N - npt && e.size(1) == 2 * D, "rope_qk: emb shape ", e.sizes(), " vs N-npt=", N - npt);
    auto q = at::empty({B, N, H, D}, qkv.options()), k = at::empty({B, N, H, D}, qkv.options());
    long long total = (long long)B * N * H * (D / 2);
    rope_qk_k<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)qkv.data_ptr(), e.data_ptr<float>(), (__half*)q.data_ptr(), (__half*)k.data_ptr(), N, H, D, (int)npt, total);
    return {q, k};
}

// one block (256 threads) per row
__global__ void swiglu_ln_k(const __half* __restrict__ gx, const __half* __restrict__ w, const __half* __restrict__ b,
                            __half* __restrict__ y, int hid, int Hp, float eps) {
    extern __shared__ float hs[];   // hid floats
    __shared__ float red[2][32];
    long long row = blockIdx.x;
    const __half* g = gx + row * 2 * Hp;
    const __half* x = g + Hp;
    float s = 0.f, s2 = 0.f;
    for (int c = threadIdx.x; c < hid; c += blockDim.x) {
        float gv = __half2float(g[c]), xv = __half2float(x[c]);
        float h = __half2float(__float2half(gv / (1.f + __expf(-gv)))) * xv;   // fp16 silu output as in eager
        h = __half2float(__float2half(h));
        hs[c] = h; s += h; s2 += h * h;
    }
    for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(0xffffffff, s, o); s2 += __shfl_xor_sync(0xffffffff, s2, o); }
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) { red[0][wid] = s; red[1][wid] = s2; }
    __syncthreads();
    if (wid == 0) {
        s = lane < blockDim.x / 32 ? red[0][lane] : 0.f;
        s2 = lane < blockDim.x / 32 ? red[1][lane] : 0.f;
        for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(0xffffffff, s, o); s2 += __shfl_xor_sync(0xffffffff, s2, o); }
        if (lane == 0) { red[0][0] = s; red[1][0] = s2; }
    }
    __syncthreads();
    float mean = red[0][0] / hid, var = fmaxf(red[1][0] / hid - mean * mean, 0.f), rstd = rsqrtf(var + eps);
    __half* yr = y + row * Hp;
    for (int c = threadIdx.x; c < Hp; c += blockDim.x)
        yr[c] = c < hid ? __float2half((hs[c] - mean) * rstd * __half2float(w[c]) + __half2float(b[c])) : __float2half(0.f);
}

at::Tensor swiglu_ln(at::Tensor gx, at::Tensor w, at::Tensor b, int64_t hid, double eps) {
    TORCH_CHECK(gx.is_contiguous() && gx.scalar_type() == at::kHalf, "swiglu_ln: contiguous fp16");
    int Hp = gx.size(-1) / 2;
    long long M = gx.numel() / gx.size(-1);
    auto sizes = gx.sizes().vec();
    sizes.back() = Hp;
    auto y = at::empty(sizes, gx.options());
    swiglu_ln_k<<<M, 256, hid * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)gx.data_ptr(), (const __half*)w.data_ptr(), (const __half*)b.data_ptr(), (__half*)y.data_ptr(),
        (int)hid, Hp, (float)eps);
    return y;
}

TORCH_LIBRARY(clfx, m) {
    m.def("rope_qk(Tensor qkv, Tensor emb, int npt, int H) -> Tensor[]");
    m.def("swiglu_ln(Tensor gx, Tensor w, Tensor b, int hid, float eps) -> Tensor");
}
TORCH_LIBRARY_IMPL(clfx, CUDA, m) { m.impl("rope_qk", rope_qk); m.impl("swiglu_ln", swiglu_ln); }

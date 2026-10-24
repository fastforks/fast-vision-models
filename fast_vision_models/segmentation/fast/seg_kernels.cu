// exp 009 custom kernels (compiled once into a shipped fatbin; size-independent, no per-shape compilation)
#include <cuda_fp16.h>
#include <stdint.h>

// Fused bilinear upsample (align_corners=False, PyTorch index math) + argmax over channels.
// in : fp16 scores, NHWC with row pitch (pixels) `pitch` and image pitch `ipitch` (pixels); only the top-left hin x win
//      region is used (lets callers pass the scores of a padded input and crop for free). C channels contiguous, C % 2 == 0.
// out: int32 [B, H, W] class index (first maximum wins, like torch.argmax).
// sh, sw: hin / H, win / W (the scale PyTorch uses when an output size is given); align = 1: align_corners=True
// (sh = (hin - 1) / (H - 1), PyTorch's area_pixel_compute_scale).
extern "C" __global__ void up_argmax_h(const __half* __restrict__ in, int32_t* __restrict__ out, int B, int hin, int win,
                                       long long ipitch, int pitch, int C, int H, int W, float sh, float sw, int align) {
    long long n = (long long)B * H * W;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int x = (int)(i % W);
        long long t = i / W;
        int y = (int)(t % H);
        int b = (int)(t / H);
        float fy = align ? y * sh : fmaxf((y + 0.5f) * sh - 0.5f, 0.f);   // align_corners: sh = (hin-1)/(H-1)
        float fx = align ? x * sw : fmaxf((x + 0.5f) * sw - 0.5f, 0.f);
        int y0 = min((int)fy, hin - 1), x0 = min((int)fx, win - 1);
        int y1 = y0 + (y0 < hin - 1), x1 = x0 + (x0 < win - 1);
        float ly = fy - y0, lx = fx - x0;
        float hy = 1.f - ly, hx = 1.f - lx;
        const __half* base = in + b * ipitch * C;
        const __half2* p00 = (const __half2*)(base + ((long long)y0 * pitch + x0) * C);
        const __half2* p01 = (const __half2*)(base + ((long long)y0 * pitch + x1) * C);
        const __half2* p10 = (const __half2*)(base + ((long long)y1 * pitch + x0) * C);
        const __half2* p11 = (const __half2*)(base + ((long long)y1 * pitch + x1) * C);
        float best = -INFINITY;
        int arg = 0;
        for (int c = 0; c < C / 2; ++c) {
            float2 a = __half22float2(p00[c]), bb = __half22float2(p01[c]);
            float2 cc = __half22float2(p10[c]), d = __half22float2(p11[c]);
            float v0 = hy * (hx * a.x + lx * bb.x) + ly * (hx * cc.x + lx * d.x);
            float v1 = hy * (hx * a.y + lx * bb.y) + ly * (hx * cc.y + lx * d.y);
            if (v0 > best) { best = v0; arg = 2 * c; }
            if (v1 > best) { best = v1; arg = 2 * c + 1; }
        }
        out[i] = arg;
    }
}

// Multi-scale deformable attention (Deformable DETR / Mask2Former pixel decoder encoder, self-attention: queries = the
// flattened multi-level tokens, no padding mask -> reference point of a query = its own pixel centre, normalized).
// value: fp16 [B, S, NH, D] (D even); off: fp16 [B, S, NH, L, P, 2] raw offsets (pixels of the target level);
// logit: fp16 [B, S, NH, L*P] pre-softmax weights; shapes: int32 [L, 2] (h, w); start: int32 [L] level start index.
// out: fp16 [B, S, NH, D]. Softmax, locations and bilinear weights in fp32; grid_sample semantics (align_corners=False,
// zero padding): x = loc_x * w - 0.5. One thread per (b, s, h, d/2).
extern "C" __global__ void msda_self_h(const __half* __restrict__ value, const __half* __restrict__ off,
                                       const __half* __restrict__ logit, const int* __restrict__ shapes,
                                       const int* __restrict__ start, __half* __restrict__ out,
                                       int B, int S, int NH, int D, int L, int P) {
    int D2 = D / 2;
    long long n = (long long)B * S * NH * D2;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int d2 = (int)(i % D2);
        long long t = i / D2;
        int h = (int)(t % NH);
        t /= NH;
        int s = (int)(t % S);
        int b = (int)(t / S);
        // reference point of query s
        int lq = 0;
        for (int l = 1; l < L; ++l) if (s >= start[l]) lq = l;
        int qi = s - start[lq], qw = shapes[2 * lq + 1], qh = shapes[2 * lq];
        float rx = ((qi % qw) + 0.5f) / qw, ry = ((qi / qw) + 0.5f) / qh;
        long long qbase = ((long long)b * S + s) * NH + h;
        const __half* lg = logit + qbase * L * P;
        const __half* of = off + qbase * L * P * 2;
        float mx = -INFINITY;
        for (int k = 0; k < L * P; ++k) mx = fmaxf(mx, __half2float(lg[k]));
        float den = 0.f;
        for (int k = 0; k < L * P; ++k) den += __expf(__half2float(lg[k]) - mx);
        float inv = 1.f / den;
        float2 acc = make_float2(0.f, 0.f);
        for (int l = 0; l < L; ++l) {
            int hl = shapes[2 * l], wl = shapes[2 * l + 1];
            const __half2* vb = (const __half2*)(value + (((long long)b * S + start[l]) * NH + h) * D) + d2;
            long long rowp = (long long)wl * NH * D2;   // half2 stride of one row of this level
            long long colp = (long long)NH * D2;
            for (int p = 0; p < P; ++p) {
                int k = l * P + p;
                float a = __expf(__half2float(lg[k]) - mx) * inv;
                float x = (rx + __half2float(of[2 * k]) / wl) * wl - 0.5f;
                float y = (ry + __half2float(of[2 * k + 1]) / hl) * hl - 0.5f;
                int x0 = (int)floorf(x), y0 = (int)floorf(y);
                float lx = x - x0, ly = y - y0;
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    int xx = x0 + (c & 1), yy = y0 + (c >> 1);
                    if (xx < 0 || yy < 0 || xx >= wl || yy >= hl) continue;
                    float wgt = a * ((c & 1) ? lx : 1.f - lx) * ((c >> 1) ? ly : 1.f - ly);
                    float2 v = __half22float2(vb[yy * rowp + xx * colp]);
                    acc.x += wgt * v.x;
                    acc.y += wgt * v.y;
                }
            }
        }
        ((__half2*)(out + qbase * D))[d2] = __floats2half2_rn(acc.x, acc.y);
    }
}

// Conv epilogue (the cuDNN fp16-accumulate convs have no fused epilogue on sm_120 without runtime compilation):
// y = act(y + bias[c] (+ res)) in place, NHWC contiguous (channel = fastest index), C even, half2 per thread, math in fp32.
// bias / res may be null; relu: 0 = none, 1 = ReLU.
extern "C" __global__ void bias_act_h(__half* __restrict__ y, const __half* __restrict__ bias, const __half* __restrict__ res,
                                      long long n2, int C2, int relu) {
    __half2* y2 = (__half2*)y;
    const __half2* b2 = (const __half2*)bias;
    const __half2* r2 = (const __half2*)res;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n2; i += (long long)gridDim.x * blockDim.x) {
        float2 v = __half22float2(y2[i]);
        if (b2) { float2 b = __half22float2(b2[i % C2]); v.x += b.x; v.y += b.y; }
        if (r2) { float2 r = __half22float2(r2[i]); v.x += r.x; v.y += r.y; }
        if (relu) { v.x = fmaxf(v.x, 0.f); v.y = fmaxf(v.y, 0.f); }
        y2[i] = __floats2half2_rn(v.x, v.y);
    }
}

// Depthwise conv, NHWC fp16 in / out, any kernel size / stride / padding / dilation (cuDNN's channels-last fp16
// depthwise kernels are 4-5x slower than bandwidth on sm_120). w: fp16 [kh*kw, C] (tap-major, channels contiguous),
// bias fp16 [C] or null, fp32 accumulation. One thread per (n, ho, wo, channel pair).
extern "C" __global__ void dwconv_h(const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ bias,
                                    __half* __restrict__ y, int N, int H, int W, int C, int Ho, int Wo, int kh, int kw,
                                    int sh, int sw, int ph, int pw, int dh, int dw, int act) {
    int C2 = C / 2;
    long long n = (long long)N * Ho * Wo * C2;
    const __half2* x2 = (const __half2*)x;
    const __half2* w2 = (const __half2*)w;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int c = (int)(i % C2);
        long long t = i / C2;
        int wo = (int)(t % Wo);
        t /= Wo;
        int ho = (int)(t % Ho);
        int b = (int)(t / Ho);
        float2 acc = bias ? __half22float2(((const __half2*)bias)[c]) : make_float2(0.f, 0.f);
        const __half2* xb = x2 + (long long)b * H * W * C2 + c;
        for (int ky = 0; ky < kh; ++ky) {
            int yy = ho * sh - ph + ky * dh;
            if (yy < 0 || yy >= H) continue;
            for (int kx = 0; kx < kw; ++kx) {
                int xx = wo * sw - pw + kx * dw;
                if (xx < 0 || xx >= W) continue;
                float2 v = __half22float2(xb[((long long)yy * W + xx) * C2]);
                float2 k = __half22float2(w2[(ky * kw + kx) * C2 + c]);
                acc.x = fmaf(v.x, k.x, acc.x);
                acc.y = fmaf(v.y, k.y, acc.y);
            }
        }
        if (act == 1) {   // exact (erf) GELU, fp32 (round 2: MSCAN Mlp dwconv -> GELU fused)
            acc.x = 0.5f * acc.x * (1.f + erff(acc.x * 0.70710678118654752f));
            acc.y = 0.5f * acc.y * (1.f + erff(acc.y * 0.70710678118654752f));
        }
        ((__half2*)y)[i] = __floats2half2_rn(acc.x, acc.y);
    }
}

// Bilinear resize, NHWC fp16 in / out (C even), PyTorch index math (area_pixel_compute_source_index; sh / sw = the
// scale PyTorch uses: (in-1)/(out-1) for align_corners, 1/scale_factor when given, else in/out). One thread per
// (b, y, x, channel pair); PyTorch's channels-last half kernel is ~4-5x slower than bandwidth on sm_120.
extern "C" __global__ void upsample_bilinear_h(const __half* __restrict__ in, __half* __restrict__ out, int B, int hin,
                                               int win, int C, int H, int W, float sh, float sw, int align) {
    int C2 = C / 2;
    long long n = (long long)B * H * W * C2;
    const __half2* in2 = (const __half2*)in;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int c = (int)(i % C2);
        long long t = i / C2;
        int x = (int)(t % W);
        t /= W;
        int y = (int)(t % H);
        int b = (int)(t / H);
        float fy = align ? y * sh : fmaxf((y + 0.5f) * sh - 0.5f, 0.f);
        float fx = align ? x * sw : fmaxf((x + 0.5f) * sw - 0.5f, 0.f);
        int y0 = min((int)fy, hin - 1), x0 = min((int)fx, win - 1);
        int y1 = y0 + (y0 < hin - 1), x1 = x0 + (x0 < win - 1);
        float ly = fy - y0, lx = fx - x0;
        float hy = 1.f - ly, hx = 1.f - lx;
        const __half2* base = in2 + (long long)b * hin * win * C2 + c;
        float2 a = __half22float2(base[((long long)y0 * win + x0) * C2]), bb = __half22float2(base[((long long)y0 * win + x1) * C2]);
        float2 cc = __half22float2(base[((long long)y1 * win + x0) * C2]), d = __half22float2(base[((long long)y1 * win + x1) * C2]);
        float vx = hy * (hx * a.x + lx * bb.x) + ly * (hx * cc.x + lx * d.x);
        float vy = hy * (hx * a.y + lx * bb.y) + ly * (hx * cc.y + lx * d.y);
        ((__half2*)out)[i] = __floats2half2_rn(vx, vy);
    }
}

// ---- round 2: Swin layer fusion (seg_fast/swin.py). One warp per row, C even, fp32 statistics (two-pass mean / var
// like PyTorch's LayerNorm up to summation order).
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void ln_row(const __half2* __restrict__ xr, const __half2* __restrict__ w2,
                                       const __half2* __restrict__ b2, __half2* __restrict__ o, int C, float eps, int lane) {
    int C2 = C / 2;
    float s = 0.f;
    for (int c = lane; c < C2; c += 32) { float2 v = __half22float2(xr[c]); s += v.x + v.y; }
    float mean = warp_sum(s) / C;
    float q = 0.f;
    for (int c = lane; c < C2; c += 32) { float2 v = __half22float2(xr[c]); v.x -= mean; v.y -= mean; q += v.x * v.x + v.y * v.y; }
    float rstd = rsqrtf(warp_sum(q) / C + eps);
    for (int c = lane; c < C2; c += 32) {
        float2 v = __half22float2(xr[c]), ww = __half22float2(w2[c]), bb = __half22float2(b2[c]);
        o[c] = __floats2half2_rn((v.x - mean) * rstd * ww.x + bb.x, (v.y - mean) * rstd * ww.y + bb.y);
    }
}

// out[r] = idx[r] >= 0 ? LayerNorm(x[idx[r]]) : 0    (Swin: LN + pad + roll + window partition in one pass)
extern "C" __global__ void ln_gather_h(const __half* __restrict__ x, const int* __restrict__ idx, const __half* __restrict__ w,
                                       const __half* __restrict__ b, __half* __restrict__ out, int R, int C, float eps) {
    int lane = threadIdx.x & 31;
    int nw = (gridDim.x * blockDim.x) >> 5;
    for (int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; r < R; r += nw) {
        int s = idx[r];
        __half2* o = (__half2*)(out + (long long)r * C);
        if (s < 0) {
            for (int c = lane; c < C / 2; c += 32) o[c] = __float2half2_rn(0.f);
            continue;
        }
        ln_row((const __half2*)(x + (long long)s * C), (const __half2*)w, (const __half2*)b, o, C, eps, lane);
    }
}

// h[t] = res[t] + o[inv[t]] (rounded to fp16 like the eager add), y[t] = LayerNorm(h[t])
// (Swin: window reverse + roll back + crop + residual + layernorm_after in one pass)
extern "C" __global__ void gather_add_ln_h(const __half* __restrict__ o, const int* __restrict__ inv, const __half* __restrict__ res,
                                           const __half* __restrict__ w, const __half* __restrict__ b, __half* __restrict__ h,
                                           __half* __restrict__ y, int N, int C, float eps) {
    int lane = threadIdx.x & 31;
    int nw = (gridDim.x * blockDim.x) >> 5;
    int C2 = C / 2;
    for (int t = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; t < N; t += nw) {
        const __half2* src = (const __half2*)(o + (long long)inv[t] * C);
        const __half2* rr = (const __half2*)(res + (long long)t * C);
        __half2* hr = (__half2*)(h + (long long)t * C);
        for (int c = lane; c < C2; c += 32) hr[c] = __hadd2(rr[c], src[c]);
        __syncwarp();
        ln_row(hr, (const __half2*)w, (const __half2*)b, (__half2*)(y + (long long)t * C), C, eps, lane);
    }
}

// ---- round 2: SegNeXt MSCA strip convolutions (seg_fast/segnext.py). NHWC fp16, C even, depthwise 1-D convs along
// one axis (axis 0: along W ("1 x k"), 1: along H ("k x 1")), zero padding k / 2, odd k <= 21, fp32 accumulation.
// Register blocking: each thread computes SEG consecutive outputs along the axis for one channel pair, loading
// SEG + KMAX - 1 inputs once (instead of SEG * k) and reusing them for up to three kernels.
#define STRIP_SEG 8
#define STRIP_KMAX 21
#define STRIP_NV (STRIP_SEG + STRIP_KMAX - 1)

template <int K>
__device__ __forceinline__ void strip_k(const float2* v, const __half2* __restrict__ wq, int C2, float2* acc) {
    constexpr int OFF = STRIP_KMAX / 2 - K / 2;
#pragma unroll
    for (int j = 0; j < K; ++j) {
        float2 ww = __half22float2(wq[j * C2]);
#pragma unroll
        for (int o = 0; o < STRIP_SEG; ++o) {
            acc[o].x = fmaf(v[o + OFF + j].x, ww.x, acc[o].x);
            acc[o].y = fmaf(v[o + OFF + j].y, ww.y, acc[o].y);
        }
    }
}

__device__ __forceinline__ void strip_any(int k, const float2* v, const __half2* __restrict__ wq, int C2, float2* acc) {
    switch (k) {
        case 1: strip_k<1>(v, wq, C2, acc); break;
        case 3: strip_k<3>(v, wq, C2, acc); break;
        case 5: strip_k<5>(v, wq, C2, acc); break;
        case 7: strip_k<7>(v, wq, C2, acc); break;
        case 9: strip_k<9>(v, wq, C2, acc); break;
        case 11: strip_k<11>(v, wq, C2, acc); break;
        case 13: strip_k<13>(v, wq, C2, acc); break;
        case 15: strip_k<15>(v, wq, C2, acc); break;
        case 17: strip_k<17>(v, wq, C2, acc); break;
        case 19: strip_k<19>(v, wq, C2, acc); break;
        default: strip_k<21>(v, wq, C2, acc); break;
    }
}

// Horizontal / vertical strip conv of up to three inputs x_q with kernels w_q (k_q = 0: unused).
// mode 0 ("split"): y_q = conv_q(x_0) (+ b_q)        -- one input read, three outputs (MSCA conv*_1)
// mode 1 ("sum")  : y_0 = base + sum_q conv_q(x_q) (+ b_q)   -- three inputs, one output (MSCA conv*_2 + the adds)
// w_q: fp16 [k_q, C] tap-major; b_q fp16 [C] or null; base NHWC or null.
extern "C" __global__ void strip3_h(const __half* __restrict__ x0, const __half* __restrict__ x1, const __half* __restrict__ x2,
                                    const __half* __restrict__ w0, const __half* __restrict__ w1, const __half* __restrict__ w2,
                                    const __half* __restrict__ b0, const __half* __restrict__ b1, const __half* __restrict__ b2,
                                    const __half* __restrict__ base, __half* __restrict__ y0, __half* __restrict__ y1,
                                    __half* __restrict__ y2, int N, int H, int W, int C, int k0, int k1, int k2, int axis,
                                    int mode) {
    const int C2 = C / 2;
    const int L = axis ? H : W;
    const long long step = axis ? (long long)W * C2 : C2;
    const int nseg = (L + STRIP_SEG - 1) / STRIP_SEG;
    const int other = axis ? W : H;
    long long n = (long long)N * other * nseg * C2;
    const __half* xs[3] = {x0, x1, x2};
    const __half* ws[3] = {w0, w1, w2};
    const __half* bs[3] = {b0, b1, b2};
    __half* ys[3] = {y0, y1, y2};
    const int ks[3] = {k0, k1, k2};
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int c = (int)(i % C2);
        long long t = i / C2;
        int seg = (int)(t % nseg);
        t /= nseg;
        int o2 = (int)(t % other);
        int b = (int)(t / other);
        int p0 = seg * STRIP_SEG;
        long long rb = axis ? ((long long)b * H * W + o2) * C2 + c : ((long long)b * H * W + (long long)o2 * W) * C2 + c;
        float2 v[STRIP_NV];
        float2 sum[STRIP_SEG];
        if (mode == 1) {
#pragma unroll
            for (int o = 0; o < STRIP_SEG; ++o)
                sum[o] = (base && p0 + o < L) ? __half22float2(((const __half2*)base)[rb + (p0 + o) * step]) : make_float2(0.f, 0.f);
        }
#pragma unroll 1
        for (int q = 0; q < 3; ++q) {
            int k = ks[q];
            if (!k) continue;
            if (q == 0 || mode == 1) {
                const __half2* xq = (const __half2*)xs[mode == 1 ? q : 0];
#pragma unroll
                for (int j = 0; j < STRIP_NV; ++j) {
                    int p = p0 - STRIP_KMAX / 2 + j;
                    v[j] = (p >= 0 && p < L) ? __half22float2(xq[rb + p * step]) : make_float2(0.f, 0.f);
                }
            }
            float2 acc[STRIP_SEG];
            float2 bb = bs[q] ? __half22float2(((const __half2*)bs[q])[c]) : make_float2(0.f, 0.f);
#pragma unroll
            for (int o = 0; o < STRIP_SEG; ++o) acc[o] = bb;
            strip_any(k, v, (const __half2*)ws[q] + c, C2, acc);
            if (mode == 1) {
#pragma unroll
                for (int o = 0; o < STRIP_SEG; ++o) { sum[o].x += acc[o].x; sum[o].y += acc[o].y; }
            } else {
                __half2* yq = (__half2*)ys[q];
#pragma unroll
                for (int o = 0; o < STRIP_SEG; ++o)
                    if (p0 + o < L) yq[rb + (p0 + o) * step] = __floats2half2_rn(acc[o].x, acc[o].y);
            }
        }
        if (mode == 1) {
            __half2* yq = (__half2*)y0;
#pragma unroll
            for (int o = 0; o < STRIP_SEG; ++o)
                if (p0 + o < L) yq[rb + (p0 + o) * step] = __floats2half2_rn(sum[o].x, sum[o].y);
        }
    }
}

// out = x * k[c] + a * l[c]   (NHWC fp16, k / l fp32 [C] (l null: 1), math in fp32; round 2: SegNeXt block residuals
// with norm1 folded away and the layer scales applied here instead of being folded into fp16 weights)
extern "C" __global__ void scale_add_h(const __half* __restrict__ a, const __half* __restrict__ x, const float* __restrict__ k,
                                       const float* __restrict__ l, __half* __restrict__ out, long long n2, int C2) {
    const __half2* a2 = (const __half2*)a;
    const __half2* x2 = (const __half2*)x;
    const float2* k2 = (const float2*)k;
    const float2* l2 = (const float2*)l;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n2; i += (long long)gridDim.x * blockDim.x) {
        float2 va = __half22float2(a2[i]), vx = __half22float2(x2[i]), kk = k2[i % C2];
        float2 ll = l2 ? l2[i % C2] : make_float2(1.f, 1.f);
        ((__half2*)out)[i] = __floats2half2_rn(fmaf(vx.x, kk.x, va.x * ll.x), fmaf(vx.y, kk.y, va.y * ll.y));
    }
}

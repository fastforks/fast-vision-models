// exp 009 (2026-10-10, launch-overhead pass): the prebuilt seg_kernels.cu kernels as torch ops (torch.ops.segx.*), so a
// TorchScript trace of the max path can run them from C++ (a raw ctypes launch is invisible to the tracer) and the CPU cost
// per launch drops from ~10 us (Python + ctypes) to ~2-3 us (dispatcher). Every shape-dependent number (output size, scale,
// grid) is computed here at run time -> one trace serves every input size.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "../seg_kernels.cu"

static inline dim3 lin_grid(long long n) {
    long long b = (n + 255) / 256;
    if (b < 1) b = 1;
    if (b > 65535ll * 8) b = 65535ll * 8;
    return dim3((unsigned)b);
}

static inline const __half* hp(const c10::optional<at::Tensor>& t) {
    return t ? (const __half*)t->data_ptr() : nullptr;
}

static inline bool nhwc_ok(const at::Tensor& t) {
    return t.scalar_type() == at::kHalf && t.dim() == 4 && t.is_contiguous(at::MemoryFormat::ChannelsLast);
}

// in place y = relu?(y + bias[c] (+ res)); y channels-last fp16, C even (falls back to aten ops otherwise)
at::Tensor bias_act_(at::Tensor y, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> res, bool relu) {
    int C = y.size(1);
    bool ok = nhwc_ok(y) && C % 2 == 0 && (!res || (res->sizes() == y.sizes() && nhwc_ok(*res)));
    if (!ok) {
        if (bias) y.add_(bias->view({1, -1, 1, 1}));
        if (res) y.add_(*res);
        if (relu) y.relu_();
        return y;
    }
    long long n2 = y.numel() / 2;
    bias_act_h<<<lin_grid(n2), 256, 0, at::cuda::getCurrentCUDAStream()>>>((__half*)y.data_ptr(), hp(bias), hp(res), n2,
                                                                             C / 2, (int)relu);
    return y;
}

// depthwise conv (groups == C), x channels-last fp16, wt fp16 [kh*kw, C] tap-major; act 0 none / 1 GELU(erf)
at::Tensor dwconv(at::Tensor x, at::Tensor wt, c10::optional<at::Tensor> bias, int64_t kh, int64_t kw,
                  std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, int64_t act) {
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int sh = stride[0], sw = stride[1], ph = padding[0], pw = padding[1], dh = dilation[0], dw = dilation[1];
    int Ho = (H + 2 * ph - dh * (kh - 1) - 1) / sh + 1, Wo = (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    auto y = at::empty({N, C, Ho, Wo}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    long long n = (long long)N * Ho * Wo * C / 2;
    dwconv_h<<<lin_grid(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)x.data_ptr(), (const __half*)wt.data_ptr(), hp(bias), (__half*)y.data_ptr(), N, H, W, C, Ho, Wo,
        (int)kh, (int)kw, sh, sw, ph, pw, dh, dw, (int)act);
    return y;
}

// Bilinear resize (+ optional add of a base tensor + ReLU), NHWC fp16, VEC channels per thread (8: 16-byte loads when
// C % 8 == 0, else 2), 32-bit index math (64-bit divisions made the round-2 kernels ALU bound), PyTorch index math
// (sh / sw as in upsample_bilinear_h). The interpolated value is rounded to fp16 before the add (as the unfused path).
template <int VEC>
__global__ void bilin_k(const __half* __restrict__ in, const __half* __restrict__ base, __half* out, int B, int hin, int win,
                        int C, int H, int W, float sh, float sw, int align, int relu) {
    const int CV = C / VEC;
    const int n = B * H * W * CV;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        int c = i % CV;
        int t = i / CV;
        int x = t % W;
        t /= W;
        int y = t % H;
        int b = t / H;
        float fy = align ? y * sh : fmaxf((y + 0.5f) * sh - 0.5f, 0.f);
        float fx = align ? x * sw : fmaxf((x + 0.5f) * sw - 0.5f, 0.f);
        int y0 = min((int)fy, hin - 1), x0 = min((int)fx, win - 1);
        int y1 = y0 + (y0 < hin - 1), x1 = x0 + (x0 < win - 1);
        float ly = fy - y0, lx = fx - x0;
        float hy = 1.f - ly, hx = 1.f - lx;
        const __half* src = in + (size_t)b * hin * win * C + c * VEC;
        __align__(16) __half va[VEC], vb[VEC], vc[VEC], vd[VEC], vo[VEC];
        if constexpr (VEC == 8) {
            *(uint4*)va = __ldg((const uint4*)(src + ((size_t)y0 * win + x0) * C));
            *(uint4*)vb = __ldg((const uint4*)(src + ((size_t)y0 * win + x1) * C));
            *(uint4*)vc = __ldg((const uint4*)(src + ((size_t)y1 * win + x0) * C));
            *(uint4*)vd = __ldg((const uint4*)(src + ((size_t)y1 * win + x1) * C));
        } else {
            *(__half2*)va = *(const __half2*)(src + ((size_t)y0 * win + x0) * C);
            *(__half2*)vb = *(const __half2*)(src + ((size_t)y0 * win + x1) * C);
            *(__half2*)vc = *(const __half2*)(src + ((size_t)y1 * win + x0) * C);
            *(__half2*)vd = *(const __half2*)(src + ((size_t)y1 * win + x1) * C);
        }
        __align__(16) __half vbase[VEC];
        if (base) {
            if constexpr (VEC == 8) *(uint4*)vbase = *(const uint4*)(base + (size_t)i * VEC);
            else *(__half2*)vbase = *(const __half2*)(base + (size_t)i * VEC);
        }
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
            float v = hy * (hx * __half2float(va[k]) + lx * __half2float(vb[k])) +
                      ly * (hx * __half2float(vc[k]) + lx * __half2float(vd[k]));
            if (base) {
                v = __half2float(vbase[k]) + __half2float(__float2half(v));
                if (relu) v = fmaxf(v, 0.f);
            }
            vo[k] = __float2half(v);
        }
        if constexpr (VEC == 8) *(uint4*)(out + (size_t)i * VEC) = *(uint4*)vo;
        else *(__half2*)(out + (size_t)i * VEC) = *(__half2*)vo;
    }
}

static void bilin_launch(const at::Tensor& x, const __half* base, at::Tensor& out, float sh, float sw, bool align, bool relu) {
    int B = x.size(0), C = x.size(1), h = x.size(2), w = x.size(3), H = out.size(2), W = out.size(3);
    auto st = at::cuda::getCurrentCUDAStream();
    if (C % 8 == 0 && ((uintptr_t)x.data_ptr()) % 16 == 0 && ((uintptr_t)out.data_ptr()) % 16 == 0 &&
        (!base || ((uintptr_t)base) % 16 == 0)) {
        long long n = (long long)B * H * W * (C / 8);
        bilin_k<8><<<lin_grid(n), 256, 0, st>>>((const __half*)x.data_ptr(), base, (__half*)out.data_ptr(), B, h, w, C, H, W,
                                                 sh, sw, (int)align, (int)relu);
    } else {
        long long n = (long long)B * H * W * (C / 2);
        bilin_k<2><<<lin_grid(n), 256, 0, st>>>((const __half*)x.data_ptr(), base, (__half*)out.data_ptr(), B, h, w, C, H, W,
                                                 sh, sw, (int)align, (int)relu);
    }
}

// bilinear resize NHWC fp16 (C even), PyTorch semantics. H, W > 0: output size; H, W <= 0: output size from the scale
// factors sf_h / sf_w (floor(in * sf), computed here so a trace stays size-generic); with a scale factor (and no
// align_corners) the index math uses 1 / sf like PyTorch, else in / out
at::Tensor upsample_bilinear(at::Tensor x, int64_t H, int64_t W, c10::optional<double> sf_h,
                             c10::optional<double> sf_w, bool align) {
    int B = x.size(0), C = x.size(1), h = x.size(2), w = x.size(3);
    if (H <= 0) H = (int64_t)std::floor(h * *sf_h);
    if (W <= 0) W = (int64_t)std::floor(w * *sf_w);
    if (!(nhwc_ok(x) && C % 2 == 0)) {
        auto xs = x.contiguous(at::MemoryFormat::ChannelsLast);
        if (sf_h && !align)
            return at::upsample_bilinear2d(xs, {H, W}, align, *sf_h, *sf_w);
        return at::upsample_bilinear2d(xs, {H, W}, align, c10::nullopt, c10::nullopt);
    }
    float sh, sw;
    if (align) {
        sh = H > 1 ? float(h - 1) / float(H - 1) : 0.f;
        sw = W > 1 ? float(w - 1) / float(W - 1) : 0.f;
    } else if (sf_h) {
        sh = (float)(1.0 / *sf_h);
        sw = (float)(1.0 / *sf_w);
    } else {
        sh = float(h) / float(H);
        sw = float(w) / float(W);
    }
    auto y = at::empty({B, C, H, W}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    TORCH_CHECK((long long)B * H * W * C < (1ll << 31) && (long long)B * h * w * C < (1ll << 31), "upsample: tensor too large");
    bilin_launch(x, nullptr, y, sh, sw, align, false);
    return y;
}

// resize x to ref's spatial size (mmseg resize(x, size=ref.shape[2:])): the size comes from a tensor -> trace-generic
at::Tensor upsample_like(at::Tensor x, at::Tensor ref, bool align) {
    return upsample_bilinear(x, ref.size(2), ref.size(3), c10::nullopt, c10::nullopt, align);
}

// size given as a CPU int64 tensor [2] (built from traced sizes by torch.stack inside a TorchScript trace: int arguments
// of custom ops are frozen into the trace, tensor arguments are not)
at::Tensor upsample_bilinear_t(at::Tensor x, at::Tensor size, bool align) {
    auto s = size.to(at::kLong).contiguous();
    return upsample_bilinear(x, s.data_ptr<int64_t>()[0], s.data_ptr<int64_t>()[1], c10::nullopt, c10::nullopt, align);
}

// argmax over C of bilinear(s[:, :hin, :win], (H, W)); s = NHWC scores [B, h, w, C] (C contiguous, even), int32 out
at::Tensor up_argmax(at::Tensor s, int64_t H, int64_t W, bool align) {
    int B = s.size(0), h = s.size(1), w = s.size(2), C = s.size(3);
    if (!(s.scalar_type() == at::kHalf && s.stride(3) == 1 && s.stride(2) == C && s.stride(1) == (long long)w * C &&
          C % 2 == 0)) {
        s = s.to(at::kHalf).contiguous();
        if (C % 2) {
            s = at::constant_pad_nd(s, {0, 1}, -INFINITY);
            C += 1;
        }
    }
    auto out = at::empty({B, H, W}, s.options().dtype(at::kInt));
    long long n = (long long)B * H * W;
    float a, b;
    if (align) {
        a = float(h - 1) / float(H > 1 ? H - 1 : 1);
        b = float(w - 1) / float(W > 1 ? W - 1 : 1);
    } else {
        a = float(h) / float(H);
        b = float(w) / float(W);
    }
    up_argmax_h<<<lin_grid(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)s.data_ptr(), (int32_t*)out.data_ptr(), B, h, w, h * w, w, C, (int)H, (int)W, a, b, (int)align);
    return out;
}

// base + resize(upsample(z, scale_factor sf), base size) (+ ReLU): one kernel when floor(h * sf) == H (then the second
// resize is the identity), else the composite with aten-equivalent steps. inplace: write into base.
at::Tensor upsample_add(at::Tensor base, at::Tensor z, double sf, bool relu, bool inplace) {
    int B = base.size(0), C = base.size(1), H = base.size(2), W = base.size(3);
    int h = z.size(2), w = z.size(3);
    bool ok = nhwc_ok(base) && nhwc_ok(z) && C % 2 == 0 && z.size(1) == C &&
              (int64_t)std::floor(h * sf) == H && (int64_t)std::floor(w * sf) == W;
    if (!ok) {
        auto u = upsample_bilinear(z, -1, -1, sf, sf, false);
        if (u.size(2) != H || u.size(3) != W) u = upsample_bilinear(u, H, W, c10::nullopt, c10::nullopt, false);
        at::Tensor y = inplace ? base.add_(u) : at::add(base, u).contiguous(at::MemoryFormat::ChannelsLast);
        return relu ? y.relu_() : y;
    }
    auto out = inplace ? base : at::empty_like(base, base.options().memory_format(at::MemoryFormat::ChannelsLast));
    TORCH_CHECK((long long)B * H * W * C < (1ll << 31), "upsample_add: tensor too large");
    bilin_launch(z, (const __half*)base.data_ptr(), out, (float)(1.0 / sf), (float)(1.0 / sf), false, relu);
    return out;
}

at::Tensor upsample_add_op(at::Tensor base, at::Tensor z, double sf, bool relu) { return upsample_add(base, z, sf, relu, false); }
at::Tensor upsample_add_inplace(at::Tensor base, at::Tensor z, double sf, bool relu) { return upsample_add(base, z, sf, relu, true); }

// out = x * k[c] + a * l[c]  (NHWC fp16, k / l fp32)
at::Tensor scale_add(at::Tensor a, at::Tensor x, at::Tensor k, c10::optional<at::Tensor> l) {
    a = a.contiguous(at::MemoryFormat::ChannelsLast);
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    auto out = at::empty_like(x, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    long long n2 = x.numel() / 2;
    scale_add_h<<<lin_grid(n2), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)a.data_ptr(), (const __half*)x.data_ptr(), k.data_ptr<float>(),
        l ? l->data_ptr<float>() : nullptr, (__half*)out.data_ptr(), n2, x.size(1) / 2);
    return out;
}

// SegNeXt MSCA strip convs (seg_fast/segnext.py): split = the three 1xk convs of one input (three outputs), sum = the three
// kx1 convs of three inputs + base (one output). w_q fp16 [k_q, C] tap-major, b_q fp16 [C]. axis 0 = along W, 1 = along H.
static void strip_launch(const at::Tensor& ref, const __half* x0, const __half* x1, const __half* x2, std::vector<at::Tensor>& w,
                         std::vector<c10::optional<at::Tensor>>& b, const __half* base, __half* y0, __half* y1, __half* y2,
                         int axis, int mode) {
    int N = ref.size(0), C = ref.size(1), H = ref.size(2), W = ref.size(3);
    int L = axis ? H : W, other = axis ? W : H;
    long long n = (long long)N * other * ((L + STRIP_SEG - 1) / STRIP_SEG) * (C / 2);
    strip3_h<<<lin_grid(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x0, x1, x2, (const __half*)w[0].data_ptr(), (const __half*)w[1].data_ptr(), (const __half*)w[2].data_ptr(),
        hp(b[0]), hp(b[1]), hp(b[2]), base, y0, y1, y2, N, H, W, C, (int)w[0].size(0), (int)w[1].size(0), (int)w[2].size(0),
        axis, mode);
}

std::vector<at::Tensor> strip3_split(at::Tensor x, at::Tensor w0, at::Tensor w1, at::Tensor w2, c10::optional<at::Tensor> b0,
                                     c10::optional<at::Tensor> b1, c10::optional<at::Tensor> b2, int64_t axis) {
    TORCH_CHECK(nhwc_ok(x) && x.size(1) % 2 == 0, "strip3_split: channels-last fp16, C even");
    std::vector<at::Tensor> w{w0, w1, w2};
    std::vector<c10::optional<at::Tensor>> b{b0, b1, b2};
    auto y0 = at::empty_like(x), y1 = at::empty_like(x), y2 = at::empty_like(x);
    strip_launch(x, (const __half*)x.data_ptr(), nullptr, nullptr, w, b, nullptr, (__half*)y0.data_ptr(), (__half*)y1.data_ptr(),
                 (__half*)y2.data_ptr(), (int)axis, 0);
    return {y0, y1, y2};
}

at::Tensor strip3_sum(at::Tensor x0, at::Tensor x1, at::Tensor x2, at::Tensor w0, at::Tensor w1, at::Tensor w2,
                      c10::optional<at::Tensor> b0, c10::optional<at::Tensor> b1, c10::optional<at::Tensor> b2,
                      c10::optional<at::Tensor> base, int64_t axis) {
    TORCH_CHECK(nhwc_ok(x0) && nhwc_ok(x1) && nhwc_ok(x2) && (!base || nhwc_ok(*base)), "strip3_sum: channels-last fp16");
    std::vector<at::Tensor> w{w0, w1, w2};
    std::vector<c10::optional<at::Tensor>> b{b0, b1, b2};
    auto y = at::empty_like(x0);
    strip_launch(x0, (const __half*)x0.data_ptr(), (const __half*)x1.data_ptr(), (const __half*)x2.data_ptr(), w, b, hp(base),
                 (__half*)y.data_ptr(), nullptr, nullptr, (int)axis, 1);
    return y;
}

// SegNeXt Ham head NMF2D (mmseg local_inference + compute_coef + reconstruction), fp32, same op order as upstream, run
// from C++: one dispatcher call instead of ~90 TorchScript ops; batch-1 products as 2-D mm (cuBLAS picks split-K + a
// reduce kernel for batch-1 bmm: 2 launches, ~2x the host cost). x [BS, D, N], bases [BS, D, R] -> [BS, D, N].
static inline at::Tensor bmm_(const at::Tensor& a, const at::Tensor& b) {
    if (a.size(0) == 1) return at::mm(a[0], b[0]).unsqueeze(0);
    return at::bmm(a, b);
}

at::Tensor nmf2d(at::Tensor x, at::Tensor bases, int64_t steps, double inv_t) {
    auto xt = x.transpose(1, 2);
    auto coef = at::softmax(bmm_(xt, bases) * inv_t, -1);
    auto step_coef = [&](const at::Tensor& b, const at::Tensor& c) {
        auto num = bmm_(xt, b);
        auto den = bmm_(c, bmm_(b.transpose(1, 2), b));
        return c * num / (den + 1e-6);
    };
    for (int64_t i = 0; i < steps; ++i) {
        coef = step_coef(bases, coef);
        auto num = bmm_(x, coef);
        auto den = bmm_(bases, bmm_(coef.transpose(1, 2), coef));
        bases = bases * num / (den + 1e-6);
    }
    coef = step_coef(bases, coef);
    return bmm_(bases, coef.transpose(1, 2));
}

TORCH_LIBRARY_FRAGMENT(segx, m) {
    m.def("nmf2d(Tensor x, Tensor bases, int steps, float inv_t) -> Tensor");
    m.def("strip3_split(Tensor x, Tensor w0, Tensor w1, Tensor w2, Tensor? b0, Tensor? b1, Tensor? b2, int axis) -> Tensor[]");
    m.def("strip3_sum(Tensor x0, Tensor x1, Tensor x2, Tensor w0, Tensor w1, Tensor w2, Tensor? b0, Tensor? b1, Tensor? b2, Tensor? base, int axis) -> Tensor");
    m.def("bias_act_(Tensor(a!) y, Tensor? bias, Tensor? res, bool relu) -> Tensor(a!)");
    m.def("dwconv(Tensor x, Tensor wt, Tensor? bias, int kh, int kw, int[] stride, int[] padding, int[] dilation, int act) -> Tensor");
    m.def("upsample_bilinear(Tensor x, int H, int W, float? sf_h, float? sf_w, bool align) -> Tensor");
    m.def("upsample_like(Tensor x, Tensor ref, bool align) -> Tensor");
    m.def("upsample_bilinear_t(Tensor x, Tensor size, bool align) -> Tensor");
    m.def("up_argmax(Tensor s, int H, int W, bool align) -> Tensor");
    m.def("scale_add(Tensor a, Tensor x, Tensor k, Tensor? l) -> Tensor");
    m.def("upsample_add(Tensor base, Tensor z, float sf, bool relu) -> Tensor");
    m.def("upsample_add_(Tensor(a!) base, Tensor z, float sf, bool relu) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(segx, CUDA, m) {
    m.impl("nmf2d", nmf2d);
    m.impl("strip3_split", strip3_split);
    m.impl("strip3_sum", strip3_sum);
    m.impl("bias_act_", bias_act_);
    m.impl("dwconv", dwconv);
    m.impl("upsample_bilinear", upsample_bilinear);
    m.impl("upsample_like", upsample_like);
    m.impl("upsample_bilinear_t", upsample_bilinear_t);
    m.impl("up_argmax", up_argmax);
    m.impl("scale_add", scale_add);
    m.impl("upsample_add", upsample_add_op);
    m.impl("upsample_add_", upsample_add_inplace);
}

// DINOv3 RoPE applied in place to q and k of a packed qkv buffer [B, N, 3, H, 64] fp16 (patch tokens only; the first P tokens
// = cls + registers are untouched). cs: fp32 table [Np, 2, 32] = cos | sin (angle of dim j and j+32 are equal).
// qs: extra factor applied to q only (all tokens, prefix included) (pass the attention scale, then call attention with scale=1).
// out[j] = x[j] c - x[j+32] s ; out[j+32] = x[j+32] c + x[j] s  (= x*cos + rotate_half(x)*sin), computed in fp32.
#include <cuda_fp16.h>
extern "C" __global__ void rope_qk_h(__half* qkv, const float* cs, int N, int P, int H, int total, float qs) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        int jj = i & 15;               // 16 half2 pairs per half row
        int r = i >> 4;
        int h = r % H; r /= H;
        int t = r & 1; r >>= 1;        // 0 = q, 1 = k
        int n = r % N; int b = r / N; int np = n - P;
        long base = ((((long)b * N + n) * 3 + t) * H + h) * 64;
        __half2* p1 = reinterpret_cast<__half2*>(qkv + base) + jj;
        __half2* p2 = reinterpret_cast<__half2*>(qkv + base + 32) + jj;
        float2 x1 = __half22float2(*p1), x2 = __half22float2(*p2);
        if (np < 0) {                  // cls / register tokens: no rotation, q only gets the scale
            if (t == 0) { *p1 = __floats2half2_rn(x1.x * qs, x1.y * qs); *p2 = __floats2half2_rn(x2.x * qs, x2.y * qs); }
            continue;
        }
        float2 c = *(reinterpret_cast<const float2*>(cs + (long)np * 64) + jj);
        float2 s = *(reinterpret_cast<const float2*>(cs + (long)np * 64 + 32) + jj);
        float m = t ? 1.f : qs;        // q is pre-scaled (softmax scale 2^-3: exact in fp16) so fp16-accumulated QK^T cannot overflow
        *p1 = __floats2half2_rn((x1.x * c.x - x2.x * s.x) * m, (x1.y * c.y - x2.y * s.y) * m);
        *p2 = __floats2half2_rn((x2.x * c.x + x1.x * s.x) * m, (x2.y * c.y + x1.y * s.y) * m);
    }
}

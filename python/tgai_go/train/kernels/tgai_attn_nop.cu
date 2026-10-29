// Attention that never materializes P.
// Q,K,V,O,dO,dQ,dK,dV: fp32 row-major [seq, nh, dh], dh == 64.
// Launch (same contract as attn_fwd / attn_bwd):
//   grid  (nh, ceil(seq/64), 1)
//   block 256
//   dynamic smem 32 KB
//
// attn_fwd_nop: online-softmax causal forward. Writes O only.
// attn_bwd_recompute: legacy cubic recompute. Kept so older launches stay
//   bitwise-identical; the lowmem training path does not call it anymore.
// attn_fwd_stats: same math as attn_fwd_nop, also writes final per-row
//   m[seq, nh] and l[seq, nh].
// attn_bwd_flash: standard flash backward. Each query tile walks K/V once.
//   dP = dO @ V^T, dS = P*(dP-D), D = rowsum(dO*O). P and dS stay in smem.
//   dQ is owned by the query tile; dK/dV are atomicAdd across query tiles.
//   Launch: grid (nh, ceil(seq/64), 1), block 256, dynamic smem 69632.

#define BT 64
#define DH 64
#define BC 32
#define NEG (-1.0e30f)

extern __shared__ __align__(16) float smem[];

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

__device__ __forceinline__ int qkv_at(int t, int h, int nh, int dh) {
    return (t * nh + h) * dh;
}

__device__ __forceinline__ float prob_of(float s, float m, float l) {
    if (!(s > -1.0e20f) || !(l > 0.f)) return 0.f;
    return __expf(s - m) * (1.f / l);
}

// [d][row] transpose of a [BT, DH] head tile.
__device__ __forceinline__ void load_QT(const float* __restrict__ src, float* __restrict__ dst,
                                        int tok0, int h, int nh, int dh, int tid, int seq) {
#pragma unroll 1
    for (int e4 = tid; e4 < 1024; e4 += 256) {
        int e = e4 << 2;
        int local = e >> 6;
        int d = e & 63;
        int tok = tok0 + local;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tok < seq) v = *reinterpret_cast<const float4*>(src + ((tok * nh + h) * dh + d));
        dst[(d + 0) * 64 + local] = v.x;
        dst[(d + 1) * 64 + local] = v.y;
        dst[(d + 2) * 64 + local] = v.z;
        dst[(d + 3) * 64 + local] = v.w;
    }
}

__device__ __forceinline__ void load_KT(const float* __restrict__ src, float* __restrict__ dst,
                                        int tok0, int h, int nh, int dh, int tid, int seq) {
#pragma unroll 1
    for (int e4 = tid; e4 < (BC * 16); e4 += 256) {
        int e = e4 << 2;
        int local = e >> 6;
        int d = e & 63;
        int tok = tok0 + local;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tok < seq) v = *reinterpret_cast<const float4*>(src + ((tok * nh + h) * dh + d));
        dst[(d + 0) * BC + local] = v.x;
        dst[(d + 1) * BC + local] = v.y;
        dst[(d + 2) * BC + local] = v.z;
        dst[(d + 3) * BC + local] = v.w;
    }
}

__device__ __forceinline__ void load_V(const float* __restrict__ src, float* __restrict__ dst,
                                       int tok0, int h, int nh, int dh, int tid, int seq) {
#pragma unroll 1
    for (int e4 = tid; e4 < (BC * 16); e4 += 256) {
        int e = e4 << 2;
        int local = e >> 6;
        int d = e & 63;
        int tok = tok0 + local;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tok < seq) v = *reinterpret_cast<const float4*>(src + ((tok * nh + h) * dh + d));
        *reinterpret_cast<float4*>(dst + e) = v;
    }
}

// [token][dim] row-major, 64 rows.
__device__ __forceinline__ void load_rowmajor(const float* __restrict__ src, float* __restrict__ dst,
                                              int row0, int h, int nh, int dh, int tid, int seq) {
#pragma unroll 1
    for (int i = tid; i < BT * (DH / 4); i += 256) {
        int row = i >> 4;
        int d4 = i & 15;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        int tok = row0 + row;
        if (tok < seq) {
            const float* gp = src + qkv_at(tok, h, nh, dh) + (d4 << 2);
            v = *reinterpret_cast<const float4*>(gp);
        }
        *reinterpret_cast<float4*>(dst + row * DH + (d4 << 2)) = v;
    }
}

// [d][row] transpose, 64 rows. Same layout as load_QT.
__device__ __forceinline__ void load_transpose(const float* __restrict__ src, float* __restrict__ dst,
                                               int row0, int h, int nh, int dh, int tid, int seq) {
    load_QT(src, dst, row0, h, nh, dh, tid, seq);
}

// 8-row x 2-col dot. Both tiles are [contraction][64].
__device__ __forceinline__ void qk_dot8(const float* __restrict__ smQ, const float* __restrict__ smK,
                                        int row0, int col0, int col1, float acc0[8], float acc1[8]) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        acc0[r] = 0.f;
        acc1[r] = 0.f;
    }
#pragma unroll 8
    for (int d = 0; d < DH; ++d) {
        float k0 = smK[d * BT + col0];
        float k1 = smK[d * BT + col1];
        float4 q0 = *reinterpret_cast<const float4*>(smQ + d * BT + row0);
        float4 q1 = *reinterpret_cast<const float4*>(smQ + d * BT + row0 + 4);
        acc0[0] += q0.x * k0;
        acc1[0] += q0.x * k1;
        acc0[1] += q0.y * k0;
        acc1[1] += q0.y * k1;
        acc0[2] += q0.z * k0;
        acc1[2] += q0.z * k1;
        acc0[3] += q0.w * k0;
        acc1[3] += q0.w * k1;
        acc0[4] += q1.x * k0;
        acc1[4] += q1.x * k1;
        acc0[5] += q1.y * k0;
        acc1[5] += q1.y * k1;
        acc0[6] += q1.z * k0;
        acc1[6] += q1.z * k1;
        acc0[7] += q1.w * k0;
        acc1[7] += q1.w * k1;
    }
}

__device__ __forceinline__ void flush_owned(float* __restrict__ dst, float* __restrict__ sm, int n0,
                                            int h, int seq, int nh, int dh, int lane, int warp,
                                            const float a0[8], const float a1[8], float coef) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        sm[lane * BT + warp * 8 + r] = coef * a0[r];
        sm[(lane + 32) * BT + warp * 8 + r] = coef * a1[r];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int u = n0 + warp * 8 + i;
        if (u < seq) {
            float* rowp = dst + qkv_at(u, h, nh, dh);
            rowp[lane] = sm[(warp * 8 + i) * BT + lane];
            rowp[lane + 32] = sm[(warp * 8 + i) * BT + lane + 32];
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Forward, no P. v4 schedule: Bc=32, Q resident, next K/V prefetched in regs.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float4 fa_gather(const float* __restrict__ src, int tok, int h, int nh,
                                            int dh, int d, int seq) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (tok < seq)
        v = *reinterpret_cast<const float4*>(src + ((tok * nh + h) * dh + d));
    return v;
}

__device__ __forceinline__ void fa_pin(float4 &v) {
    asm volatile("" : "+f"(v.x), "+f"(v.y), "+f"(v.z), "+f"(v.w));
}

__device__ __forceinline__ void fa_pref(const float* __restrict__ src, int tok0, int h, int nh, int dh,
                                        int tid, int seq, float4 &a, float4 &b) {
    int ea = tid << 2;
    int eb = (tid + 256) << 2;
    a = fa_gather(src, tok0 + (ea >> 6), h, nh, dh, ea & 63, seq);
    b = fa_gather(src, tok0 + (eb >> 6), h, nh, dh, eb & 63, seq);
    fa_pin(a);
    fa_pin(b);
}

__device__ __forceinline__ void fa_commit_KT(float* dst, int e4, float4 v) {
    int e = e4 << 2;
    int local = e >> 6;
    int d = e & 63;
    dst[(d + 0) * BC + local] = v.x;
    dst[(d + 1) * BC + local] = v.y;
    dst[(d + 2) * BC + local] = v.z;
    dst[(d + 3) * BC + local] = v.w;
}

__device__ __forceinline__ void fa_commit_V(float* dst, int e4, float4 v) {
    *reinterpret_cast<float4*>(dst + (e4 << 2)) = v;
}

template <bool FULL>
__device__ __forceinline__ void fa_tile(
    float* smQ, float* smK, float* smV, int k0, int t0, int seq, float scale,
    int lane, int row0,
    float &sx0, float &sx1, float &sx2, float &sx3, float &sx4, float &sx5, float &sx6, float &sx7,
    float &mx0, float &mx1, float &mx2, float &mx3, float &mx4, float &mx5, float &mx6, float &mx7,
    float &lx0, float &lx1, float &lx2, float &lx3, float &lx4, float &lx5, float &lx6, float &lx7,
    float &ax0, float &ax1, float &ax2, float &ax3, float &ax4, float &ax5, float &ax6, float &ax7,
    float &ay0, float &ay1, float &ay2, float &ay3, float &ay4, float &ay5, float &ay6, float &ay7) {
    sx0 = sx1 = sx2 = sx3 = sx4 = sx5 = sx6 = sx7 = 0.f;
#pragma unroll 16
    for (int d = 0; d < 64; ++d) {
        float kv = smK[d * BC + lane];
        float4 q0 = *reinterpret_cast<const float4*>(smQ + d * 64 + row0);
        float4 q1 = *reinterpret_cast<const float4*>(smQ + d * 64 + row0 + 4);
        sx0 = fmaf(q0.x, kv, sx0);
        sx1 = fmaf(q0.y, kv, sx1);
        sx2 = fmaf(q0.z, kv, sx2);
        sx3 = fmaf(q0.w, kv, sx3);
        sx4 = fmaf(q1.x, kv, sx4);
        sx5 = fmaf(q1.y, kv, sx5);
        sx6 = fmaf(q1.z, kv, sx6);
        sx7 = fmaf(q1.w, kv, sx7);
    }
    sx0 *= scale; sx1 *= scale; sx2 *= scale; sx3 *= scale;
    sx4 *= scale; sx5 *= scale; sx6 *= scale; sx7 *= scale;

#define MASK(i, sx)                                                              \
    do {                                                                         \
        int tr_ = t0 + row0 + (i);                                               \
        int u_ = k0 + lane;                                                      \
        bool ok_ = (tr_ < seq) && (u_ < seq) && (FULL || (u_ <= tr_));           \
        if (!ok_) sx = NEG;                                                      \
    } while (0)
    MASK(0, sx0); MASK(1, sx1); MASK(2, sx2); MASK(3, sx3);
    MASK(4, sx4); MASK(5, sx5); MASK(6, sx6); MASK(7, sx7);
#undef MASK

#define SOFT(sx, mx, lx, ax, ay)                                                 \
    do {                                                                         \
        float mt_ = warp_max(sx);                                               \
        float mn_ = fmaxf(mx, mt_);                                              \
        float al_ = (mx > -1.0e20f) ? __expf(mx - mn_) : 0.f;                   \
        float p_ = (sx > -1.0e20f) ? __expf(sx - mn_) : 0.f;                    \
        lx = lx * al_ + warp_sum(p_);                                           \
        ax *= al_;                                                              \
        ay *= al_;                                                              \
        mx = mn_;                                                               \
        sx = p_;                                                                \
    } while (0)
    SOFT(sx0, mx0, lx0, ax0, ay0);
    SOFT(sx1, mx1, lx1, ax1, ay1);
    SOFT(sx2, mx2, lx2, ax2, ay2);
    SOFT(sx3, mx3, lx3, ax3, ay3);
    SOFT(sx4, mx4, lx4, ax4, ay4);
    SOFT(sx5, mx5, lx5, ax5, ay5);
    SOFT(sx6, mx6, lx6, ax6, ay6);
    SOFT(sx7, mx7, lx7, ax7, ay7);
#undef SOFT

    __syncthreads();
    smK[(row0 + 0) * BC + lane] = sx0;
    smK[(row0 + 1) * BC + lane] = sx1;
    smK[(row0 + 2) * BC + lane] = sx2;
    smK[(row0 + 3) * BC + lane] = sx3;
    smK[(row0 + 4) * BC + lane] = sx4;
    smK[(row0 + 5) * BC + lane] = sx5;
    smK[(row0 + 6) * BC + lane] = sx6;
    smK[(row0 + 7) * BC + lane] = sx7;
    __syncthreads();

#pragma unroll
    for (int c = 0; c < BC; ++c) {
        float v0 = smV[c * 64 + lane];
        float v1 = smV[c * 64 + lane + 32];
        ax0 = fmaf(smK[(row0 + 0) * BC + c], v0, ax0); ay0 = fmaf(smK[(row0 + 0) * BC + c], v1, ay0);
        ax1 = fmaf(smK[(row0 + 1) * BC + c], v0, ax1); ay1 = fmaf(smK[(row0 + 1) * BC + c], v1, ay1);
        ax2 = fmaf(smK[(row0 + 2) * BC + c], v0, ax2); ay2 = fmaf(smK[(row0 + 2) * BC + c], v1, ay2);
        ax3 = fmaf(smK[(row0 + 3) * BC + c], v0, ax3); ay3 = fmaf(smK[(row0 + 3) * BC + c], v1, ay3);
        ax4 = fmaf(smK[(row0 + 4) * BC + c], v0, ax4); ay4 = fmaf(smK[(row0 + 4) * BC + c], v1, ay4);
        ax5 = fmaf(smK[(row0 + 5) * BC + c], v0, ax5); ay5 = fmaf(smK[(row0 + 5) * BC + c], v1, ay5);
        ax6 = fmaf(smK[(row0 + 6) * BC + c], v0, ax6); ay6 = fmaf(smK[(row0 + 6) * BC + c], v1, ay6);
        ax7 = fmaf(smK[(row0 + 7) * BC + c], v0, ax7); ay7 = fmaf(smK[(row0 + 7) * BC + c], v1, ay7);
    }
    __syncthreads();
}

extern "C" __global__ void attn_fwd_nop(const float* __restrict__ Q, const float* __restrict__ K,
                                        const float* __restrict__ V, float* __restrict__ O,
                                        int seq, int nh, int dh, float scale) {
    const int h = (int)blockIdx.x;
    const int t0 = (int)blockIdx.y * BT;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = warp * 8;
    if (t0 >= seq || dh != DH) return;

    float* smQ = smem;
    float* smK = smem + 4096;
    float* smV = smK + 64 * BC;

    load_QT(Q, smQ, t0, h, nh, dh, tid, seq);
    __syncthreads();

    float sx0 = 0.f, sx1 = 0.f, sx2 = 0.f, sx3 = 0.f, sx4 = 0.f, sx5 = 0.f, sx6 = 0.f, sx7 = 0.f;
    float mx0 = NEG, mx1 = NEG, mx2 = NEG, mx3 = NEG, mx4 = NEG, mx5 = NEG, mx6 = NEG, mx7 = NEG;
    float lx0 = 0.f, lx1 = 0.f, lx2 = 0.f, lx3 = 0.f, lx4 = 0.f, lx5 = 0.f, lx6 = 0.f, lx7 = 0.f;
    float ax0 = 0.f, ax1 = 0.f, ax2 = 0.f, ax3 = 0.f, ax4 = 0.f, ax5 = 0.f, ax6 = 0.f, ax7 = 0.f;
    float ay0 = 0.f, ay1 = 0.f, ay2 = 0.f, ay3 = 0.f, ay4 = 0.f, ay5 = 0.f, ay6 = 0.f, ay7 = 0.f;

    int k_end = t0 + BT;
    if (k_end > seq) k_end = seq;
    if (k_end > 0) {
        load_KT(K, smK, 0, h, nh, dh, tid, seq);
        load_V(V, smV, 0, h, nh, dh, tid, seq);
        __syncthreads();
    }
#pragma unroll 1
    for (int k0 = 0; k0 < k_end; k0 += BC) {
        int next = k0 + BC;
        float4 ka, kb, va, vb;
        bool more = next < k_end;
        if (more) {
            fa_pref(K, next, h, nh, dh, tid, seq, ka, kb);
            fa_pref(V, next, h, nh, dh, tid, seq, va, vb);
            asm volatile("" ::: "memory");
        }
        if (k0 + BC <= t0)
            fa_tile<true>(smQ, smK, smV, k0, t0, seq, scale, lane, row0,
                          sx0, sx1, sx2, sx3, sx4, sx5, sx6, sx7,
                          mx0, mx1, mx2, mx3, mx4, mx5, mx6, mx7,
                          lx0, lx1, lx2, lx3, lx4, lx5, lx6, lx7,
                          ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7,
                          ay0, ay1, ay2, ay3, ay4, ay5, ay6, ay7);
        else
            fa_tile<false>(smQ, smK, smV, k0, t0, seq, scale, lane, row0,
                           sx0, sx1, sx2, sx3, sx4, sx5, sx6, sx7,
                           mx0, mx1, mx2, mx3, mx4, mx5, mx6, mx7,
                           lx0, lx1, lx2, lx3, lx4, lx5, lx6, lx7,
                           ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7,
                           ay0, ay1, ay2, ay3, ay4, ay5, ay6, ay7);
        if (more) {
            fa_commit_KT(smK, tid, ka);
            fa_commit_KT(smK, tid + 256, kb);
            fa_commit_V(smV, tid, va);
            fa_commit_V(smV, tid + 256, vb);
            __syncthreads();
        }
    }

#define EPI(i, xx, yy, ll)                                                       \
    do {                                                                         \
        int tr_ = t0 + row0 + (i);                                               \
        if (tr_ < seq) {                                                         \
            float inv_ = (ll > 0.f) ? (1.f / ll) : 0.f;                          \
            float* orow_ = O + ((tr_ * nh + h) * dh);                            \
            orow_[lane] = (xx) * inv_;                                           \
            orow_[lane + 32] = (yy) * inv_;                                      \
        }                                                                        \
    } while (0)
    EPI(0, ax0, ay0, lx0);
    EPI(1, ax1, ay1, lx1);
    EPI(2, ax2, ay2, lx2);
    EPI(3, ax3, ay3, lx3);
    EPI(4, ax4, ay4, lx4);
    EPI(5, ax5, ay5, lx5);
    EPI(6, ax6, ay6, lx6);
    EPI(7, ax7, ay7, lx7);
#undef EPI
}

// ---------------------------------------------------------------------------
// Forward that also keeps the online-softmax row stats. O matches attn_fwd_nop.
// ---------------------------------------------------------------------------

extern "C" __global__ void attn_fwd_stats(const float* __restrict__ Q, const float* __restrict__ K,
                                          const float* __restrict__ V, float* __restrict__ O,
                                          float* __restrict__ m_out, float* __restrict__ l_out,
                                          int seq, int nh, int dh, float scale) {
    const int h = (int)blockIdx.x;
    const int t0 = (int)blockIdx.y * BT;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = warp * 8;
    if (t0 >= seq || dh != DH) return;

    float* smQ = smem;
    float* smK = smem + 4096;
    float* smV = smK + 64 * BC;

    load_QT(Q, smQ, t0, h, nh, dh, tid, seq);
    __syncthreads();

    float sx0 = 0.f, sx1 = 0.f, sx2 = 0.f, sx3 = 0.f, sx4 = 0.f, sx5 = 0.f, sx6 = 0.f, sx7 = 0.f;
    float mx0 = NEG, mx1 = NEG, mx2 = NEG, mx3 = NEG, mx4 = NEG, mx5 = NEG, mx6 = NEG, mx7 = NEG;
    float lx0 = 0.f, lx1 = 0.f, lx2 = 0.f, lx3 = 0.f, lx4 = 0.f, lx5 = 0.f, lx6 = 0.f, lx7 = 0.f;
    float ax0 = 0.f, ax1 = 0.f, ax2 = 0.f, ax3 = 0.f, ax4 = 0.f, ax5 = 0.f, ax6 = 0.f, ax7 = 0.f;
    float ay0 = 0.f, ay1 = 0.f, ay2 = 0.f, ay3 = 0.f, ay4 = 0.f, ay5 = 0.f, ay6 = 0.f, ay7 = 0.f;

    int k_end = t0 + BT;
    if (k_end > seq) k_end = seq;
    if (k_end > 0) {
        load_KT(K, smK, 0, h, nh, dh, tid, seq);
        load_V(V, smV, 0, h, nh, dh, tid, seq);
        __syncthreads();
    }
#pragma unroll 1
    for (int k0 = 0; k0 < k_end; k0 += BC) {
        int next = k0 + BC;
        float4 ka, kb, va, vb;
        bool more = next < k_end;
        if (more) {
            fa_pref(K, next, h, nh, dh, tid, seq, ka, kb);
            fa_pref(V, next, h, nh, dh, tid, seq, va, vb);
            asm volatile("" ::: "memory");
        }
        if (k0 + BC <= t0)
            fa_tile<true>(smQ, smK, smV, k0, t0, seq, scale, lane, row0,
                          sx0, sx1, sx2, sx3, sx4, sx5, sx6, sx7,
                          mx0, mx1, mx2, mx3, mx4, mx5, mx6, mx7,
                          lx0, lx1, lx2, lx3, lx4, lx5, lx6, lx7,
                          ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7,
                          ay0, ay1, ay2, ay3, ay4, ay5, ay6, ay7);
        else
            fa_tile<false>(smQ, smK, smV, k0, t0, seq, scale, lane, row0,
                           sx0, sx1, sx2, sx3, sx4, sx5, sx6, sx7,
                           mx0, mx1, mx2, mx3, mx4, mx5, mx6, mx7,
                           lx0, lx1, lx2, lx3, lx4, lx5, lx6, lx7,
                           ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7,
                           ay0, ay1, ay2, ay3, ay4, ay5, ay6, ay7);
        if (more) {
            fa_commit_KT(smK, tid, ka);
            fa_commit_KT(smK, tid + 256, kb);
            fa_commit_V(smV, tid, va);
            fa_commit_V(smV, tid + 256, vb);
            __syncthreads();
        }
    }

#define EPI(i, xx, yy, ll)                                                       \
    do {                                                                         \
        int tr_ = t0 + row0 + (i);                                               \
        if (tr_ < seq) {                                                         \
            float inv_ = (ll > 0.f) ? (1.f / ll) : 0.f;                          \
            float* orow_ = O + ((tr_ * nh + h) * dh);                            \
            orow_[lane] = (xx) * inv_;                                           \
            orow_[lane + 32] = (yy) * inv_;                                      \
        }                                                                        \
    } while (0)
    EPI(0, ax0, ay0, lx0);
    EPI(1, ax1, ay1, lx1);
    EPI(2, ax2, ay2, lx2);
    EPI(3, ax3, ay3, lx3);
    EPI(4, ax4, ay4, lx4);
    EPI(5, ax5, ay5, lx5);
    EPI(6, ax6, ay6, lx6);
    EPI(7, ax7, ay7, lx7);
#undef EPI
    if (lane == 0) {
#define ST_ML(i, mx, lx)                                                         \
        do {                                                                     \
            int tr_ = t0 + row0 + (i);                                           \
            if (tr_ < seq) {                                                     \
                int mi_ = tr_ * nh + h;                                          \
                m_out[mi_] = (mx);                                               \
                l_out[mi_] = (lx);                                               \
            }                                                                    \
        } while (0)
        ST_ML(0, mx0, lx0);
        ST_ML(1, mx1, lx1);
        ST_ML(2, mx2, lx2);
        ST_ML(3, mx3, lx3);
        ST_ML(4, mx4, lx4);
        ST_ML(5, mx5, lx5);
        ST_ML(6, mx6, lx6);
        ST_ML(7, mx7, lx7);
#undef ST_ML
    }
}

// ---------------------------------------------------------------------------
// Flash backward. Query tile is resident. Each K/V tile is visited once.
// smem floats: Q 64*72, dO 64*72, K 64*32, V 64*32, P 64*32, dS 64*32 = 17408
// = 69632 bytes. Q/dO stride 72 cuts the dK/dV smem bank conflicts.
// ---------------------------------------------------------------------------

#define QSTR 72

__device__ __forceinline__ void load_T72(const float* __restrict__ src, float* __restrict__ dst,
                                         int tok0, int h, int nh, int dh, int tid, int seq) {
#pragma unroll 1
    for (int e4 = tid; e4 < 1024; e4 += 256) {
        int e = e4 << 2;
        int local = e >> 6;
        int d = e & 63;
        int tok = tok0 + local;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tok < seq)
            v = *reinterpret_cast<const float4*>(src + ((tok * nh + h) * dh + d));
        dst[(d + 0) * QSTR + local] = v.x;
        dst[(d + 1) * QSTR + local] = v.y;
        dst[(d + 2) * QSTR + local] = v.z;
        dst[(d + 3) * QSTR + local] = v.w;
    }
}

// Fused QK and dP. sx matches fa_tile's fma order so saved (m, l) rebuild P.
__device__ __forceinline__ void flash_qk_dp(const float* __restrict__ smQ, const float* __restrict__ smDO,
                                            const float* __restrict__ smK, const float* __restrict__ smV,
                                            int row0, int lane, float sx[8], float dp[8]) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        sx[r] = 0.f;
        dp[r] = 0.f;
    }
#pragma unroll 8
    for (int d = 0; d < 64; ++d) {
        float kv = smK[d * BC + lane];
        float vv = smV[d * BC + lane];
        float4 q0 = *reinterpret_cast<const float4*>(smQ + d * QSTR + row0);
        float4 q1 = *reinterpret_cast<const float4*>(smQ + d * QSTR + row0 + 4);
        float4 o0 = *reinterpret_cast<const float4*>(smDO + d * QSTR + row0);
        float4 o1 = *reinterpret_cast<const float4*>(smDO + d * QSTR + row0 + 4);
        sx[0] = fmaf(q0.x, kv, sx[0]); dp[0] = fmaf(o0.x, vv, dp[0]);
        sx[1] = fmaf(q0.y, kv, sx[1]); dp[1] = fmaf(o0.y, vv, dp[1]);
        sx[2] = fmaf(q0.z, kv, sx[2]); dp[2] = fmaf(o0.z, vv, dp[2]);
        sx[3] = fmaf(q0.w, kv, sx[3]); dp[3] = fmaf(o0.w, vv, dp[3]);
        sx[4] = fmaf(q1.x, kv, sx[4]); dp[4] = fmaf(o1.x, vv, dp[4]);
        sx[5] = fmaf(q1.y, kv, sx[5]); dp[5] = fmaf(o1.y, vv, dp[5]);
        sx[6] = fmaf(q1.z, kv, sx[6]); dp[6] = fmaf(o1.z, vv, dp[6]);
        sx[7] = fmaf(q1.w, kv, sx[7]); dp[7] = fmaf(o1.w, vv, dp[7]);
    }
}

extern "C" __global__ void __launch_bounds__(256, 1)
attn_bwd_flash(const float* __restrict__ dO, const float* __restrict__ Q,
               const float* __restrict__ K, const float* __restrict__ V,
               const float* __restrict__ O,
               const float* __restrict__ m_in, const float* __restrict__ l_in,
               float* __restrict__ dQ, float* __restrict__ dK, float* __restrict__ dV,
               int seq, int nh, int dh, float scale) {
    const int h = (int)blockIdx.x;
    const int t0 = (int)blockIdx.y * BT;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = warp * 8;
    if (t0 >= seq || dh != DH) return;

    float* smQ = smem;
    float* smDO = smQ + 64 * QSTR;
    float* smK = smDO + 64 * QSTR;
    float* smV = smK + 64 * BC;
    float* smP = smV + 64 * BC;
    float* smDS = smP + 64 * BC;

    load_T72(Q, smQ, t0, h, nh, dh, tid, seq);
    load_T72(dO, smDO, t0, h, nh, dh, tid, seq);
    __syncthreads();

    float m[8], l[8], D[8], dq0[8], dq1[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        int tr = t0 + row0 + r;
        if (tr < seq) {
            int mi = tr * nh + h;
            m[r] = m_in[mi];
            l[r] = l_in[mi];
            float part = smDO[lane * QSTR + row0 + r] * O[qkv_at(tr, h, nh, dh) + lane]
                       + smDO[(lane + 32) * QSTR + row0 + r] * O[qkv_at(tr, h, nh, dh) + lane + 32];
            D[r] = warp_sum(part);
        } else {
            m[r] = NEG;
            l[r] = 0.f;
            D[r] = 0.f;
        }
        dq0[r] = 0.f;
        dq1[r] = 0.f;
    }

    int k_end = t0 + BT;
    if (k_end > seq) k_end = seq;
    int ntiles = (k_end + BC - 1) / BC;
    // Consecutive query tiles start on different K tiles so the atomic wave
    // is not all hammering K tile 0 at once.
    int phase = ntiles > 0 ? ((int)blockIdx.y % ntiles) : 0;
#pragma unroll 1
    for (int step = 0; step < ntiles; ++step) {
        int k0 = ((phase + step) % ntiles) * BC;
        load_KT(K, smK, k0, h, nh, dh, tid, seq);
        load_KT(V, smV, k0, h, nh, dh, tid, seq);
        __syncthreads();

        float sx[8], dp[8];
        flash_qk_dp(smQ, smDO, smK, smV, row0, lane, sx, dp);
        bool tile_full = (k0 + BC <= t0);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            sx[r] *= scale;
            int tr = t0 + row0 + r;
            int u = k0 + lane;
            bool ok = (tr < seq) && (u < seq) && (tile_full || (u <= tr));
            if (!ok) sx[r] = NEG;
            float p = prob_of(sx[r], m[r], l[r]);
            float ds = p * (dp[r] - D[r]);
            smP[(row0 + r) * BC + lane] = p;
            smDS[(row0 + r) * BC + lane] = ds;
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            float g0 = 0.f, g1 = 0.f;
            int q = row0 + r;
#pragma unroll
            for (int k = 0; k < BC; ++k) {
                float s = smDS[q * BC + k];
                g0 = fmaf(s, smK[lane * BC + k], g0);
                g1 = fmaf(s, smK[(lane + 32) * BC + k], g1);
            }
            dq0[r] += g0;
            dq1[r] += g1;
        }

#pragma unroll 1
        for (int e = 0; e < 8; ++e) {
            int idx = tid + e * 256;
            int k = idx >> 6;
            int d = idx & 63;
            float ak = 0.f, av = 0.f;
#pragma unroll 4
            for (int q = 0; q < BT; ++q) {
                float s = smDS[q * BC + k];
                float p = smP[q * BC + k];
                ak = fmaf(s, smQ[d * QSTR + q], ak);
                av = fmaf(p, smDO[d * QSTR + q], av);
            }
            int tok = k0 + k;
            if (tok < seq) {
                float* dkrow = dK + qkv_at(tok, h, nh, dh);
                float* dvrow = dV + qkv_at(tok, h, nh, dh);
                atomicAdd(dkrow + d, scale * ak);
                atomicAdd(dvrow + d, av);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        int tr = t0 + row0 + r;
        if (tr < seq) {
            float* rowp = dQ + qkv_at(tr, h, nh, dh);
            rowp[lane] = scale * dq0[r];
            rowp[lane + 32] = scale * dq1[r];
        }
    }
}

// One query tile against the K/V already in smem. dQ accumulates in registers.
// dK/dV accumulate into smDK/smDV (caller zeroed them) so two query tiles can
// share one atomicAdd. All threads of the block must enter together.
__device__ void flash_bwd_one_tile(
    float* smQ, float* smDO, float* smK, float* smV, float* smP, float* smDS,
    float* smDK, float* smDV,
    const float* __restrict__ dO, const float* __restrict__ Q, const float* __restrict__ O,
    const float* __restrict__ m_in, const float* __restrict__ l_in,
    int t0, int k0, int h, int seq, int nh, int dh, float scale,
    int tid, int lane, int row0, float dq0[8], float dq1[8]) {
    load_T72(Q, smQ, t0, h, nh, dh, tid, seq);
    load_T72(dO, smDO, t0, h, nh, dh, tid, seq);
    __syncthreads();

    float m[8], l[8], D[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        int tr = t0 + row0 + r;
        if (tr < seq) {
            int mi = tr * nh + h;
            m[r] = m_in[mi];
            l[r] = l_in[mi];
            float part = smDO[lane * QSTR + row0 + r] * O[qkv_at(tr, h, nh, dh) + lane]
                       + smDO[(lane + 32) * QSTR + row0 + r] * O[qkv_at(tr, h, nh, dh) + lane + 32];
            D[r] = warp_sum(part);
        } else {
            m[r] = NEG;
            l[r] = 0.f;
            D[r] = 0.f;
        }
    }
    float sx[8], dp[8];
    flash_qk_dp(smQ, smDO, smK, smV, row0, lane, sx, dp);
    bool tile_full = (k0 + BC <= t0);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        sx[r] *= scale;
        int tr = t0 + row0 + r;
        int u = k0 + lane;
        bool ok = (tr < seq) && (u < seq) && (tile_full || (u <= tr));
        if (!ok) sx[r] = NEG;
        float p = prob_of(sx[r], m[r], l[r]);
        float ds = p * (dp[r] - D[r]);
        smP[(row0 + r) * BC + lane] = p;
        smDS[(row0 + r) * BC + lane] = ds;
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        float g0 = 0.f, g1 = 0.f;
        int q = row0 + r;
#pragma unroll
        for (int k = 0; k < BC; ++k) {
            float s = smDS[q * BC + k];
            g0 = fmaf(s, smK[lane * BC + k], g0);
            g1 = fmaf(s, smK[(lane + 32) * BC + k], g1);
        }
        dq0[r] += g0;
        dq1[r] += g1;
    }
#pragma unroll 1
    for (int e = 0; e < 8; ++e) {
        int idx = tid + e * 256;
        int k = idx >> 6;
        int d = idx & 63;
        float ak = 0.f, av = 0.f;
#pragma unroll 4
        for (int q = 0; q < BT; ++q) {
            ak = fmaf(smDS[q * BC + k], smQ[d * QSTR + q], ak);
            av = fmaf(smP[q * BC + k], smDO[d * QSTR + q], av);
        }
        smDK[d * BC + k] += ak;
        smDV[d * BC + k] += av;
    }
    __syncthreads();
}

// Two query tiles per block. dK/dV of those tiles are summed in smem, then
// one atomicAdd. dQ stays in registers. Grid y is ceil(seq / 128).
extern "C" __global__ void __launch_bounds__(256, 2)
attn_bwd_flash_g2(const float* __restrict__ dO, const float* __restrict__ Q,
                  const float* __restrict__ K, const float* __restrict__ V,
                  const float* __restrict__ O,
                  const float* __restrict__ m_in, const float* __restrict__ l_in,
                  float* __restrict__ dQ, float* __restrict__ dK, float* __restrict__ dV,
                  int seq, int nh, int dh, float scale) {
    const int h = (int)blockIdx.x;
    const int tA = (int)blockIdx.y * (BT * 2);
    const int tB = tA + BT;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = warp * 8;
    if (tA >= seq || dh != DH) return;

    float* smQ = smem;
    float* smDO = smQ + 64 * QSTR;
    float* smK = smDO + 64 * QSTR;
    float* smV = smK + 64 * BC;
    float* smP = smV + 64 * BC;
    float* smDS = smP + 64 * BC;
    float* smDK = smDS + 64 * BC;
    float* smDV = smDK + 64 * BC;

    float dqA0[8], dqA1[8], dqB0[8], dqB1[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        dqA0[r] = 0.f;
        dqA1[r] = 0.f;
        dqB0[r] = 0.f;
        dqB1[r] = 0.f;
    }

    int last = tA + BT - 1;
    if (tB < seq) last = tB + BT - 1;
    if (last >= seq) last = seq - 1;
    int k_end = last + 1;
    int ntiles = (k_end + BC - 1) / BC;
    int phase = ntiles > 0 ? ((int)blockIdx.y % ntiles) : 0;
#pragma unroll 1
    for (int step = 0; step < ntiles; ++step) {
        int k0 = ((phase + step) % ntiles) * BC;
        for (int e = tid; e < 64 * BC; e += 256) {
            smDK[e] = 0.f;
            smDV[e] = 0.f;
        }
        __syncthreads();
        load_KT(K, smK, k0, h, nh, dh, tid, seq);
        load_KT(V, smV, k0, h, nh, dh, tid, seq);
        __syncthreads();
        if (k0 < tA + BT) {
            flash_bwd_one_tile(smQ, smDO, smK, smV, smP, smDS, smDK, smDV,
                               dO, Q, O, m_in, l_in, tA, k0, h, seq, nh, dh, scale,
                               tid, lane, row0, dqA0, dqA1);
        }
        if (tB < seq && k0 < tB + BT) {
            flash_bwd_one_tile(smQ, smDO, smK, smV, smP, smDS, smDK, smDV,
                               dO, Q, O, m_in, l_in, tB, k0, h, seq, nh, dh, scale,
                               tid, lane, row0, dqB0, dqB1);
        }
#pragma unroll 1
        for (int e = 0; e < 8; ++e) {
            int idx = tid + e * 256;
            int k = idx >> 6;
            int d = idx & 63;
            int tok = k0 + k;
            if (tok < seq) {
                atomicAdd(dK + qkv_at(tok, h, nh, dh) + d, scale * smDK[d * BC + k]);
                atomicAdd(dV + qkv_at(tok, h, nh, dh) + d, smDV[d * BC + k]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        int tr = tA + row0 + r;
        if (tr < seq) {
            float* rowp = dQ + qkv_at(tr, h, nh, dh);
            rowp[lane] = scale * dqA0[r];
            rowp[lane + 32] = scale * dqA1[r];
        }
        tr = tB + row0 + r;
        if (tB < seq && tr < seq) {
            float* rowp = dQ + qkv_at(tr, h, nh, dh);
            rowp[lane] = scale * dqB0[r];
            rowp[lane + 32] = scale * dqB1[r];
        }
    }
}

// ---------------------------------------------------------------------------
// Backward. Online softmax is recomputed; P is never loaded or stored.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void dot8_bc(const float* __restrict__ smQ, const float* __restrict__ smK,
                                        int row0, int lane, float sx[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) sx[i] = 0.f;
#pragma unroll 8
    for (int d = 0; d < 64; ++d) {
        float kv = smK[d * BC + lane];
        float4 q0 = *reinterpret_cast<const float4*>(smQ + d * 64 + row0);
        float4 q1 = *reinterpret_cast<const float4*>(smQ + d * 64 + row0 + 4);
        sx[0] = fmaf(q0.x, kv, sx[0]);
        sx[1] = fmaf(q0.y, kv, sx[1]);
        sx[2] = fmaf(q0.z, kv, sx[2]);
        sx[3] = fmaf(q0.w, kv, sx[3]);
        sx[4] = fmaf(q1.x, kv, sx[4]);
        sx[5] = fmaf(q1.y, kv, sx[5]);
        sx[6] = fmaf(q1.z, kv, sx[6]);
        sx[7] = fmaf(q1.w, kv, sx[7]);
    }
}

__device__ __forceinline__ void soft_one(float &sx, float &mx, float &lx, float &ax, float &ay) {
    float mt = warp_max(sx);
    float mn = fmaxf(mx, mt);
    float al = (mx > -1.0e20f) ? __expf(mx - mn) : 0.f;
    float p = (sx > -1.0e20f) ? __expf(sx - mn) : 0.f;
    lx = lx * al + warp_sum(p);
    ax *= al;
    ay *= al;
    mx = mn;
    sx = p;
}

// One causal online-softmax sweep over query tile q0.
// Leaves m/l and unnormalized O (ax/ay) in the arrays.
// If cap0/cap1 >= 0, raw masked scores of those Bc tiles are kept in sA/sB
// (lane's column). Q must already sit in smQ.
__device__ void sweep_stats(const float* __restrict__ K, const float* __restrict__ V,
                            float* smQ, float* smK, float* smV,
                            int q0, int h, int seq, int nh, int dh, float scale,
                            int tid, int lane, int row0,
                            int cap0, int cap1,
                            float m[8], float l[8], float ax[8], float ay[8],
                            float sA[8], float sB[8]) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        m[r] = NEG;
        l[r] = 0.f;
        ax[r] = 0.f;
        ay[r] = 0.f;
        sA[r] = NEG;
        sB[r] = NEG;
    }
    int k_end = q0 + BT;
    if (k_end > seq) k_end = seq;
#pragma unroll 1
    for (int k0 = 0; k0 < k_end; k0 += BC) {
        load_KT(K, smK, k0, h, nh, dh, tid, seq);
        load_V(V, smV, k0, h, nh, dh, tid, seq);
        __syncthreads();
        float sx[8];
        dot8_bc(smQ, smK, row0, lane, sx);
        bool full = (k0 + BC <= q0);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            int tr = q0 + row0 + r;
            int u = k0 + lane;
            sx[r] *= scale;
            bool ok = (tr < seq) && (u < seq) && (full || (u <= tr));
            if (!ok) sx[r] = NEG;
        }
        if (k0 == cap0) {
#pragma unroll
            for (int r = 0; r < 8; ++r) sA[r] = sx[r];
        }
        if (k0 == cap1) {
#pragma unroll
            for (int r = 0; r < 8; ++r) sB[r] = sx[r];
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) soft_one(sx[r], m[r], l[r], ax[r], ay[r]);
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 8; ++r) smK[(row0 + r) * BC + lane] = sx[r];
        __syncthreads();
#pragma unroll
        for (int c = 0; c < BC; ++c) {
            float v0 = smV[c * 64 + lane];
            float v1 = smV[c * 64 + lane + 32];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                float p = smK[(row0 + r) * BC + c];
                ax[r] = fmaf(p, v0, ax[r]);
                ay[r] = fmaf(p, v1, ay[r]);
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void row_D(const float* __restrict__ smDO, int row0, int lane,
                                      const float ax[8], const float ay[8], const float l[8],
                                      float D[8]) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        float part = smDO[lane * BT + row0 + r] * ax[r] + smDO[(lane + 32) * BT + row0 + r] * ay[r];
        float red = warp_sum(part);
        D[r] = (l[r] > 0.f) ? red / l[r] : 0.f;
    }
}

// dO dims [d0, d0+32) into dst[local_d][row].
__device__ void load_do_half(const float* __restrict__ dO, float* __restrict__ dst,
                             int q0, int h, int nh, int dh, int d0, int tid, int seq) {
#pragma unroll 1
    for (int i = tid; i < (64 * 8); i += 256) {
        int row = i >> 3;
        int d4 = i & 7;
        int tok = q0 + row;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tok < seq)
            v = *reinterpret_cast<const float4*>(dO + qkv_at(tok, h, nh, dh) + d0 + (d4 << 2));
        int ld = d4 << 2;
        dst[(ld + 0) * BT + row] = v.x;
        dst[(ld + 1) * BT + row] = v.y;
        dst[(ld + 2) * BT + row] = v.z;
        dst[(ld + 3) * BT + row] = v.w;
    }
}

__device__ void load_K_rows(const float* __restrict__ K, float* __restrict__ dst,
                            int k0, int h, int nh, int dh, int tid, int seq) {
#pragma unroll 1
    for (int i = tid; i < (BC * 16); i += 256) {
        int row = i >> 4;
        int d4 = i & 15;
        int tok = k0 + row;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tok < seq)
            v = *reinterpret_cast<const float4*>(K + qkv_at(tok, h, nh, dh) + (d4 << 2));
        *reinterpret_cast<float4*>(dst + row * DH + (d4 << 2)) = v;
    }
}

// dQ for the query tile this block owns. Two sweeps: stats, then dS @ K.
__device__ __noinline__ void bwd_dq(const float* __restrict__ dO, const float* __restrict__ Q,
                                    const float* __restrict__ K, const float* __restrict__ V,
                                    float* __restrict__ dQ,
                                    int t0, int h, int seq, int nh, int dh, float scale,
                                    int tid, int lane, int row0) {
    float* smQ = smem;
    float* smK = smem + 4096;
    float* smV = smK + 64 * BC;

    load_QT(Q, smQ, t0, h, nh, dh, tid, seq);
    __syncthreads();

    float m[8], l[8], ax[8], ay[8], sinkA[8], sinkB[8];
    sweep_stats(K, V, smQ, smK, smV, t0, h, seq, nh, dh, scale, tid, lane, row0,
                -1, -1, m, l, ax, ay, sinkA, sinkB);

    // D = dot(dO, O), O = ax/l. dO reuses the K/V half so Q stays put.
    float* smDO = smK;
    load_transpose(dO, smDO, t0, h, nh, dh, tid, seq);
    __syncthreads();
    float D[8];
    row_D(smDO, row0, lane, ax, ay, l, D);
    __syncthreads();

    float dq0[8], dq1[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        dq0[r] = 0.f;
        dq1[r] = 0.f;
    }

    int k_end = t0 + BT;
    if (k_end > seq) k_end = seq;
#pragma unroll 1
    for (int k0 = 0; k0 < k_end; k0 += BC) {
        load_KT(K, smK, k0, h, nh, dh, tid, seq);
        load_V(V, smV, k0, h, nh, dh, tid, seq);
        __syncthreads();
        float sx[8];
        dot8_bc(smQ, smK, row0, lane, sx);
        bool full = (k0 + BC <= t0);
        float p[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            int tr = t0 + row0 + r;
            int u = k0 + lane;
            sx[r] *= scale;
            bool ok = (tr < seq) && (u < seq) && (full || (u <= tr));
            if (!ok) sx[r] = NEG;
            p[r] = prob_of(sx[r], m[r], l[r]);
        }
        __syncthreads();

        float dp[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) dp[r] = 0.f;
        for (int half = 0; half < 2; ++half) {
            load_do_half(dO, smK, t0, h, nh, dh, half * 32, tid, seq);
            __syncthreads();
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                float a = 0.f;
#pragma unroll
                for (int d = 0; d < 32; ++d)
                    a = fmaf(smK[d * BT + row0 + r], smV[lane * 64 + half * 32 + d], a);
                dp[r] += a;
            }
            __syncthreads();
        }

        float ds[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) ds[r] = p[r] * (dp[r] - D[r]);

        float* smDS = smem + 4096;
        float* smKR = smDS + BC * DH;
#pragma unroll
        for (int r = 0; r < 8; ++r) smDS[lane * BT + row0 + r] = ds[r];
        load_K_rows(K, smKR, k0, h, nh, dh, tid, seq);
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            float g0 = 0.f, g1 = 0.f;
#pragma unroll
            for (int c = 0; c < BC; ++c) {
                float s = smDS[c * BT + row0 + r];
                g0 = fmaf(s, smKR[c * DH + lane], g0);
                g1 = fmaf(s, smKR[c * DH + lane + 32], g1);
            }
            dq0[r] += g0;
            dq1[r] += g1;
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        int tr = t0 + row0 + r;
        if (tr < seq) {
            float* rowp = dQ + qkv_at(tr, h, nh, dh);
            rowp[lane] = scale * dq0[r];
            rowp[lane + 32] = scale * dq1[r];
        }
    }
    __syncthreads();
}

// One later (or diagonal) query tile's contribution to this block's key tile.
__device__ __noinline__ void bwd_dkv_one(const float* __restrict__ dO, const float* __restrict__ Q,
                                         const float* __restrict__ K, const float* __restrict__ V,
                                         int q0, int t0, int h, int seq, int nh, int dh, float scale,
                                         int tid, int lane, int warp, int row0,
                                         float dk0[8], float dk1[8], float dv0[8], float dv1[8]) {
    float* smQ = smem;
    float* smK = smem + 4096;
    float* smV = smK + 64 * BC;

    load_QT(Q, smQ, q0, h, nh, dh, tid, seq);
    __syncthreads();

    float m[8], l[8], ax[8], ay[8], sA[8], sB[8];
    // t0 and t0+32 are the two Bc slices of the owned key tile.
    sweep_stats(K, V, smQ, smK, smV, q0, h, seq, nh, dh, scale, tid, lane, row0,
                t0, t0 + BC, m, l, ax, ay, sA, sB);

    float* smA = smem;
    float* smB = smem + BT * DH;
    load_transpose(dO, smA, q0, h, nh, dh, tid, seq);
    load_transpose(V, smB, t0, h, nh, dh, tid, seq);
    __syncthreads();
    float dp0[8], dp1[8];
    qk_dot8(smA, smB, row0, lane, lane + 32, dp0, dp1);
    float D[8];
    row_D(smA, row0, lane, ax, ay, l, D);

    float ds0[8], ds1[8], p0[8], p1[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        p0[r] = prob_of(sA[r], m[r], l[r]);
        p1[r] = prob_of(sB[r], m[r], l[r]);
        ds0[r] = p0[r] * (dp0[r] - D[r]);
        ds1[r] = p1[r] * (dp1[r] - D[r]);
    }
    __syncthreads();

    // dK: [query][key] x Q[token][dim].
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        smB[(row0 + r) * BT + lane] = ds0[r];
        smB[(row0 + r) * BT + lane + 32] = ds1[r];
    }
    load_rowmajor(Q, smA, q0, h, nh, dh, tid, seq);
    __syncthreads();
    float g0[8], g1[8];
    qk_dot8(smA, smB, row0, lane, lane + 32, g0, g1);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        dk0[r] += g0[r];
        dk1[r] += g1[r];
    }
    __syncthreads();

    // dV: same contraction with P and dO. No scale.
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        smB[(row0 + r) * BT + lane] = p0[r];
        smB[(row0 + r) * BT + lane + 32] = p1[r];
    }
    load_rowmajor(dO, smA, q0, h, nh, dh, tid, seq);
    __syncthreads();
    qk_dot8(smA, smB, row0, lane, lane + 32, g0, g1);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        dv0[r] += g0[r];
        dv1[r] += g1[r];
    }
    __syncthreads();
}

extern "C" __global__ void attn_bwd_recompute(const float* __restrict__ dO, const float* __restrict__ Q,
                                              const float* __restrict__ K, const float* __restrict__ V,
                                              float* __restrict__ dQ, float* __restrict__ dK,
                                              float* __restrict__ dV,
                                              int seq, int nh, int dh, float scale) {
    const int h = (int)blockIdx.x;
    const int t0 = (int)blockIdx.y * BT;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = warp * 8;
    if (t0 >= seq || dh != DH) return;

    bwd_dq(dO, Q, K, V, dQ, t0, h, seq, nh, dh, scale, tid, lane, row0);

    float dk0[8], dk1[8], dv0[8], dv1[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        dk0[r] = 0.f;
        dk1[r] = 0.f;
        dv0[r] = 0.f;
        dv1[r] = 0.f;
    }
#pragma unroll 1
    for (int q0 = t0; q0 < seq; q0 += BT)
        bwd_dkv_one(dO, Q, K, V, q0, t0, h, seq, nh, dh, scale, tid, lane, warp, row0,
                    dk0, dk1, dv0, dv1);

    flush_owned(dK, smem, t0, h, seq, nh, dh, lane, warp, dk0, dk1, scale);
    flush_owned(dV, smem, t0, h, seq, nh, dh, lane, warp, dv0, dv1, 1.f);
}

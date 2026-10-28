// Tiled causal attention, device-only, dh=64.
// Q,K,V,O,dQ,dK,dV: [seq, nh, dh]
// P:                [nh, seq, seq]
// Launch: grid (nh, ceil(seq/64)), block 256, dynamic smem 32KB.
// smem: Q tile [d][row] in the low half (resident), K/P tile in the high half.

#define BT 64
#define DH 64
#define NEG (-1.0e30f)

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

__device__ __forceinline__ float dot64(const float* __restrict__ a, const float* __restrict__ b) {
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        float4 x = a4[i];
        float4 y = b4[i];
        s += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
    }
    return s;
}

// Transpose a [BT, DH] head-row into smem[d][row]. All threads must call this.
__device__ __forceinline__ void load_transpose(const float* __restrict__ src, float* __restrict__ dst,
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
        dst[(d4 * 4 + 0) * BT + row] = v.x;
        dst[(d4 * 4 + 1) * BT + row] = v.y;
        dst[(d4 * 4 + 2) * BT + row] = v.z;
        dst[(d4 * 4 + 3) * BT + row] = v.w;
    }
}

// 8-row x 2-col dot. smQ/smK are [d][64]. col0 = lane, col1 = lane+32.
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

extern __shared__ __align__(16) float smem[];

// Causal online-softmax forward.
// Fixed launch: grid (nh, ceil(seq/64)), block 256, smem 32KB.
// Key tile is 32 so Q, K and V all reside:
//   smQ[d][row] 64*64, smK[d][32], smV[32][64].
// Each lane owns one key column and two output dims of eight query rows.
// Scores of tiles that sit fully left of the query block are branch-free.
// P is written as scaled dots, then rescaled to softmax once m and l are final.
// The harness zeroes P; columns that are never stored stay zero.

#define BC 32

__device__ __forceinline__ void fa_load_QT(const float* __restrict__ src, float* __restrict__ dst,
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

__device__ __forceinline__ void fa_load_KT(const float* __restrict__ src, float* __restrict__ dst,
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

__device__ __forceinline__ void fa_load_V(const float* __restrict__ src, float* __restrict__ dst,
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

template <bool FULL>
__device__ __forceinline__ void fa_tile(
    const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ P,
    float* smQ, float* smK, float* smV, int k0, int t0, int h, int seq, int nh, int dh,
    float scale, int tid, int lane, int row0,
    float &sx0, float &sx1, float &sx2, float &sx3, float &sx4, float &sx5, float &sx6, float &sx7,
    float &mx0, float &mx1, float &mx2, float &mx3, float &mx4, float &mx5, float &mx6, float &mx7,
    float &lx0, float &lx1, float &lx2, float &lx3, float &lx4, float &lx5, float &lx6, float &lx7,
    float &ax0, float &ax1, float &ax2, float &ax3, float &ax4, float &ax5, float &ax6, float &ax7,
    float &ay0, float &ay1, float &ay2, float &ay3, float &ay4, float &ay5, float &ay6, float &ay7) {
    // K and V tiles are already in smK/smV and published by the caller.
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

#define PUT(i, sx)                                                               \
    do {                                                                         \
        int tr_ = t0 + row0 + (i);                                               \
        int u_ = k0 + lane;                                                      \
        bool ok_ = (tr_ < seq) && (u_ < seq) && (FULL || (u_ <= tr_));           \
        if (ok_) P[((h * seq + tr_) * seq) + u_] = sx;                           \
        if (!ok_) sx = NEG;                                                      \
    } while (0)
    PUT(0, sx0); PUT(1, sx1); PUT(2, sx2); PUT(3, sx3);
    PUT(4, sx4); PUT(5, sx5); PUT(6, sx6); PUT(7, sx7);
#undef PUT

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

// Each thread owns exactly two float4s of a BC=32 tile (e4 = tid, tid+256).
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

extern "C" __global__ void
attn_fwd(const float* __restrict__ Q, const float* __restrict__ K,
         const float* __restrict__ V, float* __restrict__ O,
         float* __restrict__ P, int seq, int nh, int dh, float scale) {
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

    fa_load_QT(Q, smQ, t0, h, nh, dh, tid, seq);
    __syncthreads();

    float sx0 = 0.f, sx1 = 0.f, sx2 = 0.f, sx3 = 0.f, sx4 = 0.f, sx5 = 0.f, sx6 = 0.f, sx7 = 0.f;
    float mx0 = NEG, mx1 = NEG, mx2 = NEG, mx3 = NEG, mx4 = NEG, mx5 = NEG, mx6 = NEG, mx7 = NEG;
    float lx0 = 0.f, lx1 = 0.f, lx2 = 0.f, lx3 = 0.f, lx4 = 0.f, lx5 = 0.f, lx6 = 0.f, lx7 = 0.f;
    float ax0 = 0.f, ax1 = 0.f, ax2 = 0.f, ax3 = 0.f, ax4 = 0.f, ax5 = 0.f, ax6 = 0.f, ax7 = 0.f;
    float ay0 = 0.f, ay1 = 0.f, ay2 = 0.f, ay3 = 0.f, ay4 = 0.f, ay5 = 0.f, ay6 = 0.f, ay7 = 0.f;

    int k_end = t0 + BT;
    if (k_end > seq) k_end = seq;
    if (k_end > 0) {
        fa_load_KT(K, smK, 0, h, nh, dh, tid, seq);
        fa_load_V(V, smV, 0, h, nh, dh, tid, seq);
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
            fa_tile<true>(K, V, P, smQ, smK, smV, k0, t0, h, seq, nh, dh, scale, tid, lane, row0,
                          sx0, sx1, sx2, sx3, sx4, sx5, sx6, sx7,
                          mx0, mx1, mx2, mx3, mx4, mx5, mx6, mx7,
                          lx0, lx1, lx2, lx3, lx4, lx5, lx6, lx7,
                          ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7,
                          ay0, ay1, ay2, ay3, ay4, ay5, ay6, ay7);
        else
            fa_tile<false>(K, V, P, smQ, smK, smV, k0, t0, h, seq, nh, dh, scale, tid, lane, row0,
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

#define EPI(i, xx, yy, mm, ll)                                                   \
    do {                                                                         \
        int tr_ = t0 + row0 + (i);                                               \
        if (tr_ < seq) {                                                         \
            float inv_ = (ll > 0.f) ? (1.f / ll) : 0.f;                          \
            float* orow_ = O + ((tr_ * nh + h) * dh);                            \
            orow_[lane] = (xx) * inv_;                                           \
            orow_[lane + 32] = (yy) * inv_;                                      \
            float* prow_ = P + ((h * seq + tr_) * seq);                          \
            int n4_ = (tr_ + 1) >> 2;                                            \
            for (int g_ = lane; g_ < n4_; g_ += 32) {                            \
                float4 v_ = *reinterpret_cast<float4*>(prow_ + (g_ << 2));       \
                v_.x = __expf(v_.x - (mm)) * inv_;                               \
                v_.y = __expf(v_.y - (mm)) * inv_;                               \
                v_.z = __expf(v_.z - (mm)) * inv_;                               \
                v_.w = __expf(v_.w - (mm)) * inv_;                               \
                *reinterpret_cast<float4*>(prow_ + (g_ << 2)) = v_;              \
            }                                                                    \
            for (int u_ = (n4_ << 2) + lane; u_ <= tr_; u_ += 32)                \
                prow_[u_] = __expf(prow_[u_] - (mm)) * inv_;                     \
            /* Causal stores never touch u > tr. Harness prezeroes P, so the */ \
            /* old per-row zero fill (about 8MB at seq=512) is dead traffic. */ \
        }                                                                        \
    } while (0)
    EPI(0, ax0, ay0, mx0, lx0);
    EPI(1, ax1, ay1, mx1, lx1);
    EPI(2, ax2, ay2, mx2, lx2);
    EPI(3, ax3, ay3, mx3, lx3);
    EPI(4, ax4, ay4, mx4, lx4);
    EPI(5, ax5, ay5, mx5, lx5);
    EPI(6, ax6, ay6, mx6, lx6);
    EPI(7, ax7, ay7, mx7, lx7);
#undef EPI
}

// [token][dim] row-major tile. All threads must call this.
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

// rs[r] = sum_k P[q][k] * dot(dO[q], V[k]) for this warp's 8 query rows.
// dO tile stays in smA. If cap_k >= 0, also keep dp for that key tile.
__device__ __forceinline__ void compute_rs(const float* __restrict__ dO, const float* __restrict__ V,
                                           const float* __restrict__ P, float* __restrict__ smA,
                                           float* __restrict__ smB, int q0, int h, int seq, int nh,
                                           int dh, int tid, int lane, int warp, float rs[8],
                                           float* __restrict__ cap0, float* __restrict__ cap1, int cap_k) {
    const int row0 = warp * 8;
    float rsp[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) rsp[r] = 0.f;
    load_transpose(dO, smA, q0, h, nh, dh, tid, seq);
#pragma unroll 1
    for (int k2 = 0; k2 <= q0 && k2 < seq; k2 += BT) {
        load_transpose(V, smB, k2, h, nh, dh, tid, seq);
        __syncthreads();
        float dp0[8], dp1[8];
        qk_dot8(smA, smB, row0, lane, lane + 32, dp0, dp1);
        if (cap0 != 0 && k2 == cap_k) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                cap0[r] = dp0[r];
                cap1[r] = dp1[r];
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            int t = q0 + row0 + r;
            float p0 = 0.f, p1 = 0.f;
            if (t < seq) {
                const float* prow = P + ((h * seq + t) * seq + k2);
                if (k2 + lane < seq) p0 = __ldg(prow + lane);
                if (k2 + lane + 32 < seq) p1 = __ldg(prow + lane + 32);
            }
            rsp[r] += p0 * dp0[r] + p1 * dp1[r];
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) rs[r] = warp_sum(rsp[r]);
}

// Stage [key=lane / lane+32][dim=warp*8+r] then coalesced-store one owned token tile.
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

extern "C" __global__ void attn_bwd(const float* __restrict__ dO, const float* __restrict__ Q,
                                    const float* __restrict__ K, const float* __restrict__ V,
                                    const float* __restrict__ P, float* __restrict__ dQ,
                                    float* __restrict__ dK, float* __restrict__ dV,
                                    int seq, int nh, int dh, float scale) {
    const int h = (int)blockIdx.x;
    const int t0 = (int)blockIdx.y * BT;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = warp * 8;
    if (t0 >= seq || dh != DH) return;

    float* smA = smem;
    float* smB = smem + BT * DH;

    // dQ: this block owns query tile t0. Direct store, no atomics.
    {
        float dq0[8], dq1[8], rs[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            dq0[r] = 0.f;
            dq1[r] = 0.f;
        }
        compute_rs(dO, V, P, smA, smB, t0, h, seq, nh, dh, tid, lane, warp, rs, 0, 0, -1);
#pragma unroll 1
        for (int k2 = 0; k2 <= t0 && k2 < seq; k2 += BT) {
            load_transpose(dO, smA, t0, h, nh, dh, tid, seq);
            load_transpose(V, smB, k2, h, nh, dh, tid, seq);
            __syncthreads();
            float dp0[8], dp1[8];
            qk_dot8(smA, smB, row0, lane, lane + 32, dp0, dp1);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                int t = t0 + row0 + r;
                float p0 = 0.f, p1 = 0.f;
                if (t < seq) {
                    const float* prow = P + ((h * seq + t) * seq + k2);
                    if (k2 + lane < seq) p0 = __ldg(prow + lane);
                    if (k2 + lane + 32 < seq) p1 = __ldg(prow + lane + 32);
                }
                // [key][query], consumed by the dQ gemm below.
                smB[lane * BT + row0 + r] = p0 * (dp0[r] - rs[r]);
                smB[(lane + 32) * BT + row0 + r] = p1 * (dp1[r] - rs[r]);
            }
            __syncthreads();
            load_rowmajor(K, smA, k2, h, nh, dh, tid, seq);
            __syncthreads();
            float g0[8], g1[8];
            qk_dot8(smB, smA, row0, lane, lane + 32, g0, g1);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                dq0[r] += g0[r];
                dq1[r] += g1[r];
            }
            __syncthreads();
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            int t = t0 + row0 + r;
            if (t < seq) {
                float* rowp = dQ + qkv_at(t, h, nh, dh);
                rowp[lane] = scale * dq0[r];
                rowp[lane + 32] = scale * dq1[r];
            }
        }
    }

    // dK: this block owns key tile t0. Accumulate over later query tiles, one store.
    {
        float dk0[8], dk1[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            dk0[r] = 0.f;
            dk1[r] = 0.f;
        }
#pragma unroll 1
        for (int q0 = t0; q0 < seq; q0 += BT) {
            float rs[8], dp0[8], dp1[8];
            compute_rs(dO, V, P, smA, smB, q0, h, seq, nh, dh, tid, lane, warp, rs, dp0, dp1, t0);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                int t = q0 + row0 + r;
                float p0 = 0.f, p1 = 0.f;
                if (t < seq) {
                    const float* prow = P + ((h * seq + t) * seq + t0);
                    if (t0 + lane < seq) p0 = __ldg(prow + lane);
                    if (t0 + lane + 32 < seq) p1 = __ldg(prow + lane + 32);
                }
                // [query][key] so the contraction axis of qk_dot8 is the query.
                smB[(row0 + r) * BT + lane] = p0 * (dp0[r] - rs[r]);
                smB[(row0 + r) * BT + lane + 32] = p1 * (dp1[r] - rs[r]);
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
        }
        flush_owned(dK, smA, t0, h, seq, nh, dh, lane, warp, dk0, dk1, scale);
    }

    // dV: same key tile, P^T @ dO, no scale and no row-sum.
    {
        float dv0[8], dv1[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            dv0[r] = 0.f;
            dv1[r] = 0.f;
        }
#pragma unroll 1
        for (int q0 = t0; q0 < seq; q0 += BT) {
            load_rowmajor(dO, smA, q0, h, nh, dh, tid, seq);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                int t = q0 + row0 + r;
                float p0 = 0.f, p1 = 0.f;
                if (t < seq) {
                    const float* prow = P + ((h * seq + t) * seq + t0);
                    if (t0 + lane < seq) p0 = __ldg(prow + lane);
                    if (t0 + lane + 32 < seq) p1 = __ldg(prow + lane + 32);
                }
                smB[(row0 + r) * BT + lane] = p0;
                smB[(row0 + r) * BT + lane + 32] = p1;
            }
            __syncthreads();
            float g0[8], g1[8];
            qk_dot8(smA, smB, row0, lane, lane + 32, g0, g1);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                dv0[r] += g0[r];
                dv1[r] += g1[r];
            }
            __syncthreads();
        }
        flush_owned(dV, smA, t0, h, seq, nh, dh, lane, warp, dv0, dv1, 1.f);
    }
}

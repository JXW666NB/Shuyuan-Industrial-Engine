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

extern "C" __global__ void attn_fwd(const float* __restrict__ Q, const float* __restrict__ K,
                                    const float* __restrict__ V, float* __restrict__ O,
                                    float* __restrict__ P, int seq, int nh, int dh, float scale) {
    const int h = (int)blockIdx.x;
    const int t0 = (int)blockIdx.y * BT;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = warp * 8;
    const int col0 = lane;
    const int col1 = lane + 32;
    if (t0 >= seq || dh != DH) return;

    float* smQ = smem;
    float* smK = smem + BT * DH;

    // Upper triangle (and anything this tile does not overwrite) is zero.
    {
        int rows = seq - t0;
        if (rows > BT) rows = BT;
        for (int r = 0; r < rows; ++r) {
            int t = t0 + r;
            float* prow = P + ((h * seq + t) * seq);
            for (int u = t + 1 + tid; u < seq; u += 256) prow[u] = 0.f;
        }
    }

    load_transpose(Q, smQ, t0, h, nh, dh, tid, seq);
    __syncthreads();

    float m[8], l[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        m[r] = NEG;
        l[r] = 0.f;
    }

    // Pass 1: per-thread online softmax over the keys this lane owns, then warp-merge.
    for (int k0 = 0; k0 <= t0; k0 += BT) {
        load_transpose(K, smK, k0, h, nh, dh, tid, seq);
        __syncthreads();
        float a0[8], a1[8];
        qk_dot8(smQ, smK, row0, col0, col1, a0, a1);
        bool diag = (k0 == t0);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            int lr = row0 + r;
            int t = t0 + lr;
            if (t < seq) {
                if (!diag || col0 <= lr) {
                    float s = a0[r] * scale;
                    float mn = fmaxf(m[r], s);
                    l[r] = l[r] * expf(m[r] - mn) + expf(s - mn);
                    m[r] = mn;
                }
                if (!diag || col1 <= lr) {
                    float s = a1[r] * scale;
                    float mn = fmaxf(m[r], s);
                    l[r] = l[r] * expf(m[r] - mn) + expf(s - mn);
                    m[r] = mn;
                }
            }
        }
        __syncthreads();
    }

    float inv_l[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        float gm = warp_max(m[r]);
        float contrib = (l[r] == 0.f) ? 0.f : l[r] * expf(m[r] - gm);
        float gl = warp_sum(contrib);
        m[r] = gm;
        l[r] = gl;
        inv_l[r] = (gl > 0.f) ? (1.f / gl) : 0.f;
    }

    float o0[8], o1[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        o0[r] = 0.f;
        o1[r] = 0.f;
    }

    // Pass 2: rewrite probabilities, accumulate O = P @ V. Q tile is still in smQ.
    for (int k0 = 0; k0 <= t0; k0 += BT) {
        load_transpose(K, smK, k0, h, nh, dh, tid, seq);
        __syncthreads();
        float a0[8], a1[8];
        qk_dot8(smQ, smK, row0, col0, col1, a0, a1);
        bool diag = (k0 == t0);
        float p0[8], p1[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            int lr = row0 + r;
            int t = t0 + lr;
            float pv0 = 0.f, pv1 = 0.f;
            if (t < seq) {
                int u0 = k0 + col0;
                int u1 = k0 + col1;
                if (u0 < seq && (!diag || col0 <= lr)) pv0 = expf(a0[r] * scale - m[r]) * inv_l[r];
                if (u1 < seq && (!diag || col1 <= lr)) pv1 = expf(a1[r] * scale - m[r]) * inv_l[r];
                float* prow = P + ((h * seq + t) * seq + k0);
                if (u0 < seq) prow[col0] = pv0;
                if (u1 < seq) prow[col1] = pv1;
            }
            p0[r] = pv0;
            p1[r] = pv1;
        }
        __syncthreads();
        // Park the 64x64 prob tile as [k][row] over the K half.
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            smK[col0 * BT + row0 + r] = p0[r];
            smK[col1 * BT + row0 + r] = p1[r];
        }
        __syncthreads();

        for (int k = 0; k < BT; ++k) {
            int u = k0 + k;
            float v0 = 0.f, v1 = 0.f;
            if (u < seq) {
                const float* vrow = V + qkv_at(u, h, nh, dh);
                v0 = __ldg(vrow + lane);
                v1 = __ldg(vrow + lane + 32);
            }
            float4 b0 = *reinterpret_cast<const float4*>(smK + k * BT + row0);
            float4 b1 = *reinterpret_cast<const float4*>(smK + k * BT + row0 + 4);
            o0[0] += b0.x * v0;
            o1[0] += b0.x * v1;
            o0[1] += b0.y * v0;
            o1[1] += b0.y * v1;
            o0[2] += b0.z * v0;
            o1[2] += b0.z * v1;
            o0[3] += b0.w * v0;
            o1[3] += b0.w * v1;
            o0[4] += b1.x * v0;
            o1[4] += b1.x * v1;
            o0[5] += b1.y * v0;
            o1[5] += b1.y * v1;
            o0[6] += b1.z * v0;
            o1[6] += b1.z * v1;
            o0[7] += b1.w * v0;
            o1[7] += b1.w * v1;
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        int t = t0 + row0 + r;
        if (t < seq) {
            float* orow = O + qkv_at(t, h, nh, dh);
            orow[lane] = o0[r];
            orow[lane + 32] = o1[r];
        }
    }
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

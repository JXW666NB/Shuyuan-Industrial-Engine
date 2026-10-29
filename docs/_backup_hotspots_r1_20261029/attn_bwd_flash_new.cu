extern "C" __global__ void __launch_bounds__(256, 1)
attn_bwd_flash(const float* __restrict__ dO, const float* __restrict__ Q,
               const float* __restrict__ K, const float* __restrict__ V,
               const float* __restrict__ O,
               const float* __restrict__ m_in, const float* __restrict__ l_in,
               float* __restrict__ dQ, float* __restrict__ dK, float* __restrict__ dV,
               int seq, int nh, int dh, float scale) {
    const int h = (int)blockIdx.x;
    const int qbase = (int)blockIdx.y * (BT * QGROUP);
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = warp * 8;
    if (qbase >= seq || dh != DH) return;

    float* smQ = smem;
    float* smDO = smQ + 64 * QSTR;
    float* smK = smDO + 64 * QSTR;
    float* smV = smK + 64 * BC;
    float* smP = smV + 64 * BC;
    float* smDS = smP + 64 * BC;
    float* smDK = smDS + 64 * BC;
    float* smDV = smDK + 64 * BC;

    int q_last = qbase + QGROUP * BT - 1;
    if (q_last >= seq) q_last = seq - 1;
    int k_end = q_last + 1;
    int ntiles = (k_end + BC - 1) / BC;

#pragma unroll 1
    for (int kt = 0; kt < ntiles; ++kt) {
        int k0 = kt * BC;
        for (int e = tid; e < 64 * BC; e += 256) {
            smDK[e] = 0.f;
            smDV[e] = 0.f;
        }
        __syncthreads();
        load_KT(K, smK, k0, h, nh, dh, tid, seq);
        load_KT(V, smV, k0, h, nh, dh, tid, seq);
        __syncthreads();

#pragma unroll 1
        for (int qi = 0; qi < QGROUP; ++qi) {
            int t0 = qbase + qi * BT;
            if (t0 >= seq || t0 + BT <= k0) continue;
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
                int tr = t0 + row0 + r;
                if (tr < seq) {
                    float* rowp = dQ + qkv_at(tr, h, nh, dh);
                    rowp[lane] += g0;
                    rowp[lane + 32] += g1;
                }
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
                smDK[d * BC + k] += ak;
                smDV[d * BC + k] += av;
            }
            __syncthreads();
        }

#pragma unroll 1
        for (int e = 0; e < 8; ++e) {
            int idx = tid + e * 256;
            int k = idx >> 6;
            int d = idx & 63;
            int tok = k0 + k;
            if (tok < seq) {
                float* dkrow = dK + qkv_at(tok, h, nh, dh);
                float* dvrow = dV + qkv_at(tok, h, nh, dh);
                atomicAdd(dkrow + d, scale * smDK[d * BC + k]);
                atomicAdd(dvrow + d, smDV[d * BC + k]);
            }
        }
        __syncthreads();
    }

#pragma unroll 1
    for (int qi = 0; qi < QGROUP; ++qi) {
        int t0 = qbase + qi * BT;
        if (t0 >= seq) break;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            int tr = t0 + row0 + r;
            if (tr < seq) {
                float* rowp = dQ + qkv_at(tr, h, nh, dh);
                rowp[lane] *= scale;
                rowp[lane + 32] *= scale;
            }
        }
    }
}

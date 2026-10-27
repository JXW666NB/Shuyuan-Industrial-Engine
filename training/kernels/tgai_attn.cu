// Causal multi-head attention, device-only.
// Q,K,V,O,dQ,dK,dV: [seq, nh, dh] row-major
// P:                [nh, seq, seq] row-major
// One block per head (blockIdx.x). Dynamic smem holds one [seq, seq] tile.

__device__ float dot_n(const float* __restrict__ a, const float* __restrict__ b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

__device__ __forceinline__ int qkv_index(int t, int h, int d, int nh, int dh) {
    return (t * nh + h) * dh + d;
}

extern __shared__ float smem[];

extern "C" __global__ void attn_fwd(const float* __restrict__ Q, const float* __restrict__ K,
                                    const float* __restrict__ V, float* __restrict__ O,
                                    float* __restrict__ P, int seq, int nh, int dh, float scale) {
    const int h = (int)blockIdx.x;
    float* scores = smem;

    for (int idx = (int)threadIdx.x; idx < seq * seq; idx += (int)blockDim.x) {
        const int t = idx / seq;
        const int u = idx - t * seq;
        float s = 0.f;
        if (u <= t) {
            const float* q = Q + qkv_index(t, h, 0, nh, dh);
            const float* k = K + qkv_index(u, h, 0, nh, dh);
            s = dot_n(q, k, dh) * scale;
        }
        scores[idx] = s;
    }
    __syncthreads();

    for (int t = (int)threadIdx.x; t < seq; t += (int)blockDim.x) {
        float* row = scores + t * seq;
        float m = row[0];
        for (int u = 1; u <= t; ++u) m = fmaxf(m, row[u]);
        float sum = 0.f;
        for (int u = 0; u <= t; ++u) {
            float e = expf(row[u] - m);
            row[u] = e;
            sum += e;
        }
        float inv = 1.f / sum;
        for (int u = 0; u <= t; ++u) row[u] *= inv;
        for (int u = t + 1; u < seq; ++u) row[u] = 0.f;
    }
    __syncthreads();

    const int pbase = h * seq * seq;
    for (int idx = (int)threadIdx.x; idx < seq * seq; idx += (int)blockDim.x) {
        P[pbase + idx] = scores[idx];
    }

    for (int idx = (int)threadIdx.x; idx < seq * dh; idx += (int)blockDim.x) {
        const int t = idx / dh;
        const int d = idx - t * dh;
        float acc = 0.f;
        const float* row = scores + t * seq;
        for (int u = 0; u <= t; ++u) {
            acc += row[u] * V[qkv_index(u, h, d, nh, dh)];
        }
        O[qkv_index(t, h, d, nh, dh)] = acc;
    }
}

extern "C" __global__ void attn_bwd(const float* __restrict__ dO, const float* __restrict__ Q,
                                    const float* __restrict__ K, const float* __restrict__ V,
                                    const float* __restrict__ P, float* __restrict__ dQ,
                                    float* __restrict__ dK, float* __restrict__ dV,
                                    int seq, int nh, int dh, float scale) {
    const int h = (int)blockIdx.x;
    float* dS = smem;

    // dP[t,u] = dO[t]·V[u]
    // dS[t,u] = P[t,u] * (dP[t,u] - sum_{j<=t} P[t,j]*dP[t,j]); u>t stays 0
    for (int t = (int)threadIdx.x; t < seq; t += (int)blockDim.x) {
        float* row = dS + t * seq;
        const float* do_ptr = dO + qkv_index(t, h, 0, nh, dh);
        const float* p_row = P + (h * seq + t) * seq;
        float sum_pdp = 0.f;
        for (int u = 0; u <= t; ++u) {
            float dp = dot_n(do_ptr, V + qkv_index(u, h, 0, nh, dh), dh);
            row[u] = dp;
            sum_pdp += p_row[u] * dp;
        }
        for (int u = 0; u <= t; ++u) {
            row[u] = p_row[u] * (row[u] - sum_pdp);
        }
        for (int u = t + 1; u < seq; ++u) row[u] = 0.f;
    }
    __syncthreads();

    for (int idx = (int)threadIdx.x; idx < seq * dh; idx += (int)blockDim.x) {
        const int t = idx / dh;
        const int d = idx - t * dh;
        const float* ds_row = dS + t * seq;

        float acc_q = 0.f;
        for (int u = 0; u <= t; ++u) {
            acc_q += ds_row[u] * K[qkv_index(u, h, d, nh, dh)];
        }
        dQ[qkv_index(t, h, d, nh, dh)] = scale * acc_q;

        float acc_k = 0.f;
        for (int tt = t; tt < seq; ++tt) {
            acc_k += dS[tt * seq + t] * Q[qkv_index(tt, h, d, nh, dh)];
        }
        dK[qkv_index(t, h, d, nh, dh)] = scale * acc_k;

        const float* pbase = P + h * seq * seq;
        float acc_v = 0.f;
        const float* do_col = dO + h * dh + d;
        for (int tt = t; tt < seq; ++tt) {
            acc_v += pbase[tt * seq + t] * do_col[tt * nh * dh];
        }
        dV[qkv_index(t, h, d, nh, dh)] = acc_v;
    }
}

// TGAI GO forward kernels. Device-only (NVRTC, sm_70).
// Launch shapes are fixed by the scorer; every kernel strides so d / n
// larger than the block still covers the full row.

extern "C" __global__ void rmsnorm_fwd(float* out, const float* x, const float* w,
                                       int seq, int d, float eps) {
    // One block per token. 256-wide tree reduce into the 1024B dynamic buffer.
    const int t = (int)blockIdx.x;
    if (t >= seq || d <= 0) return;

    const float* row = x + (long long)t * d;
    float* orow = out + (long long)t * d;

    float local = 0.f;
    for (int i = (int)threadIdx.x; i < d; i += (int)blockDim.x) {
        float v = row[i];
        local += v * v;
    }

    extern __shared__ float red[];
    red[threadIdx.x] = local;
    __syncthreads();
    for (int stride = (int)blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }

    float inv = rsqrtf(red[0] / (float)d + eps);
    for (int i = (int)threadIdx.x; i < d; i += (int)blockDim.x) {
        orow[i] = row[i] * inv * w[i];
    }
}

__device__ __forceinline__ float silu_f(float v) {
    // v * sigmoid(v), overflow-safe for large |v|.
    if (v >= 0.f) return v / (1.f + expf(-v));
    float e = expf(v);
    return v * (e / (1.f + e));
}

extern "C" __global__ void silu_mul_fwd(float* a, const float* b, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) a[i] = silu_f(a[i]) * b[i];
}

extern "C" __global__ void moe_add_fwd(float* out, const float* e, float weight,
                                       int seq, int d, int blk) {
    int t = (int)blockIdx.x;
    int i = (int)threadIdx.x + (int)blockIdx.y * blk;
    if (t < seq && i < d) {
        long long idx = (long long)t * d + i;
        out[idx] += weight * e[idx];
    }
}

extern "C" __global__ void emb_gather_fwd(float* out, const float* emb, const int* ids,
                                          int seq, int d, int vocab, int unk) {
    int t = (int)blockIdx.x;
    if (t >= seq) return;
    int id = ids[t];
    if (id < 0 || id >= vocab) id = unk;
    const float* src = emb + (long long)id * d;
    float* dst = out + (long long)t * d;
    for (int i = (int)threadIdx.x; i < d; i += (int)blockDim.x) dst[i] = src[i];
}

__device__ __forceinline__ void rope_apply(float* x, int t, int h, int n_heads, int d_head,
                                           int pos, const float* cos_tab, const float* sin_tab) {
    if (!x) return;
    int n_pairs = d_head >> 1;
    float* base = x + (((long long)t * n_heads + h) * d_head);
    const float* ctab = cos_tab + (long long)pos * n_pairs;
    const float* stab = sin_tab + (long long)pos * n_pairs;
    for (int i = (int)threadIdx.x; i < n_pairs; i += (int)blockDim.x) {
        float c = ctab[i];
        float s = stab[i];
        float x0 = base[2 * i];
        float x1 = base[2 * i + 1];
        base[2 * i] = x0 * c - x1 * s;
        base[2 * i + 1] = x0 * s + x1 * c;
    }
}

extern "C" __global__ void rope_fwd(float* q, float* k, int seq, int n_heads, int d_head,
                                    int cache_pos, const float* cos_tab,
                                    const float* sin_tab, int max_seq) {
    int t = (int)blockIdx.x;
    int h = (int)blockIdx.y;
    if (t >= seq || h >= n_heads) return;
    int pos = cache_pos + t;
    if (pos >= max_seq) pos = max_seq - 1;
    if (pos < 0) pos = 0;
    rope_apply(q, t, h, n_heads, d_head, pos, cos_tab, sin_tab);
    rope_apply(k, t, h, n_heads, d_head, pos, cos_tab, sin_tab);
}

extern "C" __global__ void moe_route_fwd(const float* logits, const float* bias,
                                         int n_experts, int top_k,
                                         int* out_ids, float* out_w, int out_stride) {
    // top_k <= 8 and n_experts is small: one thread scans, the rest leave.
    if (threadIdx.x != 0) return;
    int t = (int)blockIdx.x;
    int kmax = top_k;
    if (kmax > 8) kmax = 8;
    if (kmax > n_experts) kmax = n_experts;
    if (kmax <= 0 || n_experts <= 0) return;

    const float* row = logits + (long long)t * n_experts;
    int ids[8];
    float scores[8];
    for (int k = 0; k < kmax; ++k) {
        int best = 0;
        float best_s = 0.f;
        int found = 0;
        for (int e = 0; e < n_experts; ++e) {
            int used = 0;
            for (int j = 0; j < k; ++j) {
                if (ids[j] == e) { used = 1; break; }
            }
            if (used) continue;
            float s = row[e] + (bias ? bias[e] : 0.f);
            // Strict '>' keeps the smaller index on a tie.
            if (!found || s > best_s) {
                best = e;
                best_s = s;
                found = 1;
            }
        }
        ids[k] = best;
        scores[k] = best_s;
    }

    float m = scores[0];
    for (int k = 1; k < kmax; ++k) if (scores[k] > m) m = scores[k];
    float exps[8];
    float sum = 0.f;
    for (int k = 0; k < kmax; ++k) {
        exps[k] = expf(scores[k] - m);
        sum += exps[k];
    }
    float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
    int* oid = out_ids + (long long)t * out_stride;
    float* ow = out_w + (long long)t * out_stride;
    for (int k = 0; k < kmax; ++k) {
        oid[k] = ids[k];
        ow[k] = exps[k] * inv;
    }
}

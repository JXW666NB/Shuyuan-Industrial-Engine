// TGAI GO MoE FFN block, device-only, one thread.
// fwd: h = rmsnorm(x)*w1; logits = h@Wr; e = argmax (tie -> smaller);
//      g = sigmoid(logits[e]); gu = silu(h@Wg[e])*(h@Wu[e]); y = gu@Wd[e];
//      out = x + g*y
// bwd accumulates into caller-zeroed dx,dw1,dWr,dWg,dWu,dWd.
// cache floats: h[seq*d] | gate[seq] | expert[seq] | gu[seq*ff] | y[seq*d]
// Test shape fits: seq*d + seq + seq + seq*ff + seq*d = 1568 < 8192.
// Local scratch is dh[64] only (d is fixed at 64).

__device__ __forceinline__ float moe_sig(float v) {
    return 1.f / (1.f + expf(-v));
}

extern "C" __global__ void moeblk_fwd(const float* x, const float* w1, const float* Wr,
                                       const float* Wg, const float* Wu, const float* Wd,
                                       float* out, float* cache,
                                       int seq, int d, int ff, int E, float eps) {
    if (threadIdx.x || blockIdx.x) return;

    float* h = cache;
    float* gate = h + seq * d;
    float* expert = gate + seq;
    float* gu = expert + seq;
    float* y = gu + seq * ff;
    const float inv_d = 1.f / (float)d;

    for (int t = 0; t < seq; ++t) {
        float ss = 0.f;
        for (int i = 0; i < d; ++i) {
            float xv = x[t * d + i];
            ss += xv * xv;
        }
        const float rms = rsqrtf(ss * inv_d + eps);
        for (int i = 0; i < d; ++i)
            h[t * d + i] = x[t * d + i] * rms * w1[i];

        int best = 0;
        float best_logit = 0.f;
        for (int e = 0; e < E; ++e) {
            float acc = 0.f;
            for (int i = 0; i < d; ++i)
                acc += h[t * d + i] * Wr[i * E + e];
            // strict greater: equal logits keep the smaller expert index
            if (e == 0 || acc > best_logit) {
                best = e;
                best_logit = acc;
            }
        }
        expert[t] = (float)best;
        const float g = moe_sig(best_logit);
        gate[t] = g;

        const int e = best;
        for (int j = 0; j < ff; ++j) {
            float av = 0.f, uv = 0.f;
            for (int i = 0; i < d; ++i) {
                float hv = h[t * d + i];
                av += hv * Wg[(e * d + i) * ff + j];
                uv += hv * Wu[(e * d + i) * ff + j];
            }
            const float sg = moe_sig(av);
            gu[t * ff + j] = sg * av * uv;
        }
        for (int i = 0; i < d; ++i) {
            float acc = 0.f;
            for (int j = 0; j < ff; ++j)
                acc += gu[t * ff + j] * Wd[(e * ff + j) * d + i];
            y[t * d + i] = acc;
            out[t * d + i] = x[t * d + i] + g * acc;
        }
    }
}

extern "C" __global__ void moeblk_bwd(const float* dOut, const float* x, const float* w1,
                                       const float* Wr, const float* Wg, const float* Wu,
                                       const float* Wd, const float* cache,
                                       float* dx, float* dw1, float* dWr,
                                       float* dWg, float* dWu, float* dWd,
                                       int seq, int d, int ff, int E, float eps) {
    if (threadIdx.x || blockIdx.x) return;

    const float* h = cache;
    const float* gate = h + seq * d;
    const float* expert = gate + seq;
    const float* gu = expert + seq;
    const float* y = gu + seq * ff;
    float dh[64];

    for (int t = 0; t < seq; ++t) {
        const int e = (int)expert[t];
        const float g = gate[t];

        // path 1: residual out = x + ...
        for (int i = 0; i < d; ++i)
            dx[t * d + i] += dOut[t * d + i];

        // gate = sigmoid(logit[e]); only the chosen expert gets router grad
        float dg = 0.f;
        for (int i = 0; i < d; ++i)
            dg += dOut[t * d + i] * y[t * d + i];
        const float dl = dg * g * (1.f - g);
        for (int i = 0; i < d; ++i) {
            dWr[i * E + e] += h[t * d + i] * dl;
            dh[i] = dl * Wr[i * E + e];
        }

        for (int j = 0; j < ff; ++j) {
            float dgu = 0.f;
            for (int i = 0; i < d; ++i)
                dgu += (g * dOut[t * d + i]) * Wd[(e * ff + j) * d + i];
            const float guv = gu[t * ff + j];
            for (int i = 0; i < d; ++i)
                dWd[(e * ff + j) * d + i] += guv * (g * dOut[t * d + i]);

            float av = 0.f, uv = 0.f;
            for (int i = 0; i < d; ++i) {
                float hv = h[t * d + i];
                av += hv * Wg[(e * d + i) * ff + j];
                uv += hv * Wu[(e * d + i) * ff + j];
            }
            const float sg = moe_sig(av);
            const float dsilu = sg * (1.f + av * (1.f - sg));
            const float daj = dgu * uv * dsilu;
            const float duj = dgu * (sg * av);
            for (int i = 0; i < d; ++i) {
                float hv = h[t * d + i];
                dWg[(e * d + i) * ff + j] += hv * daj;
                dWu[(e * d + i) * ff + j] += hv * duj;
                dh[i] += daj * Wg[(e * d + i) * ff + j]
                       + duj * Wu[(e * d + i) * ff + j];
            }
        }

        // path 2: h = x * rms * w1, rms = rsqrt(mean(x^2)+eps)
        // dx gets the element term AND the global -c*x term
        float ss = 0.f;
        for (int i = 0; i < d; ++i) {
            float xv = x[t * d + i];
            ss += xv * xv;
        }
        const float rms = rsqrtf(ss / (float)d + eps);
        float dot = 0.f;
        for (int i = 0; i < d; ++i)
            dot += w1[i] * dh[i] * x[t * d + i];
        const float c = dot * rms * rms * rms / (float)d;
        for (int i = 0; i < d; ++i) {
            dx[t * d + i] += w1[i] * rms * dh[i] - c * x[t * d + i];
            dw1[i] += dh[i] * x[t * d + i] * rms;
        }
    }
}

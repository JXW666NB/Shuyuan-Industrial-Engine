// TGAI GO full layer: causal MHA + top-1 SwiGLU MoE. Device-only, one thread.
// fwd: h1 = rmsnorm(x)*w1
//      q,k,v = h1@{Wq,Wk,Wv}; att = causal_softmax(q k^T * scale) @ v
//      o = att@Wo; x2 = x + o
//      h2 = rmsnorm(x2)*w2; logits = h2@Wr; e = argmax (tie -> smaller)
//      g = sigmoid(logits[e]); gu = silu(h2@Wg[e])*(h2@Wu[e]); y = gu@Wd[e]
//      out = x2 + g*y
// bwd accumulates the 11 grads (caller already zeroed). loss = sum(out*dOut).
//
// cache (floats), all row-major:
//   h1 q k v [seq,d] | P [nh,seq,seq] | att x2 h2 [seq,d]
//   | gate[seq] expert[seq] | gu[seq,ff] | y[seq,d]
// Default shape uses 4880 < 8192.
// bwd scratch is dynamic smem: dx2,datt,dQ,dK,dV each [seq,d] (10KB at the test size).

__device__ __forceinline__ float layer_sig(float v) {
    return 1.f / (1.f + expf(-v));
}

extern "C" __global__ void layer_fwd(
    const float* x, const float* w1,
    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
    const float* w2, const float* Wr,
    const float* Wg, const float* Wu, const float* Wd,
    float* out, float* cache,
    int seq, int d, int nh, int dh, int ff, int E, float eps, float scale) {
    if (threadIdx.x || blockIdx.x) return;

    const int sd = seq * d;
    float* h1 = cache;
    float* q = h1 + sd;
    float* k = q + sd;
    float* v = k + sd;
    float* P = v + sd;
    float* att = P + nh * seq * seq;
    float* x2 = att + sd;
    float* h2 = x2 + sd;
    float* gate = h2 + sd;
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
            h1[t * d + i] = x[t * d + i] * rms * w1[i];

        for (int j = 0; j < d; ++j) {
            float aq = 0.f, ak = 0.f, av = 0.f;
            for (int i = 0; i < d; ++i) {
                float hv = h1[t * d + i];
                aq += hv * Wq[i * d + j];
                ak += hv * Wk[i * d + j];
                av += hv * Wv[i * d + j];
            }
            q[t * d + j] = aq;
            k[t * d + j] = ak;
            v[t * d + j] = av;
        }
    }

    for (int h = 0; h < nh; ++h) {
        for (int t = 0; t < seq; ++t) {
            float sc[8];
            float m = 0.f;
            for (int u = 0; u <= t; ++u) {
                float dot = 0.f;
                const int qi = t * d + h * dh;
                const int ki = u * d + h * dh;
                for (int di = 0; di < dh; ++di)
                    dot += q[qi + di] * k[ki + di];
                sc[u] = dot * scale;
                if (u == 0 || sc[u] > m) m = sc[u];
            }
            float sum = 0.f;
            for (int u = 0; u <= t; ++u) {
                sc[u] = expf(sc[u] - m);
                sum += sc[u];
            }
            const float inv = 1.f / sum;
            for (int u = 0; u <= t; ++u)
                P[(h * seq + t) * seq + u] = sc[u] * inv;
            for (int u = t + 1; u < seq; ++u)
                P[(h * seq + t) * seq + u] = 0.f;

            for (int di = 0; di < dh; ++di) {
                float acc = 0.f;
                for (int u = 0; u <= t; ++u)
                    acc += P[(h * seq + t) * seq + u] * v[u * d + h * dh + di];
                att[t * d + h * dh + di] = acc;
            }
        }
    }

    for (int t = 0; t < seq; ++t) {
        for (int j = 0; j < d; ++j) {
            float acc = 0.f;
            for (int i = 0; i < d; ++i)
                acc += att[t * d + i] * Wo[i * d + j];
            x2[t * d + j] = x[t * d + j] + acc;
        }

        float ss = 0.f;
        for (int i = 0; i < d; ++i) {
            float xv = x2[t * d + i];
            ss += xv * xv;
        }
        const float rms = rsqrtf(ss * inv_d + eps);
        for (int i = 0; i < d; ++i)
            h2[t * d + i] = x2[t * d + i] * rms * w2[i];

        int best = 0;
        float best_logit = 0.f;
        for (int e = 0; e < E; ++e) {
            float acc = 0.f;
            for (int i = 0; i < d; ++i)
                acc += h2[t * d + i] * Wr[i * E + e];
            if (e == 0 || acc > best_logit) {
                best = e;
                best_logit = acc;
            }
        }
        expert[t] = (float)best;
        const float g = layer_sig(best_logit);
        gate[t] = g;
        const int e = best;

        for (int j = 0; j < ff; ++j) {
            float av = 0.f, uv = 0.f;
            for (int i = 0; i < d; ++i) {
                float hv = h2[t * d + i];
                av += hv * Wg[(e * d + i) * ff + j];
                uv += hv * Wu[(e * d + i) * ff + j];
            }
            gu[t * ff + j] = layer_sig(av) * av * uv;
        }
        for (int i = 0; i < d; ++i) {
            float acc = 0.f;
            for (int j = 0; j < ff; ++j)
                acc += gu[t * ff + j] * Wd[(e * ff + j) * d + i];
            y[t * d + i] = acc;
            out[t * d + i] = x2[t * d + i] + g * acc;
        }
    }
}

extern "C" __global__ void layer_bwd(
    const float* dOut, const float* x, const float* w1,
    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
    const float* w2, const float* Wr,
    const float* Wg, const float* Wu, const float* Wd,
    const float* cache,
    float* dx, float* dw1,
    float* dWq, float* dWk, float* dWv, float* dWo,
    float* dw2, float* dWr,
    float* dWg, float* dWu, float* dWd,
    int seq, int d, int nh, int dh, int ff, int E, float eps, float scale) {
    if (threadIdx.x || blockIdx.x) return;

    const int sd = seq * d;
    const float* h1 = cache;
    const float* q = h1 + sd;
    const float* k = q + sd;
    const float* v = k + sd;
    const float* P = v + sd;
    const float* att = P + nh * seq * seq;
    const float* x2 = att + sd;
    const float* h2 = x2 + sd;
    const float* gate = h2 + sd;
    const float* expert = gate + seq;
    const float* gu = expert + seq;
    const float* y = gu + seq * ff;
    const float inv_d = 1.f / (float)d;

    extern __shared__ float smem[];
    float* dx2s = smem;
    float* datts = dx2s + sd;
    float* dQs = datts + sd;
    float* dKs = dQs + sd;
    float* dVs = dKs + sd;
    for (int i = 0; i < sd; ++i) {
        dKs[i] = 0.f;
        dVs[i] = 0.f;
    }

    for (int t = 0; t < seq; ++t) {
        const int e = (int)expert[t];
        const float g = gate[t];
        float dh2[64];
        for (int i = 0; i < d; ++i) dh2[i] = 0.f;

        float dg = 0.f;
        for (int i = 0; i < d; ++i)
            dg += dOut[t * d + i] * y[t * d + i];
        const float dl = dg * g * (1.f - g);
        for (int i = 0; i < d; ++i) {
            dWr[i * E + e] += h2[t * d + i] * dl;
            dh2[i] += dl * Wr[i * E + e];
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
                float hv = h2[t * d + i];
                av += hv * Wg[(e * d + i) * ff + j];
                uv += hv * Wu[(e * d + i) * ff + j];
            }
            const float sg = layer_sig(av);
            const float dsilu = sg * (1.f + av * (1.f - sg));
            const float daj = dgu * uv * dsilu;
            const float duj = dgu * (sg * av);
            for (int i = 0; i < d; ++i) {
                float hv = h2[t * d + i];
                dWg[(e * d + i) * ff + j] += hv * daj;
                dWu[(e * d + i) * ff + j] += hv * duj;
                dh2[i] += daj * Wg[(e * d + i) * ff + j]
                        + duj * Wu[(e * d + i) * ff + j];
            }
        }

        float ss = 0.f;
        for (int i = 0; i < d; ++i) {
            float xv = x2[t * d + i];
            ss += xv * xv;
        }
        const float rms = rsqrtf(ss * inv_d + eps);
        float dot = 0.f;
        for (int i = 0; i < d; ++i)
            dot += w2[i] * dh2[i] * x2[t * d + i];
        const float c = dot * rms * rms * rms * inv_d;
        for (int i = 0; i < d; ++i) {
            const float dx2 = dOut[t * d + i] + (w2[i] * rms * dh2[i] - c * x2[t * d + i]);
            dx2s[t * d + i] = dx2;
            dx[t * d + i] += dx2;
            dw2[i] += dh2[i] * x2[t * d + i] * rms;
        }
    }

    for (int t = 0; t < seq; ++t) {
        for (int i = 0; i < d; ++i) {
            float acc = 0.f;
            for (int j = 0; j < d; ++j)
                acc += dx2s[t * d + j] * Wo[i * d + j];
            datts[t * d + i] = acc;
        }
        for (int i = 0; i < d; ++i) {
            const float ai = att[t * d + i];
            for (int j = 0; j < d; ++j)
                dWo[i * d + j] += ai * dx2s[t * d + j];
        }
    }

    for (int h = 0; h < nh; ++h) {
        for (int t = 0; t < seq; ++t) {
            float dSrow[8];
            float sum_pdp = 0.f;
            for (int u = 0; u <= t; ++u) {
                float dp = 0.f;
                const int di0 = t * d + h * dh;
                const int vi0 = u * d + h * dh;
                for (int di = 0; di < dh; ++di)
                    dp += datts[di0 + di] * v[vi0 + di];
                dSrow[u] = dp;
                sum_pdp += P[(h * seq + t) * seq + u] * dp;
            }
            for (int u = 0; u <= t; ++u)
                dSrow[u] = P[(h * seq + t) * seq + u] * (dSrow[u] - sum_pdp);

            for (int di = 0; di < dh; ++di) {
                float acc_q = 0.f;
                for (int u = 0; u <= t; ++u)
                    acc_q += dSrow[u] * k[u * d + h * dh + di];
                dQs[t * d + h * dh + di] = scale * acc_q;
            }
            for (int u = 0; u <= t; ++u) {
                const float ds = dSrow[u];
                for (int di = 0; di < dh; ++di) {
                    dKs[u * d + h * dh + di] += scale * ds * q[t * d + h * dh + di];
                    dVs[u * d + h * dh + di] += P[(h * seq + t) * seq + u] * datts[t * d + h * dh + di];
                }
            }
        }
    }

    for (int t = 0; t < seq; ++t) {
        float dh1[64];
        for (int i = 0; i < d; ++i) dh1[i] = 0.f;
        for (int j = 0; j < d; ++j) {
            const float dq = dQs[t * d + j];
            const float dk = dKs[t * d + j];
            const float dv = dVs[t * d + j];
            for (int i = 0; i < d; ++i) {
                float hv = h1[t * d + i];
                dWq[i * d + j] += hv * dq;
                dWk[i * d + j] += hv * dk;
                dWv[i * d + j] += hv * dv;
                dh1[i] += dq * Wq[i * d + j] + dk * Wk[i * d + j] + dv * Wv[i * d + j];
            }
        }

        float ss = 0.f;
        for (int i = 0; i < d; ++i) {
            float xv = x[t * d + i];
            ss += xv * xv;
        }
        const float rms = rsqrtf(ss * inv_d + eps);
        float dot = 0.f;
        for (int i = 0; i < d; ++i)
            dot += w1[i] * dh1[i] * x[t * d + i];
        const float c = dot * rms * rms * rms * inv_d;
        for (int i = 0; i < d; ++i) {
            dx[t * d + i] += w1[i] * rms * dh1[i] - c * x[t * d + i];
            dw1[i] += dh1[i] * x[t * d + i] * rms;
        }
    }
}

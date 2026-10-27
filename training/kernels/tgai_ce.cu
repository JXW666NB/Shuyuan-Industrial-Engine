/* Softmax cross-entropy, mean reduction. sm_70, device-only.
 *
 * ce_fwd: loss[0] = mean_i( logsumexp(row_i) - row_i[target_i] )
 * ce_bwd: dlogits[i, j] = (softmax(row_i)[j] - 1[j == target_i]) / N
 *
 * logsumexp / softmax subtract the row max before exp.
 * logits and dlogits are row-major (N, V). targets are int32 in [0, V).
 */

__device__ float ce_row_max(const float* row, int V) {
    float m = row[0];
    for (int j = 1; j < V; ++j) {
        m = fmaxf(m, row[j]);
    }
    return m;
}

__device__ float ce_row_sumexp(const float* row, int V, float m) {
    float sum = 0.0f;
    for (int j = 0; j < V; ++j) {
        sum += expf(row[j] - m);
    }
    return sum;
}

extern "C" __global__ void ce_fwd(const float* logits, const int* targets, float* loss,
                                  int N, int V) {
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float* row = logits + static_cast<long long>(i) * V;
        const float m = ce_row_max(row, V);
        const float sum = ce_row_sumexp(row, V, m);
        const int t = targets[i];
        acc += (m + logf(sum)) - row[t];
    }
    loss[0] = acc / static_cast<float>(N);
}

extern "C" __global__ void ce_bwd(const float* logits, const int* targets, float* dlogits,
                                  int N, int V) {
    const float inv_n = 1.0f / static_cast<float>(N);
    for (int i = 0; i < N; ++i) {
        const float* row = logits + static_cast<long long>(i) * V;
        float* grow = dlogits + static_cast<long long>(i) * V;
        const float m = ce_row_max(row, V);
        const float sum = ce_row_sumexp(row, V, m);
        const float inv_sum = 1.0f / sum;
        const int t = targets[i];
        for (int j = 0; j < V; ++j) {
            float p = expf(row[j] - m) * inv_sum;
            if (j == t) {
                p -= 1.0f;
            }
            grow[j] = p * inv_n;
        }
    }
}

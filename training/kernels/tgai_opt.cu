// Device-only AdamW + Muon. sm_70, one element / one block respectively.
// No host code, no launches, no cuda_runtime.h.

extern "C" __global__ void adamw_step(float* __restrict__ W, const float* __restrict__ G,
                                      float* __restrict__ m, float* __restrict__ v,
                                      float lr, float b1, float b2, float eps,
                                      float wd, int t, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    // Decoupled weight decay first, then moments, then bias-corrected step.
    // Moment mix matches torch lerp: m + (1-b1)*(g-m).
    float g = G[i];
    float mi = m[i];
    float vi = v[i];
    float w = W[i] * (1.f - lr * wd);

    mi = mi + (1.f - b1) * (g - mi);
    vi = vi * b2;
    vi = vi + (1.f - b2) * (g * g);

    // Integer power. t starts at 1; the suite's step is t=1.
    float p1 = 1.f, p2 = 1.f;
    for (int k = 0; k < t; ++k) {
        p1 *= b1;
        p2 *= b2;
    }
    float bc1 = 1.f - p1;
    float bc2 = 1.f - p2;
    float step_size = lr / bc1;
    float denom = sqrtf(vi) / sqrtf(bc2) + eps;

    m[i] = mi;
    v[i] = vi;
    W[i] = w - step_size * mi / denom;
}

extern "C" __global__ void muon_step(float* W, const float* G, float* B,
                                     float lr, float beta, float wd,
                                     int rows, int cols, int steps, float eps) {
    // Entire update is one 256-thread block. rows*cols <= 2048.
    // r = min(rows, cols) after the optional transpose, so r*r <= 2048.
    __shared__ float X[2048];
    __shared__ float Xn[2048];
    __shared__ float A[2048];
    __shared__ float AA[2048];
    __shared__ float red[256];

    const int tid = threadIdx.x;
    const int n = rows * cols;
    const int trans = rows > cols;
    const int r = trans ? cols : rows;
    const int cc = trans ? rows : cols;

    float local = 0.f;
    const float one_minus_beta = 1.f - beta;
    for (int i = tid; i < n; i += 256) {
        float b = beta * B[i] + one_minus_beta * G[i];
        B[i] = b;
        local += b * b;
    }
    red[tid] = local;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    const float inv = sqrtf(red[0]) + eps;
    __syncthreads();

    for (int i = tid; i < n; i += 256) {
        int row = i / cols;
        int col = i - row * cols;
        float x = B[i] / inv;
        if (trans) X[col * cc + row] = x;
        else X[row * cc + col] = x;
    }
    __syncthreads();

    const float a = 3.4445f;
    const float bq = -4.7750f;
    const float cq = 2.0315f;
    const int rr = r * r;
    const int nc = r * cc;

    for (int s = 0; s < steps; ++s) {
        for (int idx = tid; idx < rr; idx += 256) {
            int i = idx / r;
            int j = idx - i * r;
            const float* xi = X + i * cc;
            const float* xj = X + j * cc;
            float acc = 0.f;
            for (int k = 0; k < cc; ++k) acc += xi[k] * xj[k];
            A[idx] = acc;
        }
        __syncthreads();

        for (int idx = tid; idx < rr; idx += 256) {
            int i = idx / r;
            int j = idx - i * r;
            float acc = 0.f;
            for (int k = 0; k < r; ++k) acc += A[i * r + k] * A[k * r + j];
            AA[idx] = acc;
        }
        __syncthreads();

        for (int idx = tid; idx < rr; idx += 256)
            A[idx] = bq * A[idx] + cq * AA[idx];
        __syncthreads();

        for (int idx = tid; idx < nc; idx += 256) {
            int i = idx / cc;
            int j = idx - i * cc;
            float acc = a * X[idx];
            for (int k = 0; k < r; ++k) acc += A[i * r + k] * X[k * cc + j];
            Xn[idx] = acc;
        }
        __syncthreads();

        for (int idx = tid; idx < nc; idx += 256) X[idx] = Xn[idx];
        __syncthreads();
    }

    const float scale = sqrtf(fmaxf(1.f, (float)rows / (float)cols));
    for (int i = tid; i < n; i += 256) {
        int row = i / cols;
        int col = i - row * cols;
        float x = trans ? X[col * cc + row] : X[row * cc + col];
        float w = W[i];
        W[i] = w - lr * (x * scale + wd * w);
    }
}

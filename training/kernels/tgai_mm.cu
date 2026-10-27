// TGAI GO matmul, device-only (NVRTC, sm_70).
// Launch: grid=(1,1,1) block=(256,1,1). One block owns the whole product.
// Row-major float:
//   Y[m,n]  = sum_k X[m,k] * W[k,n]
//   dX[m,k] = sum_n dY[m,n] * W[k,n]     // dY @ W^T
//   dW[k,n] = sum_m X[m,k] * dY[m,n]     // X^T @ dY
// dX and dW are zeroed by the caller; each output is written by one thread.

extern "C" __global__ void mm_fwd(const float* __restrict__ X,
                                  const float* __restrict__ W,
                                  float* __restrict__ Y,
                                  int M, int K, int N) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int yn = M * N;
    for (int i = tid; i < yn; i += stride) {
        const int m = i / N;
        const int n = i - m * N;
        const float* xrow = X + m * K;
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
            acc += xrow[k] * W[k * N + n];
        }
        Y[i] = acc;
    }
}

extern "C" __global__ void mm_bwd(const float* __restrict__ dY,
                                  const float* __restrict__ X,
                                  const float* __restrict__ W,
                                  float* __restrict__ dX,
                                  float* __restrict__ dW,
                                  int M, int K, int N) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // dX[M,K] = dY[M,N] @ W[K,N]^T
    const int dxn = M * K;
    for (int i = tid; i < dxn; i += stride) {
        const int m = i / K;
        const int k = i - m * K;
        const float* dyrow = dY + m * N;
        const float* wrow = W + k * N;
        float acc = 0.f;
        for (int n = 0; n < N; ++n) {
            acc += dyrow[n] * wrow[n];
        }
        dX[i] = acc;
    }

    // dW[K,N] = X[M,K]^T @ dY[M,N]
    const int dwn = K * N;
    for (int i = tid; i < dwn; i += stride) {
        const int k = i / N;
        const int n = i - k * N;
        float acc = 0.f;
        for (int m = 0; m < M; ++m) {
            acc += X[m * K + k] * dY[m * N + n];
        }
        dW[i] = acc;
    }
}

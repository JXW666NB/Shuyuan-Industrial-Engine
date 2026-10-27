# -*- coding: utf-8 -*-
"""matmul 判分台：Y = X @ W 的前向与反向，与 PyTorch 对拍。

被测源码 device-only，导出：
  extern "C" __global__ void mm_fwd(const float* X, const float* W, float* Y,
                                    int M, int K, int N);
  extern "C" __global__ void mm_bwd(const float* dY, const float* X, const float* W,
                                    float* dX, float* dW, int M, int K, int N);

语义（行优先，全部 float）：
  Y[M,N]  = X[M,K] @ W[K,N]
  dX[M,K] = dY[M,N] @ W[K,N]^T
  dW[K,N] = X[M,K]^T @ dY[M,N]

启动（两个 kernel 相同）：grid=(1,1,1) block=(256,1,1)
  —— 整个乘法在一个 block 内完成（测试规模最多 32*64=2048 元素），可用静态 shared <= 48KB。
dX / dW 由调用方清零，逐元素直接写。

用法:
    python mm_suite.py <src.cu>
输出最后一行:
    VERDICT ok=<bool> Y=.. dX=.. dW=..
"""
import ctypes
import sys

import torch

from tgai_nvrtc import Cuda

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

dev = "cuda"
SHAPES = [(32, 64, 48), (16, 32, 16), (8, 128, 64)]


def P_(t):
    return (t.data_ptr(), ctypes.c_void_p)


def I(v):
    return (int(v), ctypes.c_int)


def err(a, b):
    return (a - b).abs().max().item()


def main(src):
    code = open(src, encoding="utf-8", errors="replace").read()
    code = code.replace("#include <cuda_runtime.h>", "").replace("#include <stddef.h>", "")
    torch.cuda.init()
    cu = Cuda()
    mod = cu.compile(code, "sm_70")

    worst = 0.0
    lines = []
    for M, K, N in SHAPES:
        torch.manual_seed(M * 1000 + K * 10 + N)
        X = torch.randn(M, K, device=dev)
        W = torch.randn(K, N, device=dev)
        dY = torch.randn(M, N, device=dev)

        Xr = X.clone().requires_grad_(True)
        Wr = W.clone().requires_grad_(True)
        Y_ref = Xr @ Wr
        (Y_ref * dY).sum().backward()

        Y = torch.zeros(M, N, device=dev)
        dX = torch.zeros(M, K, device=dev)
        dW = torch.zeros(K, N, device=dev)

        cu.launch(mod, "mm_fwd", (1, 1, 1), (256, 1, 1),
                  [P_(X), P_(W), P_(Y), I(M), I(K), I(N)])
        cu.launch(mod, "mm_bwd", (1, 1, 1), (256, 1, 1),
                  [P_(dY), P_(X), P_(W), P_(dX), P_(dW), I(M), I(K), I(N)])
        cu.sync()

        eY, eX, eW = err(Y, Y_ref.detach()), err(dX, Xr.grad), err(dW, Wr.grad)
        worst = max(worst, eY, eX, eW)
        lines.append("%dx%dx%d Y=%.2e dX=%.2e dW=%.2e" % (M, K, N, eY, eX, eW))

    print("形状: " + " | ".join(lines))
    ok = worst < 2e-4
    print("VERDICT ok=%s worst=%.3e" % (ok, worst))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

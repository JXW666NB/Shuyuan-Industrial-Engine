# -*- coding: utf-8 -*-
"""cuBLAS ctypes 桥接：让引擎侧 GEMM 与 PyTorch 走同一个 cuBLAS，保证公平对比。

用法:
    from cublas_bridge import Cublas
    cb = Cublas()
    C = cb.sgemm(A, B, out=C)      # A[M,K] @ B[K,N] -> C[M,N]，全部行优先 fp32
"""
import ctypes
import os
import sys

import torch

CUDA_R_32F = 0
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1


def _load_dll():
    cands = []
    try:
        import torch as _t
        libdir = os.path.join(os.path.dirname(_t.__file__), "lib")
        cands.append(os.path.join(libdir, "cublas64_12.dll"))
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(libdir)
    except Exception:
        pass
    for base in (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",):
        cands.append(os.path.join(base, "cublas64_12.dll"))
    for p in cands:
        if os.path.exists(p):
            return ctypes.CDLL(p), p
    raise RuntimeError("找不到 cublas64_12.dll，尝试过: %s" % cands)


class Cublas:
    def __init__(self):
        self.dll, self.path = _load_dll()
        self.handle = ctypes.c_void_p()
        st = self.dll.cublasCreate_v2(ctypes.byref(self.handle))
        if st != 0:
            raise RuntimeError("cublasCreate_v2 failed: %d" % st)
        # 让 cuBLAS 用 torch 当前 stream，避免隐式同步
        self.set_stream(torch.cuda.current_stream().cuda_stream)

    def set_stream(self, ptr):
        self.dll.cublasSetStream_v2(self.handle, ctypes.c_void_p(ptr))

    def sgemm(self, A, B, out=None, alpha=1.0, beta=0.0):
        """行优先 C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C"""
        assert A.dim() == 2 and B.dim() == 2
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "形状不匹配: %s @ %s" % (tuple(A.shape), tuple(B.shape))
        if out is None:
            out = torch.zeros(M, N, device=A.device, dtype=torch.float32)
        a = ctypes.c_float(alpha)
        b = ctypes.c_float(beta)
        # 行优先转列优先：C^T(N,M) = B^T(N,K) @ A^T(K,M)
        st = self.dll.cublasSgemm_v2(
            self.handle, CUBLAS_OP_N, CUBLAS_OP_N,
            ctypes.c_int(N), ctypes.c_int(M), ctypes.c_int(K),
            ctypes.byref(a),
            ctypes.c_void_p(B.data_ptr()), ctypes.c_int(N),
            ctypes.c_void_p(A.data_ptr()), ctypes.c_int(K),
            ctypes.byref(b),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_int(N))
        if st != 0:
            raise RuntimeError("cublasSgemm failed: %d" % st)
        return out

    def gemm_tn(self, A, B, out=None, alpha=1.0, beta=0.0):
        """行优先 C[M,N] = A[K,M]^T @ B[K,N]

        列主序视角: C_cm[N,M] = B^T(op=N, ld=N) @ A(op=T, ld=M)
        """
        K, M = A.shape
        K2, N = B.shape
        assert K == K2
        if out is None:
            out = torch.zeros(M, N, device=A.device, dtype=torch.float32)
        a = ctypes.c_float(alpha)
        b = ctypes.c_float(beta)
        st = self.dll.cublasSgemm_v2(
            self.handle, CUBLAS_OP_N, CUBLAS_OP_T,
            ctypes.c_int(N), ctypes.c_int(M), ctypes.c_int(K),
            ctypes.byref(a),
            ctypes.c_void_p(B.data_ptr()), ctypes.c_int(N),
            ctypes.c_void_p(A.data_ptr()), ctypes.c_int(M),
            ctypes.byref(b),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_int(N))
        if st != 0:
            raise RuntimeError("cublasSgemm(TN) failed: %d" % st)
        return out

    def gemm_nt(self, A, B, out=None, alpha=1.0, beta=0.0):
        """行优先 C[M,N] = A[M,K] @ B[N,K]^T

        列主序视角: C_cm[N,M] = B(op=T, ld=K) @ A(op=N, ld=K)
        """
        M, K = A.shape
        N, K2 = B.shape
        assert K == K2
        if out is None:
            out = torch.zeros(M, N, device=A.device, dtype=torch.float32)
        a = ctypes.c_float(alpha)
        b = ctypes.c_float(beta)
        st = self.dll.cublasSgemm_v2(
            self.handle, CUBLAS_OP_T, CUBLAS_OP_N,
            ctypes.c_int(N), ctypes.c_int(M), ctypes.c_int(K),
            ctypes.byref(a),
            ctypes.c_void_p(B.data_ptr()), ctypes.c_int(K),
            ctypes.c_void_p(A.data_ptr()), ctypes.c_int(K),
            ctypes.byref(b),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_int(N))
        if st != 0:
            raise RuntimeError("cublasSgemm(NT) failed: %d" % st)
        return out

    def close(self):
        if self.handle:
            self.dll.cublasDestroy_v2(self.handle)
            self.handle = None


def _main():
    torch.cuda.init()
    cb = Cublas()
    print("cuBLAS:", cb.path)
    print("%-22s %-10s %-10s %-10s %s" % ("shape(MxKxN)", "torch ms", "cublas ms", "err", "TFLOPS(cublas)"))
    print("-" * 74)
    import time
    # 先自证两个转置变体
    for tag, fn, ref in (
        ("gemm_tn", lambda A, B: cb.gemm_tn(A, B), lambda A, B: A.t() @ B),
        ("gemm_nt", lambda A, B: cb.gemm_nt(A, B), lambda A, B: A @ B.t()),
    ):
        for K, M, N in ((64, 32, 48), (256, 128, 96)):
            A = torch.randn(K, M, device="cuda") if tag == "gemm_tn" else torch.randn(M, K, device="cuda")
            B = torch.randn(K, N, device="cuda") if tag == "gemm_tn" else torch.randn(N, K, device="cuda")
            C = fn(A, B)
            R = ref(A, B)
            e = (C - R).abs().max().item() / (R.abs().max().item() + 1e-12)
            print("  %s %dx%dx%d err=%.2e %s" % (tag, K, M, N, e, "OK" if e < 1e-5 else "BAD"))
    print()

    for M, K, N in ((512, 1024, 3072), (512, 1024, 4096), (2048, 1024, 4096),
                    (512, 4096, 1024), (128, 1024, 1024)):
        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")
        C = torch.zeros(M, N, device="cuda")
        it = max(3, int(2e9 / (2 * M * K * N)))

        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(it):
            C = torch.matmul(A, B)
        torch.cuda.synchronize(); t_torch = (time.perf_counter() - t0) / it

        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(it):
            cb.sgemm(A, B, out=C)
        torch.cuda.synchronize(); t_cb = (time.perf_counter() - t0) / it

        ref = torch.matmul(A, B)
        e = (C - ref).abs().max().item() / (ref.abs().max().item() + 1e-12)
        tflops = 2 * M * K * N / t_cb / 1e12
        print("%-22s %-10.3f %-10.3f %-10.2e %.2f"
              % ("%dx%dx%d" % (M, K, N), t_torch * 1e3, t_cb * 1e3, e, tflops))
    cb.close()


if __name__ == "__main__":
    try:
        _main()
    except Exception as ex:
        print("FAILED:", type(ex).__name__, ex)
        sys.exit(1)

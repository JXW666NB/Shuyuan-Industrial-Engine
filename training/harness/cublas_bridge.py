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

    def _strided_batched(self, transa, transb, m, n, k, A, lda, stride_a, B, ldb, stride_b,
                         C, ldc, stride_c, batch, beta):
        """cublasSgemmStridedBatched. A/B/C are device pointers (ints)."""
        fn = getattr(self, "_strided_fn", None)
        if fn is None:
            fn = self.dll.cublasSgemmStridedBatched
            fn.restype = ctypes.c_int
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
                ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
                ctypes.c_int,
            ]
            self._strided_fn = fn
        alpha = ctypes.c_float(1.0)
        b = ctypes.c_float(beta)
        st = fn(
            self.handle, transa, transb,
            ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
            ctypes.byref(alpha),
            ctypes.c_void_p(A), ctypes.c_int(lda), ctypes.c_longlong(stride_a),
            ctypes.c_void_p(B), ctypes.c_int(ldb), ctypes.c_longlong(stride_b),
            ctypes.byref(b),
            ctypes.c_void_p(C), ctypes.c_int(ldc), ctypes.c_longlong(stride_c),
            ctypes.c_int(batch))
        if st != 0:
            raise RuntimeError("cublasSgemmStridedBatched failed: %d" % st)

    def gemm_tn_batched(self, A, B, out, beta=0.0):
        """行优先、按 batch 对齐：C[b,M,N] = A[b,K,M]^T @ B[b,K,N]。

        与 gemm_tn 同一套列主序转置，stride 把 E 个专家一次发出。
        """
        batch, K, M = A.shape
        b2, K2, N = B.shape
        assert batch == b2 and K == K2 and tuple(out.shape) == (batch, M, N)
        assert A.is_contiguous() and B.is_contiguous() and out.is_contiguous()
        self._strided_batched(
            CUBLAS_OP_N, CUBLAS_OP_T, N, M, K,
            B.data_ptr(), N, K * N,
            A.data_ptr(), M, K * M,
            out.data_ptr(), N, M * N, batch, beta)
        return out

    def gemm_nt_batched(self, A, B, out, beta=0.0):
        """行优先：C[b,M,N] = A[b,M,K] @ B[b,N,K]^T。beta=1 可累加进同一 C。"""
        batch, M, K = A.shape
        b2, N, K2 = B.shape
        assert batch == b2 and K == K2 and tuple(out.shape) == (batch, M, N)
        assert A.is_contiguous() and B.is_contiguous() and out.is_contiguous()
        self._strided_batched(
            CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
            B.data_ptr(), K, N * K,
            A.data_ptr(), K, M * K,
            out.data_ptr(), N, M * N, batch, beta)
        return out

    def sgemm_batched(self, A, B, out, beta=0.0):
        """行优先：C[b,M,N] = A[b,M,K] @ B[b,K,N]。"""
        batch, M, K = A.shape
        b2, K2, N = B.shape
        assert batch == b2 and K == K2 and tuple(out.shape) == (batch, M, N)
        assert A.is_contiguous() and B.is_contiguous() and out.is_contiguous()
        self._strided_batched(
            CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            B.data_ptr(), N, K * N,
            A.data_ptr(), K, M * K,
            out.data_ptr(), N, M * N, batch, beta)
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

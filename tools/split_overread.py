# -*- coding: utf-8 -*-
"""See whether split GEMM reads past the matrix into the tail of the allocation."""
import numpy as np
import tgai_go.train as T
T.set_default_device("cuda")
from tgai_go.train import _cuda
rt = _cuda()
rt.ensure()
rt._split16_state()["on"] = True

rng = np.random.RandomState(0)
K, M, N = 4992, 256, 256
G = (rng.randn(K, M) * 0.05).astype(np.float32)
H = (rng.randn(K, N) * 0.05).astype(np.float32)
g = T.tensor(G)._gpu
h = T.tensor(H)._gpu

def gemm(buf_g, buf_h):
    C = rt.empty((M, N), dtype=np.float32)
    # gemm_tn: C[M,N] = A[K,M].T @ B[K,N]
    rt._split16_state()["on"] = False
    rt._sgemm(0, 1, N, M, K, buf_h.ptr, N, buf_g.ptr, M, C.ptr, N)
    rt.sync()
    fp = rt.d2h(C).copy()
    rt._split16_state()["on"] = True
    cache = getattr(rt, "_split_cache", None)
    if cache is not None:
        cache.clear()
    rt.sgemm_split16(0, 1, N, M, K, buf_h.ptr, N, buf_g.ptr, M, C.ptr, N)
    rt.sync()
    sp = rt.d2h(C).copy()
    rel = np.max(np.abs(sp - fp)) / max(np.max(np.abs(fp)), 1e-12)
    print("rel %.3e" % rel)
    return rel

print("tight", end=" ")
gemm(g, h)

def padded(tail):
    # row-major [K, M] but allocate extra columns so linear tail sits after each row? 
    # gemm_tn treats A as contiguous [K,M] with ldb=M. Extra elements must be AFTER the
    # whole matrix, i.e. a flat buffer of K*M + tail.
    n = K * M
    big = rt.full((n + 8192,), float(tail), dtype=np.float32)
    # copy G into the prefix
    src = T.tensor(G)._gpu
    rt._d2d_bytes(big.ptr, src.ptr, n * 4)
    # wrap a shape view. gemm uses raw ptr + lda, shape unused.
    class B(object):
        pass
    b = B()
    b.ptr = big.ptr
    b.n = n
    return b

print("tail0 G", end=" ")
gemm(padded(0), h)
print("tail1 G", end=" ")
gemm(padded(1), h)
print("tail0 H", end=" ")
gemm(g, padded(0))
print("tail7 H", end=" ")
gemm(g, padded(7))

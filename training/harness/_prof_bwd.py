# -*- coding: utf-8 -*-
"""微基准：定位 backward 里最慢的那类 op（在 seq=2048/V=32779 的真实形状上）。"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

S, D, V = 2048, 1024, 32779
T.set_device("cuda")


def bench(name, fn, n=1):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print("  %-34s %8.4f s" % (name, dt))
    return dt, out


print("形状: X[%d,%d]  W[%d,%d]" % (S, D, V, D))

# 1) A @ W.T  （logits）
g = np.random.RandomState(0)
X = (g.randn(S, D) * 0.02).astype(np.float32)
W = (g.randn(V, D) * 0.02).astype(np.float32)
dlog = (g.randn(S, V) * 0.01).astype(np.float32)

x = T.Tensor(X, requires_grad=True)
w = T.Tensor(W, requires_grad=True)
t0 = time.perf_counter()
y = x @ w.T
t_fwd = time.perf_counter() - t0
print("  %-34s %8.4f s" % ("fwd  A @ W.T", t_fwd))

c = (g.randn(*y.shape) * 0.01).astype(np.float32)
t0 = time.perf_counter()
loss = (y * T.Tensor(c)).sum()
t_sum = time.perf_counter() - t0
print("  %-34s %8.4f s" % ("fwd  (y*c).sum()", t_sum))

t0 = time.perf_counter()
loss.backward()
t_bwd = time.perf_counter() - t0
print("  %-34s %8.4f s" % ("bwd  A@W.T 整体", t_bwd))

# 2) 单独看大 GEMM：dlog @ W 与 dlog.T @ X
t0 = time.perf_counter()
_ = T.Tensor(dlog) @ T.Tensor(W)
print("  %-34s %8.4f s" % ("fwd  dlog @ W  (2048x32779x1024)", time.perf_counter() - t0))

t0 = time.perf_counter()
_ = T.Tensor(np.ascontiguousarray(dlog.T)) @ T.Tensor(X)
print("  %-34s %8.4f s" % ("fwd  dlog.T @ X (32779x2048x1024)", time.perf_counter() - t0))

# 3) embedding 反向（scatter-add 到 V x D）
emb = T.nn.Embedding(V, D)
idx = np.random.RandomState(1).randint(0, 256, size=S).astype(np.int64)
t0 = time.perf_counter()
h = emb(idx)
l2 = (h * T.Tensor(g.randn(S, D).astype(np.float32))).sum()
l2.backward()
print("  %-34s %8.4f s" % ("embedding fwd+bwd", time.perf_counter() - t0))

print()
print("签名：p.grad 大小 =", None if w.grad is None else np.asarray(w.grad).shape)

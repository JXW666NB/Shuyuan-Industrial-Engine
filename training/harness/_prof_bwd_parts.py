# -*- coding: utf-8 -*-
"""把 seq=8192 的 backward 拆成部件，逐个隔离计时（每次前后 sync）。"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

D, NH, FF, E, V, SEQ = 1024, 16, 4096, 8, 32779, 8192
F = T.functional
T.set_device("cuda")
rt = T._cuda()
rng = np.random.RandomState(0)


def bench(name, fn, n=1):
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    rt.sync()
    dt = (time.perf_counter() - t0) / n
    print("  %-30s %8.4f s" % (name, dt))
    return dt, out


print("seq=%d 各部件隔离计时" % SEQ)
N = SEQ

# ── 1) cross_entropy 前向 + 反向（[8192, 32779]）
x = (rng.randn(N, V).astype(np.float32) * 0.02)
tgt = rng.randint(0, V, size=N).astype(np.int64)
t = T.Tensor(x, requires_grad=True)
bench("CE 前向 [%d,%d]" % (N, V), lambda: F.cross_entropy(t, tgt))
loss = F.cross_entropy(t, tgt)
bench("CE 反向", lambda: loss.backward())

# ── 2) logits matmul 前向 + 反向
h = T.Tensor((rng.randn(N, D).astype(np.float32) * 0.02), requires_grad=True)
W = T.Tensor((rng.randn(V, D).astype(np.float32) * 0.02), requires_grad=True)
bench("logits matmul 前向", lambda: h @ W.T)
y2 = h @ W.T
bench("logits matmul 反向", lambda: (y2 * T.Tensor((rng.randn(N, V).astype(np.float32) * 0.01))).sum().backward())

# ── 3) attention（lowmem）前向 + 反向
dh = 64
qq = T.Tensor((rng.randn(NH, N, dh).astype(np.float32) * 0.1), requires_grad=True)
kk = T.Tensor((rng.randn(NH, N, dh).astype(np.float32) * 0.1), requires_grad=True)
vv = T.Tensor((rng.randn(NH, N, dh).astype(np.float32) * 0.1), requires_grad=True)
bench("attn lowmem 前向", lambda: F.scaled_dot_product_attention(qq, kk, vv, causal=True, lowmem=True))
o = F.scaled_dot_product_attention(qq, kk, vv, causal=True, lowmem=True)
c = T.Tensor((rng.randn(NH, N, dh).astype(np.float32) * 0.1))
bench("attn lowmem 反向", lambda: (o * c).sum().backward())

# ── 4) 单专家 FFN（gather 后的 1/8 token 数）
M = N // 2      # 每个专家实际 token 数约 N*topk/E = N/4；取 N/2 作上界
g_ = T.Tensor((rng.randn(M, D).astype(np.float32) * 0.02), requires_grad=True)
wg = T.Linear(D, FF)
wu = T.Linear(D, FF)
wd = T.Linear(FF, D)
bench("MoE 单专家 前向(%d tok)" % M,
      lambda: wd(F.silu(wg(g_)) * wu(g_)))
out = wd(F.silu(wg(g_)) * wu(g_))
bench("MoE 单专家 反向", lambda: (out * T.Tensor((rng.randn(M, D).astype(np.float32)))).sum().backward())

# ── 5) embedding 反向（scatter-add 到 V x D）
emb = T.nn.Embedding(V, D)
idx = rng.randint(0, 256, size=N).astype(np.int64)
bench("embedding 前向", lambda: emb(idx))
hh = emb(idx)
bench("embedding 反向", lambda: (hh * T.Tensor((rng.randn(N, D).astype(np.float32)))).sum().backward())

# ── 6) rmsnorm 前向+反向
rn = T.nn.RMSNorm(D)
z = T.Tensor((rng.randn(N, D).astype(np.float32)), requires_grad=True)
bench("rmsnorm 前向", lambda: rn(z))
zo = rn(z)
bench("rmsnorm 反向", lambda: (zo * T.Tensor((rng.randn(N, D).astype(np.float32)))).sum().backward())

# ── 7) 参数清梯度（每步都要做，看看贵不贵）
params = [g_, wg.weight, wu.weight, wd.weight, emb.weight, rn.weight]
def clear():
    for p in params:
        p.grad = None
bench("清梯度 x%d" % len(params), clear)

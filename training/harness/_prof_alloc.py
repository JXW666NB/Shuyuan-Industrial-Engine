# -*- coding: utf-8 -*-
"""量 cuMemAlloc/cuMemFree 的真实成本 + 统计一步训练里的分配次数。"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402
from tgai_go.train import cuda as C  # noqa: E402

T.set_device("cuda")
rt = T._cuda()
print("runtime:", type(rt).__name__)

# 1) 直接量 alloc/free
for nbytes_gb in (0.001, 0.01, 0.1):
    n = int(nbytes_gb * 2**30 // 4)
    shape = (max(1, n),)
    t0 = time.perf_counter()
    K = 50
    for _ in range(K):
        b = rt.empty(shape)
        rt.free(b.ptr)
    dt = (time.perf_counter() - t0) / K
    print("  alloc+free %8.2f MB: %8.3f ms/次" % (nbytes_gb * 1024, dt * 1000))

# 2) 统计一步里的分配次数
cnt = {"n": 0}
orig_alloc = rt._alloc


def counting_alloc(nbytes):
    cnt["n"] += 1
    return orig_alloc(nbytes)


rt._alloc = counting_alloc

D, NH, FF, E, TOPK, V, SEQ = 1024, 16, 4096, 8, 2, 32779, 2048
F = T.functional
T.manual_seed(7)
emb = T.nn.Embedding(V, D)
n1, n2 = T.nn.RMSNorm(D), T.nn.RMSNorm(D)
q, k, v, o = (T.nn.Linear(D, D) for _ in range(4))
moe = T.nn.MoE(D, FF, E, top_k=TOPK, gate="softmax")
nf = T.nn.RMSNorm(D)
params = [emb.weight, n1.weight, n2.weight, q.weight, k.weight, v.weight, o.weight, nf.weight]
for _n, p in moe.named_parameters():
    params.append(p)
opt = T.optim.AdamW(params, lr=3e-4)

g = np.random.RandomState(0)
x = (g.randint(0, 256, size=(1, SEQ))).astype(np.int64)
y = (g.randint(0, 256, size=(1, SEQ))).astype(np.int64)
B, S, DH = 1, SEQ, D // NH

for step in range(2):
    cnt["n"] = 0
    t0 = time.perf_counter()
    h = emb(x)
    n = n1(h)
    qq = q(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
    kk = k(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
    vv = v(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
    oo = F.scaled_dot_product_attention(qq, kk, vv, causal=True)
    oo = oo.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
    h = h + o(oo).reshape(B, S, D)
    h = h + moe(n2(h)).reshape(B, S, D)
    hn = nf(h)
    logits = hn.reshape(B * S, D) @ emb.weight.T
    loss = F.cross_entropy(logits, y.reshape(-1))
    n_alloc_fwd = cnt["n"]
    for p in params:
        p.grad = None
    loss.backward()
    n_alloc_all = cnt["n"]
    opt.step()
    dt = time.perf_counter() - t0
    print("  step%d: 总 %.3f s | 分配次数 前向 %d、含反向 %d"
          % (step, dt, n_alloc_fwd, n_alloc_all))

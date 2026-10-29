# -*- coding: utf-8 -*-
"""Wall-clock split of one real step: forward / backward / AdamW. No per-node sync."""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

D, NH, FF, E, TOPK, V, SEQ = 1024, 16, 4096, 8, 2, 32779, int(sys.argv[1]) if len(sys.argv) > 1 else 8192
F = T.functional
T.set_device("cuda")
rt = T._cuda()
T.manual_seed(7)
raw = (np.arange((SEQ + 1) * 3) % 256).astype(np.int64)
x, y = raw[:SEQ].reshape(1, SEQ), raw[1:SEQ + 1].reshape(1, SEQ)
emb = T.nn.Embedding(V, D)
n1, n2, nf = T.nn.RMSNorm(D), T.nn.RMSNorm(D), T.nn.RMSNorm(D)
q, k, v, o = (T.nn.Linear(D, D) for _ in range(4))
moe = T.nn.MoE(D, FF, E, top_k=TOPK, gate="softmax")
params = [emb.weight, n1.weight, n2.weight, q.weight, k.weight, v.weight, o.weight, nf.weight]
for _n, p in moe.named_parameters():
    params.append(p)
opt = T.optim.AdamW(params, lr=3e-4)
B, S, DH = 1, SEQ, D // NH
F.sdpa_lowmem_ready()

gemm_ms = [0.0]
gemm_n = [0]
gemm_on = [False]
orig = rt._grouped_sgemm


def wrapped(specs):
    if not gemm_on[0]:
        return orig(specs)
    rt.sync()
    t0 = time.perf_counter()
    r = orig(specs)
    rt.sync()
    gemm_ms[0] += (time.perf_counter() - t0) * 1e3
    gemm_n[0] += 1
    return r


rt._grouped_sgemm = wrapped


def forward_step():
    h = emb(x)
    n = n1(h)
    qq = q(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
    kk = k(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
    vv = v(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
    oo = F.scaled_dot_product_attention(qq, kk, vv, causal=True, lowmem=True)
    oo = oo.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
    h = h + o(oo).reshape(B, S, D)
    h = h + moe(n2(h)).reshape(B, S, D)
    hn = nf(h)
    logits = hn.reshape(B * S, D) @ emb.weight.T
    return F.cross_entropy(logits, y.reshape(-1))


print("seq", SEQ, "warmup")
loss = forward_step()
for p in params:
    p.grad = None
loss.backward()
opt.step()
rt.sync()
gemm_ms[0] = 0.0
gemm_n[0] = 0

def tick():
    rt.sync()
    return time.perf_counter()

for step_i in range(3):
    t0 = tick()
    loss = forward_step()
    t1 = tick()
    for p in params:
        p.grad = None
    t2 = tick()
    loss.backward()
    t3 = tick()
    opt.step()
    t4 = tick()
    print("step%d forward %.3f zero %.3f backward %.3f adamw %.3f total %.3f ms" % (
        step_i, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3,
        (t4 - t0) * 1e3))
print("loss", float(np.asarray(loss.to_numpy()).reshape(-1)[0]))
gemm_on[0] = True
loss = forward_step()
for p in params:
    p.grad = None
loss.backward()
rt.sync()
print("grouped_sgemm calls %d  %.2f ms (each call synced)" % (gemm_n[0], gemm_ms[0]))
print("pool", rt.pool_stats() if hasattr(rt, "pool_stats") else "n/a")
print("DONE")

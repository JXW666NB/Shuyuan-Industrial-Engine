# -*- coding: utf-8 -*-
"""验证假设：seq=8192 时池上限（12 GiB）导致每步 trim，拖慢训练。
用法: python pool_cap_bench.py <cap_gb>
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

CAP = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
T.set_device("cuda")
rt = T._cuda()
rt._pool_cap = int(CAP * (1024 ** 3))
print("池上限 = %.0f GiB" % CAP)

D, NH, FF, E, TOPK, V, SEQ = 1024, 16, 4096, 8, 2, 32779, 8192
F = T.functional
T.manual_seed(7)

txt = ""
for p in ("pelican_on_bike.html", "bike_on_pelican.html"):
    try:
        txt += open("/home/featurize/tgai/_corpus/" + p, encoding="utf-8", errors="replace").read()
    except Exception:
        pass
raw = np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).astype(np.int64) % 256
if len(raw) < SEQ + 2:
    raw = np.tile(raw, (SEQ // max(1, len(raw))) + 2)
ids = raw[: SEQ + 1].reshape(1, SEQ + 1)
x, y = ids[:, :-1], ids[:, 1:]

emb = T.nn.Embedding(V, D)
n1, n2 = T.nn.RMSNorm(D), T.nn.RMSNorm(D)
q, k, v, o = (T.nn.Linear(D, D) for _ in range(4))
moe = T.nn.MoE(D, FF, E, top_k=TOPK, gate="softmax")
nf = T.nn.RMSNorm(D)
params = [emb.weight, n1.weight, n2.weight, q.weight, k.weight, v.weight, o.weight, nf.weight]
for _n, p in moe.named_parameters():
    params.append(p)
opt = T.optim.AdamW(params, lr=3e-4)
B, S, DH = 1, SEQ, D // NH


def one_step():
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
    loss = F.cross_entropy(logits, y.reshape(-1))
    for p in params:
        p.grad = None
    loss.backward()
    return float(np.asarray(loss.to_numpy()).reshape(-1)[0])


times = []
for i in range(4):
    rt.sync()
    t0 = time.perf_counter()
    lossv = one_step()
    opt.step()
    rt.sync()
    dt = time.perf_counter() - t0
    times.append(dt)
    print("  step%d %.3f s  loss=%.4f  pool_hits=%s" % (
        i, dt, lossv, getattr(rt, "_pool_hits", "-")))

steady = times[1:]
print("稳态平均 %.3f s（后 3 步）| 池 cached=%.2f GiB" % (
    sum(steady) / len(steady), getattr(rt, "_pool_cached_bytes", 0) / 2**30))

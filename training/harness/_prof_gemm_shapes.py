# -*- coding: utf-8 -*-
"""Split backward gemm_nt/gemm_tn time by shape. One warmed step, seq from argv."""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

D, NH, FF, E, TOPK, V = 1024, 16, 4096, 8, 2, 32779
SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
F = T.functional
T.set_device("cuda")
rt = T._cuda()
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


loss = forward_step()
opt.zero_grad()
loss.backward()
rt.sync()
print("warmup loss", float(np.asarray(loss.to_numpy()).reshape(-1)[0]))

stats = {}
orig_nt, orig_tn, orig_sg = rt.gemm_nt, rt.gemm_tn, rt.sgemm


def wrap(kind, fn):
    def inner(A, B):
        rt.sync()
        t0 = time.perf_counter()
        out = fn(A, B)
        rt.sync()
        dt = time.perf_counter() - t0
        key = (kind, tuple(int(s) for s in A.shape), tuple(int(s) for s in B.shape))
        rec = stats.get(key)
        if rec is None:
            stats[key] = [1, dt]
        else:
            rec[0] += 1
            rec[1] += dt
        return out
    return inner


rt.gemm_nt = wrap("nt", orig_nt)
rt.gemm_tn = wrap("tn", orig_tn)
rt.sgemm = wrap("nn", orig_sg)

loss = forward_step()
opt.zero_grad()
t0 = time.perf_counter()
loss.backward()
rt.sync()
print("bwd wall %.4f" % (time.perf_counter() - t0))
rows = sorted(stats.items(), key=lambda kv: -kv[1][1])
total = sum(v[1] for v in stats.values())
print("gemm sum %.4f calls %d" % (total, sum(v[0] for v in stats.values())))
for key, (n, dt) in rows:
    print("%6.4f  x%-3d  %-2s  %s @ %s" % (dt, n, key[0], key[1], key[2]))

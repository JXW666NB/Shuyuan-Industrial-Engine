# -*- coding: utf-8 -*-
"""Forward-only kernel time by launch name, plus cublas gemm. seq from argv."""
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
rt.sync()
print("warmup", float(np.asarray(loss.to_numpy()).reshape(-1)[0]))

kern = {}
orig_launch = rt.launch


def launch(name, grid, block, args, shared=0, module=None):
    rt.sync()
    t0 = time.perf_counter()
    out = orig_launch(name, grid, block, args, shared=shared, module=module)
    rt.sync()
    kern[name] = kern.get(name, 0.0) + (time.perf_counter() - t0)
    return out


rt.launch = launch
gemm = {}
for kind, fn in (("nn", rt.sgemm), ("nt", rt.gemm_nt), ("tn", rt.gemm_tn)):
    def wrap(fn=fn, kind=kind):
        def inner(A, B):
            rt.sync()
            t0 = time.perf_counter()
            out = fn(A, B)
            rt.sync()
            gemm[kind] = gemm.get(kind, 0.0) + (time.perf_counter() - t0)
            return out
        return inner
    setattr(rt, {"nn": "sgemm", "nt": "gemm_nt", "tn": "gemm_tn"}[kind], wrap())

rt.sync()
t0 = time.perf_counter()
loss = forward_step()
rt.sync()
print("fwd wall %.4f" % (time.perf_counter() - t0))
print("gemm", " ".join("%s=%.4f" % (k, gemm[k]) for k in sorted(gemm)))
for name, dt in sorted(kern.items(), key=lambda kv: -kv[1])[:25]:
    print("%7.4f  %s" % (dt, name))

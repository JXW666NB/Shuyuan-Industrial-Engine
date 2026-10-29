# -*- coding: utf-8 -*-
"""Split the embedding backward node into emb_bwd vs _acc."""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

D, NH, FF, E, TOPK, V, SEQ = 1024, 16, 4096, 8, 2, 32779, 8192
F = T.functional
T.set_device("cuda")
rt = T._cuda()
T.manual_seed(7)

raw = np.arange(SEQ + 1, dtype=np.int64) % 256
x, y = raw[:-1].reshape(1, SEQ), raw[1:].reshape(1, SEQ)
emb = T.nn.Embedding(V, D)
n1 = T.nn.RMSNorm(D)
q, k, v, o = (T.nn.Linear(D, D) for _ in range(4))
moe = T.nn.MoE(D, FF, E, top_k=TOPK, gate="softmax")
nf = T.nn.RMSNorm(D)
B, S, DH = 1, SEQ, D // NH
F.sdpa_lowmem_ready()

calls = []
orig_emb = rt.emb_bwd
orig_add = rt.add
orig_gemm_tn = rt.gemm_tn
orig_sgemm = rt.sgemm


def wrap(name, fn):
    def inner(*a, **k):
        rt.sync()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        rt.sync()
        shape = getattr(r, "shape", None)
        calls.append((name, (time.perf_counter() - t0) * 1e3, shape))
        return r
    return inner


rt.emb_bwd = wrap("emb_bwd", orig_emb)
rt.add = wrap("add", orig_add)
rt.gemm_tn = wrap("gemm_tn", orig_gemm_tn)
rt.sgemm = wrap("sgemm", orig_sgemm)


def step():
    h = emb(x)
    n = n1(h)
    qq = q(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
    kk = k(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
    vv = v(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
    oo = F.scaled_dot_product_attention(qq, kk, vv, causal=True, lowmem=True)
    oo = oo.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
    h = h + o(oo).reshape(B, S, D)
    h = h + moe(n1(h) * 0 + n1(h)).reshape(B, S, D) if False else h + moe(T.nn.RMSNorm(D)(h) if False else n1.__class__(D)(h) if False else __import__("tgai_go.train", fromlist=["nn"]).nn.RMSNorm(D)(h))
    return h


# simpler forward matching the bench, built explicitly
n2 = T.nn.RMSNorm(D)


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


print("warmup")
loss = forward_step()
loss.backward()
rt.sync()
print("warmup calls", len(calls))
calls.clear()
print("profile")
loss = forward_step()
# find emb closure
seen, stack, nodes = set(), [loss], []
while stack:
    nd = stack.pop()
    if id(nd) in seen:
        continue
    seen.add(id(nd))
    nodes.append(nd)
    stack.extend(nd._parents)
for nd in nodes:
    bw = getattr(nd, "_backward", None)
    if bw is None:
        continue
    code = bw.__code__
    if code.co_firstlineno == 2221:
        print("found emb bw", code.co_filename, "free", list(bw.__code__.co_freevars))

t0 = time.perf_counter()
loss.backward()
rt.sync()
print("full bwd %.3f s" % (time.perf_counter() - t0))
# aggregate
agg = {}
for name, ms, shape in calls:
    agg.setdefault(name, [0.0, 0, None])
    agg[name][0] += ms
    agg[name][1] += 1
    agg[name][2] = shape
for name, (ms, n, shape) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    print("  %-10s n=%3d  %8.2f ms  last=%s" % (name, n, ms, shape))
print("DONE")

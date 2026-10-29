# -*- coding: utf-8 -*-
"""Sync around major kernels so the wall time is the kernel, not the profiler artifact."""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
D, NH, FF, E, TOPK, V = 1024, 16, 4096, 8, 2, 32779
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

acc = {}
calls = {}


def wrap(name, fn):
    def inner(*a, **k):
        rt.sync()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        rt.sync()
        acc[name] = acc.get(name, 0.0) + (time.perf_counter() - t0) * 1e3
        calls[name] = calls.get(name, 0) + 1
        return r
    return inner


for name in (
    "attn_nop_fwd_stats", "attn_nop_bwd_flash", "emb_fwd", "emb_bwd",
    "grouped_nt", "grouped_nn", "grouped_tn", "gemm_nt", "gemm_tn", "sgemm",
    "rmsnorm_fwd", "rmsnorm_bwd", "ce_fwd", "ce_bwd",
    "permute", "add", "mul", "clone", "softmax_lastdim", "log_softmax_lastdim",
    "moe_gather_rows", "moe_scatter_rows", "moe_row_scale", "moe_row_dot",
    "moe_topk", "moe_compact", "silu_mul_fwd", "silu_mul_bwd",
    "transpose2d", "swap_last2", "adamw", "zeros",
):
    if hasattr(rt, name):
        setattr(rt, name, wrap(name, getattr(rt, name)))


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


print("warmup", SEQ)
loss = forward_step()
for p in params:
    p.grad = None
loss.backward()
opt.step()
rt.sync()
acc.clear()
rt.sync()
t0 = time.perf_counter()
loss = forward_step()
for p in params:
    p.grad = None
loss.backward()
opt.step()
rt.sync()
total = (time.perf_counter() - t0) * 1e3
print("step %.2f ms" % total)
s = 0.0
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print("  %-22s %8.2f ms  n=%d" % (k, v, calls.get(k, 0)))
    s += v
print("  covered %.2f ms  other %.2f ms" % (s, total - s))
print("loss", float(np.asarray(loss.to_numpy()).reshape(-1)[0]))
st = rt.pool_stats()
print("peak_GiB %.2f live_GiB %.2f" % (st["peak_bytes"] / 1024**3, st["peak_live_bytes"] / 1024**3))
print("DONE")

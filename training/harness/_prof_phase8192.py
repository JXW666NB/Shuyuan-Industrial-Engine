# -*- coding: utf-8 -*-
"""seq=8192 MoE（lowmem）的一步：分段计时，定位该优化哪个 kernel。"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

D, NH, FF, E, TOPK, V, SEQ = 1024, 16, 4096, 8, 2, 32779, 8192
F = T.functional
T.set_device("cuda")
T.manual_seed(7)

acc = {}


_RT = [None]


def wrap(name, fn, *a, **kw):
    # kernel 是异步的，必须同步才能真正归因到该阶段
    if _RT[0] is None:
        import tgai_go.train as _T
        _RT[0] = _T._cuda()
    rt = _RT[0]
    rt.sync()
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    rt.sync()
    acc[name] = acc.get(name, 0.0) + (time.perf_counter() - t0)
    return out


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

for step in range(2):
    for kk in list(acc.keys()):
        acc[kk] = 0.0
    t_all = time.perf_counter()

    h = wrap("embedding", emb, x)
    n = wrap("rmsnorm", n1, h)
    qq = wrap("qkv_proj", lambda: (q(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH),
                                   k(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH),
                                   v(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)))
    oo = wrap("sdpa_lowmem", F.scaled_dot_product_attention, qq[0], qq[1], qq[2], causal=True, lowmem=True)
    oo = wrap("attn_reshape", lambda: oo.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D))
    h2 = wrap("out_proj+res", lambda: h + o(oo).reshape(B, S, D))
    n2h = wrap("rmsnorm2", n2, h2)
    mo = wrap("moe", moe, n2h)
    h3 = wrap("moe_res", lambda: h2 + mo.reshape(B, S, D))
    hn = wrap("final_norm", nf, h3)
    logits = wrap("logits_matmul", lambda: hn.reshape(B * S, D) @ emb.weight.T)
    loss = wrap("cross_entropy", F.cross_entropy, logits, y.reshape(-1))
    for p in params:
        p.grad = None
    wrap("backward", loss.backward)
    wrap("opt_step", opt.step)
    dt = time.perf_counter() - t_all

    if step == 1:
        print("seq=%d 步时间 %.3f s" % (SEQ, dt))
        print("%-16s %8s %7s" % ("阶段", "秒", "占比"))
        for kk_, vv_ in sorted(acc.items(), key=lambda kv: -kv[1]):
            print("%-16s %8.3f %6.1f%%" % (kk_, vv_, 100 * vv_ / dt))

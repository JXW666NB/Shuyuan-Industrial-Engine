# -*- coding: utf-8 -*-
"""显存账：把「参数 + 优化器状态」「活张量峰值」「池驻留」分开量，判断 A3 该往哪打。"""
import sys

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

D, NH, FF, E, TOPK, V = 1024, 16, 4096, 8, 2, 32779
F = T.functional
T.set_device("cuda")
rt = T._cuda()
T.manual_seed(7)
rt.empty_cache()
rt._pool_peak_live = 0

txt = ""
for p in ("pelican_on_bike.html", "bike_on_pelican.html"):
    try:
        txt += open("/home/featurize/tgai/_corpus/" + p, encoding="utf-8", errors="replace").read()
    except Exception:
        pass
raw = np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).astype(np.int64) % 256

emb = T.nn.Embedding(V, D)
n1, n2 = T.nn.RMSNorm(D), T.nn.RMSNorm(D)
q, k, v, o = (T.nn.Linear(D, D) for _ in range(4))
moe = T.nn.MoE(D, FF, E, top_k=TOPK, gate="softmax")
nf = T.nn.RMSNorm(D)
params = [emb.weight, n1.weight, n2.weight, q.weight, k.weight, v.weight, o.weight, nf.weight]
for _n, p in moe.named_parameters():
    params.append(p)
opt = T.optim.AdamW(params, lr=3e-4)

n_par = sum(int(np.prod(p.shape)) for p in params)
print("参数量 %.2f M" % (n_par / 1e6))
print("参数 + 梯度 + AdamW 状态（m,v）≈ %.2f GiB" % (n_par * 4 * 4 / 2**30))

B, DH = 1, D // NH
for SEQ in (2048, 8192):
    raw2 = raw if len(raw) >= SEQ + 2 else np.tile(raw, (SEQ // max(1, len(raw))) + 2)
    ids = raw2[: SEQ + 1].reshape(1, SEQ + 1)
    x, y = ids[:, :-1], ids[:, 1:]
    S = SEQ
    rt.empty_cache()
    rt._pool_peak_live = 0

    def step():
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
        opt.step()

    step()          # 预热（让池稳住）
    rt.empty_cache()
    rt._pool_peak_live = 0
    step()
    rt.sync()
    st = rt.pool_stats() if hasattr(rt, "pool_stats") else {}
    live = getattr(rt, "_pool_peak_live", 0) / 2**30
    cached = getattr(rt, "_pool_cached_bytes", 0) / 2**30
    print("\nseq=%d" % SEQ)
    print("  活张量峰值（不含池驻留）≈ %.2f GiB" % live)
    print("  池中空闲块            ≈ %.2f GiB" % cached)
    print("  pool_stats: %s" % (st,))

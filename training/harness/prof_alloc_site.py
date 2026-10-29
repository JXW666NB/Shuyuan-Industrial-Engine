# -*- coding: utf-8 -*-
"""分配点归因：在显存峰值时刻，哪些调用点还活着、各占多少。

做法：给 rt._alloc 打桩，用调用栈的 (文件:行) 当标签，维护「每个标签的存活字节」；
每当存活总量刷新高水位，就把当时的分标签快照记下来。最后打印峰值时刻的 Top 名单。

开销：每次分配都要抽栈（约几十微秒），只用于分析，不进生产路径。
"""
import sys
import traceback

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

D, NH, FF, E, TOPK, V, SEQ = 1024, 16, 4096, 8, 2, 32779, int(sys.argv[1]) if len(sys.argv) > 1 else 8192
F = T.functional
T.set_device("cuda")
rt = T._cuda()

live = {}          # tag -> bytes
alloc_of = {}      # id(buffer) -> (tag, bytes)
peak_total = [0]
peak_snapshot = [None]

_orig = rt._alloc


def _tag():
    st = traceback.extract_stack()
    # 跳过本文件与 cuda.py 的帧，取第一个业务帧
    for fr in reversed(st[:-2]):
        fn = fr.filename.replace("\\", "/")
        if fn.endswith("prof_alloc_site.py") or fn.endswith("train/cuda.py"):
            continue
        return "%s:%d" % (fn.split("/")[-1], fr.lineno)
    return "?"


def _alloc(nbytes):
    buf = _orig(nbytes)
    try:
        t = _tag()
        key = id(buf)
        alloc_of[key] = (t, int(nbytes))
        live[t] = live.get(t, 0) + int(nbytes)
        tot = sum(live.values())
        if tot > peak_total[0]:
            peak_total[0] = tot
            peak_snapshot[0] = dict(live)
    except Exception:
        pass
    return buf


rt._alloc = _alloc

# 让 buffer 释放时能同步扣减：包里走的是 GpuBuf.__del__ / 池回收，这里用 gc 回调近似
import weakref  # noqa: E402


def _on_gone(key):
    info = alloc_of.pop(key, None)
    if info:
        t, nb = info
        live[t] = live.get(t, 0) - nb


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
B, S, DH = 1, SEQ, D // NH

emb = T.nn.Embedding(V, D)
n1, n2 = T.nn.RMSNorm(D), T.nn.RMSNorm(D)
q, k, v, o = (T.nn.Linear(D, D) for _ in range(4))
moe = T.nn.MoE(D, FF, E, top_k=TOPK, gate="softmax")
nf = T.nn.RMSNorm(D)
params = [emb.weight, n1.weight, n2.weight, q.weight, k.weight, v.weight, o.weight, nf.weight]
for _n, p in moe.named_parameters():
    params.append(p)
opt = T.optim.AdamW(params, lr=3e-4)


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
    import os as _os
    _c = int(_os.environ.get("CHUNK", "0") or 0)
    if _c:
        loss = F.linear_cross_entropy(hn.reshape(B * S, D), emb.weight, y.reshape(-1), chunk=_c)
    else:
        loss = F.cross_entropy(hn.reshape(B * S, D) @ emb.weight.T, y.reshape(-1))
    for p in params:
        p.grad = None
    loss.backward()
    opt.step()


for _ in range(2):
    step()          # 预热
live.clear()
alloc_of.clear()
peak_total[0] = 0
peak_snapshot[0] = None
step()
rt.sync()

snap = peak_snapshot[0] or {}
print("seq=%d 分配点归因（峰值时刻 %.2f GiB）" % (SEQ, peak_total[0] / 2**30))
print("%-46s %10s %6s" % ("调用点", "MiB", "占比"))
tot = sum(snap.values()) or 1
for tag, nb in sorted(snap.items(), key=lambda kv: -kv[1])[:12]:
    print("%-46s %10.1f %5.1f%%" % (tag, nb / 2**20, 100.0 * nb / tot))

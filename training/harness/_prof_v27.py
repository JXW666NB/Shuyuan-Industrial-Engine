# -*- coding: utf-8 -*-
"""分段计时：找出 tgai_go.train CUDA 后端在 seq=2048 下的真实热点。"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

D, NH, FF, V, SEQ, LAYERS = 1024, 16, 4096, 32779, 2048, 1
F = T.functional
T.set_device("cuda")
T.manual_seed(7)


def tic():
    return time.perf_counter()


def rep(name, t0, extra=""):
    dt = time.perf_counter() - t0
    print("  %-26s %8.3f s %s" % (name, dt, extra))
    return dt


txt = ""
for p in ("pelican_on_bike.html", "bike_on_pelican.html"):
    try:
        txt += open("/home/featurize/tgai/_corpus/" + p, encoding="utf-8", errors="replace").read()
    except Exception:
        pass
raw = np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).astype(np.int64) % 256
if len(raw) < SEQ + 2:
    raw = np.tile(raw, (SEQ // max(1, len(raw))) + 2)
ids = raw[:SEQ + 1].reshape(1, SEQ + 1)
x, y = ids[:, :-1], ids[:, 1:]

emb = T.nn.Embedding(V, D)
n1, n2 = T.nn.RMSNorm(D), T.nn.RMSNorm(D)
q, k, v, o = (T.nn.Linear(D, D) for _ in range(4))
wg, wu, wd = T.nn.Linear(D, FF), T.nn.Linear(D, FF), T.nn.Linear(FF, D)
nf = T.nn.RMSNorm(D)
params = [emb.weight, n1.weight, n2.weight, q.weight, k.weight, v.weight, o.weight,
          wg.weight, wu.weight, wd.weight, nf.weight]

print("参数就绪，开始分段（seq=%d, D=%d, FF=%d, V=%d）" % (SEQ, D, FF, V))
B, DH = 1, D // NH
acc = {}

t0 = tic()
h = emb(x)
acc["embedding"] = rep("embedding", t0)

t0 = tic()
n = n1(h)
acc["rmsnorm"] = rep("rmsnorm", t0)

t0 = tic()
qq = q(n.reshape(B * SEQ, D))
kk = k(n.reshape(B * SEQ, D))
vv = v(n.reshape(B * SEQ, D))
acc["qkv_linear"] = rep("qkv 三个 Linear", t0)

t0 = tic()
qq2 = qq.reshape(B, SEQ, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, SEQ, DH)
kk2 = kk.reshape(B, SEQ, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, SEQ, DH)
vv2 = vv.reshape(B, SEQ, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, SEQ, DH)
acc["permute"] = rep("reshape+permute ×3", t0)

t0 = tic()
oa = F.scaled_dot_product_attention(qq2, kk2, vv2, causal=True)
acc["sdpa"] = rep("SDPA(含 P 物化)", t0, "P 约 %.1f GB" % (B * NH * SEQ * SEQ * 4 / 2**30))

t0 = tic()
o2 = oa.reshape(B, NH, SEQ, DH).permute(0, 2, 1, 3).reshape(B * SEQ, D)
h = h + o(o2).reshape(B, SEQ, D)
acc["attn_out"] = rep("attn out + 残差", t0)

t0 = tic()
n2h = n2(h)
gu = F.silu(wg(n2h.reshape(B * SEQ, D))) * wu(n2h.reshape(B * SEQ, D))
h = h + wd(gu).reshape(B, SEQ, D)
acc["ffn"] = rep("FFN(silu_mul)", t0)

t0 = tic()
hn = nf(h)
logits = hn.reshape(B * SEQ, D) @ emb.weight.T
acc["logits"] = rep("final norm + logits", t0, "logits %.0f MB" % (SEQ * V * 4 / 2**20))

t0 = tic()
loss = F.cross_entropy(logits, y.reshape(-1))
acc["ce"] = rep("cross_entropy", t0)

t0 = tic()
for p in params:
    p.grad = None
import cProfile, pstats
_pr = cProfile.Profile()
_pr.enable()
loss.backward()
_pr.disable()
acc["backward"] = rep("backward 全部", t0)
_pr.dump_stats("/tmp/bwd.prof")
_st = pstats.Stats("/tmp/bwd.prof")
print("--- 反向 top15（按累计时间）---")
_st.sort_stats("cumtime").print_stats(15)

t0 = tic()
opt = T.optim.AdamW(params, lr=3e-4)
opt.step()
acc["opt"] = rep("AdamW.step", t0)

print()
total = sum(acc.values())
print("合计 %.2f s" % total)
for kk_, vv_ in sorted(acc.items(), key=lambda kv: -kv[1]):
    print("  %-16s %7.3f s  %5.1f%%" % (kk_, vv_, 100 * vv_ / total))

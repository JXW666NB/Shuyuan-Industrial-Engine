# -*- coding: utf-8 -*-
import sys, time
import numpy as np
sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T
D, NH, FF, E, TOPK, V, SEQ = 1024, 16, 4096, 8, 2, 32779, 8192
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

def sync():
    rt.sync()

# warmup
h = emb(x); n = n1(h)
qq = q(n.reshape(B*S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B*NH, S, DH)
kk = k(n.reshape(B*S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B*NH, S, DH)
vv = v(n.reshape(B*S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B*NH, S, DH)
oo = F.scaled_dot_product_attention(qq, kk, vv, causal=True, lowmem=True)
oo = oo.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B*S, D)
h = h + o(oo).reshape(B, S, D)
h = h + moe(n2(h)).reshape(B, S, D)
hn = nf(h)
logits = hn.reshape(B*S, D) @ emb.weight.T
loss = F.cross_entropy(logits, y.reshape(-1))
sync()

def tick(label, fn):
    sync(); t = time.perf_counter(); out = fn(); sync()
    print("%-16s %.4f" % (label, time.perf_counter() - t))
    return out

h = tick("emb", lambda: emb(x))
n = tick("rmsnorm", lambda: n1(h))
def qkv():
    qq = q(n.reshape(B*S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B*NH, S, DH)
    kk = k(n.reshape(B*S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B*NH, S, DH)
    vv = v(n.reshape(B*S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B*NH, S, DH)
    return qq, kk, vv
qq, kk, vv = tick("qkv+layout", qkv)
oo = tick("attn", lambda: F.scaled_dot_product_attention(qq, kk, vv, causal=True, lowmem=True))
def proj():
    z = oo.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B*S, D)
    return o(z).reshape(B, S, D)
po = tick("outproj", proj)
h2 = tick("resid", lambda: h + po)
hm = tick("moe", lambda: moe(n2(h2)))
h3 = tick("resid2", lambda: h2 + hm)
hn = tick("finalnorm", lambda: nf(h3))
lg = tick("logits", lambda: hn.reshape(B*S, D) @ emb.weight.T)
tick("ce", lambda: F.cross_entropy(lg, y.reshape(-1)))

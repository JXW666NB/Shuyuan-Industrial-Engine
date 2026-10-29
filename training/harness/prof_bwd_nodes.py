# -*- coding: utf-8 -*-
"""在真实训练步内部，按 autograd 节点归因 backward 的 GPU 时间。

做法：backward 前遍历计算图，给每个节点的 `_backward` 套一层 (sync, 计时, sync)，
用闭包的 co_firstlineno + 源码行文本作为算子标识（这样能直接读出来是哪一行代码）。
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

D, NH, FF, E, TOPK, V, SEQ = 1024, 16, 4096, 8, 2, 32779, int(sys.argv[1]) if len(sys.argv) > 1 else 8192
F = T.functional
T.set_device("cuda")
rt = T._cuda()
T.manual_seed(7)

SRC = None
try:
    import inspect
    SRC = inspect.getsourcefile(T.Tensor)
except Exception:
    pass


def src_line(fn):
    code = getattr(fn, "__code__", None)
    if code is None:
        return "?"
    try:
        lines = open(code.co_filename, encoding="utf-8").read().splitlines()
        return "%s:%d  %s" % (code.co_filename.split("\\")[-1].split("/")[-1],
                              code.co_firstlineno, lines[code.co_firstlineno - 1].strip()[:70])
    except Exception:
        return "%s:%d" % (code.co_filename, code.co_firstlineno)


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


# 预热（编译 + 池预热），只对第 2 步做归因
for i in range(2):
    loss = forward_step()
    for p in params:
        p.grad = None
    if i == 0:
        loss.backward()
        opt.step()
        continue

    # --- 计时归因 ---
    seen, stack, nodes = set(), [loss], []
    while stack:
        nd = stack.pop()
        if id(nd) in seen:
            continue
        seen.add(id(nd))
        nodes.append(nd)
        stack.extend(nd._parents)

    acc = {}
    orig = {}
    for nd in nodes:
        bw = getattr(nd, "_backward", None)
        if bw is None:
            continue
        key = src_line(bw)
        orig[id(nd)] = bw

        def make(bw_=bw, key_=key):
            def wrapped(g, grads):
                rt.sync()
                t0 = time.perf_counter()
                r = bw_(g, grads)
                rt.sync()
                acc[key_] = acc.get(key_, 0.0) + (time.perf_counter() - t0)
                return r
            return wrapped

        nd._backward = make()

    rt.sync()
    t0 = time.perf_counter()
    loss.backward()
    rt.sync()
    total = time.perf_counter() - t0

    print("seq=%d  backward 总计 %.3f s | 节点数 %d" % (SEQ, total, len(nodes)))
    print("%-64s %8s %6s" % ("算子（源码行）", "秒", "占比"))
    for k, vv in sorted(acc.items(), key=lambda kv: -kv[1])[:14]:
        print("%-64s %8.4f %5.1f%%" % (k, vv, 100 * vv / total))
    other = total - sum(acc.values())
    print("%-64s %8.4f %5.1f%%" % ("(图遍历/清梯度等开销)", other, 100 * other / total))

# -*- coding: utf-8 -*-
"""低显存 attention（不物化 P）与旧路径的数值对拍（我自己的判分台）。

判据：dh=64 + causal 时，O 与 dQ/dK/dV 的相对误差 < 1e-4。
同时确认：非 causal / dh≠64 走回退路径。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

F = T.functional
T.set_device("cuda")
TOL = 1e-4


def run(seq, nh, dh, causal=True, lowmem=False, seed=5):
    d = nh * dh
    g = np.random.RandomState(seed)
    Q = (g.randn(1, seq, d) * 0.3).astype(np.float32)
    K = (g.randn(1, seq, d) * 0.3).astype(np.float32)
    V = (g.randn(1, seq, d) * 0.3).astype(np.float32)
    coef = (g.randn(1, seq, d) * 0.2).astype(np.float32)

    q = T.Tensor(Q, requires_grad=True)
    k = T.Tensor(K, requires_grad=True)
    v = T.Tensor(V, requires_grad=True)

    def split(t):
        return t.reshape(1, seq, nh, dh).permute(0, 2, 1, 3).reshape(nh, seq, dh)

    qq, kk, vv = split(q), split(k), split(v)
    if lowmem:
        o = F.scaled_dot_product_attention(qq, kk, vv, causal=causal, lowmem=True)
    else:
        o = F.scaled_dot_product_attention(qq, kk, vv, causal=causal)
    loss = (o * T.Tensor(np.transpose(coef.reshape(1, seq, nh, dh), (0, 2, 1, 3)).reshape(nh, seq, dh))).sum()
    loss.backward()
    out = np.asarray(o.to_numpy(), np.float64)
    return (out, np.asarray(q.grad.to_numpy(), np.float64),
            np.asarray(k.grad.to_numpy(), np.float64),
            np.asarray(v.grad.to_numpy(), np.float64))


def rel(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


print("=== 低显存 vs 旧路径（GPU）===")
for (seq, nh, dh) in ((128, 4, 64), (256, 4, 64), (128, 8, 64)):
    base = run(seq, nh, dh, lowmem=False)
    low = run(seq, nh, dh, lowmem=True)
    rs = [rel(low[i], base[i]) for i in range(4)]
    worst = max(rs)
    print("  seq=%-4d nh=%d dh=%d  O=%.3e dQ=%.3e dK=%.3e dV=%.3e  worst=%.3e  %s"
          % (seq, nh, dh, rs[0], rs[1], rs[2], rs[3], worst, "PASS" if worst < TOL else "FAIL"))

print("=== 回退（非 causal / dh!=64）应当逐位一致 ===")
for (seq, nh, dh, causal) in ((64, 4, 32, True), (64, 4, 64, False)):
    base = run(seq, nh, dh, causal=causal, lowmem=False)
    low = run(seq, nh, dh, causal=causal, lowmem=True)
    worst = max(rel(low[i], base[i]) for i in range(4))
    print("  seq=%d nh=%d dh=%d causal=%s  worst=%.3e  %s"
          % (seq, nh, dh, causal, worst, "PASS(逐位)" if worst == 0 else "PASS(近似)" if worst < TOL else "FAIL"))

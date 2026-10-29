# -*- coding: utf-8 -*-
"""Smoke: grouped GEMM matches gemm_nt/nn/tn, CE bwd stays close to the old formula."""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

T.set_device("cuda")
rt = T._cuda()
rng = np.random.RandomState(0)


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def check_grouped():
    worst = 0.0
    shapes = [(200, 128, 64), (17, 32, 40), (4894, 1024, 256), (300, 64, 64)]
    for M, K, N in shapes:
        A = rt.h2d(rng.randn(M, K).astype(np.float32))
        B = rt.h2d(rng.randn(N, K).astype(np.float32))
        C = rt.grouped_nt([A], [B])[0]
        R = rt.gemm_nt(A, B)
        worst = max(worst, rel(rt.d2h(C), rt.d2h(R)))
        A2 = rt.h2d(rng.randn(M, K).astype(np.float32))
        B2 = rt.h2d(rng.randn(K, N).astype(np.float32))
        C2 = rt.grouped_nn([A2], [B2])[0]
        R2 = rt.sgemm(A2, B2)
        worst = max(worst, rel(rt.d2h(C2), rt.d2h(R2)))
        A3 = rt.h2d(rng.randn(K, M).astype(np.float32))
        B3 = rt.h2d(rng.randn(K, N).astype(np.float32))
        C3 = rt.grouped_tn([A3], [B3])[0]
        R3 = rt.gemm_tn(A3, B3)
        worst = max(worst, rel(rt.d2h(C3), rt.d2h(R3)))
    # two different M in one launch
    As = [rt.h2d(rng.randn(m, 32).astype(np.float32)) for m in (10, 40, 7)]
    Bs = [rt.h2d(rng.randn(16, 32).astype(np.float32)) for _ in range(3)]
    Cs = rt.grouped_nt(As, Bs)
    for A, B, C in zip(As, Bs, Cs):
        worst = max(worst, rel(rt.d2h(C), rt.d2h(rt.gemm_nt(A, B))))
    print("grouped_gemm worst_rel %.3e %s" % (worst, "PASS" if worst < 1e-5 else "FAIL"))
    return worst < 1e-5


def check_ce():
    N, V = 4, 300
    logits = rng.randn(N, V).astype(np.float32) * 0.5
    tgt = rng.randint(0, V, size=(N,)).astype(np.int32)
    # reference: stable softmax on host, same formula
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    p[np.arange(N), tgt] -= 1.0
    ref = (p / N).astype(np.float32)
    g = rt.ce_bwd(rt.h2d(logits), rt.h2d(tgt, dtype=np.int32), N, V)
    got = rt.d2h(g)
    r = rel(got, ref)
    # V<=256 path still used by parity; just make sure it runs
    logits2 = rng.randn(3, 8).astype(np.float32)
    tgt2 = np.array([0, 1, 7], np.int32)
    m2 = logits2.max(axis=1, keepdims=True)
    e2 = np.exp(logits2 - m2)
    p2 = e2 / e2.sum(axis=1, keepdims=True)
    p2[np.arange(3), tgt2] -= 1.0
    ref2 = (p2 / 3).astype(np.float32)
    g2 = rt.ce_bwd(rt.h2d(logits2), rt.h2d(tgt2, dtype=np.int32), 3, 8)
    r2 = rel(rt.d2h(g2), ref2)
    print("ce_bwd V=300 rel %.3e  V=8 rel %.3e" % (r, r2))
    # bandwidth-ish timing at the training shape, after warmup
    N, V = 8192, 32779
    big = rt.h2d(rng.randn(128, V).astype(np.float32))  # not full N, just compile
    tg = rt.h2d(rng.randint(0, V, size=(128,)).astype(np.int32), dtype=np.int32)
    rt.ce_bwd(big, tg, 128, V)
    rt.sync()
    logits = rt.h2d(rng.randn(N, V).astype(np.float32) * 0.2)
    targets = rt.h2d(rng.randint(0, V, size=(N,)).astype(np.int32), dtype=np.int32)
    for _ in range(2):
        rt.ce_bwd(logits, targets, N, V)
    rt.sync()
    ts = []
    for _ in range(3):
        rt.sync()
        t0 = time.perf_counter()
        rt.ce_bwd(logits, targets, N, V)
        rt.sync()
        ts.append(time.perf_counter() - t0)
    print("ce_bwd N=8192 V=32779 ms %s" % ", ".join("%.2f" % (t * 1e3) for t in ts))
    return r < 5e-4 and r2 < 5e-4


ok = check_grouped() and check_ce()
print("SMOKE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

# -*- coding: utf-8 -*-
"""2×fp16 拆分 GEMM 的单点精度/速度，以及一步梯度对拍。

在 4090 上：
    cd /home/featurize/tgai/e3
    PYTHONPATH=/home/featurize/tgai/e3 python3 -u /home/featurize/tgai/e3/split_gemm_test.py --mode both
"""
import argparse
import os
import sys
import time

import numpy as np

os.environ.pop("TG_FP16", None)
os.environ.setdefault("TG_CORPUS", "/home/featurize/tgai/e3/SFT/persona.jsonl")


def rel(got, ref):
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    denom = float(np.max(np.abs(ref)))
    if denom == 0.0:
        return float(np.max(np.abs(got)))
    return float(np.max(np.abs(got - ref)) / denom)


def _l2rel(got, ref):
    got = np.asarray(got, np.float64).ravel()
    ref = np.asarray(ref, np.float64).ravel()
    denom = float(np.linalg.norm(ref))
    if denom == 0.0:
        return float(np.linalg.norm(got))
    return float(np.linalg.norm(got - ref) / denom)


def set_split(rt, on):
    st = rt._split16_state()
    st["on"] = bool(on)
    st["warned"] = False
    cache = getattr(rt, "_split_cache", None)
    if cache is None:
        rt._split_cache = {}
    else:
        cache.clear()
    st["hits"] = 0
    st["misses"] = 0
    return st


def sync_ms(rt, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    rt.sync()
    return (time.perf_counter() - t0) / float(iters) * 1e3


def gemm_suite(rt, T):
    rng = np.random.RandomState(0)
    shapes = [(1280, 4992, 256), (4992, 256, 1280)]
    rows = []
    for M, K, N in shapes:
        A = (rng.randn(M, K) * 0.05).astype(np.float32)
        B = (rng.randn(K, N) * 0.05).astype(np.float32)
        tA, tB = T.tensor(A), T.tensor(B)
        set_split(rt, False)
        C32 = tA @ tB
        rt.sync()
        ref = np.asarray(C32.to_numpy(), np.float32)
        set_split(rt, True)
        Csp = tA @ tB
        rt.sync()
        got = np.asarray(Csp.to_numpy(), np.float32)
        err = rel(got, ref)
        # 冷启动：清缓存后只算一次，含拆分。
        set_split(rt, True)
        rt.sync()
        t0 = time.perf_counter()
        Cold = tA @ tB
        rt.sync()
        cold = (time.perf_counter() - t0) * 1e3
        st = set_split(rt, True)
        warm = sync_ms(rt, lambda: tA @ tB, 3, 10)
        hits, misses = int(st["hits"]), int(st["misses"])
        set_split(rt, False)
        base = sync_ms(rt, lambda: tA @ tB, 3, 10)
        print("SHAPE %d %d %d rel %.3e cold_ms %.3f warm_ms %.3f fp32_ms %.3f "
              "speedup %.3f hits %d misses %d nan %s"
              % (M, K, N, err, cold, warm, base, base / warm if warm else 0.0,
                 hits, misses,
                 "yes" if not np.isfinite(got).all() else "no"))
        rows.append((M, K, N, err, cold, warm, base, hits, misses))
        # 原地缩放后缓存必须失效，否则还会用旧的 hi/lo。
        set_split(rt, True)
        _ = (tA @ tB)
        rt.sync()
        rt.scale_inplace(tA._gpu, 1.1)
        C2 = tA @ tB
        rt.sync()
        set_split(rt, False)
        C2r = tA @ tB
        rt.sync()
        err_mut = rel(C2.to_numpy(), C2r.to_numpy())
        print("MUTATE scale 1.1 rel %.3e" % err_mut)
        rows.append(("mutate", M, K, N, err_mut))
        del tA, tB, C32, Csp, Cold, C2, C2r

    # 反向用的 nt / tn。列主序转置标志和 sgemm 不同，单独对一下。
    M, K, N = 1280, 4992, 256
    rng = np.random.RandomState(1)
    A = (rng.randn(M, K) * 0.05).astype(np.float32)
    BT = (rng.randn(N, K) * 0.05).astype(np.float32)   # gemm_nt 的 B 是 [N,K]
    AT = (rng.randn(K, M) * 0.05).astype(np.float32)   # gemm_tn 的 A 是 [K,M]
    B = (rng.randn(K, N) * 0.05).astype(np.float32)
    tA, tBT, tAT, tB = T.tensor(A), T.tensor(BT), T.tensor(AT), T.tensor(B)
    for name, fn in (
            ("gemm_nt", lambda: rt.gemm_nt(tA._gpu, tBT._gpu)),
            ("gemm_tn", lambda: rt.gemm_tn(tAT._gpu, tB._gpu))):
        set_split(rt, False)
        R = fn()
        rt.sync()
        set_split(rt, True)
        G = fn()
        rt.sync()
        err = rel(G.to_numpy(), R.to_numpy())
        print("OP %s rel %.3e" % (name, err))
        rows.append((name, err))

    # 小 GEMM 不该拆。结果应与 fp32 一致。
    A = (rng.randn(64, 64) * 0.05).astype(np.float32)
    B = (rng.randn(64, 64) * 0.05).astype(np.float32)
    tA, tB = T.tensor(A), T.tensor(B)
    set_split(rt, False)
    R = tA @ tB
    rt.sync()
    st = set_split(rt, True)
    G = tA @ tB
    rt.sync()
    err = rel(G.to_numpy(), R.to_numpy())
    print("SMALL 64 64 64 rel %.3e misses %d (expect 0)" % (err, st["misses"]))
    rows.append(("small", err, st["misses"]))
    return rows


def grad_suite(rt, T):
    sys.path.insert(0, "/home/featurize/tgai/e3")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
    import exam2_chat10m_tgai as E
    text, src = E.load_corpus()
    toks, stoi = E.build_vocab(text)
    x, y = E.build_examples(text, stoi, E.CFG["MAXLEN"])
    cfg = dict(E.CFG, V=len(toks), S=int(x.shape[1]))
    print("corpus %s dialogs %d S %d V %d" % (src, x.shape[0], cfg["S"], cfg["V"]))
    F = T.functional

    def one(split):
        set_split(rt, split)
        p = E.make_model(cfg, np.random.RandomState(0))
        opt = T.optim.AdamW(list(p.values()), lr=2e-3)
        logits = E.forward(p, x, cfg)
        loss = F.cross_entropy(
            logits.reshape(cfg["S"] * x.shape[0], cfg["V"]),
            T.tensor(y.reshape(-1)), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        rt.sync()
        lv = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
        logits_np = np.asarray(logits.to_numpy(), np.float32).copy()
        grads = {}
        for k, t in p.items():
            g = t.grad
            if g is None:
                continue
            grads[k] = np.asarray(g.to_numpy(), np.float32).copy()
        return lv, grads, logits_np

    loss32, g32, z32 = one(False)
    loss32b, g32b, z32b = one(False)
    wfp = 0.0
    wfpk = ""
    for k in g32:
        if k not in g32b:
            continue
        e2 = _l2rel(g32b[k], g32[k])
        if e2 > wfp:
            wfp, wfpk = e2, k
    print("FP32_VS_FP32 LOGITS l2 %.3e WORST_L2 %.3e (%s)" % (_l2rel(z32b, z32), wfp, wfpk))
    losssp, gsp, zsp = one(True)
    print("LOSS fp32 %.8f split %.8f rel %.3e" % (loss32, losssp, rel(losssp, loss32)))
    print("LOGITS rel_inf %.3e rel_l2 %.3e" % (rel(zsp, z32), _l2rel(zsp, z32)))
    worst = 0.0
    worst_k = ""
    n = 0
    bad = 0
    for k in g32:
        if k not in gsp:
            print("MISSING %s" % k)
            continue
        e = rel(gsp[k], g32[k])
        e2 = _l2rel(gsp[k], g32[k])
        if e2 > worst:
            worst = e2
            worst_k = k
        if e2 > 1e-5:
            bad += 1
        n += 1
        if n <= 8:
            print("GRAD %-48s inf %.3e l2 %.3e" % (k, e, e2))
    print("GRAD_COUNT %d WORST_L2 %.3e (%s) ABOVE_1E-5 %d" % (n, worst, worst_k, bad))
    os.environ["TG_FP16_SPLIT_CACHE"] = "0"
    loss0, g0, z0 = one(True)
    w0, w0k, b0 = 0.0, "", 0
    for k in g32:
        if k not in g0:
            continue
        e2 = _l2rel(g0[k], g32[k])
        if e2 > w0:
            w0, w0k = e2, k
        if e2 > 1e-5:
            b0 += 1
    print("NOCACHE LOSS rel %.3e LOGITS inf %.3e l2 %.3e WORST_L2 %.3e (%s) ABOVE %d"
          % (rel(loss0, loss32), rel(z0, z32), _l2rel(z0, z32), w0, w0k, b0))
    return loss32, losssp, worst, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gemm", "grad", "both"], default="both")
    a = ap.parse_args()
    import tgai_go.train as T
    T.set_default_device("cuda")
    from tgai_go.train import _cuda
    rt = _cuda()
    rt.ensure()
    if a.mode in ("gemm", "both"):
        gemm_suite(rt, T)
    if a.mode in ("grad", "both"):
        grad_suite(rt, T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Isolate emb / attn / moe backward pieces. Not part of the score harness."""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

T.set_device("cuda")
rt = T._cuda()
rt.ensure_attn_nop()


def bench(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    rt.sync()
    return (time.perf_counter() - t0) / n


def bench_emb(seq, mode):
    d, V = 1024, 32779
    g = rt.full((seq, d), 0.01)
    rng = np.random.RandomState(0)
    if mode == "skew":
        ids_np = (np.arange(seq) % 256).astype(np.int32)
    elif mode == "byte":
        ids_np = rng.randint(0, 256, size=seq).astype(np.int32)
    else:
        ids_np = rng.randint(0, V, size=seq).astype(np.int32)
    ids = rt.h2d(ids_np, dtype=np.int32)
    # touch unique count
    nunq = int(np.unique(ids_np).size)
    dt = bench(lambda: rt.emb_bwd(g, ids, seq, d, (V, d)))
    print("emb seq=%d mode=%s uniq=%d  %.3f ms" % (seq, mode, nunq, dt * 1e3))


def bench_memset(nfloats):
    buf = rt.empty((nfloats,), dtype=np.float32)
    dt = bench(lambda: rt._memset32(buf, 0))
    print("memset %.1f MB  %.3f ms" % (nfloats * 4 / 1e6, dt * 1e3))


def bench_attn(seq, nh=16, dh=64):
    rng = np.random.RandomState(1)
    shape = (seq, nh, dh)
    q = rt.h2d(rng.randn(*shape).astype(np.float32))
    k = rt.h2d(rng.randn(*shape).astype(np.float32))
    v = rt.h2d(rng.randn(*shape).astype(np.float32))
    scale = 1.0 / (dh ** 0.5)
    o, m, lse = rt.attn_nop_fwd_stats(q, k, v, scale)
    do = rt.h2d(rng.randn(*shape).astype(np.float32) * 0.01)
    rt.sync()
    dt = bench(lambda: rt.attn_nop_bwd_flash(do, q, k, v, o, m, lse, scale), n=3, warm=1)
    print("attn_bwd seq=%d  %.3f ms" % (seq, dt * 1e3))


print("warmup compile")
bench_emb(128, "skew")
bench_memset(32779 * 1024)
bench_memset(8192 * 1024)
for seq in (2048, 8192):
    for mode in ("skew", "byte", "uniform"):
        bench_emb(seq, mode)
for seq in (2048, 8192):
    bench_attn(seq)
print("DONE")

# -*- coding: utf-8 -*-
"""Isolate gemm_tn / sgemm shapes used by dW."""
import os
import numpy as np
os.environ.pop("TG_FP16", None)
import tgai_go.train as T
T.set_default_device("cuda")
from tgai_go.train import _cuda
rt = _cuda()
rt.ensure()

def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))

def run(name, fn):
    rt._split16_state()["on"] = False
    R = fn()
    rt.sync()
    rt._split16_state()["on"] = True
    if getattr(rt, "_split_cache", None) is not None:
        rt._split_cache.clear()
    G = fn()
    rt.sync()
    print("%-20s rel %.3e" % (name, rel(G.to_numpy(), R.to_numpy())))

rng = np.random.RandomState(0)
for scale in (0.05, 1.0, 1e-4):
    G = T.tensor((rng.randn(4992, 256) * scale).astype(np.float32))
    H = T.tensor((rng.randn(4992, 256) * scale).astype(np.float32))
    run("tn256 scale %.1e" % scale, lambda: T.tensor(rt.d2h(rt.gemm_tn(G._gpu, H._gpu))))
    # tensor() of d2h is silly; gemm_tn returns GpuBuf. Wrap properly.
print("--- direct buf ---")
for scale in (0.05, 1.0, 1e-3):
    g = (rng.randn(4992, 256) * scale).astype(np.float32)
    h = (rng.randn(4992, 256) * scale).astype(np.float32)
    G, H = T.tensor(g), T.tensor(h)
    rt._split16_state()["on"] = False
    R = rt.gemm_tn(G._gpu, H._gpu)
    rt.sync()
    rn = rt.d2h(R)
    rt._split16_state()["on"] = True
    rt._split_cache.clear()
    S = rt.gemm_tn(G._gpu, H._gpu)
    rt.sync()
    sn = rt.d2h(S)
    print("tn256 scale %.1e rel %.3e maxref %.3e maxdiff %.3e" % (
        scale, rel(sn, rn), np.max(np.abs(rn)), np.max(np.abs(sn - rn))))

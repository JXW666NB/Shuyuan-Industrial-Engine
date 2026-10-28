# -*- coding: utf-8 -*-
"""M4-Muon 归因：NS5 的 float32（CPU/GPU）与 float64 参考三方对照。

目的：判断 Muon 的 7.0e-07 是"实现错误"还是"float32 固有舍入"。
判据逻辑：
  - 若 CPU-fp32 与 GPU-fp32 各自都离 float64 参考很近（~1e-7 量级），
    而两者互相之间差 7e-7 → 说明是双方各自舍入的累积，属固有；
  - 若某一方离 fp64 参考明显更远 → 那一方有实现问题。
"""
import sys

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402


def ns5_fp64(g, steps=5, eps=1e-7):
    """与实现同公式的 float64 参考（这是"真值"的近似）。"""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = np.array(g, dtype=np.float64)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = np.ascontiguousarray(x.T)
        transposed = True
    nrm = np.sqrt(np.sum(x * x, dtype=np.float64))
    x = x / (nrm + eps)
    for _ in range(steps):
        s = x @ x.T
        y = b * s + c * (s @ s)
        x = a * x + y @ x
    if transposed:
        x = np.ascontiguousarray(x.T)
    return x


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


rng = np.random.RandomState(0)
T.set_device("cuda")
rt = T._cuda()

print("shape          cpu_vs_fp64   gpu_vs_fp64   gpu_vs_cpu")
for shape in ((16, 32), (32, 16), (64, 64), (128, 64)):
    g = (rng.randn(*shape) * 0.1).astype(np.float32)
    ref = ns5_fp64(g)
    cpu = np.asarray(T._newton_schulz5(g.copy()), np.float64)
    buf = rt.h2d(g)
    out = T._newton_schulz5_gpu(rt, buf)
    gpu = np.asarray(out.to_numpy() if hasattr(out, "to_numpy") else rt.d2h(out), np.float64)
    print("%-14s %11.3e %13.3e %12.3e"
          % ("%dx%d" % shape, rel(cpu, ref), rel(gpu, ref), rel(gpu, cpu)))

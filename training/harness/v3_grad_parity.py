# -*- coding: utf-8 -*-
"""B4 反向判分台：V3（非 CSA）的梯度 vs 官方实现。

真值来源：`training/oracle/v3_tiny_grads.npz`（官方 `model_v3.py` 导出，112 个张量，
上游权重固定种子 7，loss=4.653289）。

判据：
- 自证：同侧跑两次，每个参数的梯度必须**逐位一致**（差 0）
- 考核：与官方梯度逐参数比较，**相对误差 < 1e-3**

用法：python v3_grad_parity.py --selftest | --compare
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, HERE)

import tgai_go.train as T            # noqa: E402

ORACLE = os.path.join(REPO, "training", "oracle", "v3_tiny_")
TOL = 1e-3


def _np(x):
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy(), np.float64)
    return np.asarray(x, np.float64)


def build_and_grad():
    """跑一次前向+反向，返回 {参数名: 梯度 ndarray}。"""
    import v3_ref
    cfg, W = v3_ref.load_oracle(ORACLE)
    ids = np.load(ORACLE + "ids.npy")
    upstream = np.load(ORACLE + "upstream.npy")

    m = v3_ref.V3Auto(cfg, W)
    logits = m.forward(ids)                       # [1,S,V]
    up = T.Tensor(np.asarray(upstream, np.float32))
    loss = (logits * up).sum() if hasattr((logits * up), "sum") else None
    for p in m.parameters():
        p.grad = None
    loss.backward()
    out = {}
    for k, p in m.w.items():
        g = getattr(p, "grad", None)
        if g is not None:
            out[k] = _np(g)
    return out, float(_np(loss).reshape(-1)[0])


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    d = float(np.max(np.abs(a - b)))
    s = float(np.max(np.abs(b))) + 1e-12
    return d / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    T.set_device("cpu")

    if a.selftest:
        g1, l1 = build_and_grad()
        g2, l2 = build_and_grad()
        keys = sorted(set(g1) & set(g2))
        worst = max((float(np.max(np.abs(g1[k] - g2[k]))) for k in keys), default=0.0)
        print("[判分台自证] V3 反向同侧两次 worst=%.3e（%d 个张量）" % (worst, len(keys)))
        ok = worst == 0.0 and l1 == l2
        print("  VERDICT ok=%s（必须恰好为 0）" % ok)
        return 0 if ok else 1

    if a.compare:
        ref = np.load(ORACLE + "grads.npz")
        got, loss = build_and_grad()
        print("[B4 反向考核] V3（非 CSA）梯度 vs 官方实现")
        print("  loss = %.6f（官方 4.653289）" % loss)
        rows = []
        for k in sorted(ref.files):
            if k not in got:
                rows.append((k, float("inf")))
                continue
            rows.append((k, rel(got[k], ref[k])))
        rows.sort(key=lambda r: -r[1])
        miss = [k for k in ref.files if k not in got]
        worst_k, worst_v = rows[0]
        print("  官方 %d 个张量，我们给出 %d 个，缺 %d 个" % (len(ref.files), len(got), len(miss)))
        if miss:
            print("  缺失:", miss[:6])
        print("  最差: %s = %.3e" % (worst_k, worst_v))
        for k, v in rows[:5]:
            print("    %-42s %.3e" % (k, v))
        ok = (not miss) and worst_v < TOL
        print("  VERDICT ok=%s（判据 <%.0e）" % (ok, TOL))
        return 0 if ok else 1

    print("请指定 --selftest 或 --compare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

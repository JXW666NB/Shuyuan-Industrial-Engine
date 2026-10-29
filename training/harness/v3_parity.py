# -*- coding: utf-8 -*-
"""B4 判分台：TGAI V3（非 CSA）前向 vs 官方实现（oracle）。

判据：logits 相对误差 < 1e-4。
自证：同侧跑两次必须恰好 0。

用法: python v3_parity.py --selftest | --compare
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
import v3_ref                        # noqa: E402

ORACLE = os.path.join(REPO, "training", "oracle", "v3_tiny_")
TOL = 1e-4


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def run():
    cfg, W = v3_ref.load_oracle(ORACLE)
    ids = np.load(ORACLE + "ids.npy")
    m = v3_ref.V3Forward(cfg, W)
    return np.asarray(m.forward(ids).to_numpy(), np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    T.set_device("cpu")

    if a.selftest:
        r1 = run()
        r2 = run()
        w = rel(r2, r1)
        print("[判分台自证] V3 同侧两次 worst=%.3e" % w)
        ok = w == 0.0
        print("  VERDICT ok=%s（必须恰好为 0）" % ok)
        return 0 if ok else 1

    if a.compare:
        ref = np.load(ORACLE + "logits.npy")
        got = run()
        r = rel(got, ref)
        print("[B4 考核] V3（非 CSA）前向 vs 官方实现")
        print("  oracle logits %s | ours %s" % (ref.shape, got.shape))
        print("  相对误差 = %.3e" % r)
        ok = r < TOL
        print("  VERDICT ok=%s（判据 <%.0e）" % (ok, TOL))
        return 0 if ok else 1

    print("请指定 --selftest 或 --compare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

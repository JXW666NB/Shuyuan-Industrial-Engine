# -*- coding: utf-8 -*-
"""M4 判分台：优化器在 CUDA 后端与 CPU 后端的参数更新对齐。

两个优化器各跑 3 步（覆盖一阶/二阶动量的偏差校正）：
    AdamW([lin1.weight, lin2.weight], lr=0.05)
    Muon ([lin1.weight, lin2.weight], lr=0.02)

判据（M4）：更新后的参数相对误差 < 1e-7（要求很高，因为两步都应是同一个 float32 算法）。

设计原则 —— **判分台先自证**：
    --selftest   CPU vs CPU 必须**恰好为 0**
    --cuda       CPU vs CUDA，用于考核

用法：
    python train_opt_parity.py --selftest
    python train_opt_parity.py --cuda
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 3)[0])

import tgai_go.train as T            # noqa: E402

D, C, N = 8, 4, 12
SEED = 555
TOL = 1e-7
STEPS = 3


def _arr(x):
    if x is None:
        return None
    a = x.to_numpy() if hasattr(x, "to_numpy") else x
    return np.array(a, dtype=np.float64)


def run(kind, device):
    if hasattr(T, "set_device"):
        T.set_device(device)
    np.random.seed(SEED)
    lin1 = T.nn.Linear(D, D)
    lin2 = T.nn.Linear(D, C)
    params = [lin1.weight, lin2.weight]

    if kind == "adamw":
        opt = T.optim.AdamW(params, lr=0.05)
    elif kind == "muon":
        opt = T.optim.Muon(params, lr=0.02)
    else:
        raise ValueError(kind)

    g = np.random.RandomState(11)
    X = (g.randn(N, D) * 0.5).astype(np.float32)
    tgt = g.randint(0, C, size=N).astype(np.int64)

    losses = []
    for _ in range(STEPS):
        x = T.Tensor(X)
        y = lin2(T.functional.silu(lin1(x)))
        loss = T.functional.cross_entropy(y, tgt)
        losses.append(float(_arr(loss).reshape(-1)[0]))
        lin1.zero_grad()
        lin2.zero_grad()
        loss.backward()
        opt.step()

    out = {"loss": np.array(losses), "lin1": _arr(lin1.weight), "lin2": _arr(lin2.weight)}
    if hasattr(T, "set_device"):
        T.set_device("cpu")
    return out


def rel(a, b):
    if a is None or b is None:
        return float("inf")
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        print("[判分台自证] CPU vs CPU（同 seed，两次独立运行）")
        worst = 0.0
        for kind in ("adamw", "muon"):
            r1 = run(kind, "cpu")
            r2 = run(kind, "cpu")
            for k in ("loss", "lin1", "lin2"):
                r = rel(r2[k], r1[k])
                worst = max(worst, r)
                print("  %-6s %-5s %.3e" % (kind, k, r))
        ok = worst == 0.0
        print("  VERDICT ok=%s（必须恰好为 0）" % ok)
        return 0 if ok else 1

    if a.cuda:
        if not hasattr(T, "set_device"):
            print("CUDA 后端尚未实现")
            return 2
        allok = True
        print("[M4 考核] CPU vs CUDA（每个优化器 %d 步）" % STEPS)
        for kind in ("adamw", "muon"):
            ref = run(kind, "cpu")
            try:
                cur = run(kind, "cuda")
            except Exception as ex:
                print("  %-6s ERROR %r" % (kind, ex))
                allok = False
                continue
            rl = rel(cur["loss"], ref["loss"])
            r1 = rel(cur["lin1"], ref["lin1"])
            r2 = rel(cur["lin2"], ref["lin2"])
            worst = max(rl, r1, r2)
            flag = "PASS" if worst < TOL else "FAIL"
            print("  %-6s %s  worst=%.3e  (loss=%.3e lin1=%.3e lin2=%.3e)"
                  % (kind, flag, worst, rl, r1, r2))
            allok = allok and worst < TOL
        print("  VERDICT ok=%s（判据 < %.0e）" % (allok, TOL))
        return 0 if allok else 1

    print("请指定 --selftest 或 --cuda")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

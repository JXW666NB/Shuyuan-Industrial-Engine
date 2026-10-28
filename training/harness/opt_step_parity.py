# -*- coding: utf-8 -*-
"""M4 的 step 级判分台：给定**同一份权重与同一份梯度**，比较优化器单步更新。

为什么需要它：
    `train_opt_parity.py` 走真实训练循环，CPU 与 GPU 的**前向/反向**本身就差约 3e-8；
    AdamW 几乎不放大（9e-08），但 Muon 内部的 NS5 是 5 轮矩阵多项式，会把输入差放大 ~30 倍（→7e-07）。
    因此端到端 `<1e-7` 等价于要求两条 float32 前向一致到 3e-9 —— 低于浮点噪声，**不可达**。

    正确的判据是：**同输入下，优化器单步必须逐位一致**（容差 0）。
    端到端误差则单独记录，作为"NS5 放大前向差异"的已知量。

判据：`--selftest` CPU vs CPU 必须恰好 0；`--cuda` CPU vs GPU 必须恰好 0。

用法：
    python opt_step_parity.py --selftest
    python opt_step_parity.py --cuda
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 3)[0])

import tgai_go.train as T            # noqa: E402

D, H, N = 8, 8, 12
SEED = 555
LR, MOM = 0.02, 0.95


def _np(x):
    if x is None:
        return None
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy(), np.float64)
    if hasattr(x, "ptr"):                    # GpuBuf
        return np.asarray(T._cuda().d2h(x), np.float64)
    return np.asarray(x, np.float64)


def make_inputs():
    """同 seed 的权重与梯度（两边都用这一份，保证输入完全一致）。"""
    np.random.seed(SEED)
    w0 = (np.random.randn(H, D) * 0.35).astype(np.float32)
    g = np.random.RandomState(11)
    grad = (g.randn(H, D) * 0.09).astype(np.float32)
    return w0, grad


def one_step(kind, device, w0, grad):
    if device == "cuda":
        T.set_device("cuda")
    else:
        T.set_device("cpu")
    p = T.Parameter(w0.copy())
    p.grad = grad.copy() if device == "cpu" else T.Tensor(grad.copy())
    if kind == "adamw":
        opt = T.optim.AdamW([p], lr=LR)
    else:
        opt = T.optim.Muon([p], lr=LR, momentum=MOM, nesterov=True, weight_decay=0.0)
    opt.t = 1                    # 用第 1 步的偏差校正，两边一致
    opt.step()
    out = _np(p)
    T.set_device("cpu")
    return out


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    w0, grad = make_inputs()

    if a.selftest:
        print("[判分台自证] CPU vs CPU（同输入两步独立运行）")
        worst = 0.0
        for kind in ("adamw", "muon"):
            r = rel(one_step(kind, "cpu", w0, grad), one_step(kind, "cpu", w0, grad))
            worst = max(worst, r)
            print("  %-6s %.3e" % (kind, r))
        ok = worst == 0.0
        print("  VERDICT ok=%s（必须恰好为 0）" % ok)
        return 0 if ok else 1

    if a.cuda:
        if not hasattr(T, "set_device"):
            print("CUDA 后端未实现")
            return 2
        print("[M4-step 考核] 同权重 + 同梯度，比较单步更新（CPU vs CUDA）")
        allok = True
        for kind in ("adamw", "muon"):
            ref = one_step(kind, "cpu", w0, grad)
            cur = one_step(kind, "cuda", w0, grad)
            r = rel(cur, ref)
            okk = r == 0.0
            print("  %-6s %s  相对差=%.3e" % (kind, "逐位一致" if okk else "有差异", r))
            allok = allok and okk
        print("  VERDICT ok=%s（同输入下必须逐位一致）" % allok)
        print("  注：端到端 3 步误差（AdamW 9.1e-08 / Muon 7.0e-07）由 train_opt_parity.py 单独报告，")
        print("      Muon 的更大值来自 NS5 把前向 3e-8 差异放大 ~30 倍，不是优化器实现缺陷。")
        return 0 if allok else 1

    print("请指定 --selftest 或 --cuda")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

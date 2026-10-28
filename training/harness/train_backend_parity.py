# -*- coding: utf-8 -*-
"""M1 判分台：tgai_go.train 的 CUDA 后端与 CPU 后端对齐。

判据（M1）：Linear + cross_entropy 一个 step，loss 与梯度相对误差 < 1e-6。

设计原则 —— **判分台先自证**：
    --selftest   CPU vs CPU（两次独立运行）必须**逐位为 0**；不为 0 说明判分台本身有问题。
    --cuda       CPU vs CUDA 后端才允许用于考核。

契约（Grok 实现需要满足）：
    import tgai_go.train as T
    T.set_device("cuda")     # 之后新建的 Parameter / Module 都在 GPU 上
    ... 同一份模型与训练代码，不改一行 ...
    T.set_device("cpu")
    另外 Parameter / Tensor 上的 .to_numpy() 在 CUDA 张量上必须可用（内部做 D2H）。

用法：
    python train_backend_parity.py --selftest
    python train_backend_parity.py --cuda
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 3)[0])   # 让 tgai_go 可导入

import tgai_go.train as T            # noqa: E402

N, D, C = 16, 8, 4                   # 样本数 / 输入维度 / 类别数
SEED = 1234


def build_data():
    g = np.random.RandomState(7)
    X = g.randn(N, D).astype(np.float32)
    tgt = g.randint(0, C, size=N).astype(np.int64)
    return X, tgt


def one_step(device):
    """在指定 device 上：建模型 → 前向 → CE → 反向，返回 (loss, dW, W)。"""
    if hasattr(T, "set_device"):
        T.set_device(device)
    np.random.seed(SEED)             # 同一个初始化种子
    lin = T.nn.Linear(D, C)
    X, tgt = build_data()
    x = T.Tensor(X)
    y = lin(x)
    loss = T.functional.cross_entropy(y, tgt)
    lin.zero_grad()
    loss.backward()
    W = np.array(lin.weight.to_numpy(), dtype=np.float64)
    g = lin.weight.grad
    if g is None:
        dW = None
    else:
        dW = np.array(g.to_numpy() if hasattr(g, "to_numpy") else g, dtype=np.float64)
    lv = float(np.array(loss.to_numpy(), dtype=np.float64).reshape(-1)[0])
    if hasattr(T, "set_device"):
        T.set_device("cpu")
    return lv, dW, W


def rel(a, b):
    if a is None or b is None:
        return float("inf")
    d = np.max(np.abs(a - b))
    s = np.max(np.abs(b)) + 1e-12
    return float(d / s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()

    ref_loss, ref_dW, ref_W = one_step("cpu")

    if a.selftest:
        l2, d2, w2 = one_step("cpu")
        rl, rw, rd = rel(np.array(l2), np.array(ref_loss)), rel(w2, ref_W), rel(d2, ref_dW)
        print("[判分台自证] CPU vs CPU（同 seed，两次独立运行）")
        print("  loss 差 = %.3e" % rl)
        print("  W    差 = %.3e" % rw)
        print("  dW   差 = %.3e" % rd)
        ok = max(rl, rw, rd) == 0.0
        print("  VERDICT ok=%s（必须恰好为 0，否则判分台本身不可信）" % ok)
        return 0 if ok else 1

    if a.cuda:
        if not hasattr(T, "set_device"):
            print("CUDA 后端尚未实现：tgai_go.train 里没有 set_device()")
            print("（这是 M1 的交付内容，见 goal 里的契约）")
            return 2
        try:
            c_loss, c_dW, c_W = one_step("cuda")
        except Exception as ex:
            print("CUDA 后端调用失败：%r" % (ex,))
            return 3
        rl = rel(np.array(c_loss), np.array(ref_loss))
        rw = rel(c_W, ref_W)
        rd = rel(c_dW, ref_dW)
        print("[M1 考核] CPU vs CUDA 后端（同 seed、同一份模型代码）")
        print("  loss  CPU=%.8f  CUDA=%.8f  rel=%.3e" % (ref_loss, c_loss, rl))
        print("  W     rel=%.3e" % rw)
        print("  dW    rel=%.3e" % rd)
        ok = max(rl, rw, rd) < 1e-6
        print("  VERDICT ok=%s（判据 < 1e-6）" % ok)
        return 0 if ok else 1

    print("请指定 --selftest 或 --cuda")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

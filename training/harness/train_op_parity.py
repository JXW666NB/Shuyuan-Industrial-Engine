# -*- coding: utf-8 -*-
"""M2 判分台：逐个算子对比 CUDA 后端与 CPU 后端。

按**用户可见的公共 API** 测，不按 kernel 名测：
    rmsnorm   → tgai_go.train.nn.RMSNorm
    silu_mul  → functional.silu(a) * u（SwiGLU 的门）
    embedding → tgai_go.train.nn.Embedding（含重复下标）
    ce        → functional.cross_entropy（含 ignore_index）

判据（M2）：每个算子的前向输出与梯度，相对误差 < 1e-5。

设计原则 —— **判分台先自证**：
    --selftest   CPU vs CPU（两次独立运行）每一项必须**恰好为 0**；否则判分台不可信。
    --cuda       CPU vs CUDA，用于考核。

注：rope / moe_add 目前**没有**在 CPU 后端里作为公共算子暴露（它们属于 attention / MoE 模块），
按纪律不在 M2 虚列判据，等 M3 接模块时一并测 —— 这一点写在 ACCEPTANCE 里。

用法：
    python train_op_parity.py --selftest
    python train_op_parity.py --cuda
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 3)[0])

import tgai_go.train as T            # noqa: E402

SEED = 2024
TOL = 1e-5


def _arr(x):
    """把 Tensor / ndarray 统一成 float64 numpy。"""
    if x is None:
        return None
    a = x.to_numpy() if hasattr(x, "to_numpy") else x
    return np.array(a, dtype=np.float64)


def _coef(*shape):
    """固定系数张量，让 loss 的梯度不至于全 0 或强对称。"""
    g = np.random.RandomState(99)
    return (g.randn(*shape) * 0.5).astype(np.float32)


# ─────────────── 四个算子的 case ───────────────
def case_rmsnorm():
    g = np.random.RandomState(1)
    X = (g.randn(16, 64) * 0.5).astype(np.float32)
    nrm = T.nn.RMSNorm(64)
    # 让 weight 不是全 1，才能测到 dw
    nrm.weight = T.Parameter((g.randn(64) * 0.1 + 1.0).astype(np.float32))
    x = T.Tensor(X, requires_grad=True)
    y = nrm(x)
    c = _coef(*_arr(y).shape)
    loss = (y * T.Tensor(c)).sum()
    nrm.zero_grad()
    if x.grad is not None:
        x.grad = None
    loss.backward()
    return {"y": _arr(y), "dx": _arr(x.grad), "dw": _arr(nrm.weight.grad)}


def case_silu_mul():
    g = np.random.RandomState(2)
    A = (g.randn(16, 64) * 0.8).astype(np.float32)
    U = (g.randn(16, 64) * 0.8).astype(np.float32)
    a = T.Tensor(A, requires_grad=True)
    u = T.Tensor(U, requires_grad=True)
    y = T.functional.silu(a) * u
    c = _coef(*_arr(y).shape)
    loss = (y * T.Tensor(c)).sum()
    loss.backward()
    return {"y": _arr(y), "da": _arr(a.grad), "du": _arr(u.grad)}


def case_embedding():
    g = np.random.RandomState(3)
    idx = np.array([0, 3, 3, 7, 7, 7, 31, 0, 15, 15, 2, 9], dtype=np.int64)   # 含重复下标
    emb = T.nn.Embedding(32, 16)
    emb.weight = T.Parameter((g.randn(32, 16) * 0.1).astype(np.float32))
    y = emb(idx)
    c = _coef(*_arr(y).shape)
    loss = (y * T.Tensor(c)).sum()
    emb.zero_grad()
    loss.backward()
    return {"y": _arr(y), "dw": _arr(emb.weight.grad)}


def case_ce():
    g = np.random.RandomState(4)
    D, C, N = 32, 8, 16
    lin = T.nn.Linear(D, C)
    lin.weight = T.Parameter((g.randn(C, D) * 0.2).astype(np.float32))
    X = (g.randn(N, D) * 0.5).astype(np.float32)
    tgt = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    y = lin(T.Tensor(X))
    loss = T.functional.cross_entropy(y, tgt)
    lin.zero_grad()
    loss.backward()
    return {"loss": _arr(loss).reshape(1), "dw": _arr(lin.weight.grad)}


CASES = [
    ("rmsnorm", case_rmsnorm, ("y", "dx", "dw")),
    ("silu_mul", case_silu_mul, ("y", "da", "du")),
    ("embedding", case_embedding, ("y", "dw")),
    ("cross_entropy", case_ce, ("loss", "dw")),
]


def run_all(device):
    if hasattr(T, "set_device"):
        T.set_device(device)
    outs = {}
    for name, fn, _ in CASES:
        np.random.seed(SEED)
        try:
            outs[name] = fn()
        except Exception as ex:
            outs[name] = {"__error__": repr(ex)}
    if hasattr(T, "set_device"):
        T.set_device("cpu")
    return outs


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

    ref = run_all("cpu")

    if a.selftest:
        again = run_all("cpu")
        print("[判分台自证] CPU vs CPU（同 seed，两次独立运行）")
        worst = 0.0
        for name, _, fields in CASES:
            for f in fields:
                r = rel(again[name].get(f), ref[name].get(f))
                worst = max(worst, r)
                print("  %-14s %-6s %.3e" % (name, f, r))
        ok = worst == 0.0
        print("  VERDICT ok=%s（必须恰好为 0）" % ok)
        return 0 if ok else 1

    if a.cuda:
        if not hasattr(T, "set_device"):
            print("CUDA 后端尚未实现（无 set_device）")
            return 2
        cur = run_all("cuda")
        print("[M2 考核] CPU vs CUDA 后端（同 seed、同一份模型代码）")
        allok = True
        for name, _, fields in CASES:
            if "__error__" in cur[name]:
                print("  %-14s ERROR %s" % (name, cur[name]["__error__"][:110]))
                allok = False
                continue
            worst = max(rel(cur[name].get(f), ref[name].get(f)) for f in fields)
            flag = "PASS" if worst < TOL else "FAIL"
            detail = "  ".join("%s=%.2e" % (f, rel(cur[name].get(f), ref[name].get(f)))
                               for f in fields)
            print("  %-14s %s  worst=%.2e  (%s)" % (name, flag, worst, detail))
            allok = allok and worst < TOL
        print("  VERDICT ok=%s（判据：每项 < %.0e）" % (allok, TOL))
        return 0 if allok else 1

    print("请指定 --selftest 或 --cuda")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

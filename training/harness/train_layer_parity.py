# -*- coding: utf-8 -*-
"""M3 判分台：三层 attention 小模型，CPU 后端 vs CUDA 后端的梯度对齐。

模型（三层 attention 堆叠，检验 reshape / head 拆分 / SDPA 反向 / 投影反向）：
    x[B,S,D]
      → Linear(D,D) → CausalSelfAttention(D,nh)
      → Linear(D,D) → CausalSelfAttention(D,nh)
      → Linear(D,D) → CausalSelfAttention(D,nh)
      → Linear(D,V) → cross_entropy

判据（M3）：loss 与**所有参数梯度**相对误差 < 1e-4。

设计原则 —— **判分台先自证**：
    --selftest   CPU vs CPU 必须**恰好为 0**
    --cuda       CPU vs CUDA，用于考核

用法：
    python train_layer_parity.py --selftest
    python train_layer_parity.py --cuda
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 3)[0])

import tgai_go.train as T            # noqa: E402

B, S, D, NH, V = 2, 16, 32, 4, 8
SEED = 777
TOL = 1e-4


def _arr(x):
    if x is None:
        return None
    a = x.to_numpy() if hasattr(x, "to_numpy") else x
    return np.array(a, dtype=np.float64)


def build(n_layers=3):
    """三层 attention 堆叠 + 输出头。

    CPU 后端的 Linear 只吃 2D，所以这里手动 reshape，不依赖后端扩展。
    """
    lins = [T.nn.Linear(D, D) for _ in range(n_layers)]
    attns = [T.nn.CausalSelfAttention(D, NH) for _ in range(n_layers)]
    head = T.nn.Linear(D, V)
    return lins, attns, head


def forward_stack(x, lins, attns, head):
    h = x
    for lin, att in zip(lins, attns):
        h = lin(h.reshape(B * S, D)).reshape(B, S, D)
        h = att(h)
    return head(h.reshape(B * S, D)).reshape(B, S, V)


def one_run(device):
    if hasattr(T, "set_device"):
        T.set_device(device)
    np.random.seed(SEED)
    lins, attns, head = build()
    g = np.random.RandomState(31)
    X = (g.randn(B, S, D) * 0.5).astype(np.float32)
    tgt = g.randint(0, V, size=B * S).astype(np.int64)

    x = T.Tensor(X, requires_grad=True)
    y = forward_stack(x, lins, attns, head)      # [B,S,V]
    loss = T.functional.cross_entropy(y.reshape(B * S, V), tgt)

    for m in list(lins) + list(attns) + [head]:
        m.zero_grad()
    loss.backward()

    out = {"loss": _arr(loss).reshape(1)}
    for tag, m in ([("lin%d" % i, m) for i, m in enumerate(lins)] +
                   [("attn%d" % i, m) for i, m in enumerate(attns)] +
                   [("head", head)]):
        for name, p in m.named_parameters():
            gg = p.grad
            out[tag + "." + name] = _arr(gg.to_numpy() if hasattr(gg, "to_numpy") else gg)
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

    ref = one_run("cpu")
    print("参数量：%d 个张量" % len(ref))

    if a.selftest:
        again = one_run("cpu")
        worst = 0.0
        for k in ref:
            r = rel(again.get(k), ref[k])
            worst = max(worst, r)
        print("[判分台自证] CPU vs CPU，%d 项" % len(ref))
        print("  worst = %.3e" % worst)
        ok = worst == 0.0
        print("  VERDICT ok=%s（必须恰好为 0）" % ok)
        return 0 if ok else 1

    if a.cuda:
        if not hasattr(T, "set_device"):
            print("CUDA 后端尚未实现")
            return 2
        try:
            cur = one_run("cuda")
        except Exception as ex:
            print("CUDA 后端调用失败：%r" % (ex,))
            return 3
        worst, worst_k = 0.0, ""
        bad = []
        for k in ref:
            r = rel(cur.get(k), ref[k])
            if r > worst:
                worst, worst_k = r, k
            if r >= TOL:
                bad.append((k, r))
        print("[M3 考核] CPU vs CUDA（三层 attention 堆叠）")
        print("  loss rel = %.3e" % rel(cur.get("loss"), ref["loss"]))
        print("  参数量 = %d，最差项 = %s (%.3e)" % (len(ref), worst_k, worst))
        if bad:
            for k, r in bad[:8]:
                print("    FAIL %-40s %.3e" % (k, r))
        ok = worst < TOL
        print("  VERDICT ok=%s（判据 < %.0e）" % (ok, TOL))
        return 0 if ok else 1

    print("请指定 --selftest 或 --cuda")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

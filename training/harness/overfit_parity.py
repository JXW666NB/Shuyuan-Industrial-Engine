# -*- coding: utf-8 -*-
"""过拟合对照：**同一份训练循环**分别跑 TGAI GO(CUDA) 与 PyTorch，逐 step 比 loss。

这是"TGAI GO 能否替代 PyTorch"的核心判据：
- 若两者 loss 曲线逐步吻合（<1e-3 相对差），说明前向/反向/优化器都对；
- 若两条曲线都降到接近 0，说明"过拟合测试"通过（能排除大部分梯度/更新类 BUG）。

同一份 `train_once(bk, ...)` 被两个后端调用 —— bk 既可以是 `tgai_go.train`，
也可以是真正的 `torch`。**这正是"改个导入就迁移"的最小证明**。

用法（在 4090 上）：
    python overfit_parity.py --steps 400
    python overfit_parity.py --steps 400 --tgai-device cpu
"""
import argparse
import sys

import numpy as np

V, D, S, B = 64, 32, 8, 4          # 32 个预测目标 ⇒ 过拟合很快看得到


def train_once(bk, ids, tgt, steps, lr, init, device=None):
    """一份代码，两个后端。init = (emb_w, head_w, head_b)，由 numpy 固定种子生成后灌入。

    ★ 初始化必须两后端逐位相同，否则逐 step 比 loss 没有意义（我第一版就栽在这）。
    """
    if device is not None and hasattr(bk, "set_device"):
        bk.set_device(device)
    nn, F, optim = bk.nn, bk.functional, bk.optim

    emb = nn.Embedding(V, D)
    head = nn.Linear(D, V)
    # ★ 必须用 load_state_dict：CUDA 下 p.data 是 None（值在 p._gpu），直接赋 .data 不生效。
    emb.load_state_dict({"weight": init[0]})
    head.load_state_dict({"weight": init[1], "bias": init[2]})
    # 自检：确认初始化真的灌进去了（否则逐 step 比 loss 毫无意义）
    _sd = {"emb": emb.state_dict(), "head": head.state_dict()}
    import numpy as _np
    assert _np.allclose(_sd["emb"]["weight"], init[0]), "emb 初始化没生效"
    assert _np.allclose(_sd["head"]["weight"], init[1]), "head.weight 初始化没生效"
    assert _np.allclose(_sd["head"]["bias"], init[2]), "head.bias 初始化没生效"

    opt = optim.AdamW([emb.weight, head.weight, head.bias], lr=lr)
    losses = []
    for _ in range(steps):
        x = emb(ids).reshape(B * S, D)
        logits = head(x)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss).reshape(-1)[0]))
    return losses


def train_once_torch(ids, tgt, steps, lr, init):
    """同一个循环，换成真 PyTorch。"""
    import torch
    import torch.nn as tnn
    import torch.nn.functional as tF
    import torch.optim as topt

    emb = tnn.Embedding(V, D)
    head = tnn.Linear(D, V)
    with torch.no_grad():
        emb.weight.copy_(torch.tensor(init[0]))
        head.weight.copy_(torch.tensor(init[1]))
        head.bias.copy_(torch.tensor(init[2]))
    opt = topt.AdamW(list(emb.parameters()) + list(head.parameters()), lr=lr)
    ids_t = torch.tensor(ids, dtype=torch.long)
    tgt_t = torch.tensor(tgt, dtype=torch.long)
    out = []
    for _ in range(steps):
        x = emb(ids_t).reshape(B * S, D)
        logits = head(x)
        loss = tF.cross_entropy(logits, tgt_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        out.append(float(loss))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tgai-device", default="cuda")
    a = ap.parse_args()

    g = np.random.RandomState(0)
    ids = g.randint(0, V, size=(B, S)).astype(np.int64)
    # ★ 标签必须**由 id 决定**（可学映射），否则同一个 id 配不同标签，loss 永远到不了 0。
    tgt = ((ids * 7 + 3) % V).reshape(-1).astype(np.int64)

    # 统一初始化（numpy 固定种子），两个后端逐位相同
    gi = np.random.RandomState(1234)
    init = ((gi.randn(V, D) * 0.02).astype(np.float32),
            (gi.randn(V, D) / np.sqrt(D)).astype(np.float32),
            np.zeros(V, np.float32))

    import tgai_go.train as tg
    la = train_once(tg, ids, tgt, a.steps, a.lr, init, device=a.tgai_device)
    lb = train_once_torch(ids, tgt, a.steps, a.lr, init)

    print("steps=%d lr=%g  TGAI device=%s" % (a.steps, a.lr, a.tgai_device))
    print("  TGAI  loss %.6f -> %.6f" % (la[0], la[-1]))
    print("  torch loss %.6f -> %.6f" % (lb[0], lb[-1]))
    for i in range(0, a.steps, max(1, a.steps // 6)):
        print("    step %3d   tgai %.6f   torch %.6f" % (i, la[i], lb[i]))
    print("    step %3d   tgai %.6f   torch %.6f" % (a.steps - 1, la[-1], lb[-1]))

    rel = max(abs(x - y) / (abs(y) + 1e-12) for x, y in zip(la, lb))
    print("  逐 step 最大相对差 = %.3e" % rel)
    overfit_ok = la[-1] < 0.05 and lb[-1] < 0.05
    print("  过拟合（两边都 <0.05）: %s" % overfit_ok)
    print("  VERDICT", "PASS" if (rel < 1e-3 and overfit_ok) else "FAIL")
    return 0 if (rel < 1e-3 and overfit_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

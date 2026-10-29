# -*- coding: utf-8 -*-
"""示例 ①：一个"PyTorch 风格"的训练脚本，**只改导入**就跑在 TGAI GO 上。

这段代码除了第一行 `import tgai_go.train as torch` 之外，写法与 PyTorch 完全一致：
`nn.Embedding` / `nn.Linear` / `optim.AdamW` / `loss.backward()` / `opt.zero_grad()`。

用途：
1. 证明"改个导入名就能迁移"（在本库支持的算子范围内）；
2. 顺带是一个**过拟合测试**：固定同一批数据反复训练，loss 应该一路降到接近 0。

跑法（CUDA）：
    python examples/torch_style_overfit.py            # 默认 cuda
    python examples/torch_style_overfit.py --cpu      # 无 GPU 时
"""
import argparse
import sys

import numpy as np

import tgai_go.train as torch                       # ← 唯一的"迁移改动"
from tgai_go.train import nn, optim
from tgai_go.train import functional as F       # 与 torch.nn.functional 对应

V, D, S, B = 128, 64, 16, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    dev = "cpu" if a.cpu else "cuda"
    torch.set_device(dev)

    g = np.random.RandomState(0)
    ids = g.randint(0, V, size=(B, S)).astype(np.int64)
    tgt = ((ids * 7 + 3) % V).reshape(-1).astype(np.int64)   # 由 id 决定的可学映射 ⇒ 能过拟合到 0

    emb = nn.Embedding(V, D)
    head = nn.Linear(D, V)
    # ★ CUDA 下要给参数赋初值，必须用 load_state_dict（p.data 为 None，直接赋值不生效）
    gi = np.random.RandomState(1234)
    emb.load_state_dict({'weight': (gi.randn(V, D) * 0.02).astype(np.float32)})
    head.load_state_dict({'weight': (gi.randn(V, D) / np.sqrt(D)).astype(np.float32),
                          'bias': np.zeros(V, np.float32)})
    params = [emb.weight, head.weight, head.bias]
    opt = optim.AdamW(params, lr=3e-3)

    print("device=%s  steps=%d  V=%d D=%d B*S=%d" % (dev, a.steps, V, D, B * S))
    first = None
    for step in range(a.steps):
        x = emb(ids).reshape(B * S, D)
        logits = head(x)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        v = float(loss.to_numpy().reshape(-1)[0])
        if first is None:
            first = v
        if step % 10 == 0 or step == a.steps - 1:
            print("  step %3d  loss %.6f" % (step, v))
    last = v
    print("首末 loss: %.6f -> %.6f   下降 %.1f%%" % (first, last, (1 - last / first) * 100))
    print("过拟合判据（fixed batch, 应趋近 0）:", "PASS" if last < 0.05 else "FAIL")


if __name__ == "__main__":
    sys.exit(main())

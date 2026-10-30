# -*- coding: utf-8 -*-
"""BF16 天花板实验：在任何机器上，用 PyTorch 测出 BF16 相对 fp32 的真实收益。

为什么要先做这个：我们引擎目前只有 fp16 路径，没有 BF16。
在动引擎之前，必须先用 PyTorch 回答两个问题：
  1. 这张卡上 BF16 **硬件**到底快多少（GEMM 层面）；
  2. 整步训练开 autocast(bf16) 后，速度与 loss 相对 fp32 差多少。

用法（任意有 PyTorch 的机器）：
    python tools/bench_bf16_ceiling.py                 # 默认 600 步
    python tools/bench_bf16_ceiling.py --steps 200

判定：
  * 没有 BF16 硬件（算力 < 8.0）→ 直接告诉你"不要用"，并给出实测证据；
  * 有硬件 → 打印 fp32/fp16/bf16 三档的 ms/step 与 loss，loss 偏差 ≤2% 才算可用。
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "examples"))


def gemm_bench(torch, M=4992, K=256, N=1280, iters=20):
    """同形状三种精度的 GEMM 计时（都带 sync）。"""
    out = {}
    for name, dt in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        try:
            a = torch.randn(M, K, device="cuda", dtype=dt) * 0.05
            b = torch.randn(K, N, device="cuda", dtype=dt) * 0.05
            for _ in range(3):
                c = a @ b
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                c = a @ b
            torch.cuda.synchronize()
            out[name] = (time.perf_counter() - t0) / iters * 1e3
        except Exception as exc:  # noqa: BLE001
            out[name] = float("nan")
            print("  %s 失败: %s" % (name, str(exc)[:70]))
    return out


def run_twin(torch, steps, mode):
    """用 torch 孪生脚本跑 steps 步。mode: fp32 / fp16 / bf16。"""
    import importlib
    if "exam2_chat10m_torch" in sys.modules:
        del sys.modules["exam2_chat10m_torch"]
    E = importlib.import_module("exam2_chat10m_torch")

    import numpy as np
    text, _ = E.load_corpus()
    toks, stoi = E.build_vocab(text)
    x, y = E.build_examples(text, stoi, E.CFG["MAXLEN"])
    cfg = dict(E.CFG, V=len(toks), S=int(x.shape[1]))
    p = E.make_model(cfg, np.random.RandomState(0))
    w = {k: torch.nn.Parameter(torch.tensor(v)) for k, v in p.items()}
    opt = torch.optim.AdamW(list(w.values()), lr=2e-3)
    xt, yt = torch.tensor(x), torch.tensor(y)
    F = torch.nn.functional
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mode)

    def one():
        with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
            logits = E.forward(w, xt, cfg)
            loss = F.cross_entropy(logits.reshape(cfg["S"] * x.shape[0], cfg["V"]),
                                   yt.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.detach())

    for _ in range(3):                      # 热身
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lv = None
    for _ in range(steps):
        lv = one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1e3, lv


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--modes", default="fp32,bf16")
    a = ap.parse_args(argv)

    import torch
    if not torch.cuda.is_available():
        print("没有 CUDA 设备")
        return 1
    cap = torch.cuda.get_device_capability(0)
    print("设备: %s" % torch.cuda.get_device_name(0))
    print("算力: %s   torch %s   cuda %s" % (cap, torch.__version__, torch.version.cuda))
    hw_bf16 = cap[0] >= 8
    print("BF16 硬件: %s（算力 %d.%d；sm_80+ 才有真 BF16）"
          % ("有" if hw_bf16 else "没有，PyTorch 会软件模拟", cap[0], cap[1]))

    print("\n== 单 GEMM（4992x256x1280）==")
    g = gemm_bench(torch)
    print("%-8s %10s %10s" % ("精度", "ms", "相对 fp32"))
    for k, v in g.items():
        print("%-8s %10.3f %10.2fx" % (k, v, (g["fp32"] / v) if v == v and v else float("nan")))

    print("\n== %d 步训练（torch 孪生脚本，同代码）==" % a.steps)
    res = {}
    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        try:
            ms, lv = run_twin(torch, a.steps, mode)
            res[mode] = (ms, lv)
            print("%-6s %9.2f ms/step   末 loss %.6f" % (mode, ms, lv))
        except Exception as exc:  # noqa: BLE001
            print("%-6s 失败: %s" % (mode, str(exc)[:100]))

    if "fp32" in res:
        base_ms, base_loss = res["fp32"]
        print("\n== 结论 ==")
        for mode, (ms, lv) in res.items():
            if mode == "fp32":
                continue
            dloss = abs(lv - base_loss) / max(abs(base_loss), 1e-12) * 100
            print("%-6s 速度 %.2fx，loss 偏差 %.2f%%——%s"
                  % (mode, base_ms / ms, dloss,
                     "可用" if dloss <= 2.0 and ms < base_ms else "不可用"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

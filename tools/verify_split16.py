# -*- coding: utf-8 -*-
"""独立验收 2×fp16 拆分 GEMM（不采信实现者的自测）。

两次进程分别跑，结果写 JSON，再对比：
    python tools/verify_split16.py --mode fp32  --out v_fp32.json
    python tools/verify_split16.py --mode split --out v_split.json
    python tools/verify_split16.py --compare v_fp32.json v_split.json

每个模式都会实测：
  1. 单 GEMM（两个形状）与 fp64 参考的最大相对误差、耗时；
  2. 一步训练的 loss + 若干参数梯度（用于跨模式对拍）；
  3. 20 步训练的每步耗时（热身 3 步后算）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "examples"))


def rel(C, ref):
    C = np.asarray(C, np.float64)
    return float(np.abs(C - ref).max() / max(np.abs(ref).max(), 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp32", "split"], default="fp32")
    ap.add_argument("--out", default="")
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args(argv)

    if a.mode == "split":
        os.environ["TG_FP16_SPLIT"] = "1"
    else:
        os.environ.pop("TG_FP16_SPLIT", None)
        os.environ.pop("TG_FP16", None)

    import tgai_go.train as T
    T.set_default_device("cuda")
    from tgai_go.train import _cuda
    import exam2_chat10m_tgai as E

    rt = _cuda()
    F = T.functional
    res = {"mode": a.mode, "shapes": {}}

    # ---------- 1. 单 GEMM ----------
    rng = np.random.RandomState(0)
    for (M, K, N) in [(1280, 4992, 256), (4992, 256, 1280)]:
        A = (rng.randn(M, K) * 0.05).astype(np.float32)
        B = (rng.randn(K, N) * 0.05).astype(np.float32)
        ref = A.astype(np.float64) @ B.astype(np.float64)
        tA, tB = T.tensor(A), T.tensor(B)
        C = (tA @ tB)
        rt.sync()
        err = rel(C.to_numpy(), ref)
        t0 = time.perf_counter()
        for _ in range(5):
            C = (tA @ tB)
        rt.sync()
        dt = (time.perf_counter() - t0) / 5 * 1e3
        res["shapes"]["%dx%dx%d" % (M, K, N)] = {"relerr": err, "ms": dt}
        del tA, tB, C

    # ---------- 2. 一步训练的 loss + 梯度 ----------
    text, src = E.load_corpus()
    toks, stoi = E.build_vocab(text)
    x, y = E.build_examples(text, stoi, E.CFG["MAXLEN"])
    cfg = dict(E.CFG, V=len(toks), S=int(x.shape[1]))
    p = E.make_model(cfg, np.random.RandomState(0))
    opt = T.optim.AdamW(list(p.values()), lr=2e-3)
    logits = E.forward(p, x, cfg)
    loss = F.cross_entropy(logits.reshape(cfg["S"] * x.shape[0], cfg["V"]),
                           T.tensor(y.reshape(-1)), ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    res["loss"] = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
    keys = ["token_embedding.weight", "blocks.0.attn.q_proj.weight",
            "blocks.0.moe.experts.0.w1.weight", "blocks.7.attn.out_proj.weight",
            "blocks.3.moe.experts.0.w3.weight", "final_norm.weight"]
    grads = {}
    for k in keys:
        g = p[k].grad
        if g is not None:
            arr = np.asarray(g.to_numpy() if hasattr(g, "to_numpy") else g, np.float32)
            grads[k] = {"sum": float(arr.sum()), "absmax": float(np.abs(arr).max()),
                        "l2": float(np.sqrt((arr.astype(np.float64) ** 2).sum()))}
    res["grads"] = grads
    opt.step()

    # ---------- 3. 20 步耗时（新模型，避免被上面的状态影响） ----------
    p2 = E.make_model(cfg, np.random.RandomState(1))
    opt2 = T.optim.AdamW(list(p2.values()), lr=2e-3)
    for _ in range(3):
        lg = E.forward(p2, x, cfg)
        l2 = F.cross_entropy(lg.reshape(cfg["S"] * x.shape[0], cfg["V"]),
                             T.tensor(y.reshape(-1)), ignore_index=-100)
        opt2.zero_grad(); l2.backward(); opt2.step()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(a.steps):
        lg = E.forward(p2, x, cfg)
        l2 = F.cross_entropy(lg.reshape(cfg["S"] * x.shape[0], cfg["V"]),
                             T.tensor(y.reshape(-1)), ignore_index=-100)
        opt2.zero_grad(); l2.backward(); opt2.step()
    rt.sync()
    res["ms_per_step"] = (time.perf_counter() - t0) / a.steps * 1e3

    print(json.dumps({k: v for k, v in res.items() if k != "grads"}, ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("已写 %s" % a.out)
    return 0


def compare(f1, f2):
    A = json.load(open(f1, encoding="utf-8"))
    B = json.load(open(f2, encoding="utf-8"))
    print("fp32 = %s   split = %s" % (A["mode"], B["mode"]))
    print("\n== 单 GEMM ==")
    print("%-18s %12s %12s %10s %10s" % ("形状", "fp32 relerr", "split relerr", "fp32 ms", "split ms"))
    for k in A["shapes"]:
        a, b = A["shapes"][k], B["shapes"].get(k, {})
        print("%-18s %12.3e %12.3e %10.3f %10.3f"
              % (k, a["relerr"], b.get("relerr", float("nan")), a["ms"], b.get("ms", float("nan"))))
    print("\n== 单步 loss ==")
    print("fp32 %.8f   split %.8f   相对差 %.2e"
          % (A["loss"], B["loss"], abs(A["loss"] - B["loss"]) / max(abs(A["loss"]), 1e-12)))
    print("\n== 梯度对拍（l2 相对差）==")
    print("%-42s %12s" % ("参数", "相对差"))
    worst = 0.0
    for k in A["grads"]:
        if k not in B["grads"]:
            print("%-42s 缺失" % k); continue
        ga, gb = A["grads"][k]["l2"], B["grads"][k]["l2"]
        d = abs(ga - gb) / max(abs(ga), 1e-12)
        worst = max(worst, d)
        print("%-42s %12.3e" % (k, d))
    print("最差 %.3e（判据 ≤1e-5）" % worst)
    print("\n== 每步耗时 ==")
    print("fp32 %.2f ms   split %.2f ms   加速 %.2fx"
          % (A["ms_per_step"], B["ms_per_step"], A["ms_per_step"] / max(B["ms_per_step"], 1e-9)))
    ok = worst <= 1e-5 and B["ms_per_step"] <= A["ms_per_step"]
    print("\nVERDICT: %s" % ("PASS" if ok else "FAIL（精度或速度不达标）"))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--compare":
        raise SystemExit(compare(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())

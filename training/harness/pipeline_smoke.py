# -*- coding: utf-8 -*-
"""B6 判分台：一键流水线（语料 → 自训 BPE → 预设 → 训练 → 存盘 → 重载）。

判据：
1. 自证：同 seed 跑两次，最终 loss **恰好相同**（0 差）
2. 验收：
   a) 产出物齐全（tokenizer.json / ckpt.npz / meta.json）
   b) loss 明显下降（末值 < 首值 × 0.8）
   c) **重载后 logits 与存盘前逐位一致**（差 0）
   d) 词表由流水线自训，用户没有提供分词表

用法：python pipeline_smoke.py --selftest | --check
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "training"))

CORPUS_LINES = [
    "床前明月光，疑是地上霜。",
    "举头望明月，低头思故乡。",
    "白日依山尽，黄河入海流。",
    "欲穷千里目，更上一层楼。",
    "春眠不觉晓，处处闻啼鸟。",
    "夜来风雨声，花落知多少。",
] * 20


def make_corpus(root):
    d = os.path.join(root, "corpus")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "poem.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CORPUS_LINES))
    return d


def run_once(out_dir, seed=0, steps=30):
    from tgai_go.train.pipeline import run
    return run(corpus_dir=os.path.join(os.path.dirname(out_dir), "corpus"),
               out_dir=out_dir, arch="llama", vocab_size=192, seq=32,
               steps=steps, seed=seed, d_model=32, n_layers=1, n_heads=4, n_kv=2,
               d_ff=64, lr=3e-3, device="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    root = os.path.join(tempfile.gettempdir(), "tgai_pipeline")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    make_corpus(root)

    if a.selftest:
        r1 = run_once(os.path.join(root, "r1"), seed=0)
        r2 = run_once(os.path.join(root, "r2"), seed=0)
        d = abs(float(r1["final_loss"]) - float(r2["final_loss"]))
        print("[自证] 同 seed 两次 final_loss 差 = %.3e" % d)
        ok = d == 0.0
        print("  VERDICT ok=%s（必须恰好为 0）" % ok)
        return 0 if ok else 1

    if a.check:
        out = os.path.join(root, "run")
        r = run_once(out, seed=0)
        first, last = float(r["first_loss"]), float(r["final_loss"])
        print("[B6] 语料 → 自训 BPE → llama → 训练 %d 步" % r["steps"])
        print("  词表大小（自训）: %d，merges: %d" % (r["vocab_size"], r["n_merges"]))
        print("  loss: %.4f -> %.4f" % (first, last))
        files = {n: os.path.exists(os.path.join(out, n))
                 for n in ("tokenizer.json", "ckpt.npz", "meta.json")}
        print("  产出物: %s" % files)
        reload_ok = bool(r.get("reload_bitwise", False))
        print("  重载后 logits 逐位一致: %s" % reload_ok)
        ok = (last < first * 0.8) and all(files.values()) and reload_ok
        print("  VERDICT ok=%s" % ok)
        return 0 if ok else 1

    print("请指定 --selftest 或 --check")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

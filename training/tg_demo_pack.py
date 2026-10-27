# -*- coding: utf-8 -*-
"""生成一个"引擎可完整加载"的 demo .TG（含 rope 表），用于验证 .TG 数值通路。

模型: L 层、d、nh、ff、V、E、max_seq；权重命名严格按 tgai_format.h 的参数名约定。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tgai_pack import TGWriter  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tgtest")

L, D, NH, FF, V, E, TOPK, MAXSEQ = 2, 32, 4, 64, 40, 2, 2, 128
DH = D // NH
THETA = 10000.0


def rope_tables(max_seq, dh, theta):
    """真模型里 rope 表是 [1,1,max_seq,d_head]（1024 维模型为 64），
    即每行 d_head 宽；前一半是真值，后一半重复（兼容两种索引方式）。"""
    half = dh // 2
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    inv = theta ** (-2.0 * np.arange(half, dtype=np.float64) / dh)
    ang = pos * inv[None, :]
    c = np.cos(ang)
    s_ = np.sin(ang)
    return (np.concatenate([c, c], 1).astype(np.float32),
            np.concatenate([s_, s_], 1).astype(np.float32))


def build():
    rng = np.random.default_rng(1234)
    w = {}
    w["token_embedding.weight"] = (rng.standard_normal((V, D)) * 0.02).astype(np.float32)
    cos, sin = rope_tables(MAXSEQ, DH, THETA)
    for i in range(L):
        w[f"blocks.{i}.ln1.weight"] = np.ones(D, np.float32)
        w[f"blocks.{i}.ln2.weight"] = np.ones(D, np.float32)
        g = 1.0 / np.sqrt(D)
        w[f"blocks.{i}.attn.q_proj.weight"] = (rng.standard_normal((D, D)) * g).astype(np.float32)
        w[f"blocks.{i}.attn.k_proj.weight"] = (rng.standard_normal((D, D)) * g).astype(np.float32)
        w[f"blocks.{i}.attn.v_proj.weight"] = (rng.standard_normal((D, D)) * g).astype(np.float32)
        w[f"blocks.{i}.attn.out_proj.weight"] = (rng.standard_normal((D, D)) * g).astype(np.float32)
        # rope 表: 规范是 [1,1,max_seq,dh/2]
        w[f"blocks.{i}.attn.rope.cos"] = cos.reshape(1, 1, MAXSEQ, DH)
        w[f"blocks.{i}.attn.rope.sin"] = sin.reshape(1, 1, MAXSEQ, DH)
        w[f"blocks.{i}.moe.router.weight"] = (rng.standard_normal((E, D)) * g).astype(np.float32)
        w[f"blocks.{i}.moe.gate.weight"] = (rng.standard_normal((E, D)) * g).astype(np.float32)
        w[f"blocks.{i}.moe._qb_bias"] = np.zeros(E, np.float32)
        ge = 1.0 / np.sqrt(D)
        for e in range(E):
            w[f"blocks.{i}.moe.experts.{e}.w1.weight"] = (rng.standard_normal((FF, D)) * ge).astype(np.float32)
            w[f"blocks.{i}.moe.experts.{e}.w2.weight"] = (rng.standard_normal((FF, D)) * ge).astype(np.float32)
            w[f"blocks.{i}.moe.experts.{e}.w3.weight"] = (rng.standard_normal((D, FF)) * (1.0 / np.sqrt(FF))).astype(np.float32)
    w["final_norm.weight"] = np.ones(D, np.float32)
    return w


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "demo_rope.tg")
    w = build()
    tokens = [chr(ord("a") + i) for i in range(V - 3)] + ["<pad>", "<bos>", "<eos>"]
    wr = TGWriter(n_layers=L, d_model=D, n_heads=NH, d_ff=FF, vocab=V, max_seq=MAXSEQ,
                  n_experts=E, n_activated=TOPK, rope_theta=int(THETA),
                  pad=V - 3, bos=V - 2, eos=V - 1, unk=0, weight_dtype=0)
    wr.add_weights(w)
    wr.set_tokenizer(tokens, merges=[], special_ids=[V - 3, V - 2, V - 1, 0])
    info = wr.write(path)
    np.savez(os.path.join(OUT, "demo_rope_weights.npz"), **{k: v for k, v in w.items()})
    print("写出: %s  %d B  权重 %d 项（含 rope 表，dtype=FP32 便于逐位比对）"
          % (info["path"], info["size"], info["n_weights"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""BPE 训练（最朴素实现）+ 与引擎 tg_encode 的逐 token 对拍。

引擎编码规则（源码 tgai_engine.c:tg_encode 核对）：
  1) 特殊标记做贪心最长匹配，命中直接出 id
  2) 普通文本：' ' → space_token（若词表里有）；再按 UTF-8 字符切成 base 序列
  3) 反复合并 **rank 最小** 的相邻对（rank = merges 里的顺序）
  4) 未命中的连续段按字节回退 <bN>

用法:
    python bpe_train.py             # 训练 + 打包 + 对拍
"""
import ctypes
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tgai_pack import TGWriter                    # noqa: E402
from tg_load_test import TGHeader                 # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tgtest")
DLL = r"D:\TGAI\libtgai_engine_gpu.dll"

TEXT = ("<svg width=\"800\" height=\"600\"><circle cx=\"40\" cy=\"40\" r=\"20\"/></svg>"
        "<svg width=\"800\" height=\"600\"><circle cx=\"40\" cy=\"40\" r=\"20\"/></svg>")


def train_bpe(text, target_vocab):
    chars = sorted(set(text))
    vocab = list(chars)
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = [stoi[c] for c in text]
    merges = []
    while len(vocab) < target_vocab:
        pairs = Counter(zip(ids, ids[1:]))
        if not pairs:
            break
        (a, b), cnt = pairs.most_common(1)[0]
        if cnt < 2:
            break
        new_tok = vocab[a] + vocab[b]
        vocab.append(new_tok)
        new_id = len(vocab) - 1
        merges.append((a, b))
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return vocab, merges


def encode_ref(text, vocab, merges):
    stoi = {c: i for i, c in enumerate(vocab)}
    rank = {p: r for r, p in enumerate(merges)}
    ab = {}
    for (a, b) in merges:
        s = vocab[a] + vocab[b]
        ab[(a, b)] = stoi[s]
    ids = [stoi[ch] for ch in text]
    while True:
        best = None
        for i in range(len(ids) - 1):
            r = rank.get((ids[i], ids[i + 1]))
            if r is not None and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        i = best[1]
        ids[i:i + 2] = [ab[(ids[i], ids[i + 1])]]
    return ids


def build_dummy_tg(path, vocab, merges):
    """只为 tg_encode 服务的迷你模型（权重随意，必须字段齐全）"""
    D, NH, FF, V, E, MAXSEQ = 32, 4, 64, len(vocab), 1, 64
    rng = np.random.default_rng(0)
    w = {"token_embedding.weight": (rng.standard_normal((V, D)) * 0.02).astype(np.float32)}
    DH = D // NH
    half = DH // 2
    pos = np.arange(MAXSEQ, dtype=np.float64)[:, None]
    inv = 10000.0 ** (-2.0 * np.arange(half, dtype=np.float64) / DH)
    ang = pos * inv[None, :]
    c = np.concatenate([np.cos(ang)] * 2, 1).astype(np.float32)
    s = np.concatenate([np.sin(ang)] * 2, 1).astype(np.float32)
    for i in range(1):
        w[f"blocks.{i}.ln1.weight"] = np.ones(D, np.float32)
        w[f"blocks.{i}.ln2.weight"] = np.ones(D, np.float32)
        g = 1.0 / np.sqrt(D)
        for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
            w[f"blocks.{i}.attn.{p}.weight"] = (rng.standard_normal((D, D)) * g).astype(np.float32)
        w[f"blocks.{i}.attn.rope.cos"] = c.reshape(1, 1, MAXSEQ, DH)
        w[f"blocks.{i}.attn.rope.sin"] = s.reshape(1, 1, MAXSEQ, DH)
        w[f"blocks.{i}.moe.router.weight"] = np.zeros((E, D), np.float32)
        w[f"blocks.{i}.moe.gate.weight"] = np.zeros((E, D), np.float32)
        w[f"blocks.{i}.moe._qb_bias"] = np.zeros(E, np.float32)
        for e in range(E):
            w[f"blocks.{i}.moe.experts.{e}.w1.weight"] = (rng.standard_normal((FF, D)) * g).astype(np.float32)
            w[f"blocks.{i}.moe.experts.{e}.w2.weight"] = (rng.standard_normal((FF, D)) * g).astype(np.float32)
            w[f"blocks.{i}.moe.experts.{e}.w3.weight"] = (rng.standard_normal((D, FF)) / np.sqrt(FF)).astype(np.float32)
    w["final_norm.weight"] = np.ones(D, np.float32)

    wr = TGWriter(n_layers=1, d_model=D, n_heads=NH, d_ff=FF, vocab=V, max_seq=MAXSEQ,
                  n_experts=E, n_activated=1, rope_theta=10000,
                  pad=0, bos=0, eos=0, unk=0, weight_dtype=0)
    wr.add_weights(w)
    wr.set_tokenizer(vocab, merges=merges, special_ids=[0, 0, 0, 0])
    wr.write(path)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    vocab, merges = train_bpe(TEXT, target_vocab=len(set(TEXT)) + 24)
    print("训练完成: 词表 %d, merges %d" % (len(vocab), len(merges)))
    print("  merges:", merges[:8], "..." if len(merges) > 8 else "")
    print("  新 token:", [t for t in vocab[len(set(TEXT)):][:8]])

    tg = os.path.join(OUT, "bpe_demo.tg")
    build_dummy_tg(tg, vocab, merges)

    lib = ctypes.CDLL(DLL)
    lib.tg_load.restype = ctypes.c_void_p
    lib.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
    lib.tg_encode.restype = ctypes.c_int
    lib.tg_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                              ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int]
    lib.tg_free.argtypes = [ctypes.c_void_p]

    m = lib.tg_load(tg.encode(), 0, None)
    assert m, "load 失败"

    ok_all = True
    for s in (TEXT[:60], TEXT, "<circle cx=\"40\"", "svg", "<svg width=\"8\" height=\"6\">"):
        buf = (ctypes.c_int32 * 512)()
        n = lib.tg_encode(m, s.encode("utf-8"), buf, 512, 0)
        got = [buf[i] for i in range(n)]
        ref = encode_ref(s, vocab, merges)
        same = got == ref
        ok_all = ok_all and same
        print("\n文本: %r" % s[:40])
        print("  engine: %s" % got[:16])
        print("  ours  : %s" % ref[:16])
        print("  一致: %s" % same)
    lib.tg_free(m)
    print("\nVERDICT ok=%s" % ok_all)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

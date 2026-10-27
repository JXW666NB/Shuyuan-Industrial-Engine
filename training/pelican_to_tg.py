# -*- coding: utf-8 -*-
"""把训练好的鹈鹕 char-LM 打成 .TG，并用真引擎生成文本（端到端闭环）。

用法:
    python pelican_to_tg.py pack       # 训练完先打包
    python pelican_to_tg.py gen [N]    # 用引擎生成 N 个字符
"""
import ctypes
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tgai_pack import TGWriter                    # noqa: E402
from tg_load_test import TGHeader                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_tgtest")
DLL = r"D:\TGAI\libtgai_engine_gpu.dll"
CKPT = os.path.join(OUT, "pelican_lm.pt")
TG = os.path.join(OUT, "pelican.tg")

# 与 charlm_torch.py 的训练配置一致
NL, D, NH = 4, 512, 16
DH = D // NH
FF = 1024
SEQ = 1024
THETA = 10000.0
CORPUS = [os.path.join(HERE, "_corpus", "pelican_on_bike.html"),
          os.path.join(HERE, "_corpus", "bike_on_pelican.html")]


def rope_tables(max_seq, dh, theta):
    half = dh // 2
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    inv = theta ** (-2.0 * np.arange(half, dtype=np.float64) / dh)
    ang = pos * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    return (np.concatenate([c, c], 1).astype(np.float32),
            np.concatenate([s, s], 1).astype(np.float32))


def load_ckpt():
    sd = torch.load(CKPT, map_location="cpu")
    return {k: v.numpy() for k, v in sd.items()}


def pack():
    sd = load_ckpt()
    emb = sd["emb"]                                  # [V,D]
    V = emb.shape[0]
    cos, sin = rope_tables(SEQ, DH, THETA)
    w = {"token_embedding.weight": emb,
         "final_norm.weight": sd["w3"]}
    for i in range(NL):
        Wqkv = sd[f"Wqkv_{i}"]                        # [D, 3D] = [in,out]
        w[f"blocks.{i}.ln1.weight"] = sd[f"w1_{i}"]
        w[f"blocks.{i}.attn.q_proj.weight"] = Wqkv[:, :D].T
        w[f"blocks.{i}.attn.k_proj.weight"] = Wqkv[:, D:2 * D].T
        w[f"blocks.{i}.attn.v_proj.weight"] = Wqkv[:, 2 * D:].T
        w[f"blocks.{i}.attn.out_proj.weight"] = sd[f"Wo_{i}"].T
        w[f"blocks.{i}.attn.rope.cos"] = cos.reshape(1, 1, SEQ, DH)
        w[f"blocks.{i}.attn.rope.sin"] = sin.reshape(1, 1, SEQ, DH)
        w[f"blocks.{i}.ln2.weight"] = sd[f"w2_{i}"]
        w[f"blocks.{i}.moe.router.weight"] = np.zeros((1, D), np.float32)
        w[f"blocks.{i}.moe.gate.weight"] = np.zeros((1, D), np.float32)
        w[f"blocks.{i}.moe._qb_bias"] = np.zeros(1, np.float32)
        w[f"blocks.{i}.moe.experts.0.w1.weight"] = sd[f"Wg_{i}"].T   # [FF,D]
        w[f"blocks.{i}.moe.experts.0.w2.weight"] = sd[f"Wu_{i}"].T
        w[f"blocks.{i}.moe.experts.0.w3.weight"] = sd[f"Wd_{i}"].T   # [D,FF]

    text = "".join(open(p, encoding="utf-8").read() for p in CORPUS)
    vocab = sorted(set(text))
    assert len(vocab) == V, "词表不一致: %d vs %d" % (len(vocab), V)

    wr = TGWriter(n_layers=NL, d_model=D, n_heads=NH, d_ff=FF, vocab=V, max_seq=SEQ,
                  n_experts=1, n_activated=1, rope_theta=int(THETA),
                  pad=0, bos=0, eos=0, unk=0, weight_dtype=0)
    wr.add_weights(w)
    wr.set_tokenizer(vocab, merges=[], special_ids=[0, 0, 0, 0])
    info = wr.write(TG)
    print("打包完成: %s  %.1f MB  权重 %d 项  V=%d"
          % (info["path"], info["size"] / 2**20, info["n_weights"], V))
    with open(os.path.join(OUT, "pelican_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    return 0


def gen(n_chars=400, prompt_frac=0.06):
    lib = ctypes.CDLL(DLL)
    lib.tg_load.restype = ctypes.c_void_p
    lib.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
    lib.tg_config.restype = ctypes.POINTER(TGHeader)
    lib.tg_config.argtypes = [ctypes.c_void_p]
    lib.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
    lib.tg_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                               ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.tg_free.argtypes = [ctypes.c_void_p]

    m = lib.tg_load(TG.encode(), 0, None)
    assert m, "引擎加载失败"
    h = lib.tg_config(m).contents
    vocab = json.load(open(os.path.join(OUT, "pelican_vocab.json"), encoding="utf-8"))
    print("引擎读到: L=%d d=%d nh=%d ff=%d V=%d seq=%d E=%d topk=%d"
          % (h.n_layers, h.d_model, h.n_heads, h.d_ff, h.vocab_size,
             h.max_seq_len, h.n_experts, h.n_activated))
    stoi = {c: i for i, c in enumerate(vocab)}

    A = open(CORPUS[0], encoding="utf-8").read()
    B = open(CORPUS[1], encoding="utf-8").read()
    div = 0
    while div < min(len(A), len(B)) and A[div] == B[div]:
        div += 1
    for name, txt in (("pelican_on_bike", A), ("bike_on_pelican", B)):
        cut = min(div + 40, len(txt) // 2)
        ids = [stoi[c] for c in txt[:cut]]
        for _ in range(n_chars):
            arr = (ctypes.c_int32 * len(ids))(*ids)
            ptr = lib.tg_forward(m, arr, len(ids), None, 0)
            logits = np.ctypeslib.as_array(ptr, shape=(h.vocab_size,))
            nxt = int(np.argmax(logits))
            ids.append(nxt)
        out = "".join(vocab[i] for i in ids[cut:])
        truth = txt[cut:cut + n_chars]
        same = sum(1 for a, b in zip(out, truth) if a == b)
        print("\n=== %s ===" % name)
        print("引擎生成 %d 字符，逐字符匹配 %d/%d = %.1f%%"
              % (n_chars, same, len(truth), same / max(1, len(truth)) * 100))
        print("片段: " + out[:150].replace("\n", "⏎"))
    lib.tg_free(m)
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pack"
    if mode == "pack":
        sys.exit(pack())
    sys.exit(gen(int(sys.argv[2]) if len(sys.argv) > 2 else 400))

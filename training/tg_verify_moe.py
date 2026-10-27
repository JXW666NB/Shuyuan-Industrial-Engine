# -*- coding: utf-8 -*-
"""验证多专家路由 + qb_bias：E=2, topk=2，随机 router 权重与非零偏置。

引擎语义（源码核对）：
  logits = h2 @ router.weight^T + _qb_bias      （bias 在 top-k 之前加）
  top-k → softmax（仅选中项）→ 加权专家输出
  gate.weight 引擎不读（遗留项）
"""
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tgai_pack import TGWriter                    # noqa: E402
from tg_load_test import TGHeader                 # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tgtest")
DLL = r"D:\TGAI\libtgai_engine_gpu.dll"
L, D, NH, FF, V, MAXSEQ = 1, 32, 4, 64, 40, 128
E, TOPK = 2, 2
DH = D // NH
THETA = 10000.0
EPS = 1e-6


def rmsnorm(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def rope_tables(max_seq, dh, theta):
    half = dh // 2
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    inv = theta ** (-2.0 * np.arange(half, dtype=np.float64) / dh)
    ang = pos * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    return np.concatenate([c, c], 1).astype(np.float32), np.concatenate([s, s], 1).astype(np.float32)


def build():
    rng = np.random.default_rng(21)
    w = {"token_embedding.weight": (rng.standard_normal((V, D)) * 0.05).astype(np.float32)}
    cos, sin = rope_tables(MAXSEQ, DH, THETA)
    for i in range(L):
        w[f"blocks.{i}.ln1.weight"] = np.ones(D, np.float32)
        w[f"blocks.{i}.ln2.weight"] = np.ones(D, np.float32)
        g = 1.0 / np.sqrt(D)
        for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
            w[f"blocks.{i}.attn.{p}.weight"] = (rng.standard_normal((D, D)) * g).astype(np.float32)
        w[f"blocks.{i}.attn.rope.cos"] = cos.reshape(1, 1, MAXSEQ, DH)
        w[f"blocks.{i}.attn.rope.sin"] = sin.reshape(1, 1, MAXSEQ, DH)
        w[f"blocks.{i}.moe.router.weight"] = (rng.standard_normal((E, D)) * (0.5 / np.sqrt(D))).astype(np.float32)
        w[f"blocks.{i}.moe.gate.weight"] = (rng.standard_normal((E, D)) * g).astype(np.float32)
        w[f"blocks.{i}.moe._qb_bias"] = (rng.standard_normal(E) * 0.1).astype(np.float32)
        for e in range(E):
            w[f"blocks.{i}.moe.experts.{e}.w1.weight"] = (rng.standard_normal((FF, D)) * g).astype(np.float32)
            w[f"blocks.{i}.moe.experts.{e}.w2.weight"] = (rng.standard_normal((FF, D)) * g).astype(np.float32)
            w[f"blocks.{i}.moe.experts.{e}.w3.weight"] = (rng.standard_normal((D, FF)) / np.sqrt(FF)).astype(np.float32)
    w["final_norm.weight"] = np.ones(D, np.float32)
    return w


def ref(w, ids):
    emb = w["token_embedding.weight"]
    x = emb[ids].astype(np.float64)
    for i in range(L):
        h = rmsnorm(x, w[f"blocks.{i}.ln1.weight"])
        q = h @ w[f"blocks.{i}.attn.q_proj.weight"].T
        k = h @ w[f"blocks.{i}.attn.k_proj.weight"].T
        v = h @ w[f"blocks.{i}.attn.v_proj.weight"].T
        cos = w[f"blocks.{i}.attn.rope.cos"].reshape(MAXSEQ, DH)[:, :DH // 2]
        sin = w[f"blocks.{i}.attn.rope.sin"].reshape(MAXSEQ, DH)[:, :DH // 2]

        def rot(t):
            t = t.reshape(len(ids), NH, DH).copy()
            for p in range(len(ids)):
                ev = t[p, :, 0::2].copy()
                od = t[p, :, 1::2].copy()
                t[p, :, 0::2] = ev * cos[p] - od * sin[p]
                t[p, :, 1::2] = ev * sin[p] + od * cos[p]
            return t.reshape(len(ids), D)

        q, k = rot(q), rot(k)
        qh = q.reshape(len(ids), NH, DH).transpose(1, 0, 2)
        kh = k.reshape(len(ids), NH, DH).transpose(1, 0, 2)
        vh = v.reshape(len(ids), NH, DH).transpose(1, 0, 2)
        sc = qh @ kh.transpose(0, 2, 1) / np.sqrt(DH)
        mask = np.tril(np.ones((len(ids), len(ids)), bool))
        sc = np.where(mask, sc, -np.inf)
        sc = sc - sc.max(-1, keepdims=True)
        pr = np.exp(sc)
        pr = pr / pr.sum(-1, keepdims=True)
        at = (pr @ vh).transpose(1, 0, 2).reshape(len(ids), D)
        x = x + at @ w[f"blocks.{i}.attn.out_proj.weight"].T

        h2 = rmsnorm(x, w[f"blocks.{i}.ln2.weight"])
        logits = h2 @ w[f"blocks.{i}.moe.router.weight"].T + w[f"blocks.{i}.moe._qb_bias"]
        acc = np.zeros_like(x)
        for t in range(len(ids)):
            idx = np.argsort(-logits[t])[:TOPK]           # 同分取小下标由 argsort 稳定性近似
            sel = logits[t][idx]
            sel = sel - sel.max()
            p = np.exp(sel)
            p = p / p.sum()
            for j, e in enumerate(idx):
                gu = h2[t] @ w[f"blocks.{i}.moe.experts.{e}.w1.weight"].T
                up = h2[t] @ w[f"blocks.{i}.moe.experts.{e}.w2.weight"].T
                silu = gu / (1 + np.exp(-gu))
                acc[t] = acc[t] + p[j] * ((silu * up) @ w[f"blocks.{i}.moe.experts.{e}.w3.weight"].T)
        x = x + acc
    hn = rmsnorm(x, "final_norm.weight" if False else w["final_norm.weight"])
    return hn @ w["token_embedding.weight"].T


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "verify_moe.tg")
    w = build()
    tokens = [chr(ord("a") + i) for i in range(V - 3)] + ["<pad>", "<bos>", "<eos>"]
    wr = TGWriter(n_layers=L, d_model=D, n_heads=NH, d_ff=FF, vocab=V, max_seq=MAXSEQ,
                  n_experts=E, n_activated=TOPK, rope_theta=int(THETA),
                  pad=V - 3, bos=V - 2, eos=V - 1, unk=0, weight_dtype=0)
    wr.add_weights(w)
    wr.set_tokenizer(tokens)
    wr.write(path)

    lib = ctypes.CDLL(DLL)
    lib.tg_load.restype = ctypes.c_void_p
    lib.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
    lib.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
    lib.tg_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                               ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.tg_free.argtypes = [ctypes.c_void_p]

    m = lib.tg_load(path.encode(), 0, None)
    assert m, "load 失败"
    ids = np.array([1, 2, 3, 4], np.int32)
    ptr = lib.tg_forward(m, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(ids), None, 0)
    got = np.ctypeslib.as_array(ptr, shape=(V,)).copy()
    refv = ref(w, ids)[-1]
    e = np.abs(got - refv).max() / (np.abs(refv).max() + 1e-12)
    print("E=%d topk=%d  got[0,:5]=%s" % (E, TOPK, np.round(got[:5], 5).tolist()))
    print("             ref[0,:5]=%s" % np.round(refv[:5], 5).tolist())
    print("             最大相对误差 = %.3e" % e)
    lib.tg_free(m)
    ok = e < 1e-4
    print("VERDICT ok=%s rel=%.3e" % (ok, e))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

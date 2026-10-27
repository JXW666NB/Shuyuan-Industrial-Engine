# -*- coding: utf-8 -*-
"""分层验证 .TG 数值通路：从"整层恒等"开始，逐层加回 attention / MoE。

stage 0: q/k/v/o = 0、专家 w1/w2/w3 = 0、router/gate/qb = 0
         ⇒ 每层输出 = 输入（残差恒等）
         ⇒ logits = rmsnorm(emb) * final_norm @ emb^T   （lm_head 与 embedding 共享）
stage 1: 加上 attention（q/k/v/o 非零，MoE 仍为 0）
stage 2: 全开（MoE 也开）

用法: python tg_verify_layers.py <stage 0|1|2>
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
L, D, NH, FF, V, E, TOPK, MAXSEQ = 1, 32, 4, 64, 40, 1, 1, 128
DH = D // NH
THETA = 10000.0
EPS = 1e-6          # 引擎 rmsnorm 用的是 1e-6


def rope_tables(max_seq, dh, theta):
    half = dh // 2
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    inv = theta ** (-2.0 * np.arange(half, dtype=np.float64) / dh)
    ang = pos * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    return np.concatenate([c, c], 1).astype(np.float32), np.concatenate([s, s], 1).astype(np.float32)


def build(stage):
    rng = np.random.default_rng(7)
    w = {}
    w["token_embedding.weight"] = (rng.standard_normal((V, D)) * 0.05).astype(np.float32)
    cos, sin = rope_tables(MAXSEQ, DH, THETA)
    for i in range(L):
        w[f"blocks.{i}.ln1.weight"] = np.ones(D, np.float32)
        w[f"blocks.{i}.ln2.weight"] = np.ones(D, np.float32)
        on = 1.0 if stage >= 1 else 0.0
        g = on / np.sqrt(D)
        w[f"blocks.{i}.attn.q_proj.weight"] = (rng.standard_normal((D, D)) * g).astype(np.float32)
        w[f"blocks.{i}.attn.k_proj.weight"] = (rng.standard_normal((D, D)) * g).astype(np.float32)
        w[f"blocks.{i}.attn.v_proj.weight"] = (rng.standard_normal((D, D)) * g).astype(np.float32)
        w[f"blocks.{i}.attn.out_proj.weight"] = (rng.standard_normal((D, D)) * g).astype(np.float32)
        w[f"blocks.{i}.attn.rope.cos"] = cos.reshape(1, 1, MAXSEQ, DH)
        w[f"blocks.{i}.attn.rope.sin"] = sin.reshape(1, 1, MAXSEQ, DH)
        w[f"blocks.{i}.moe.router.weight"] = np.zeros((E, D), np.float32)
        w[f"blocks.{i}.moe.gate.weight"] = np.zeros((E, D), np.float32)
        w[f"blocks.{i}.moe._qb_bias"] = np.zeros(E, np.float32)
        onm = 1.0 if stage >= 2 else 0.0
        for e in range(E):
            w[f"blocks.{i}.moe.experts.{e}.w1.weight"] = (rng.standard_normal((FF, D)) * onm / np.sqrt(D)).astype(np.float32)
            w[f"blocks.{i}.moe.experts.{e}.w2.weight"] = (rng.standard_normal((FF, D)) * onm / np.sqrt(D)).astype(np.float32)
            w[f"blocks.{i}.moe.experts.{e}.w3.weight"] = (rng.standard_normal((D, FF)) * onm / np.sqrt(FF)).astype(np.float32)
    w["final_norm.weight"] = np.ones(D, np.float32)
    return w


def rmsnorm(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def ref_logits(w, ids):
    emb = w["token_embedding.weight"]
    x = emb[ids].astype(np.float64)                     # [T,D]
    for i in range(L):
        h = rmsnorm(x, w[f"blocks.{i}.ln1.weight"])
        # 权重约定为 [out, in]（与 PyTorch nn.Linear 一致）⇒ 要转置
        q = h @ w[f"blocks.{i}.attn.q_proj.weight"].T
        k = h @ w[f"blocks.{i}.attn.k_proj.weight"].T
        v = h @ w[f"blocks.{i}.attn.v_proj.weight"].T
        # RoPE（成对旋转，表取前 dh/2 列）
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
        # 因果注意力
        qh = q.reshape(len(ids), NH, DH).transpose(1, 0, 2)
        kh = k.reshape(len(ids), NH, DH).transpose(1, 0, 2)
        vh = v.reshape(len(ids), NH, DH).transpose(1, 0, 2)
        sc = qh @ kh.transpose(0, 2, 1) / np.sqrt(DH)
        mask = np.tril(np.ones((len(ids), len(ids)), bool))
        sc = np.where(mask, sc, -np.inf)
        sc = sc - sc.max(-1, keepdims=True)
        p = np.exp(sc)
        p = p / p.sum(-1, keepdims=True)
        at = (p @ vh).transpose(1, 0, 2).reshape(len(ids), D)
        x = x + at @ w[f"blocks.{i}.attn.out_proj.weight"].T
        h2 = rmsnorm(x, w[f"blocks.{i}.ln2.weight"])
        if stage_of(w) >= 2:
            acc = np.zeros_like(x)
            for e in range(E):
                g = 1.0 / E if E else 0.0
                gu = (h2 @ w[f"blocks.{i}.moe.experts.{e}.w1.weight"].T)
                silu = gu / (1 + np.exp(-gu))
                up = h2 @ w[f"blocks.{i}.moe.experts.{e}.w2.weight"].T
                acc = acc + (silu * up) @ w[f"blocks.{i}.moe.experts.{e}.w3.weight"].T * g
            x = x + acc
    hn = rmsnorm(x, w["final_norm.weight"])
    return hn @ w["token_embedding.weight"].T


_STAGE = 0


def stage_of(_w):
    return _STAGE


def main(stage):
    global _STAGE
    _STAGE = stage
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "verify_s%d.tg" % stage)
    w = build(stage)
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
    lib.tg_config.restype = ctypes.POINTER(TGHeader)
    lib.tg_config.argtypes = [ctypes.c_void_p]
    lib.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
    lib.tg_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                               ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.tg_free.argtypes = [ctypes.c_void_p]

    m = lib.tg_load(path.encode(), 0, None)
    assert m, "load 失败"
    ids = np.array([1, 2, 3, 4], np.int32)
    ptr = lib.tg_forward(m, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                         len(ids), None, 0)
    # 引擎只返回最后一个位置的 logits（长度 V）
    got = np.ctypeslib.as_array(ptr, shape=(V,)).reshape(1, V).copy()
    ref = ref_logits(w, ids)[-1:, :]
    e = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)
    print("stage=%d  logits: got[0,:5]=%s" % (stage, np.round(got[0, :5], 5).tolist()))
    print("           ref[0,:5]=%s" % (np.round(ref[0, :5], 5).tolist()))
    print("           got 范围 [%.4g, %.4g]  ref 范围 [%.4g, %.4g]"
          % (got.min(), got.max(), ref.min(), ref.max()))
    print("           最大相对误差 = %.3e" % e)
    lib.tg_free(m)
    ok = e < 1e-4
    print("VERDICT stage=%d ok=%s rel=%.3e" % (stage, ok, e))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 0))

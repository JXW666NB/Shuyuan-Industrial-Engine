# -*- coding: utf-8 -*-
"""纯 CPU 侧验证（不需要 GPU）：
  1) 我们写的 chat meta 段能被仓库自带的 parse_tg_chat_meta 正确解析
  2) CPU 引擎（libtgai_engine.dll）能加载我们的 .TG 并前向
  3) CPU 前向的 logits 与 NumPy 参考一致

用法: python tg_cpu_verify.py
"""
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "_repos", "Shuyuan-Industrial-Engine")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_pack import TGWriter                          # noqa: E402
from tg_load_test import TGHeader                       # noqa: E402

CPU_DLL = r"D:\TGAI\01_源码\TGAI-GO-wt\cpu\build\libtgai_engine.dll"
OUT = os.path.join(HERE, "_tgtest")
L, D, NH, FF, V, E, TOPK, MAXSEQ = 1, 32, 4, 64, 40, 1, 1, 128
DH = D // NH
THETA = 10000.0
EPS = 1e-6
META = {
    5: {"temperature": "0.7", "top_p": "0.9", "max_tokens": "512"},   # GEN_DEFAULTS
    4: ["<pad>", "<bos>", "<eos>"],                                        # SPECIAL_TOKENS
    2: ["<|end|>", "</s>"],                                                # STOP_STRINGS
}


def rmsnorm(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def rope_tables(max_seq, dh, theta):
    half = dh // 2
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    inv = theta ** (-2.0 * np.arange(half, dtype=np.float64) / dh)
    ang = pos * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    return (np.concatenate([c, c], 1).astype(np.float32),
            np.concatenate([s, s], 1).astype(np.float32))


def build():
    rng = np.random.default_rng(42)
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
        w[f"blocks.{i}.moe.router.weight"] = np.zeros((E, D), np.float32)
        w[f"blocks.{i}.moe.gate.weight"] = np.zeros((E, D), np.float32)
        w[f"blocks.{i}.moe._qb_bias"] = np.zeros(E, np.float32)
        for e in range(E):
            w[f"blocks.{i}.moe.experts.{e}.w1.weight"] = (rng.standard_normal((FF, D)) * g).astype(np.float32)
            w[f"blocks.{i}.moe.experts.{e}.w2.weight"] = (rng.standard_normal((FF, D)) * g).astype(np.float32)
            w[f"blocks.{i}.moe.experts.{e}.w3.weight"] = (rng.standard_normal((D, FF)) / np.sqrt(FF)).astype(np.float32)
    w["final_norm.weight"] = np.ones(D, np.float32)
    return w


def ref_logits(w, ids):
    x = w["token_embedding.weight"][ids].astype(np.float64)
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
        gu = np.maximum(h2 @ w[f"blocks.{i}.moe.experts.0.w1.weight"].T, 0) * 0  # 占位
        a = h2 @ w[f"blocks.{i}.moe.experts.0.w1.weight"].T
        u = h2 @ w[f"blocks.{i}.moe.experts.0.w2.weight"].T
        silu = a / (1 + np.exp(-a))
        x = x + (silu * u) @ w[f"blocks.{i}.moe.experts.0.w3.weight"].T
    hn = rmsnorm(x, w["final_norm.weight"])
    return hn @ w["token_embedding.weight"].T


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "cpu_verify.tg")
    w = build()
    tokens = [chr(ord("a") + i) for i in range(V - 3)] + ["<pad>", "<bos>", "<eos>"]
    wr = TGWriter(n_layers=L, d_model=D, n_heads=NH, d_ff=FF, vocab=V, max_seq=MAXSEQ,
                  n_experts=E, n_activated=TOPK, rope_theta=int(THETA),
                  pad=V - 3, bos=V - 2, eos=V - 1, unk=0, weight_dtype=0)
    wr.add_weights(w)
    wr.set_tokenizer(tokens)
    wr.set_chat_meta(META)
    info = wr.write(path)
    print("[1] 打包: %s  %d B  (含 chat meta)" % (info["path"], info["size"]))

    # ---- chat meta 解析（仓库自带 python 解析器）----
    ok_meta = False
    try:
        try:
            from tgai_go._meta import parse_tg_chat_meta
        except Exception:
            import _meta_ref as _mref          # 远端没有仓库包时的备用
            parse_tg_chat_meta = _mref.parse_tg_chat_meta
        m = parse_tg_chat_meta(path)
        present = m.get("present", False)
        print("[2] chat meta present =", present, " keys =", [k for k in m if k != "present"])
        gd = m.get("gen_defaults")
        sp = m.get("special_tokens")
        st = m.get("stop_strings")
        print("    gen_defaults =", repr(gd)[:80])
        print("    special_tokens =", repr(sp)[:80])
        print("    stop_strings =", repr(st)[:80])
        ok_meta = present
    except Exception as ex:
        print("[2] 解析失败:", type(ex).__name__, str(ex)[:120])

    # ---- CPU 引擎前向 ----
    ok_fwd = False
    if not os.path.exists(CPU_DLL):
        print("[3] 找不到 CPU 引擎:", CPU_DLL)
    else:
        lib = ctypes.CDLL(CPU_DLL)
        lib.tg_load.restype = ctypes.c_void_p
        lib.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
        lib.tg_config.restype = ctypes.POINTER(TGHeader)
        lib.tg_config.argtypes = [ctypes.c_void_p]
        lib.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
        lib.tg_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                   ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        lib.tg_free.argtypes = [ctypes.c_void_p]
        m2 = lib.tg_load(path.encode(), 0, None)
        if not m2:
            print("[3] CPU 引擎加载失败")
        else:
            h = lib.tg_config(m2).contents
            print("[3] CPU 引擎读到: L=%d d=%d nh=%d ff=%d V=%d seq=%d E=%d topk=%d"
                  % (h.n_layers, h.d_model, h.n_heads, h.d_ff, h.vocab_size,
                     h.max_seq_len, h.n_experts, h.n_activated))
            ids = np.array([1, 2, 3, 4], np.int32)
            ptr = lib.tg_forward(m2, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                 len(ids), None, 0)
            if not ptr:
                print("[3] CPU 前向返回 NULL")
            else:
                got = np.ctypeslib.as_array(ptr, shape=(V,)).copy()
                ref = ref_logits(w, ids)[-1]
                e = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)
                print("[4] CPU 引擎 logits vs NumPy 参考: 最大相对误差 = %.3e" % e)
                print("    got[:5] = %s" % np.round(got[:5], 5).tolist())
                print("    ref[:5] = %s" % np.round(ref[:5], 5).tolist())
                ok_fwd = e < 1e-4
            lib.tg_free(m2)

    ok = ok_meta and ok_fwd
    print("\nVERDICT ok=%s  chat_meta=%s  cpu_forward=%s" % (ok, ok_meta, ok_fwd))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

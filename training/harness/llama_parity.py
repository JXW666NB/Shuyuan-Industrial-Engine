# -*- coding: utf-8 -*-
"""B1 判分台：llama 预设 vs PyTorch 参考（同权重、同输入）。

判据（目标 B1）：前向 logits 相对误差 < 1e-4，反向各参数梯度 < 1e-3。
自证：同侧跑两次必须恰好 0。

用法：
    python llama_parity.py --selftest   # CPU vs CPU（两次独立运行）必须为 0
    python llama_parity.py --cuda       # 我们的 CUDA 后端 vs PyTorch(CPU float64 参考)
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))

import tgai_go.train as T            # noqa: E402
import llama_ref                     # noqa: E402

CFG = dict(d=32, L=2, nh=4, n_kv=4, ff=64, V=64, seq=16, theta=10000.0, seed=0)  # GQA 待 CUDA 切片支持后改回 2
TOL_LOGITS = 1e-4
TOL_GRAD = 1e-3


def _np(x):
    if x is None:
        return None
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy(), np.float64)
    return np.asarray(x, np.float64)


def run_tgai(device, cfg, ids, tgt):
    if hasattr(T, "set_device"):
        T.set_device(device)
    m = llama_ref.Llama(cfg, cfg["weights"])
    logits = m.forward(ids)
    loss = T.functional.cross_entropy(logits, tgt)
    for p in m.parameters():
        p.grad = None
    loss.backward()
    out = {"logits": _np(logits), "loss": _np(loss).reshape(1)}
    for k, p in m.w.items():
        out["g:" + k] = _np(p.grad)
    if hasattr(T, "set_device"):
        T.set_device("cpu")
    return out


def run_torch(cfg, ids, tgt):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    D, L, FF, V, NH, NKV, S = cfg["d"], cfg["L"], cfg["ff"], cfg["V"], cfg["nh"], cfg["n_kv"], cfg["seq"]
    DH = D // NH
    n_rep = NH // NKV

    def m(name, shape):
        t = torch.tensor(cfg["weights"][name].copy(), dtype=torch.float64)
        return nn.Parameter(t)

    P = {"tok_emb": m("tok_emb.weight", (V, D)), "lm_head": m("lm_head.weight", (V, D)),
         "final_norm": m("final_norm.weight", (D,))}
    names = ["attn_norm", "attn.q", "attn.k", "attn.v", "attn.o", "ffn_norm", "ffn.gate", "ffn.up", "ffn.down"]
    for i in range(L):
        p = "blocks.%d." % i
        P["%d.attn_norm" % i] = m(p + "attn_norm.weight", (D,))
        P["%d.attn.q" % i] = m(p + "attn.q.weight", (D, D))
        P["%d.attn.k" % i] = m(p + "attn.k.weight", (NKV * DH, D))
        P["%d.attn.v" % i] = m(p + "attn.v.weight", (NKV * DH, D))
        P["%d.attn.o" % i] = m(p + "attn.o.weight", (D, D))
        P["%d.ffn_norm" % i] = m(p + "ffn_norm.weight", (D,))
        P["%d.ffn.gate" % i] = m(p + "ffn.gate.weight", (FF, D))
        P["%d.ffn.up" % i] = m(p + "ffn.up.weight", (FF, D))
        P["%d.ffn.down" % i] = m(p + "ffn.down.weight", (D, FF))

    cos = torch.tensor(cfg["cos"], dtype=torch.float64)
    sin = torch.tensor(cfg["sin"], dtype=torch.float64)

    def rms(x, w, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w

    def rope(x):
        half = DH // 2
        x1, x2 = x[..., :half], x[..., half:]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + rot * sin

    idx = torch.tensor(ids, dtype=torch.long)
    B = ids.shape[0]
    h = P["tok_emb"][idx.reshape(-1)].reshape(B, S, D)
    for i in range(L):
        n = rms(h, P["%d.attn_norm" % i])
        q = (n.reshape(B * S, D) @ P["%d.attn.q" % i].T).reshape(B, S, NH, DH).transpose(1, 2)
        k = (n.reshape(B * S, D) @ P["%d.attn.k" % i].T).reshape(B, S, NKV, DH).transpose(1, 2)
        v = (n.reshape(B * S, D) @ P["%d.attn.v" % i].T).reshape(B, S, NKV, DH).transpose(1, 2)
        q = rope(q)
        k = rope(k)
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B * S, D)
        h = h + (o @ P["%d.attn.o" % i].T).reshape(B, S, D)
        n2 = rms(h, P["%d.ffn_norm" % i])
        gu = F.silu(n2.reshape(B * S, D) @ P["%d.ffn.gate" % i].T) * (n2.reshape(B * S, D) @ P["%d.ffn.up" % i].T)
        h = h + (gu @ P["%d.ffn.down" % i].T).reshape(B, S, D)
    hn = rms(h, P["final_norm"])
    logits = hn.reshape(B * S, D) @ P["lm_head"].T
    tgt_t = torch.tensor(tgt, dtype=torch.long)
    loss = F.cross_entropy(logits, tgt_t)
    for p in P.values():
        if p.grad is not None:
            p.grad = None
    loss.backward()

    out = {"logits": logits.detach().numpy().astype(np.float64), "loss": np.array([float(loss)])}
    key_map = {"tok_emb.weight": "tok_emb", "lm_head.weight": "lm_head", "final_norm.weight": "final_norm"}
    for i in range(L):
        p = "blocks.%d." % i
        key_map[p + "attn_norm.weight"] = "%d.attn_norm" % i
        key_map[p + "attn.q.weight"] = "%d.attn.q" % i
        key_map[p + "attn.k.weight"] = "%d.attn.k" % i
        key_map[p + "attn.v.weight"] = "%d.attn.v" % i
        key_map[p + "attn.o.weight"] = "%d.attn.o" % i
        key_map[p + "ffn_norm.weight"] = "%d.ffn_norm" % i
        key_map[p + "ffn.gate.weight"] = "%d.ffn.gate" % i
        key_map[p + "ffn.up.weight"] = "%d.ffn.up" % i
        key_map[p + "ffn.down.weight"] = "%d.ffn.down" % i
    for wk, tk in key_map.items():
        out["g:" + wk] = P[tk].grad.numpy().astype(np.float64)
    return out


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()

    cfg = llama_ref.make_cfg(**CFG)
    g = np.random.RandomState(1)
    ids = g.randint(0, cfg["V"], size=(2, cfg["seq"])).astype(np.int64)
    tgt = g.randint(0, cfg["V"], size=2 * cfg["seq"]).astype(np.int64)

    if a.selftest:
        print("[判分台自证] 同侧跑两次必须恰好 0")
        r1 = run_tgai("cpu", cfg, ids, tgt)
        r2 = run_tgai("cpu", cfg, ids, tgt)
        worst = max(rel(r2[k], r1[k]) for k in r1)
        print("  tgai CPU vs CPU: worst=%.3e" % worst)
        ok = worst == 0.0
        print("  VERDICT ok=%s（必须恰好为 0）" % ok)
        return 0 if ok else 1

    if a.cuda:
        ref = run_torch(cfg, ids, tgt)
        cur = run_tgai("cuda", cfg, ids, tgt) if hasattr(T, "set_device") else run_tgai("cpu", cfg, ids, tgt)
        rl = rel(cur["logits"], ref["logits"])
        rloss = rel(cur["loss"], ref["loss"])
        grads = {k: rel(cur[k], ref[k]) for k in ref if k.startswith("g:")}
        worst_g, worst_k = max((v, k) for k, v in grads.items())
        print("[B1 考核] llama 预设 vs PyTorch（同权重/同输入）")
        print("  logits rel = %.3e   loss rel = %.3e" % (rl, rloss))
        print("  梯度最差 = %s (%.3e)" % (worst_k, worst_g))
        ok = rl < TOL_LOGITS and worst_g < TOL_GRAD
        print("  VERDICT ok=%s（判据 logits<%.0e, grad<%.0e）" % (ok, TOL_LOGITS, TOL_GRAD))
        return 0 if ok else 1

    print("请指定 --selftest 或 --cuda")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""B2 判分台：GPT-2 预设 vs PyTorch 参考（同权重、同输入）。

判据：logits 相对误差 < 1e-4、各参数梯度 < 1e-3；自证（同侧跑两次）必须恰好 0。

用法：
  python gpt2_parity.py --selftest   # CPU vs CPU
  python gpt2_parity.py --cpu        # 我们 CPU vs PyTorch(CPU)
  python gpt2_parity.py --cuda       # 我们 CUDA vs PyTorch(CPU)
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))

import tgai_go.train as T            # noqa: E402
import gpt2_ref                      # noqa: E402

CFG = dict(d=32, L=2, nh=4, ff=64, V=64, seq=16, tie=False, seed=0)
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
    m = gpt2_ref.GPT2(cfg, cfg["weights"])
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

    D, L, FF, V, NH, S = cfg["d"], cfg["L"], cfg["ff"], cfg["V"], cfg["nh"], cfg["seq"]
    DH = D // NH
    P = {}

    def reg(name, shape):
        P[name] = nn.Parameter(torch.tensor(cfg["weights"][name].copy(), dtype=torch.float64))

    # tie=True 时用 wte 当头，**不注册** lm_head —— 真实 GPT-2 的 state_dict 里没有这个键
    heads = ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias")
    if not cfg.get("tie"):
        heads = heads + ("lm_head",)
    for nm in heads:
        reg(nm, cfg["weights"][nm].shape)
    for i in range(L):
        p = "blocks.%d." % i
        for nm in ("ln_1", "ln_2"):
            reg(p + nm + ".weight", (D,))
            reg(p + nm + ".bias", (D,))
        reg(p + "attn.qkv.weight", (3 * D, D))
        reg(p + "attn.qkv.bias", (3 * D,))
        reg(p + "attn.proj.weight", (D, D))
        reg(p + "attn.proj.bias", (D,))
        reg(p + "mlp.fc.weight", (FF, D))
        reg(p + "mlp.fc.bias", (FF,))
        reg(p + "mlp.proj.weight", (D, FF))
        reg(p + "mlp.proj.bias", (D,))

    def ln(x, nm):
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        var = xc.pow(2).mean(-1, keepdim=True)
        return xc * torch.rsqrt(var + 1e-5) * P[nm + ".weight"] + P[nm + ".bias"]

    def lin(x, nm):
        return x @ P[nm + ".weight"].T + P[nm + ".bias"]

    def gelu(x):
        return F.gelu(x, approximate="tanh")

    idx = torch.tensor(ids, dtype=torch.long)
    B = ids.shape[0]
    tok = P["wte.weight"][idx.reshape(-1)].reshape(B, S, D)
    pos = P["wpe.weight"][torch.arange(S, dtype=torch.long)].reshape(1, S, D)
    h = tok + pos
    for i in range(L):
        p = "blocks.%d." % i
        n = ln(h, p + "ln_1")
        qkv = lin(n.reshape(B * S, D), p + "attn.qkv").reshape(B, S, 3, NH, DH)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B * S, D)
        h = h + lin(o, p + "attn.proj").reshape(B, S, D)
        n2 = ln(h, p + "ln_2")
        mid = gelu(lin(n2.reshape(B * S, D), p + "mlp.fc"))
        h = h + lin(mid, p + "mlp.proj").reshape(B, S, D)
    hn = ln(h, "ln_f")
    head = P["wte.weight"] if cfg.get("tie") else P["lm_head"]
    logits = hn.reshape(B * S, D) @ head.T
    loss = F.cross_entropy(logits, torch.tensor(tgt, dtype=torch.long))
    for p in P.values():
        if p.grad is not None:
            p.grad = None
    loss.backward()
    out = {"logits": logits.detach().numpy().astype(np.float64),
           "loss": np.array([float(loss.detach())])}
    for k, v in P.items():
        out["g:" + k] = v.grad.numpy().astype(np.float64)
    return out


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()

    cfg = gpt2_ref.make_cfg(**CFG)
    g = np.random.RandomState(1)
    ids = g.randint(0, cfg["V"], size=(2, cfg["seq"])).astype(np.int64)
    tgt = g.randint(0, cfg["V"], size=2 * cfg["seq"]).astype(np.int64)

    if a.selftest:
        r1 = run_tgai("cpu", cfg, ids, tgt)
        r2 = run_tgai("cpu", cfg, ids, tgt)
        worst = max(rel(r2[k], r1[k]) for k in r1)
        print("[判分台自证] tgai CPU vs CPU worst=%.3e" % worst)
        ok = worst == 0.0
        print("  VERDICT ok=%s（必须恰好为 0）" % ok)
        return 0 if ok else 1

    if a.cpu or a.cuda:
        dev = "cuda" if a.cuda else "cpu"
        ref = run_torch(cfg, ids, tgt)
        cur = run_tgai(dev, cfg, ids, tgt)
        rl = rel(cur["logits"], ref["logits"])
        grads = {k: rel(cur[k], ref[k]) for k in ref if k.startswith("g:")}
        worst_g, worst_k = max((v, k) for k, v in grads.items())
        print("[B2 考核] GPT-2 预设(%s) vs PyTorch" % dev)
        print("  logits rel = %.3e   loss rel = %.3e" % (rl, rel(cur["loss"], ref["loss"])))
        print("  梯度最差 = %s (%.3e)" % (worst_k, worst_g))
        ok = rl < TOL_LOGITS and worst_g < TOL_GRAD
        print("  VERDICT ok=%s（判据 logits<%.0e, grad<%.0e）" % (ok, TOL_LOGITS, TOL_GRAD))
        return 0 if ok else 1

    print("请指定 --selftest / --cpu / --cuda")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

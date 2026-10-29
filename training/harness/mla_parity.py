# -*- coding: utf-8 -*-
"""B5 判分台：MLA（隐空间注意力）vs PyTorch 参考实现。

判据：logits 相对误差 < 1e-4；自证：同侧两次必须恰好 0。

范围（诚实边界）：本步只验 **MLA 的结构**（隐空间压缩、nope/rope 拆分、k_rope 全头共享、
可选 Q 低秩、输出投影）。RoPE 本身与完整 DeepSeek MoE 路由**不在本步**。

用法：python mla_parity.py --selftest | --cpu | --cuda
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, HERE)

import tgai_go.train as T            # noqa: E402
import mla_ref                       # noqa: E402

TOL = 1e-4
CFG = dict(d=64, nh=4, kv_lora=32, qk_nope=16, qk_rope=8, v_head=16,
           V=64, seq=16, q_lora=False, seed=0)


def _np(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float64)


def run_tgai(device, cfg, ids):
    T.set_device(device)
    m = mla_ref.MLA(cfg, cfg["weights"])
    out = _np(m.forward(ids))
    T.set_device("cpu")
    return out


def run_torch(cfg, ids):
    import torch

    D, NH = cfg["d"], cfg["nh"]
    NOPE, ROPE, VH, KL = cfg["qk_nope"], cfg["qk_rope"], cfg["v_head"], cfg["kv_lora"]
    W = cfg["weights"]
    t = {k: torch.tensor(v.copy(), dtype=torch.float64) for k, v in W.items()}
    idx = torch.tensor(ids, dtype=torch.long)
    B, S = ids.shape

    x = t["embed.weight"][idx.reshape(-1)].reshape(B * S, D)
    ckv = x @ t["kv_dkv.weight"].T
    k_nope = (ckv @ t["kv_uk.weight"].T).reshape(B, S, NH, NOPE)
    k_rope = ckv @ t["kv_kr.weight"].T                      # [N, ROPE] 全头共享
    v = (ckv @ t["kv_uv.weight"].T).reshape(B, S, NH, VH)

    if cfg.get("q_lora"):
        qh = (x @ t["q_dq.weight"].T) @ t["q_uq.weight"].T
    else:
        qh = x @ t["q_proj.weight"].T
    q = qh.reshape(B, S, NH, NOPE + ROPE)

    qn = q[..., :NOPE].transpose(1, 2)                      # [B,NH,S,NOPE]
    qr = q[..., NOPE:].transpose(1, 2)
    kn = k_nope.transpose(1, 2)
    vh = v.transpose(1, 2)
    kr = k_rope.reshape(B, S, ROPE).unsqueeze(1).expand(B, NH, S, ROPE)

    qcat = torch.cat([qn, qr], dim=-1)
    kcat = torch.cat([kn, kr], dim=-1)
    o = torch.nn.functional.scaled_dot_product_attention(
        qcat, kcat, vh, is_causal=True, scale=1.0 / (NOPE + ROPE) ** 0.5)
    o = o.transpose(1, 2).reshape(B * S, NH * VH)
    out = (o @ t["o_proj.weight"].T) @ t["lm_head.weight"].T
    return out.reshape(B, S, cfg["V"]).numpy().astype(np.float64)


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

    cfg = mla_ref.make_cfg(**CFG)
    g = np.random.RandomState(1)
    ids = g.randint(0, cfg["V"], size=(2, cfg["seq"])).astype(np.int64)

    if a.selftest:
        r1 = run_tgai("cpu", cfg, ids)
        r2 = run_tgai("cpu", cfg, ids)
        w = rel(r2, r1)
        print("[判分台自证] MLA 同侧两次 worst=%.3e" % w)
        ok = w == 0.0
        print("  VERDICT ok=%s（必须恰好为 0）" % ok)
        return 0 if ok else 1

    if a.cpu or a.cuda:
        dev = "cuda" if a.cuda else "cpu"
        ref = run_torch(cfg, ids)
        got = run_tgai(dev, cfg, ids)
        r = rel(got, ref)
        print("[B5 考核] MLA（隐空间注意力，%s）vs PyTorch" % dev)
        print("  logits rel = %.3e" % r)
        ok = r < TOL
        print("  VERDICT ok=%s（判据 <%.0e）" % (ok, TOL))
        return 0 if ok else 1

    print("请指定 --selftest / --cpu / --cuda")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

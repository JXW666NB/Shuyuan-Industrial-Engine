# -*- coding: utf-8 -*-
"""推理程序 ②：用 PyTorch 跑 `.pt` 模型（对照组）。

与 `infer_tgai.py` 是**同一个模型、同一套超参**，只是用 PyTorch 执行，
用来对照 TGAI GO 引擎的输出是否一致。

用法：
    python infer_torch.py                      # 交互式（考试二：对话）
    python infer_torch.py --task pelican       # 考试一：从空 prompt 续写
    python infer_torch.py --prompt "你是谁"      # 单次提问
"""
import argparse
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BOS, EOS = 0, 2


def pick(*cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def rms(x, w, eps=1e-6):
    m = (x * x).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(m + eps) * w


def silu(x):
    return x * torch.sigmoid(x)


def rope(x, cos, sin):
    dh = x.shape[-1]
    P = torch.zeros(dh, dh)
    for j in range(dh):
        if j % 2 == 0:
            P[j + 1, j] = -1.0
        else:
            P[j - 1, j] = 1.0
    return x * cos + (x @ P) * sin


def forward(w, ids, cfg):
    D, NH, L, S, V = cfg["D"], cfg["NH"], cfg["L"], cfg["S"], cfg["V"]
    B = ids.shape[0]
    DH = D // NH
    oh = torch.nn.functional.one_hot(ids.reshape(-1), V).float()
    x = (oh @ w["token_embedding.weight"]).reshape(B, S, D)
    half = DH // 2
    ang = (torch.arange(S, dtype=torch.float64)[:, None]
           * (10000.0 ** (-2.0 * torch.arange(half, dtype=torch.float64) / DH))[None, :])
    cos = torch.repeat_interleave(torch.cos(ang), 2, dim=1).float()[None].repeat(B * NH, 1, 1)
    sin = torch.repeat_interleave(torch.sin(ang), 2, dim=1).float()[None].repeat(B * NH, 1, 1)
    for i in range(L):
        b = "blocks.%d." % i
        hf = rms(x, w[b + "ln1.weight"]).reshape(B * S, D)
        q = (hf @ w[b + "attn.q_proj.weight"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        kk = (hf @ w[b + "attn.k_proj.weight"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        vv = (hf @ w[b + "attn.v_proj.weight"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        o = torch.nn.functional.scaled_dot_product_attention(
            rope(q, cos, sin), rope(kk, cos, sin), vv, is_causal=True)
        o = o.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
        x = x + (o @ w[b + "attn.out_proj.weight"].T).reshape(B, S, D)
        hf = rms(x, w[b + "ln2.weight"]).reshape(B * S, D)
        y = silu(hf @ w[b + "moe.experts.0.w1.weight"].T) * (hf @ w[b + "moe.experts.0.w2.weight"].T)
        x = x + (y @ w[b + "moe.experts.0.w3.weight"].T).reshape(B, S, D)
    z = rms(x, w["final_norm.weight"]).reshape(B * S, D)
    return (z @ w["token_embedding.weight"].T).reshape(B, S, V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["chat", "pelican"], default="chat")
    ap.add_argument("--pt", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-tokens", type=int, default=48)
    a = ap.parse_args()

    root = os.path.dirname(HERE)
    if a.task == "chat":
        pt = pick(a.pt, os.path.join(root, "V100", "models", "chat10m.pt"),
                  os.path.join(root, "4090", "models", "chat10m.pt"))
    else:
        pt = pick(a.pt, os.path.join(root, "V100", "models", "pelican.pt"),
                  os.path.join(root, "4090", "models", "pelican.pt"))
    if not pt:
        raise SystemExit("找不到 .pt 文件，请用 --pt 指定")

    blob = torch.load(pt, map_location="cpu")
    w = {k: torch.tensor(v, dtype=torch.float32) for k, v in blob["params"].items()}
    toks = blob["vocab"]
    meta = blob["meta"]
    S = int(meta["S"])
    cfg = dict(D=meta["D"], NH=meta["NH"], L=meta["L"], S=S, V=meta["V"])
    stoi = {c: i for i, c in enumerate(toks)}

    def encode(text):
        return [stoi.get(c, stoi["<unk>"]) for c in text]

    def gen(prompt_text, max_tokens):
        out = []
        base = [BOS] + encode(prompt_text)
        for _ in range(max_tokens):
            cur = base + out
            if len(cur) >= S:
                break
            ids = torch.zeros(1, S, dtype=torch.long)
            ids[0, :len(cur)] = torch.tensor(cur)
            with torch.no_grad():
                lg = forward(w, ids, cfg)[0, len(cur) - 1]
            nxt = int(torch.argmax(lg))
            if nxt == EOS:
                break
            out.append(nxt)
        return "".join(toks[i] for i in out)

    print("模型: %s" % pt)
    if a.task == "pelican":
        if a.prompt is not None:
            print(gen(a.prompt, a.max_tokens))
        else:
            for _ in range(3):
                print(" -> %r" % gen("", a.max_tokens))
        return 0

    def ask(q):
        return gen("<用户>\n%s\n<TGAI><回答>\n" % q, a.max_tokens)

    if a.prompt:
        print("Q: %s\nA: %s" % (a.prompt, ask(a.prompt)))
        return 0
    print("输入问题，回车发送（空行退出）")
    while True:
        try:
            q = input("你 > ").strip()
        except EOFError:
            break
        if not q:
            break
        print("小模型 > %s" % ask(q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

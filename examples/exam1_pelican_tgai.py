# -*- coding: utf-8 -*-
"""考试一：只输出「鹈鹕骑自行车」或「自行车骑行鹈鹕」的模型（**非预设架构**）。

本文件与 `exam1_pelican_torch.py` 的区别**只有两处**：
  ① 文件头的导入 / 保存实现（SHIM 段）
  ② 无其它差异 —— 模型定义、前向、训练循环、生成校验逐字节相同。

用 diff 验证：
    diff exam1_pelican_tgai.py exam1_pelican_torch.py
"""
import argparse
import os
import sys

import numpy as np

# ═══════════════ SHIM ①：导入（PyT 版就换成 torch）═══════════════
import tgai_go.train as torch
# ══════════════════════════════════════════════════════════════

T = torch
# 两边都成立：torch 走 nn.functional，TGAI 回落到 functional
F = getattr(getattr(T, "nn", None), "functional", None) or getattr(T, "functional")

# ★ 引擎 add_bos 加的是 id 0（pad），不是我们的 <bos>=1；
#   引擎对空 prompt 给的是 [0, 0]，所以训练必须**从这个上下文出发**。
#
# ★★ 关键：两句话如果共用同一个前缀 [0,0]，位置 1 的"下一词"有歧义
#    （可能是"鹈"也可能是"自"）⇒ 模型只能折中输出 <pad>，永远学不出来。
#    ⇒ 主句（鹈鹕骑自行车）从 [0,0] 出发；第二句用"喂首字"的方式单独成一条。
PREFIX = [0, 0]                                   # 引擎 encode("", add_bos=True) -> [0, 0]
PRIMED = [("", "鹈鹕骑自行车<eos>"),               # 空 prompt ⇒ 这句
          ("鹈鹕骑", "自行车<eos>")]               # 喂 "鹈鹕骑" ⇒ 接着说出"自行车"


# ═══════════════ SHIM ②：保存（.TG 只有 TGAI 能写）═══════════════
def save_artifacts(path, params, vocab, meta):
    import os as _os
    from tgai_go.train.export import save_tg
    # TG_NEXP：把单专家复制成 N 份（用于验证引擎是否支持 n_experts==1）
    ne = int(_os.environ.get("TG_NEXP", "1"))
    if ne > 1:
        for i in range(meta["L"]):
            src = "blocks.%d.moe.experts.0." % i
            for e in range(1, ne):
                for w in ("w1", "w2", "w3"):
                    params["blocks.%d.moe.experts.%d.%s.weight" % (i, e, w)] = params[src + w + ".weight"]
    import os as _os2
    _pv = int(_os2.environ.get("TG_PAD_VOCAB", "0"))
    if _pv and len(vocab) < _pv:
        add = _pv - len(vocab)
        vocab = list(vocab) + ["<x%d>" % i for i in range(add)]
        w = params["token_embedding.weight"]
        params["token_embedding.weight"] = np.vstack(
            [w, np.zeros((add, w.shape[1]), np.float32)])
        meta = dict(meta, V=_pv)
        print("词表补齐: %d -> %d" % (_pv - add, _pv))
    save_tg(path, params, n_layers=meta["L"], d_model=meta["D"], n_heads=meta["NH"],
            d_ff=meta["FF"], vocab_tokens=vocab, merges=[], max_seq=64,   # ★ 不能只写训练长度 9：
            #   引擎按 max_seq 决定 RoPE 表长度，太短会拒绝 generate(max_tokens>max_seq)
            n_experts=ne, n_activated=1, rope_theta=10000.0,
            special_ids=[0, 1, 2, 3], weight_dtype=0)
# ══════════════════════════════════════════════════════════════


def build_vocab():
    chars = sorted(set("".join(pm + tx for pm, tx in PRIMED)
                    .replace("<eos>", "").replace("<bos>", "")))
    toks = ["<pad>", "<bos>", "<eos>", "<unk>"] + chars
    return toks, {c: i for i, c in enumerate(toks)}


def encode(text, stoi):
    out, i = [], 0
    while i < len(text):
        if text.startswith("<bos>", i):
            out.append(stoi["<bos>"]); i += 5
        elif text.startswith("<eos>", i):
            out.append(stoi["<eos>"]); i += 5
        else:
            out.append(stoi.get(text[i], stoi["<unk>"])); i += 1
    return out


def rms(x, w, eps=1e-6):
    m = (x * x).mean(dim=-1, keepdim=True)
    return x * ((m + T.tensor(np.array([eps], np.float32))) ** -0.5) * w


def silu(x):
    return x * ((T.tensor(np.array([1.0], np.float32)) + (-x).exp()) ** -1.0)


def rope(x, cos, sin):
    """**交错式** RoPE：out[2k] = -x[2k+1], out[2k+1] = x[2k]。

    引擎（V3 血统）用交错式；之前用"半切"时引擎能跑但输出错句
    （'自行车骑自行车' 而不是 '鹈鹕骑自行车'）—— 这就是约定不一致。
    用置换矩阵实现：rot = x @ P，两个后端都支持、且可求导。
    """
    dh = int(x.shape[-1])
    P = np.zeros((dh, dh), np.float32)
    for j in range(dh):
        if j % 2 == 0:
            P[j, j + 1] = -1.0
        else:
            P[j, j - 1] = 1.0
    rot = x @ T.tensor(P)
    return x * cos + rot * sin


def make_model(cfg, rng):
    D, NH, FF, V, L = cfg["D"], cfg["NH"], cfg["FF"], cfg["V"], cfg["L"]
    k = 1.0 / np.sqrt(D)
    P = T.nn.Parameter

    def mk(a):
        return P(T.tensor(np.asarray(a, np.float32)))

    p = {"token_embedding.weight": mk(rng.randn(V, D) * 0.02),
         "final_norm.weight": mk(np.ones(D, np.float32))}
    for i in range(L):
        b = "blocks.%d." % i
        p[b + "ln1.weight"] = mk(np.ones(D, np.float32))
        p[b + "ln2.weight"] = mk(np.ones(D, np.float32))
        for nm in ("q_proj", "k_proj", "v_proj", "out_proj"):
            p[b + "attn.%s.weight" % nm] = mk(rng.randn(D, D) * k)
        p[b + "moe.router.weight"] = mk(rng.randn(1, D) * k)
        p[b + "moe.gate.weight"] = mk(np.zeros((1, D), np.float32))
        p[b + "moe._qb_bias"] = mk(np.zeros(1, np.float32))
        p[b + "moe.experts.0.w1.weight"] = mk(rng.randn(FF, D) * k)
        p[b + "moe.experts.0.w2.weight"] = mk(rng.randn(FF, D) * k)
        p[b + "moe.experts.0.w3.weight"] = mk(rng.randn(D, FF) / np.sqrt(FF))
    return p


def forward(p, ids, cfg):
    D, NH, L, S, V = cfg["D"], cfg["NH"], cfg["L"], cfg["S"], cfg["V"]
    B = int(ids.shape[0])
    DH = D // NH
    x = (T.tensor(np.eye(V, dtype=np.float32)[ids.reshape(-1)])
         @ p["token_embedding.weight"]).reshape(B, S, D)

    half = DH // 2
    ang = (np.arange(S, dtype=np.float64)[:, None]
           * (10000.0 ** (-2.0 * np.arange(half, dtype=np.float64) / DH))[None, :])
    cos = T.tensor(np.tile(np.repeat(np.cos(ang), 2, axis=1).astype(np.float32), (B * NH, 1, 1)))
    sin = T.tensor(np.tile(np.repeat(np.sin(ang), 2, axis=1).astype(np.float32), (B * NH, 1, 1)))

    for i in range(L):
        b = "blocks.%d." % i
        hf = rms(x, p[b + "ln1.weight"]).reshape(B * S, D)
        q = (hf @ p[b + "attn.q_proj.weight"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        kk = (hf @ p[b + "attn.k_proj.weight"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        vv = (hf @ p[b + "attn.v_proj.weight"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        o = F.scaled_dot_product_attention(rope(q, cos, sin), rope(kk, cos, sin), vv, is_causal=True)
        o = o.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
        x = x + (o @ p[b + "attn.out_proj.weight"].T).reshape(B, S, D)

        hf = rms(x, p[b + "ln2.weight"]).reshape(B * S, D)
        y = silu(hf @ p[b + "moe.experts.0.w1.weight"].T) * (hf @ p[b + "moe.experts.0.w2.weight"].T)
        x = x + (y @ p[b + "moe.experts.0.w3.weight"].T).reshape(B, S, D)

    z = rms(x, p["final_norm.weight"]).reshape(B * S, D)
    return (z @ p["token_embedding.weight"].T).reshape(B, S, V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", default="pelican")
    a = ap.parse_args()
    T.set_default_device(a.device)

    toks, stoi = build_vocab()
    # 训练序列 = 引擎实际的上下文（[0,0]）+ （可选首字）+ 句子 + <eos>
    seqs = [list(PREFIX) + encode(pm, stoi) + encode(tx, stoi) for pm, tx in PRIMED]
    L = max(len(s) for s in seqs)
    cfg = dict(V=len(toks), D=128, NH=4, L=2, FF=256, S=L)

    ids = np.zeros((len(seqs), L), np.int64)
    tgt = np.zeros((len(seqs), L), np.int64)
    for r, sq in enumerate(seqs):
        n = len(sq)                       # 两句长度不同（8 / 9），按各自长度填，其余留 <pad>
        ids[r, :n] = sq
        tgt[r, :n - 1] = sq[1:]
        tgt[r, n - 1] = stoi["<eos>"]

    p = make_model(cfg, np.random.RandomState(0))
    n_par = sum(int(np.prod(tuple(x.shape))) for x in p.values())
    print("device=%s  vocab=%d  seq=%d  参数量=%.3fM" % (a.device, cfg["V"], L, n_par / 1e6))
    opt = T.optim.AdamW(list(p.values()), lr=3e-3)

    first = None
    for step in range(a.steps):
        logits = forward(p, ids, cfg)
        # 两边通用：目标先转成后端 Tensor（torch 的 CE 只接受 Tensor）
        loss = F.cross_entropy(logits.reshape(ids.shape[0] * cfg["S"], cfg["V"]),
                               T.tensor(tgt.reshape(-1)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        v = float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy")
                             else loss.detach().cpu().numpy()).reshape(-1)[0])
        if first is None:
            first = v
        if step % 150 == 0 or step == a.steps - 1:
            print("  step %3d  loss %.6f" % (step, v))
    print("loss %.6f -> %.6f" % (first, v))

    print("--- 贪心生成 6 次（从引擎同款上下文 %r 起）---" % (PREFIX,))
    good = 0
    for t in range(6):
        out = list(PREFIX)
        while len(out) < cfg["S"]:
            cur = np.zeros((1, cfg["S"]), np.int64)
            cur[0, :len(out)] = out[-cfg["S"]:]
            lg = forward(p, cur, cfg)
            lg = np.asarray(lg.to_numpy() if hasattr(lg, "to_numpy") else lg.detach().cpu().numpy())
            nxt = int(np.argmax(lg[0, len(out) - 1]))
            if nxt == stoi["<eos>"]:
                break
            out.append(nxt)
        text = "".join(toks[i] for i in out[len(PREFIX):])
        ok = text in ("鹈鹕骑自行车", "自行车骑行鹈鹕")
        good += 1 if ok else 0
        print("  %d: %r  %s" % (t + 1, text, "PASS" if ok else "FAIL"))
    print("合规 %d/6" % good)

    raw = {}
    for k, val in p.items():
        raw[k] = np.asarray(val.to_numpy() if hasattr(val, "to_numpy")
                            else val.detach().cpu().numpy(), np.float32)
    out_path = a.out + (".TG" if a.out.endswith("pelican") else "")
    save_artifacts(out_path, raw, toks, cfg)
    print("已保存:", out_path, os.path.getsize(out_path), "字节")
    return 0 if good >= 6 else 1


if __name__ == "__main__":
    sys.exit(main())

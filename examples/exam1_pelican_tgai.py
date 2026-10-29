# -*- coding: utf-8 -*-
"""考试一（重做版）：只输出「鹈鹕骑自行车」或「自行车骑行鹈鹕」的非预设架构模型。

TGAI 版与 PyTorch 版**只有 SHIM 段不同**（导入 + 保存格式），其余逐字节相同。

产物：
  pelican.TG           模型（TGAI 格式）
  pelican.tokens.txt   TGAI GO 训出来的分词表（`id<TAB>token`）
  pelican.TG.meta.json 训练元信息
  log_e1_*.csv         逐步 loss / ppl / 耗时
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# === SHIM-IMPORT-BEGIN ===
import tgai_go.train as torch
# === SHIM-END ===

T = torch
F = getattr(getattr(T, "nn", None), "functional", None) or getattr(T, "functional")

# 引擎 encode("", add_bos=True) == [0, 0]；两句话共用同一前缀会让位置 1 有歧义，
# 所以主句独占 [0,0]，第二句用"喂首字"的方式单独成一条。
PREFIX = [0, 0]
PRIMED = [("", "鹈鹕骑自行车<eos>"),
          ("鹈鹕骑", "自行车<eos>")]

CFG = dict(D=128, NH=4, L=2, FF=256)


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


_ROPE_CACHE = {}


def rope(x, cos, sin, dh):
    """交错式 RoPE，用置换矩阵实现（两个后端都支持、可求导）。

    ★ 必须是 x @ P.T —— 用 x @ P 会得到半切配对，和引擎对不上（上一轮定位到的真 BUG）。
    """
    if dh not in _ROPE_CACHE:
        P = np.zeros((dh, dh), np.float32)
        for j in range(dh):
            if j % 2 == 0:
                P[j + 1, j] = -1.0
            else:
                P[j - 1, j] = 1.0
        _ROPE_CACHE[dh] = T.tensor(P)
    return x * cos + (x @ _ROPE_CACHE[dh]) * sin


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
    oh = T.tensor(np.eye(V, dtype=np.float32)[np.asarray(ids).reshape(-1)])
    x = (oh @ p["token_embedding.weight"]).reshape(B, S, D)

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
        o = F.scaled_dot_product_attention(rope(q, cos, sin, DH), rope(kk, cos, sin, DH), vv, is_causal=True)
        o = o.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
        x = x + (o @ p[b + "attn.out_proj.weight"].T).reshape(B, S, D)

        hf = rms(x, p[b + "ln2.weight"]).reshape(B * S, D)
        y = silu(hf @ p[b + "moe.experts.0.w1.weight"].T) * (hf @ p[b + "moe.experts.0.w2.weight"].T)
        x = x + (y @ p[b + "moe.experts.0.w3.weight"].T).reshape(B, S, D)

    z = rms(x, p["final_norm.weight"]).reshape(B * S, D)
    return (z @ p["token_embedding.weight"].T).reshape(B, S, V)


# === SHIM-SAVE-BEGIN ===
def save_artifacts(path, params, vocab, meta):
    from tgai_go.train.export import save_tg
    save_tg(path, params, n_layers=meta["L"], d_model=meta["D"], n_heads=meta["NH"],
            d_ff=meta["FF"], vocab_tokens=vocab, merges=[], max_seq=64,
            n_experts=1, n_activated=1, rope_theta=10000.0,
            special_ids=[0, 1, 2, 3], weight_dtype=0)
    with open(os.path.splitext(path)[0] + ".tokens.txt", "w", encoding="utf-8") as f:
        for i, t in enumerate(vocab):
            f.write("%d\t%s\n" % (i, t))
# === SHIM-END ===


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="pelican")
    ap.add_argument("--log-csv", default="")
    a = ap.parse_args()
    T.set_default_device(a.device)

    toks, stoi = build_vocab()
    seqs = [list(PREFIX) + encode(pm, stoi) + encode(tx, stoi) for pm, tx in PRIMED]
    S = max(len(s) for s in seqs)
    cfg = dict(CFG, V=len(toks), S=S)

    ids = np.zeros((len(seqs), S), np.int64)
    # ★ 填充位的 target 设为 -100（不计损失）：否则模型会把大量 pad 当成要学的目标，
    #   推理时偏向输出 <pad>（这是实测踩到的坑）。
    tgt = np.full((len(seqs), S), -100, np.int64)
    for r, sq in enumerate(seqs):
        n = len(sq)
        ids[r, :n] = sq
        tgt[r, :n - 1] = sq[1:]
        tgt[r, n - 1] = stoi["<eos>"]
        # ★ 前缀内部位不计损失：两个 pad 输入完全相同，若强迫 pos0→pad、pos1→鹈，
        #   模型只能给出打平的输出（实测 <pad> 与 鹈 都是 6.24），贪心必选 pad。
        tgt[r, :len(PREFIX) - 1] = -100

    p = make_model(cfg, np.random.RandomState(0))
    n_par = sum(int(np.prod(tuple(x.shape))) for x in p.values())
    print("device=%s  词表=%d  序列=%d  参数量=%.3fM" % (a.device, cfg["V"], S, n_par / 1e6))
    opt = T.optim.AdamW(list(p.values()), lr=a.lr)

    log = open(a.log_csv, "w", encoding="utf-8") if a.log_csv else None
    if log:
        log.write("step,loss,ppl,elapsed_s\n")
    t0 = time.time()
    first = None
    v = float("nan")
    for step in range(a.steps):
        logits = forward(p, ids, cfg)
        loss = F.cross_entropy(logits.reshape(len(seqs) * S, cfg["V"]), T.tensor(tgt.reshape(-1)),
                               ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        v = float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy")
                             else loss.detach().cpu().numpy()).reshape(-1)[0])
        if first is None:
            first = v
        if log:
            log.write("%d,%.6f,%.4f,%.3f\n" % (step, v, float(np.exp(min(v, 50))), time.time() - t0))
        if step % 100 == 0 or step == a.steps - 1:
            print("  step %4d  loss %.6f  ppl %.3f" % (step, v, float(np.exp(min(v, 50)))))
    if log:
        log.close()
    print("loss %.6f -> %.6f" % (first, v))

    print("--- 贪心生成 6 次（从 %r 起）---" % (PREFIX,))
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

    raw = {k: np.asarray(val.to_numpy() if hasattr(val, "to_numpy")
                         else val.detach().cpu().numpy(), np.float32) for k, val in p.items()}
    out_path = a.out + (".TG" if a.out.endswith("pelican") else "")
    save_artifacts(out_path, raw, toks, cfg)
    print("已保存:", out_path, os.path.getsize(out_path), "字节")
    with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"vocab": len(toks), "seq": S, "params_M": n_par / 1e6,
                   "steps": a.steps, "lr": a.lr, "device": a.device,
                   "first_loss": first, "final_loss": v, "compliant": good}, f,
                  ensure_ascii=False, indent=1)
    return 0 if good == 6 else 1


if __name__ == "__main__":
    sys.exit(main())

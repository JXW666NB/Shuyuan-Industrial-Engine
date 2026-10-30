# -*- coding: utf-8 -*-
"""考试二（重做版）：一个 10M 参数的**对话**小模型（非预设架构，手写）。

TGAI 版与 PyTorch 版**只有 SHIM 段不同**（导入 + 保存格式），其余逐字节相同。

产物：
  chat10m.TG           模型（TGAI 格式）
  chat10m.tokens.txt   TGAI GO 训出来的分词表（`id<TAB>token`）
  chat10m.TG.meta.json 训练元信息
  log_e2_*.csv         逐步 loss / ppl / 耗时

语料：V100 的 `D:\SFT语料库\40_角色扮演\TGAI自我认知语料_clean.jsonl`（九标记格式）。
训练是 SFT：只对**回答**部分计损失（prompt 位置 target = -100）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# === SHIM-IMPORT-BEGIN ===
import torch
# === SHIM-END ===

T = torch
F = getattr(getattr(T, "nn", None), "functional", None) or getattr(T, "functional")

# 引擎 encode("", add_bos=True) == [0] + 内容；训练必须与之一致 ⇒ 每条样本前缀一个 0。
BOS = 0

_ENV_CORPUS = os.environ.get("TG_CORPUS", "")
CORPUS_FILES = [
    _ENV_CORPUS,
    r"D:\SFT语料库\40_角色扮演\TGAI自我认知语料_clean.jsonl",
]
FALLBACK = [
    ("你是谁", "我是 JXW 训练的 AI 助手。"),
    ("你的父亲是谁", "我是由 JXW 开发的语言模型，没有具体的父亲。"),
    ("创造你的人是谁", "我是由 JXW 创造的。"),
    ("推荐一本书", "可以看看平凡的世界。"),
    ("你好", "你好，有什么可以帮你的吗。"),
]

CFG = dict(D=256, NH=8, L=8, FF=1280, MAXLEN=64)


def load_corpus(max_chars=60000):
    """读 SFT 语料（九标记格式），只保留短回答；读不到就用内置样例。"""
    chunks, total = [], 0
    for fp in CORPUS_FILES:
        if not fp or not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line).get("text", "")
                except Exception:
                    continue
                if "<TGAI>" not in t:
                    continue
                ans = t.split("<TGAI>")[1].replace("<回答>", "").strip()
                if not ans or len(ans) > 40:
                    continue
                chunks.append(t + "\n")
                total += len(t) + 1
                if total >= max_chars:
                    break
        if total >= max_chars:
            break
    if chunks:
        return "".join(chunks), "files"
    return "\n".join("<用户>\n%s\n<TGAI><回答>\n%s" % (u, a) for u, a in FALLBACK), "builtin"


def build_vocab(text, limit=4000):
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    chars = [c for c, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:limit - 4]
    toks = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(chars)
    return toks, {c: i for i, c in enumerate(toks)}


def encode(text, stoi):
    return [stoi.get(c, stoi["<unk>"]) for c in text]


def build_examples(text, stoi, seq):
    """构造与引擎 `build_prompt` 逐字符一致的训练序列。

    引擎：`<用户>\n{q}<TGAI><回答>\n`（`<TGAI>` 前**不换行**）。

    ★ 损失用**全位置**（只忽略第 0 位），不是只算回答：
      字符级 10M 小模型如果只对回答计损，学不到上下文，实测会退化成复读
      （"象象象象嗨嗨嗨嗨…"）。全位置损失才对得上之前验证过的结果。
    """
    pairs = []
    for chunk in text.split("<用户>")[1:]:
        body = "<用户>" + chunk
        if "<TGAI>" not in body:
            continue
        qpart, _, apart = body.partition("<TGAI>")
        q = qpart.replace("<用户>", "", 1).strip()
        a = apart.replace("<回答>", "").strip()
        if q and a:
            pairs.append((q, a))
    if not pairs:
        return None, None

    eos = stoi["<eos>"]
    rows = []
    for q, a in pairs:
        sq = [BOS] + encode("<用户>\n%s<TGAI><回答>\n" % q, stoi) + encode(a, stoi) + [eos]
        if len(sq) > seq:
            sq = sq[:seq]
            sq[-1] = eos
        rows.append(sq)
    S = max(len(r) for r in rows)
    X = np.zeros((len(rows), S), np.int64)
    Y = np.full((len(rows), S), -100, np.int64)
    for r, sq in enumerate(rows):
        n = len(sq)
        X[r, :n] = sq
        Y[r, :n - 1] = sq[1:]
        Y[r, n - 1] = eos
        Y[r, 0] = -100          # 首个 BOS 位不计损失
    return X, Y


def rms(x, w, eps=1e-6):
    m = (x * x).mean(dim=-1, keepdim=True)
    return x * ((m + T.tensor(np.array([eps], np.float32))) ** -0.5) * w


_EMB_CACHE = {}
_IS_TORCH = getattr(T, "__name__", "") == "torch"


def emb_lookup(ids, w, V, D):
    """按 id 取词向量：用后端的融合 gather（CUDA 上是 1 个 kernel）。

    之前是 `np.eye(V)[ids] @ W`：每步要造一个 [B*S, V] 的稠密矩阵（10M 模型是 7 MB），
    再做一次稠密 GEMM —— 换成 gather 后这两样都省了。
    """
    m = _EMB_CACHE.get(id(w))
    if m is None:
        m = T.nn.Embedding(V, D)
        m.weight = w
        _EMB_CACHE[id(w)] = m
    idx = T.tensor(ids) if _IS_TORCH else ids
    return m(idx)


_LN_CACHE = {}


def ln(x, w, eps=1e-6):
    """RMSNorm：优先用后端融合内核（CUDA 上 1 个 kernel，组合实现要 6 个）。

    实测（[78,64,256] 前向+反向）：组合 1.063 ms → 融合 0.299 ms（3.6×）。
    权重仍然是我们 dict 里的 Parameter（共享同一个对象），导出不受影响。
    """
    key = id(w)
    m = _LN_CACHE.get(key)
    if m is None:
        # TG_NO_LN_FUSED=1：强制走组合式，用于和没有融合 RMSNorm 的 PyTorch 版本做公平对照
        if not hasattr(T.nn, "RMSNorm") or os.environ.get("TG_NO_LN_FUSED"):
            return _rms_composed(x, w, eps)
        m = T.nn.RMSNorm(int(w.numel()), eps=eps)
        m.weight = w
        _LN_CACHE[key] = m
    return m(x)


def _rms_composed(x, w, eps=1e-6):
    m = (x * x).mean(dim=-1, keepdim=True)
    return x * ((m + T.tensor(np.array([eps], np.float32))) ** -0.5) * w


def silu(x):
    # torch 有融合 silu；TGAI 的 CUDA 后端走下面的组合式（只在没有 glu 时用）
    if hasattr(F, "silu"):
        return F.silu(x)
    return x * ((T.tensor(np.array([1.0], np.float32)) + (-x).exp()) ** -1.0)


def glu(a, b):
    """out = silu(a) * b。TGAI 有融合 kernel（1 个），其它后端退回组合式。"""
    if hasattr(T, "silu_mul"):
        return T.silu_mul(a, b)
    return silu(a) * b


_ROPE_CACHE = {}
_ROPE_TS = {}


def rope(x, cos, sin, dh):
    """交错式 RoPE（置换矩阵）。必须是 x @ P.T，用 x @ P 是半切配对，与引擎不一致。

    后端有融合内核（TGAI CUDA：fwd/bwd 各 1 个 kernel）时走融合；torch 上没有
    `fused_rope` 这个属性，自动落回下面的组合式 —— 双子文件保持只差 SHIM 段。
    """
    if hasattr(T, "fused_rope"):
        try:
            return T.fused_rope(x, cos, sin)
        except Exception:
            pass
    if dh not in _ROPE_CACHE:
        P = np.zeros((dh, dh), np.float32)
        for j in range(dh):
            if j % 2 == 0:
                P[j + 1, j] = -1.0
            else:
                P[j - 1, j] = 1.0
        _ROPE_CACHE[dh] = T.tensor(P)
    return x * cos + (x @ _ROPE_CACHE[dh]) * sin


def resadd(x, y):
    """残差加：后端有原地内核就用（省一次分配），否则 x + y。两者逐位相同。"""
    f = getattr(T, "fused_add_", None)
    if f is not None:
        try:
            return f(x, y)
        except Exception:
            pass
    return x + y


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
    x = emb_lookup(ids, p["token_embedding.weight"], V, D).reshape(B, S, D)

    # ★ cos/sin 每步都一样：缓存起来，别每步重算 + 重新上传（[B*NH,S,DH]×2 ≈ 10 MB/步）
    half = DH // 2
    key = (S, DH, NH, B)
    cs = _ROPE_TS.get(key)
    if cs is None:
        ang = (np.arange(S, dtype=np.float64)[:, None]
               * (10000.0 ** (-2.0 * np.arange(half, dtype=np.float64) / DH))[None, :])
        cos = T.tensor(np.tile(np.repeat(np.cos(ang), 2, axis=1).astype(np.float32), (B * NH, 1, 1)))
        sin = T.tensor(np.tile(np.repeat(np.sin(ang), 2, axis=1).astype(np.float32), (B * NH, 1, 1)))
        cs = (cos, sin)
        _ROPE_TS[key] = cs
    cos, sin = cs

    for i in range(L):
        b = "blocks.%d." % i
        hf = ln(x, p[b + "ln1.weight"]).reshape(B * S, D)
        q = (hf @ p[b + "attn.q_proj.weight"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        kk = (hf @ p[b + "attn.k_proj.weight"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        vv = (hf @ p[b + "attn.v_proj.weight"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        o = F.scaled_dot_product_attention(rope(q, cos, sin, DH), rope(kk, cos, sin, DH), vv, is_causal=True)
        o = o.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
        x = resadd(x, (o @ p[b + "attn.out_proj.weight"].T).reshape(B, S, D))

        hf = ln(x, p[b + "ln2.weight"]).reshape(B * S, D)
        y = glu(hf @ p[b + "moe.experts.0.w1.weight"].T, hf @ p[b + "moe.experts.0.w2.weight"].T)
        x = resadd(x, (y @ p[b + "moe.experts.0.w3.weight"].T).reshape(B, S, D))

    z = ln(x, p["final_norm.weight"]).reshape(B * S, D)
    return (z @ p["token_embedding.weight"].T).reshape(B, S, V)


# === SHIM-SAVE-BEGIN ===
def save_artifacts(path, params, vocab, meta):
    torch.save({"params": {k: torch.tensor(v) for k, v in params.items()},
                "vocab": vocab, "meta": meta}, path)
    with open(os.path.splitext(path)[0] + ".tokens.txt", "w", encoding="utf-8") as f:
        for i, t in enumerate(vocab):
            f.write("%d\t%s\n" % (i, t))
# === SHIM-END ===


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="chat10m.pt")
    ap.add_argument("--log-csv", default="")
    a = ap.parse_args()
    T.set_default_device(a.device)

    text, src = load_corpus()
    toks, stoi = build_vocab(text)
    x, y = build_examples(text, stoi, CFG["MAXLEN"])
    if x is None:
        raise SystemExit("语料里没有可用对话")
    cfg = dict(CFG, V=len(toks), S=int(x.shape[1]))
    print("语料=%s  %d 字 → 词表 %d，对话 %d 条，序列 %d"
          % (src, len(text), cfg["V"], x.shape[0], cfg["S"]))

    p = make_model(cfg, np.random.RandomState(0))
    n_par = sum(int(np.prod(tuple(t.shape))) for t in p.values())
    print("device=%s  参数量=%.3fM（含词表 %.3fM）"
          % (a.device, n_par / 1e6, cfg["V"] * cfg["D"] / 1e6))
    opt = T.optim.AdamW(list(p.values()), lr=a.lr)

    log = open(a.log_csv, "w", encoding="utf-8") if a.log_csv else None
    if log:
        log.write("step,loss,ppl,elapsed_s\n")
    t0 = time.time()
    first = None
    v = float("nan")
    for step in range(a.steps):
        logits = forward(p, x, cfg)
        loss = F.cross_entropy(logits.reshape(cfg["S"] * x.shape[0], cfg["V"]),
                               T.tensor(y.reshape(-1)), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        v = float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy")
                             else loss.detach().cpu().numpy()).reshape(-1)[0])
        if first is None:
            first = v
        if log:
            log.write("%d,%.6f,%.4f,%.3f\n" % (step, v, float(np.exp(min(v, 50))), time.time() - t0))
        if step % 150 == 0 or step == a.steps - 1:
            print("  step %4d  loss %.6f  ppl %.3f" % (step, v, float(np.exp(min(v, 50)))))
    if log:
        log.close()
    print("loss %.6f -> %.6f" % (first, v))

    print("--- 抽样回答（语料内问题，贪心）---")
    probes = [c.split("<TGAI>")[0].strip() for c in text.split("<用户>")[1:4]]
    for u in probes:
        cur = [BOS] + encode("<用户>\n%s\n<TGAI><回答>\n" % u, stoi)
        out = []
        while len(cur) < cfg["S"]:
            pad = np.zeros((1, cfg["S"]), np.int64)
            pad[0, :len(cur)] = cur[-cfg["S"]:]
            lg = forward(p, pad, cfg)
            lg = np.asarray(lg.to_numpy() if hasattr(lg, "to_numpy") else lg.detach().cpu().numpy())
            nxt = int(np.argmax(lg[0, len(cur) - 1]))
            if nxt == stoi["<eos>"]:
                break
            cur.append(nxt)
            out.append(nxt)
        print("  Q: %-12s A: %s" % (u[:12], "".join(toks[i] for i in out)[:80]))

    raw = {k: np.asarray(val.to_numpy() if hasattr(val, "to_numpy")
                         else val.detach().cpu().numpy(), np.float32) for k, val in p.items()}
    out_path = a.out
    save_artifacts(out_path, raw, toks, cfg)
    print("已保存:", out_path, os.path.getsize(out_path), "字节")
    with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"vocab": cfg["V"], "seq": cfg["S"], "params_M": n_par / 1e6,
                   "dialogues": int(x.shape[0]), "corpus": src, "steps": a.steps,
                   "lr": a.lr, "device": a.device, "first_loss": first, "final_loss": v},
                  f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

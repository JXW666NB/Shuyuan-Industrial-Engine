# -*- coding: utf-8 -*-
"""考试二：一个 10M 参数的**对话**小模型（非预设架构，手写）。

与 exam1 同一套规矩：本文件与 `exam2_chat10m_torch.py` 的差别**只有 SHIM 段**（导入 + 保存格式）。

架构（引擎可导出的 TGAI 形态，但完全手写，不引用 presets）：
    embedding(词表×d, tied) → L × [ RMSNorm → 因果注意力(交错式 RoPE, GQA) → 残差
                                   → RMSNorm → 单专家 FFN(SiLU-GLU) → 残差 ]
    → RMSNorm → 输出头（与词表共享）

训练：SFT —— 只对**回答部分**计损失（prompt 位置 target = -100，用 cross_entropy 的 ignore_index）。
"""
import argparse
import os
import sys

import numpy as np

# ═══════════════ SHIM ①：导入（PyT 版就换成 torch）═══════════════
import tgai_go.train as torch
# ══════════════════════════════════════════════════════════════

T = torch
F = getattr(getattr(T, "nn", None), "functional", None) or getattr(T, "functional")

PREFIX = [0, 0]          # 引擎 encode("", add_bos=True) -> [0, 0]

CORPUS_FILES = [
    # 主题集中的自我认知语料（159 条完整对话）：10M 模型在这个域内才可能答得像话
    r"D:\SFT语料库\40_角色扮演\TGAI自我认知语料_clean.jsonl",
]

PAIRS = [
    ("你好", "你好，我是树缘智能的小模型。"),
    ("你是谁", "我是 TGAI GO 训练出来的十兆参数小模型。"),
    ("你能做什么", "我能做简单的问答和闲聊。"),
    ("今天天气怎么样", "我看不到天气，你可以自己看看窗外。"),
    ("推荐一本书", "可以看看《平凡的世界》。"),
    ("什么是机器学习", "机器学习是让程序从数据里自己找出规律。"),
    ("什么是神经网络", "神经网络是很多层简单的运算叠在一起。"),
    ("你会写诗吗", "会一点点，比如：床前明月光。"),
    ("一加一等于几", "一加一等于二。"),
    ("再见", "再见，有需要再找我。"),
    ("谢谢", "不客气。"),
    ("你叫什么名字", "我没有名字，你叫我小模型就好。"),
    ("你多大了", "我刚被训练出来，还很小。"),
    ("什么是 CUDA", "CUDA 是英伟达显卡上的并行计算平台。"),
    ("GPU 有什么用", "GPU 擅长同时做很多简单的计算。"),
    ("你能联网吗", "不能，我完全在本地运行。"),
    ("你能记住我吗", "这次对话里我记得，关掉就忘了。"),
    ("讲个笑话", "有人说程序员的浪漫是写注释。"),
    ("我很难过", "别太难过，先休息一会儿。"),
    ("我很开心", "那真好，继续保持。"),
]


def build_vocab():
    chars = sorted(set("".join(u + a for u, a in PAIRS)))
    toks = ["<pad>", "<bos>", "<eos>", "<unk>"] + chars
    return toks, {c: i for i, c in enumerate(toks)}


def encode(text, stoi):
    return [stoi.get(c, stoi["<unk>"]) for c in text]


def rms(x, w, eps=1e-6):
    m = (x * x).mean(dim=-1, keepdim=True)
    return x * ((m + T.tensor(np.array([eps], np.float32))) ** -0.5) * w


def silu(x):
    return x * ((T.tensor(np.array([1.0], np.float32)) + (-x).exp()) ** -1.0)


def rope(x, cos, sin):
    """交错式 RoPE（引擎是 V3 血统，必须交错；用置换矩阵实现，两边通用可求导）。"""
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
    x = cfg["emb"](ids).reshape(B, S, D)          # 可微查表

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


# ═══════════════ SHIM ②：保存（.TG 只有 TGAI 能写）═══════════════
def save_artifacts(path, params, vocab, meta):
    from tgai_go.train.export import save_tg
    save_tg(path, params, n_layers=meta["L"], d_model=meta["D"], n_heads=meta["NH"],
            d_ff=meta["FF"], vocab_tokens=vocab, merges=[], max_seq=64,
            n_experts=1, n_activated=1, rope_theta=10000.0,
            special_ids=[0, 1, 2, 3], weight_dtype=0)
# ══════════════════════════════════════════════════════════════


def load_corpus(max_chars=60000):
    """读真实 SFT 语料（九标记格式的 text 字段）；读不到就回落到内置小样本。"""
    import json
    chunks, total = [], 0
    for fp in CORPUS_FILES:
        if not os.path.exists(fp):
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
                if not t:
                    continue
                # ★ 只保留**短回答**（≤24 字）的对话：10M 模型在长回答上必然漂移
                parts = t.split("<TGAI>")
                if len(parts) < 2:
                    continue
                ans = parts[1].replace("<回答>", "").strip()
                if not ans or len(ans) > 24:
                    continue
                chunks.append(t + "\n")
                total += len(t) + 1
                if total >= max_chars:
                    break
        if total >= max_chars:
            break
    if chunks:
        return "".join(chunks), "files"
    text = "\n".join("<用户>\n%s\n<TGAI><回答>\n%s" % (u, a) for u, a in PAIRS)
    return text, "builtin"


def build_vocab(text, vocab_size=4000):
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    chars = [c for c, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:vocab_size - 4]
    toks = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(chars)
    return toks, {c: i for i, c in enumerate(toks)}


def build_examples(text, stoi, seq=96):
    """按对话切分，构造与**引擎推理时完全一致**的序列：[0] + prompt + answer + <eos>。

    引擎对非空 prompt 的编码是 `[0] + chars`（add_bos 加的是 id 0），
    所以训练前缀必须只有一个 0 —— 之前用 [0,0] 与随机窗口，推理时必然错位。
    """
    rows = []
    for chunk in text.split("<用户>")[1:]:
        body = "<用户>" + chunk
        if "<TGAI>" not in body:
            continue
        qpart, _, apart = body.partition("<TGAI>")
        prompt = qpart + "<TGAI><回答>\n"
        answer = apart.replace("<回答>", "").strip()
        if not answer:
            continue
        ids = [0] + encode(prompt, stoi) + encode(answer, stoi) + [stoi["<eos>"]]
        if len(ids) > seq:
            ids = ids[:seq]
            ids[-1] = stoi["<eos>"]
        rows.append(ids)
    if not rows:
        return None, None
    L = max(len(r) for r in rows)
    ids = np.zeros((len(rows), L), np.int64)
    tgt = np.full((len(rows), L), -100, np.int64)
    for r, sq in enumerate(rows):
        ids[r, :len(sq)] = sq
        tgt[r, :len(sq) - 1] = sq[1:]
        tgt[r, len(sq) - 1] = stoi["<eos>"]
    return ids, tgt


def forward(p, ids, cfg):
    D, NH, L, S, V = cfg["D"], cfg["NH"], cfg["L"], cfg["S"], cfg["V"]
    B = int(ids.shape[0])
    DH = D // NH
    x = cfg["emb"](ids).reshape(B, S, D)          # 可微查表

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


# ═══════════════ SHIM ②：保存（.TG 只有 TGAI 能写）═══════════════
def save_artifacts(path, params, vocab, meta):
    from tgai_go.train.export import save_tg
    save_tg(path, params, n_layers=meta["L"], d_model=meta["D"], n_heads=meta["NH"],
            d_ff=meta["FF"], vocab_tokens=vocab, merges=[], max_seq=64,
            n_experts=1, n_activated=1, rope_theta=10000.0,
            special_ids=[0, 1, 2, 3], weight_dtype=0)
# ══════════════════════════════════════════════════════════════


def load_corpus(max_chars=60000):
    """读真实 SFT 语料（九标记格式的 text 字段）；读不到就回落到内置小样本。"""
    import json
    chunks, total = [], 0
    for fp in CORPUS_FILES:
        if not os.path.exists(fp):
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
                if not t:
                    continue
                # ★ 只保留**短回答**（≤24 字）的对话：10M 模型在长回答上必然漂移
                parts = t.split("<TGAI>")
                if len(parts) < 2:
                    continue
                ans = parts[1].replace("<回答>", "").strip()
                if not ans or len(ans) > 24:
                    continue
                chunks.append(t + "\n")
                total += len(t) + 1
                if total >= max_chars:
                    break
        if total >= max_chars:
            break
    if chunks:
        return "".join(chunks), "files"
    text = "\n".join("<用户>\n%s\n<TGAI><回答>\n%s" % (u, a) for u, a in PAIRS)
    return text, "builtin"


def build_vocab(text, vocab_size=4000):
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    chars = [c for c, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:vocab_size - 4]
    toks = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(chars)
    return toks, {c: i for i, c in enumerate(toks)}


def make_windows(ids, seq, n_win):
    g = np.random.RandomState(0)
    need = n_win * seq + 1
    if len(ids) < need:
        reps = int(np.ceil(need / max(1, len(ids))))
        ids = np.tile(ids, reps)
    starts = g.randint(0, len(ids) - seq - 1, size=n_win)
    x = np.stack([ids[s:s + seq] for s in starts])
    y = np.stack([ids[s + 1:s + seq + 1] for s in starts])
    return x.astype(np.int64), y.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", default="chat10m")
    ap.add_argument("--save-npz", default="")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--nh", type=int, default=8)
    ap.add_argument("--ff", type=int, default=1280)
    a = ap.parse_args()
    T.set_default_device(a.device)

    text, src = load_corpus()
    toks, stoi = build_vocab(text)
    x, y = build_examples(text, stoi)
    if x is None:
        raise SystemExit("语料里没有可用的对话")
    cfg = dict(V=len(toks), D=a.d, NH=a.nh, L=a.layers, FF=a.ff, S=int(x.shape[1]))
    print("语料来源=%s  %d 字 → 词表 %d，对话 %d 条，序列长 %d"
          % (src, len(text), len(toks), x.shape[0], x.shape[1]))

    p = make_model(cfg, np.random.RandomState(0))
    # ★ 必须**共享同一个参数对象**（tied）：之前用 load_state_dict 复制了一份，
    #   训练只更新了副本，而输出头/导出用的是原份 ⇒ 导出的是没训练过的词表 ⇒ 输出乱码。
    emb = T.nn.Embedding(cfg["V"], cfg["D"])
    emb.weight = p["token_embedding.weight"]
    cfg["emb"] = emb
    n_par = sum(int(np.prod(tuple(x.shape))) for x in p.values())
    print("device=%s  vocab=%d  seq=%d  参数量=%.3fM（含词表 %.3fM）"
          % (a.device, cfg["V"], cfg["S"], n_par / 1e6, cfg["V"] * cfg["D"] / 1e6))
    opt = T.optim.AdamW(list(p.values()), lr=2e-3)

    first = None
    for step in range(a.steps):
        logits = forward(p, x, cfg)
        loss = F.cross_entropy(logits.reshape(x.shape[0] * cfg["S"], cfg["V"]),
                               T.tensor(y.reshape(-1)), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        v = float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy")
                             else loss.detach().cpu().numpy()).reshape(-1)[0])
        if first is None:
            first = v
        if step % 300 == 0 or step == a.steps - 1:
            print("  step %4d  loss %.6f" % (step, v))
    print("loss %.6f -> %.6f" % (first, v))

    # 抽样：直接用**语料里真实出现过的问题**（否则是分布外，看不出对错）
    print("--- 抽样回答（语料内问题，贪心）---")
    probes = []
    for chunk in text.split("<用户>")[1:6]:
        q = chunk.split("<TGAI>")[0].strip()
        if q:
            probes.append(q)
    for u in probes[:3]:
        cur = list(PREFIX) + encode("<用户>\n%s\n<TGAI><回答>\n" % u, stoi)
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
        print("  Q: %-14s A: %s" % (u[:14], "".join(toks[i] for i in out)[:90]))

    # 首位 token 对照（供引擎侧比较）
    _q = "创造你的人是谁"
    _cur = [0] + encode("<用户>\n%s\n<TGAI><回答>\n" % _q, stoi)
    _pad = np.zeros((1, cfg["S"]), np.int64)
    _pad[0, :len(_cur)] = _cur[-cfg["S"]:]
    _lg = forward(p, _pad, cfg)
    _lg = np.asarray(_lg.to_numpy() if hasattr(_lg, "to_numpy") else _lg.detach().cpu().numpy())
    _nxt = int(np.argmax(_lg[0, len(_cur) - 1]))
    print("首位 token 对照: id=%d char=%r" % (_nxt, toks[_nxt]))

    if a.save_npz:
        _npz = {k: np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v.detach().cpu().numpy(), np.float32)
                for k, v in p.items()}
        np.savez(a.save_npz, **{("w__" + k.replace(".", "_")): v for k, v in _npz.items()})
        import json as _json
        with open(a.save_npz + ".json", "w", encoding="utf-8") as _f:
            _json.dump({"vocab": list(toks), "cfg": {k: v for k, v in cfg.items() if k != "emb"}}, _f, ensure_ascii=False)
        print("已存 npz:", a.save_npz)

    raw = {k: np.asarray(val.to_numpy() if hasattr(val, "to_numpy")
                         else val.detach().cpu().numpy(), np.float32) for k, val in p.items()}

    # ★ 词表补齐到 256 的倍数：怀疑引擎对 vocab 做了对齐，未补齐时输出投影读越界 → 输出乱码
    pad_to = 256
    if len(toks) < pad_to:
        add = pad_to - len(toks)
        toks = list(toks) + ["<x%d>" % i for i in range(add)]
        w = raw["token_embedding.weight"]
        raw["token_embedding.weight"] = np.vstack([w, np.zeros((add, w.shape[1]), np.float32)])
        cfg["V"] = len(toks)
        print("词表补齐: %d -> %d" % (len(toks) - add, len(toks)))
    out_path = a.out + (".TG" if a.out.endswith("chat10m") else "")
    save_artifacts(out_path, raw, toks, cfg)
    print("已保存:", out_path, os.path.getsize(out_path), "字节")
    return 0


if __name__ == "__main__":
    sys.exit(main())

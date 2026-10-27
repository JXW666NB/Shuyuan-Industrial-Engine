# -*- coding: utf-8 -*-
"""鹈鹕过拟合 —— PyTorch 参照实现（先证明任务可学，再做引擎侧）。

语料: _corpus/pelican_on_bike.html + bike_on_pelican.html（约 24.6k 字符，词表 85）
模型: char-level, d=256, nh=8, dh=32, ff=1024, E=1(等价 dense), 1 层, V=85
      参数约 1.09M
目标: 训练到 loss≈0，采样能复现两份文档

用法: python charlm_torch.py [steps]
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = [os.path.join(HERE, "_corpus", "pelican_on_bike.html"),
          os.path.join(HERE, "_corpus", "bike_on_pelican.html")]

_a = sys.argv
STEPS_ARG = int(_a[1]) if len(_a) > 1 else 1500
NL = int(_a[2]) if len(_a) > 2 else 2
D = int(_a[3]) if len(_a) > 3 else 256
NH = int(_a[4]) if len(_a) > 4 else 8
FF = int(_a[5]) if len(_a) > 5 else 512
SEQ = int(_a[6]) if len(_a) > 6 else 256
DH = D // NH
EPS = 1e-5
SCALE = 1.0 / (DH ** 0.5)
LR = 3e-3
STEPS = STEPS_ARG
dev = "cuda"


def load_corpus():
    parts = []
    for p in CORPUS:
        with open(p, encoding="utf-8") as f:
            parts.append(f.read())
    text = parts[0] + parts[1]
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return text, vocab, stoi, ids


def main():
    text, vocab, stoi, ids = load_corpus()
    V = len(vocab)
    print("语料: %d 字符, 词表 %d, 序列长度 %d" % (len(text), V, SEQ))

    torch.manual_seed(0)
    p = {"emb": torch.randn(V, D, device=dev) * 0.02,
         "w3": torch.ones(D, device=dev),
         "head": torch.randn(D, V, device=dev) * (1.0 / D ** 0.5)}
    for L in range(NL):
        p[f"w1_{L}"] = torch.ones(D, device=dev)
        p[f"Wqkv_{L}"] = torch.randn(D, 3 * D, device=dev) * (1.0 / D ** 0.5)
        p[f"Wo_{L}"] = torch.randn(D, D, device=dev) * (1.0 / D ** 0.5)
        p[f"w2_{L}"] = torch.ones(D, device=dev)
        p[f"Wg_{L}"] = torch.randn(D, FF, device=dev) * (1.0 / D ** 0.5)
        p[f"Wu_{L}"] = torch.randn(D, FF, device=dev) * (1.0 / D ** 0.5)
        p[f"Wd_{L}"] = torch.randn(FF, D, device=dev) * (1.0 / FF ** 0.5)
    nparam = sum(t.numel() for t in p.values())
    print("参数量: %.2f M" % (nparam / 1e6))

    for k in p:
        p[k].requires_grad_(True)
    opt = torch.optim.AdamW(list(p.values()), lr=LR, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS + 1,
                                                pct_start=0.05, anneal_strategy="cos",
                                                div_factor=10.0, final_div_factor=10000.0)

    cos = torch.cos(torch.arange(SEQ, device=dev).float().unsqueeze(1) *
                    (10000.0 ** (-torch.arange(0, DH, 2, device=dev).float() / DH))).view(1, SEQ, 1, DH // 2)
    sin = torch.sin(torch.arange(SEQ, device=dev).float().unsqueeze(1) *
                    (10000.0 ** (-torch.arange(0, DH, 2, device=dev).float() / DH))).view(1, SEQ, 1, DH // 2)

    def rot(t):                       # t: [B,seq,nh,dh]
        ev = t[..., 0::2].clone()
        od = t[..., 1::2].clone()
        o = t.clone()
        o[..., 0::2] = ev * cos - od * sin
        o[..., 1::2] = ev * sin + od * cos
        return o

    def forward(idx):                 # idx [B,seq]
        B = idx.shape[0]
        x = p["emb"][idx]                                  # [B,seq,D]
        for L in range(NL):
            h = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * p[f"w1_{L}"]
            qkv = h @ p[f"Wqkv_{L}"]
            q, k, v = qkv[..., :D], qkv[..., D:2 * D], qkv[..., 2 * D:]
            q = rot(q.view(B, SEQ, NH, DH))
            k = rot(k.view(B, SEQ, NH, DH))
            v = v.view(B, SEQ, NH, DH)
            at = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                v.transpose(1, 2), is_causal=True, scale=SCALE)
            at = at.transpose(1, 2).reshape(B, SEQ, D)
            x2 = x + at @ p[f"Wo_{L}"]
            h2 = x2 * torch.rsqrt((x2 * x2).mean(-1, keepdim=True) + EPS) * p[f"w2_{L}"]
            gu = F.silu(h2 @ p[f"Wg_{L}"]) * (h2 @ p[f"Wu_{L}"])
            x = x2 + gu @ p[f"Wd_{L}"]
        hn = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * p["w3"]
        return hn @ p["head"], x

    def sample(prompt, n=200, temp=0.8, greedy=False):
        idx = torch.tensor([[stoi[c] for c in prompt]], device=dev)
        out = list(prompt)
        with torch.no_grad():
            for _ in range(n):
                seq = idx[:, -SEQ:]
                pad = SEQ - seq.shape[1]
                if pad > 0:
                    seq = torch.cat([torch.zeros(1, pad, dtype=torch.long, device=dev), seq], 1)
                logits, _ = forward(seq)
                lg = logits[0, -1] / temp
                if greedy:
                    nxt = int(torch.argmax(lg).item())
                else:
                    probs = torch.softmax(lg, dim=-1)
                    nxt = torch.multinomial(probs, 1).item()
                out.append(vocab[nxt])
                idx = torch.cat([idx, torch.tensor([[nxt]], device=dev)], 1)
        return "".join(out)

    BATCH = int(_a[7]) if len(_a) > 7 else 8
    rng = torch.Generator(device="cpu").manual_seed(7)
    t0 = time.perf_counter()
    hist = []
    for step in range(STEPS + 1):
        starts = torch.randint(0, len(text) - SEQ - 1, (BATCH,), generator=rng).tolist()
        idx = torch.tensor([[stoi[c] for c in text[s:s + SEQ]] for s in starts], device=dev)
        tgt = torch.tensor([[stoi[c] for c in text[s + 1:s + SEQ + 1]] for s in starts], device=dev)
        logits, _ = forward(idx)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(loss.item())
        if step % 100 == 0:
            print("step %-5d loss=%.4f  (%.0fs)" % (step, loss.item(), time.perf_counter() - t0))
        if loss.item() < 1e-5:
            print("step %-5d loss=%.6f  已收敛，提前停止" % (step, loss.item()))
            break

    print("\n最终 loss = %.6f（最近 20 步均值 %.6f）" % (hist[-1], sum(hist[-20:]) / 20))

    # ---- 全语料 teacher-forced 准确率（真正的记忆度量）----
    with torch.no_grad():
        W = 32
        starts_all = list(range(0, len(text) - SEQ - 1, W))
        hit = tot = 0
        for i in range(0, len(starts_all), 16):
            chunk = starts_all[i:i + 16]
            idx = torch.tensor([[stoi[c] for c in text[s0:s0 + SEQ]] for s0 in chunk], device=dev)
            tgt = torch.tensor([[stoi[c] for c in text[s0 + 1:s0 + SEQ + 1]] for s0 in chunk], device=dev)
            lg, _ = forward(idx)
            pred = lg.argmax(-1)
            hit += (pred == tgt).sum().item(); tot += tgt.numel()
        print("全语料 teacher-forced 逐字符准确率 = %.2f%%（%d/%d）" % (hit / tot * 100, hit, tot))

    # ---- 保存模型 ----
    os.makedirs(os.path.join(HERE, "_tgtest"), exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in p.items()},
               os.path.join(HERE, "_tgtest", "pelican_lm.pt"))
    print("模型已存: _tgtest/pelican_lm.pt")

    # ---- 复现检查 ----
    def match_rate(prompt, target_text, n=400):
        s = sample(prompt, n=n, temp=0.1)     # 低温度 = 近似贪心
        hit = 0
        for i, ch in enumerate(target_text[:n]):
            if i + len(prompt) < len(s) and s[i + len(prompt)] == target_text[i]:
                hit += 1
            else:
                break
        return hit / n

    A = open(CORPUS[0], encoding="utf-8").read()
    B = open(CORPUS[1], encoding="utf-8").read()
    div = 0
    while div < min(len(A), len(B)) and A[div] == B[div]:
        div += 1
    print("两份文档在第 %d 个字符处分叉" % div)
    for path in CORPUS:
        txt = open(path, encoding="utf-8").read()
        cut = min(div + 40, len(txt) // 2)
        prompt = txt[:cut]
        s = sample(prompt, n=400, greedy=True)
        same = sum(1 for a, b in zip(s[len(prompt):], txt[len(prompt):]) if a == b)
        rate = same / max(1, len(s) - len(prompt))
        print("\n=== %s ===" % os.path.basename(path))
        print("逐字符匹配率: %.1f%%" % (rate * 100))
        print("生成片段(前 200 字):")
        print(s[:200].replace("\n", "⏎"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

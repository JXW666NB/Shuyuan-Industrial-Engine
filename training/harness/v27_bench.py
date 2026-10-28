# -*- coding: utf-8 -*-
"""V2.7 宽度 + 长上下文的训练对比：tgai_go.train(后端) vs PyTorch。

**注意：这是稠密 FFN 版本，不是 V2.7 的 MoE。** 用于回答
「CUDA 后端在真实宽度、真实长上下文下能不能训，和 PyTorch 差多少」。

配置（用户指定）：d=1024 / nh=16 / ff=4096 / seq=8192
    层数默认 2（长上下文下 Python 编排很慢，先小步验证），可用 --layers 调
    V 按 V2.7 的 32779 pad（embedding 形状与真实模型一致）

输出三项（用户要的判据）：
    loss 曲线 / 训练吞吐 tokens/s / 峰值显存

用法：
    python v27_bench.py --backend tgai  --layers 2 --steps 3
    python v27_bench.py --backend torch --layers 2 --steps 3
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))

CORPUS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "_corpus")

D, NH, FF, V, SEQ = 1024, 16, 4096, 32779, 8192
PEAK = {"vram_gb": None}


def make_batch(seq, vocab_used, seed=0):
    """用 pelican 语料做字节级 id（词表远小于 V，embedding 其余行不使用）。"""
    txt = ""
    for p in ("pelican_on_bike.html", "bike_on_pelican.html"):
        f = os.path.join(CORPUS, p)
        if os.path.exists(f):
            txt += open(f, encoding="utf-8", errors="replace").read()
    if not txt:
        rng = np.random.RandomState(0)
        ids = rng.randint(0, vocab_used, size=seq + 1).astype(np.int64)
    else:
        raw = np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).astype(np.int64) % vocab_used
        if len(raw) < seq + 2:
            raw = np.tile(raw, (seq // max(1, len(raw))) + 2)
        ids = raw[: seq + 1]
    return ids[:-1].reshape(1, seq), ids[1:].reshape(1, seq)


def peak_gb(device="cuda"):
    """峰值显存：PyTorch 用它自己的统计；我们没有统计就用 nvidia-smi 兜底。"""
    try:
        import torch
        return torch.cuda.max_memory_allocated() / 2**30
    except Exception:
        return None


def run_torch(layers, steps, seq):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = "cuda"
    torch.manual_seed(7)

    class RMSNorm(nn.Module):
        """torch 2.2 没有 nn.RMSNorm（2.4 才有），这里给等价实现，eps 与引擎一致 1e-6。"""
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.w = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            v = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(v + self.eps) * self.w

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(D, D, bias=False)
            self.k = nn.Linear(D, D, bias=False)
            self.v = nn.Linear(D, D, bias=False)
            self.o = nn.Linear(D, D, bias=False)

        def forward(self, x):
            B, S, _ = x.shape
            h = NH
            dh = D // h
            q = self.q(x).reshape(B, S, h, dh).transpose(1, 2)
            k = self.k(x).reshape(B, S, h, dh).transpose(1, 2)
            v = self.v(x).reshape(B, S, h, dh).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).reshape(B, S, D)
            return self.o(y)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = RMSNorm(D)
            self.att = Attn()
            self.n2 = RMSNorm(D)
            self.wg = nn.Linear(D, FF, bias=False)
            self.wu = nn.Linear(D, FF, bias=False)
            self.wd = nn.Linear(FF, D, bias=False)

        def forward(self, x):
            x = x + self.att(self.n1(x))
            h = self.n2(x)
            gu = F.silu(self.wg(h)) * self.wu(h)
            return x + self.wd(gu)

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, D)
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.nf = RMSNorm(D)

        def forward(self, ids):
            x = self.emb(ids)
            for b in self.blocks:
                x = b(x)
            return self.nf(x) @ self.emb.weight.T

    m = LM().to(dev)
    nparam = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    x, y = make_batch(seq, 256)
    xt = torch.tensor(x, device=dev)
    yt = torch.tensor(y, device=dev)

    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = None
    for i in range(steps):
        if i == 0:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        logits = m(xt)
        loss = F.cross_entropy(logits.reshape(-1, V), yt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print("参数量: %.1f M" % (nparam / 1e6))
    print("loss:", " -> ".join("%.4f" % v for v in losses))
    print("吞吐: %.1f tokens/s（%.2f s/step）" % (steps * seq / dt, dt / steps))
    print("峰值显存: %.2f GB" % peak_gb())
    return losses


def run_tgai(layers, steps, seq):
    """用 tgai_go.train 跑同样的配置；rmsnorm 走 nn.RMSNorm（已接 GPU kernel）。"""
    import tgai_go.train as T
    F = T.functional

    if hasattr(T, "set_device"):
        T.set_device("cuda")
    T.manual_seed(7)

    emb = T.nn.Embedding(V, D)
    blocks = []
    for _ in range(layers):
        blocks.append(dict(
            n1=T.nn.RMSNorm(D),
            q=T.nn.Linear(D, D), k=T.nn.Linear(D, D), v=T.nn.Linear(D, D), o=T.nn.Linear(D, D),
            n2=T.nn.RMSNorm(D),
            wg=T.nn.Linear(D, FF), wu=T.nn.Linear(D, FF), wd=T.nn.Linear(FF, D),
        ))
    nf = T.nn.RMSNorm(D)

    params = [emb.weight, nf.weight]
    for b in blocks:
        for key in ("n1", "n2"):
            params.append(b[key].weight)
        for key in ("q", "k", "v", "o", "wg", "wu", "wd"):
            params.append(b[key].weight)
    nparam = sum(int(np.prod(p.shape)) for p in params)
    opt = T.optim.AdamW(params, lr=3e-4)

    x, y = make_batch(seq, 256)
    losses = []
    t0 = None
    B, DH = 1, D // NH
    for i in range(steps):
        if i == 0:
            t0 = time.perf_counter()
        h = emb(x)                                     # [B,S,D]
        for b in blocks:
            n = b["n1"](h)
            q = b["q"](n.reshape(B * seq, D))
            k = b["k"](n.reshape(B * seq, D))
            v = b["v"](n.reshape(B * seq, D))
            qq = q.reshape(B, seq, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, seq, DH)
            kk = k.reshape(B, seq, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, seq, DH)
            vv = v.reshape(B, seq, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, seq, DH)
            o = F.scaled_dot_product_attention(qq, kk, vv, causal=True)
            o = o.reshape(B, NH, seq, DH).permute(0, 2, 1, 3).reshape(B * seq, D)
            h = h + b["o"](o).reshape(B, seq, D)
            n2 = b["n2"](h)
            gu = F.silu(b["wg"](n2.reshape(B * seq, D))) * b["wu"](n2.reshape(B * seq, D))
            h = h + b["wd"](gu).reshape(B, seq, D)
        hn = nf(h)
        logits = hn.reshape(B * seq, D) @ emb.weight.T
        loss = F.cross_entropy(logits, y.reshape(-1))
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()
        losses.append(float(np.asarray(loss.to_numpy()).reshape(-1)[0]))
    dt = time.perf_counter() - t0
    print("参数量: %.1f M" % (nparam / 1e6))
    print("loss:", " -> ".join("%.4f" % v for v in losses))
    print("吞吐: %.1f tokens/s（%.2f s/step）" % (steps * seq / dt, dt / steps))
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "tgai"], required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seq", type=int, default=SEQ)
    a = ap.parse_args()
    print("配置: d=%d nh=%d ff=%d V=%d seq=%d layers=%d steps=%d（稠密 FFN，非 MoE）"
          % (D, NH, FF, V, a.seq, a.layers, a.steps))
    if a.backend == "torch":
        run_torch(a.layers, a.steps, a.seq)
    else:
        run_tgai(a.layers, a.steps, a.seq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

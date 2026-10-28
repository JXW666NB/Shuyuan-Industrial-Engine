# -*- coding: utf-8 -*-
"""V2.7 形状的 MoE 训练对比：tgai_go.train vs PyTorch。

配置：d=1024 / nh=16 / ff=4096 / **E=8 top-2** / V=32779 / seq 可调 / 层数可调
两边**同样结构、同样 init 尺度、同样超参**，跑 N 步，比：
    loss 曲线 / 训练吞吐 tokens/s / 峰值显存

用法：
    python v27_moe_bench.py --backend tgai  --layers 1 --steps 3 --seq 2048
    python v27_moe_bench.py --backend torch --layers 1 --steps 3 --seq 2048
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))

D, NH, FF, E, TOPK, V = 1024, 16, 4096, 8, 2, 32779
CORPUS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "_corpus")


def make_batch(seq):
    txt = ""
    for p in ("pelican_on_bike.html", "bike_on_pelican.html"):
        f = os.path.join(CORPUS, p)
        if os.path.exists(f):
            txt += open(f, encoding="utf-8", errors="replace").read()
    if txt:
        raw = np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).astype(np.int64)
    else:
        raw = np.random.RandomState(0).randint(0, 256, size=seq * 4).astype(np.int64)
    raw = raw % 256
    if len(raw) < seq + 2:
        raw = np.tile(raw, (seq // max(1, len(raw))) + 2)
    ids = raw[: seq + 1]
    return ids[:-1].reshape(1, seq), ids[1:].reshape(1, seq)


# ─────────────────────────── PyTorch ───────────────────────────
def run_torch(layers, steps, seq):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = "cuda"
    torch.manual_seed(7)
    INIT = 0.02                                   # 与 tgai 的 Embedding/Linear 尺度对齐

    class RMS(nn.Module):
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
            self.q, self.k, self.v, self.o = (nn.Linear(D, D, bias=False) for _ in range(4))

        def forward(self, x):
            B, S, _ = x.shape
            dh = D // NH
            q = self.q(x).reshape(B, S, NH, dh).transpose(1, 2)
            k = self.k(x).reshape(B, S, NH, dh).transpose(1, 2)
            v = self.v(x).reshape(B, S, NH, dh).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o(y.transpose(1, 2).reshape(B, S, D))

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = nn.Linear(D, E, bias=False)
            self.wg = nn.ParameterList([nn.Parameter(torch.randn(FF, D) * INIT) for _ in range(E)])
            self.wu = nn.ParameterList([nn.Parameter(torch.randn(FF, D) * INIT) for _ in range(E)])
            self.wd = nn.ParameterList([nn.Parameter(torch.randn(D, FF) * INIT) for _ in range(E)])

        def forward(self, x):
            B, S, _ = x.shape
            flat = x.reshape(B * S, D)
            logits = self.router(flat)                      # [N,E]
            probs = F.softmax(logits, dim=-1)
            topv, topi = probs.topk(TOPK, dim=-1)           # [N,K]
            out = torch.zeros_like(flat)
            for e in range(E):
                sel = (topi == e)                           # [N,K] bool
                hit = sel.any(dim=-1)                       # [N]
                if not bool(hit.any()):
                    continue
                w = (topv * sel).sum(dim=-1)                # [N]
                h = flat[hit]
                gu = F.silu(h @ self.wg[e].T) * (h @ self.wu[e].T)
                out[hit] += gu @ self.wd[e].T * w[hit].unsqueeze(-1)
            return out.reshape(B, S, D)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = RMS(D), RMS(D)
            self.att, self.moe = Attn(), MoE()

        def forward(self, x):
            x = x + self.att(self.n1(x))
            return x + self.moe(self.n2(x))

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, D)
            nn.init.normal_(self.emb.weight, std=INIT)
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.nf = RMS(D)

        def forward(self, ids):
            x = self.emb(ids)
            for b in self.blocks:
                x = b(x)
            return self.nf(x) @ self.emb.weight.T

    m = LM().to(dev)
    nparam = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    x, y = make_batch(seq)
    xt = torch.tensor(x, device=dev)
    yt = torch.tensor(y, device=dev)

    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
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
    print("吞吐: %.1f tokens/s（%.3f s/step）" % (steps * seq / dt, dt / steps))
    print("峰值显存: %.2f GB" % (torch.cuda.max_memory_allocated() / 2**30))
    return losses


# ─────────────────────────── tgai_go.train ───────────────────────────
def run_tgai(layers, steps, seq, attn="lowmem"):
    import tgai_go.train as T

    F = T.functional
    T.set_device("cuda")
    T.manual_seed(7)

    emb = T.nn.Embedding(V, D)
    blocks = []
    for _ in range(layers):
        blocks.append(dict(n1=T.nn.RMSNorm(D), q=T.nn.Linear(D, D), k=T.nn.Linear(D, D),
                           v=T.nn.Linear(D, D), o=T.nn.Linear(D, D), n2=T.nn.RMSNorm(D),
                           moe=T.nn.MoE(D, FF, E, top_k=TOPK, gate="softmax")))
    nf = T.nn.RMSNorm(D)

    params = [emb.weight, nf.weight]
    for b in blocks:
        params += [b["n1"].weight, b["q"].weight, b["k"].weight, b["v"].weight, b["o"].weight,
                   b["n2"].weight]
        for _n, p in b["moe"].named_parameters():
            params.append(p)
    nparam = sum(int(np.prod(p.shape)) for p in params)
    opt = T.optim.AdamW(params, lr=3e-4)

    x, y = make_batch(seq)
    B, S, DH = 1, seq, D // NH
    # 编译不计入 s/step。主模块在 set_device 里已经编过；nop 内核只在 lowmem 时加载。
    if attn == "lowmem":
        F.sdpa_lowmem_ready()
    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
        h = emb(x)
        for b in blocks:
            n = b["n1"](h)
            q = b["q"](n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
            k = b["k"](n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
            v = b["v"](n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
            if attn == "lowmem":
                o = F.scaled_dot_product_attention(q, k, v, causal=True, lowmem=True)
            else:
                o = F.scaled_dot_product_attention(q, k, v, causal=True)
            o = o.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
            h = h + b["o"](o).reshape(B, S, D)
            h = h + b["moe"](b["n2"](h)).reshape(B, S, D)
        hn = nf(h)
        logits = hn.reshape(B * S, D) @ emb.weight.T
        loss = F.cross_entropy(logits, y.reshape(-1))
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()
        losses.append(float(np.asarray(loss.to_numpy()).reshape(-1)[0]))
    dt = time.perf_counter() - t0
    print("参数量: %.1f M" % (nparam / 1e6))
    print("loss:", " -> ".join("%.4f" % v for v in losses))
    print("吞吐: %.1f tokens/s（%.3f s/step）" % (steps * seq / dt, dt / steps))
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "tgai"], required=True)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--attn", choices=["lowmem", "legacy"], default="lowmem",
                    help="仅 tgai：lowmem 不物化 P（默认）；legacy 是原来的 bmm+softmax")
    a = ap.parse_args()
    print("配置: d=%d nh=%d ff=%d E=%d top%d V=%d seq=%d layers=%d steps=%d（MoE） attn=%s"
          % (D, NH, FF, E, TOPK, V, a.seq, a.layers, a.steps,
             a.attn if a.backend == "tgai" else "torch"))
    if a.backend == "torch":
        run_torch(a.layers, a.steps, a.seq)
    else:
        run_tgai(a.layers, a.steps, a.seq, attn=a.attn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

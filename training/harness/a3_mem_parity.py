# -*- coding: utf-8 -*-
"""A3 判据：用**双方分配器的高水位**比显存，而不是 nvidia-smi 轮询。

为什么改口径：nvidia-smi 每 100 ms 采一次，跑 3 步只要 0.15 s，
采样点可能落在初始化中/结束后，同一份代码能读出 2.90 与 6.17 GiB 两个数（我实测过）。
分配器高水位是确定性的，且两边含义一致：
  tgai  : CudaRuntime.pool_stats()['peak_bytes']
  torch : torch.cuda.max_memory_allocated()

用法: python a3_mem_parity.py --backend tgai|torch --seq 2048
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tgai", "torch"], required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=0)
    a = ap.parse_args()

    D, NH, FF, E, TOPK, V = 1024, 16, 4096, 8, 2, 32779
    B, S, DH = 1, a.seq, D // NH

    txt = ""
    for p in ("pelican_on_bike.html", "bike_on_pelican.html"):
        try:
            txt += open("/home/featurize/tgai/_corpus/" + p, encoding="utf-8", errors="replace").read()
        except Exception:
            pass
    raw = np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).astype(np.int64) % 256
    if len(raw) < S + 2:
        raw = np.tile(raw, (S // max(1, len(raw))) + 2)
    ids = raw[: S + 1].reshape(1, S + 1)
    x, y = ids[:, :-1], ids[:, 1:]

    if a.backend == "torch":
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        dev = "cuda"
        torch.manual_seed(7)

        class MoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = nn.Linear(D, E, bias=False)
                self.experts = nn.ModuleList([
                    nn.ModuleDict({"gate": nn.Linear(D, FF, bias=False),
                                   "up": nn.Linear(D, FF, bias=False),
                                   "down": nn.Linear(FF, D, bias=False)})
                    for _ in range(E)])

            def forward(self, t):
                shape = t.shape
                flat = t.reshape(-1, D)
                probs = F.softmax(self.router(flat), dim=-1)
                top, idx = torch.topk(probs, TOPK, dim=-1)
                w = top / top.sum(dim=-1, keepdim=True)
                out = torch.zeros_like(flat)
                for e in range(E):
                    sel = (idx == e)
                    rows = sel.any(dim=-1)
                    if not bool(rows.any()):
                        continue
                    h = flat[rows]
                    ew = w[rows][sel[rows]].unsqueeze(-1)
                    g = self.experts[e]["gate"](h)
                    u = self.experts[e]["up"](h)
                    o = self.experts[e]["down"](F.silu(g) * u)
                    out[rows] = out[rows] + ew * o
                return out.reshape(shape)

        class RMSNorm(nn.Module):
            """torch 2.2.2 没有 nn.RMSNorm（2.4 才加），手写同语义版本。"""

            def __init__(self, dd, eps=1e-6):
                super().__init__()
                self.w = nn.Parameter(torch.ones(dd))
                self.eps = eps

            def forward(self, t):
                return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

        emb = nn.Embedding(V, D).to(dev)
        n1 = RMSNorm(D).to(dev)
        n2 = RMSNorm(D).to(dev)
        q = nn.Linear(D, D, bias=False).to(dev)
        k = nn.Linear(D, D, bias=False).to(dev)
        v = nn.Linear(D, D, bias=False).to(dev)
        o = nn.Linear(D, D, bias=False).to(dev)
        moe = MoE().to(dev)
        nf = RMSNorm(D).to(dev)
        params = list(emb.parameters()) + list(n1.parameters()) + list(n2.parameters())
        params += list(q.parameters()) + list(k.parameters()) + list(v.parameters())
        params += list(o.parameters()) + list(moe.parameters()) + list(nf.parameters())
        opt = torch.optim.AdamW(params, lr=3e-4)
        xt = torch.tensor(x, dtype=torch.long, device=dev)
        yt = torch.tensor(y.reshape(-1), dtype=torch.long, device=dev)

        def step():
            h = emb(xt)
            n = n1(h)
            qq = q(n).reshape(B, S, NH, DH).transpose(1, 2)
            kk = k(n).reshape(B, S, NH, DH).transpose(1, 2)
            vv = v(n).reshape(B, S, NH, DH).transpose(1, 2)
            oo = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
            oo = oo.transpose(1, 2).reshape(B * S, D)
            h = h + o(oo).reshape(B, S, D)
            h = h + moe(n2(h)).reshape(B, S, D)
            hn = nf(h)
            logits = hn.reshape(B * S, D) @ emb.weight.T
            loss = F.cross_entropy(logits, yt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return float(loss)

        for _ in range(2):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        print("seq=%d backend=torch 分配器峰值 = %.3f GiB (%.0f MB)"
              % (S, peak / 2**30, peak / 1e6))
        return

    import tgai_go.train as T
    F = T.functional
    T.set_device("cuda")
    rt = T._cuda()
    T.manual_seed(7)

    emb = T.nn.Embedding(V, D)
    n1, n2 = T.nn.RMSNorm(D), T.nn.RMSNorm(D)
    q, k, v, o = (T.nn.Linear(D, D) for _ in range(4))
    moe = T.nn.MoE(D, FF, E, top_k=TOPK, gate="softmax")
    nf = T.nn.RMSNorm(D)
    params = [emb.weight, n1.weight, n2.weight, q.weight, k.weight, v.weight, o.weight, nf.weight]
    for _n, p in moe.named_parameters():
        params.append(p)
    opt = T.optim.AdamW(params, lr=3e-4)

    def step():
        h = emb(x)
        n = n1(h)
        qq = q(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        kk = k(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        vv = v(n.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        oo = F.scaled_dot_product_attention(qq, kk, vv, causal=True, lowmem=True)
        oo = oo.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
        h = h + o(oo).reshape(B, S, D)
        h = h + moe(n2(h)).reshape(B, S, D)
        hn = nf(h)
        if a.chunk:
            loss = F.linear_cross_entropy(hn.reshape(B * S, D), emb.weight, y.reshape(-1), chunk=a.chunk)
        else:
            loss = F.cross_entropy(hn.reshape(B * S, D) @ emb.weight.T, y.reshape(-1))
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()
        return float(np.asarray(loss.to_numpy()).reshape(-1)[0])

    for _ in range(2):
        step()
    rt.sync()
    rt.empty_cache()
    rt._pool_peak = 0
    rt._pool_peak_live = 0
    step()
    rt.sync()
    st = rt.pool_stats()
    print("seq=%d backend=tgai chunk=%d 分配器峰值 = %.3f GiB (%.0f MB)  [live_peak=%.3f]"
          % (S, a.chunk, st["peak_bytes"] / 2**30, st["peak_bytes"] / 1e6,
             st["peak_live_bytes"] / 2**30))


if __name__ == "__main__":
    main()

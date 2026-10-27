# -*- coding: utf-8 -*-
"""单层 Transformer 判分台：attention + MoE FFN 的前向与反向。

被测源码 device-only，导出：
  layer_fwd(x,w1,Wq,Wk,Wv,Wo,w2,Wr,Wg,Wu,Wd, out,cache,
            seq,d,nh,dh,ff,E,eps,scale)
  layer_bwd(dOut,x,w1,Wq,Wk,Wv,Wo,w2,Wr,Wg,Wu,Wd, cache,
            dx,dw1,dWq,dWk,dWv,dWo,dw2,dWr,dWg,dWu,dWd,
            seq,d,nh,dh,ff,E,eps,scale)
启动: grid=(1,1,1) block=(1,1,1)，动态共享内存 48KB
cache: 调用方分配 8192 float，布局自定（前向写、反向读）

前向语义:
  h1 = rmsnorm(x)*w1
  q,k,v = h1@Wq, h1@Wk, h1@Wv    （按 [seq,nh,dh] 解释）
  att = softmax_causal(q k^T * scale) @ v ;  o = att@Wo ;  x2 = x + o
  h2 = rmsnorm(x2)*w2 ;  logits = h2@Wr
  e = argmax(logits) ; g = sigmoid(logits[e])
  gu = silu(h2@Wg[e]) * (h2@Wu[e]) ; y = gu@Wd[e] ; out = x2 + g*y
反向: loss = sum(out*dOut)，输出全部 11 个梯度（调用方已清零，只累加）

用法:
    python layer_suite.py <src.cu> [seq d nh dh ff E]
"""
import ctypes
import sys

import torch
import torch.nn.functional as F

from tgai_nvrtc import Cuda

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 8
D = int(sys.argv[3]) if len(sys.argv) > 3 else 64
NH = int(sys.argv[4]) if len(sys.argv) > 4 else 4
DH = D // NH
FF = int(sys.argv[5]) if len(sys.argv) > 5 else 64
E = int(sys.argv[6]) if len(sys.argv) > 6 else 2
EPS = 1e-5
SCALE = 1.0 / (DH ** 0.5)
CACHE = 8192
SMEM = 48 * 1024
dev = "cuda"


def P_(t):
    return (t.data_ptr(), ctypes.c_void_p)


def I(v):
    return (int(v), ctypes.c_int)


def Fv(v):
    return (float(v), ctypes.c_float)


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)


def main(src):
    code = open(src, encoding="utf-8", errors="replace").read()
    code = code.replace("#include <cuda_runtime.h>", "").replace("#include <stddef.h>", "")
    torch.cuda.init()
    cu = Cuda()
    mod = cu.compile(code, "sm_70")

    torch.manual_seed(11)
    x = torch.randn(SEQ, D, device=dev)
    w1 = torch.randn(D, device=dev)
    Wq = torch.randn(D, D, device=dev) * 0.1
    Wk = torch.randn(D, D, device=dev) * 0.1
    Wv = torch.randn(D, D, device=dev) * 0.1
    Wo = torch.randn(D, D, device=dev) * 0.1
    w2 = torch.randn(D, device=dev)
    Wr = torch.randn(D, E, device=dev)
    Wg = torch.randn(E, D, FF, device=dev) * 0.1
    Wu = torch.randn(E, D, FF, device=dev) * 0.1
    Wd = torch.randn(E, FF, D, device=dev) * 0.1
    dOut = torch.randn(SEQ, D, device=dev)

    # ---------- 参考 ----------
    ps = [t.clone().requires_grad_(True) for t in (x, w1, Wq, Wk, Wv, Wo, w2, Wr, Wg, Wu, Wd)]
    xr, w1r, Wqr, Wkr, Wvr, Wor, w2r, Wrr, Wgr, Wur, Wdr = ps

    ms = (xr * xr).mean(dim=1, keepdim=True)
    h1 = xr * torch.rsqrt(ms + EPS) * w1r
    q = h1 @ Wqr
    k = h1 @ Wkr
    v = h1 @ Wvr
    qh = q.view(SEQ, NH, DH).transpose(0, 1)      # [nh,seq,dh]
    kh = k.view(SEQ, NH, DH).transpose(0, 1)
    vh = v.view(SEQ, NH, DH).transpose(0, 1)
    scores = (qh @ kh.transpose(1, 2)) * SCALE    # [nh,seq,seq]
    mask = torch.tril(torch.ones(SEQ, SEQ, device=dev, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    P = torch.softmax(scores, dim=-1)
    at = (P @ vh).transpose(0, 1).reshape(SEQ, D)
    o = at @ Wor
    x2 = xr + o
    ms2 = (x2 * x2).mean(dim=1, keepdim=True)
    h2 = x2 * torch.rsqrt(ms2 + EPS) * w2r
    logits = h2 @ Wrr
    e = logits.argmax(dim=1)
    gate = torch.sigmoid(logits.gather(1, e.unsqueeze(1))).squeeze(1)
    a = torch.einsum("td,tdf->tf", h2, Wgr[e])
    u = torch.einsum("td,tdf->tf", h2, Wur[e])
    gu = F.silu(a) * u
    y = torch.einsum("tf,tfd->td", gu, Wdr[e])
    out_ref = x2 + gate.unsqueeze(1) * y
    (out_ref * dOut).sum().backward()

    # ---------- 被测 ----------
    out = torch.zeros(SEQ, D, device=dev)
    cache = torch.zeros(CACHE, device=dev)
    args_f = [P_(x), P_(w1), P_(Wq), P_(Wk), P_(Wv), P_(Wo),
              P_(w2), P_(Wr), P_(Wg), P_(Wu), P_(Wd), P_(out), P_(cache),
              I(SEQ), I(D), I(NH), I(DH), I(FF), I(E), Fv(EPS), Fv(SCALE)]
    cu.launch(mod, "layer_fwd", (1, 1, 1), (1, 1, 1), args_f, shared=SMEM)
    cu.sync()

    grads = {n: torch.zeros_like(t) for n, t in
             (("dx", x), ("dw1", w1), ("dWq", Wq), ("dWk", Wk), ("dWv", Wv),
              ("dWo", Wo), ("dw2", w2), ("dWr", Wr), ("dWg", Wg), ("dWu", Wu), ("dWd", Wd))}
    args_b = [P_(dOut), P_(x), P_(w1), P_(Wq), P_(Wk), P_(Wv), P_(Wo),
              P_(w2), P_(Wr), P_(Wg), P_(Wu), P_(Wd), P_(cache),
              P_(grads["dx"]), P_(grads["dw1"]), P_(grads["dWq"]), P_(grads["dWk"]),
              P_(grads["dWv"]), P_(grads["dWo"]), P_(grads["dw2"]), P_(grads["dWr"]),
              P_(grads["dWg"]), P_(grads["dWu"]), P_(grads["dWd"]),
              I(SEQ), I(D), I(NH), I(DH), I(FF), I(E), Fv(EPS), Fv(SCALE)]
    cu.launch(mod, "layer_bwd", (1, 1, 1), (1, 1, 1), args_b, shared=SMEM)
    cu.sync()

    refs = {"dx": ps[0].grad, "dw1": ps[1].grad, "dWq": ps[2].grad, "dWk": ps[3].grad,
            "dWv": ps[4].grad, "dWo": ps[5].grad, "dw2": ps[6].grad, "dWr": ps[7].grad,
            "dWg": ps[8].grad, "dWu": ps[9].grad, "dWd": ps[10].grad}

    print("seq=%d d=%d nh=%d dh=%d ff=%d E=%d  专家分布=%s"
          % (SEQ, D, NH, DH, FF, E, e.cpu().tolist()))
    worst = 0.0
    e_out = (out - out_ref.detach()).abs().max().item()
    print("  out   rel=%.2e" % (e_out / (out_ref.detach().abs().max().item() + 1e-12)))
    for n, g in grads.items():
        r = rel(g, refs[n])
        worst = max(worst, r)
        print("  %-4s rel=%.2e" % (n, r))

    ok = worst < 5e-4
    print("VERDICT ok=%s worst_rel=%.3e" % (ok, worst))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

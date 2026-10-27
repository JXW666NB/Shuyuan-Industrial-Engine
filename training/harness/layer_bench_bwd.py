# -*- coding: utf-8 -*-
"""反向基准：attention 块 + MoE FFN 块的 backward，我们 vs PyTorch autograd。

规模同前: seq=512, d=1024, nh=16, dh=64, ff=4096, E=8, 每专家 128 行（合计 1024 行）
只计反向时间（torch 侧用 retain_graph，图在计时外构建）。

用法: python layer_bench_bwd.py <bwd.cu> <attn.cu>
"""
import ctypes
import sys

import torch
import torch.nn.functional as F

from cublas_bridge import Cublas
from tgai_nvrtc import Cuda

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

dev = "cuda"
SEQ, D, NH, DH, FF, E, ROWS = 512, 1024, 16, 64, 4096, 8, 128
MROWS = E * ROWS
SCALE = 1.0 / (DH ** 0.5)
SMEM = 32 * 1024
BQ = 64


def P_(t):
    return (t.data_ptr(), ctypes.c_void_p)


def I(v):
    return (int(v), ctypes.c_int)


def Fv(v):
    return (float(v), ctypes.c_float)


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main(bwd_src, attn_src):
    torch.cuda.init()
    cu = Cuda()
    cb = Cublas()

    def load(src):
        code = open(src, encoding="utf-8", errors="replace").read()
        code = code.replace("#include <cuda_runtime.h>", "").replace("#include <stddef.h>", "")
        return cu.compile(code, "sm_70")

    mb, ma = load(bwd_src), load(attn_src)
    gy = (SEQ + BQ - 1) // BQ

    torch.manual_seed(0)
    # ---- 固定张量（前向快照，等价于训练里保存的中间量）----
    x = torch.randn(SEQ, D, device=dev) * 0.5
    w1 = torch.randn(D, device=dev)
    h = torch.randn(SEQ, D, device=dev) * 0.5
    Wqkv = torch.randn(D, 3 * D, device=dev) * 0.05
    dqkv = torch.randn(SEQ, 3 * D, device=dev) * 0.05
    Wo = torch.randn(D, D, device=dev) * 0.05
    att = torch.randn(SEQ, D, device=dev) * 0.5
    datt = torch.randn(SEQ, D, device=dev) * 0.5
    q = torch.randn(SEQ, NH, DH, device=dev) * 0.5
    k = torch.randn(SEQ, NH, DH, device=dev) * 0.5
    v = torch.randn(SEQ, NH, DH, device=dev) * 0.5
    Pmat = torch.rand(NH, SEQ, SEQ, device=dev)
    dO = torch.randn(SEQ, NH, DH, device=dev) * 0.5
    dx2 = torch.randn(SEQ, D, device=dev) * 0.5

    dx = torch.zeros(SEQ, D, device=dev)
    dw1 = torch.zeros(D, device=dev)
    dWqkv = torch.zeros(D, 3 * D, device=dev)
    dWo = torch.zeros(D, D, device=dev)
    dq = torch.zeros(SEQ, NH, DH, device=dev)
    dk = torch.zeros(SEQ, NH, DH, device=dev)
    dv = torch.zeros(SEQ, NH, DH, device=dev)
    dh = torch.zeros(SEQ, D, device=dev)

    def ours_attn_bwd():
        dx.copy_(dx2)
        cb.gemm_tn(att, dx2, out=dWo)
        datt_buf = cb.gemm_nt(dx2, Wo)
        cu.launch(ma, "attn_bwd", (NH, gy, 1), (256, 1, 1),
                  [P_(datt_buf), P_(q), P_(k), P_(v), P_(Pmat),
                   P_(dq), P_(dk), P_(dv), I(SEQ), I(NH), I(DH), Fv(SCALE)], shared=SMEM)
        cu.launch(mb, "rope_bwd", (SEQ, NH, 1), (128, 1, 1),
                  [P_(dq), P_(dk), I(SEQ), I(NH), I(DH), I(0), P_(q), P_(k), I(SEQ)])
        dqkv_buf = torch.cat([dq.reshape(SEQ, D), dk.reshape(SEQ, D), dv.reshape(SEQ, D)], dim=1)
        cb.gemm_tn(h, dqkv_buf, out=dWqkv)
        dh_buf = cb.gemm_nt(dqkv_buf, Wqkv)
        cu.launch(mb, "rmsnorm_bwd", (SEQ, 1, 1), (256, 1, 1),
                  [P_(dh_buf), P_(x), P_(w1), P_(dx), P_(dw1), I(SEQ), I(D), Fv(1e-5)], shared=1024)

    # ---- torch 侧 ----
    xr = x.clone().requires_grad_(True)
    w1r = w1.clone().requires_grad_(True)
    hr = h.clone().requires_grad_(True)
    Wqkvr = Wqkv.clone().requires_grad_(True)
    Wor = Wo.clone().requires_grad_(True)
    attr_ = att.clone().requires_grad_(True)
    qr = q.clone().requires_grad_(True)
    kr = k.clone().requires_grad_(True)
    vr = v.clone().requires_grad_(True)

    def torch_forward():
        hh = hr * torch.rsqrt((xr * xr).mean(1, keepdim=True) + 1e-5) * w1r
        dq_ = hh @ Wqkvr
        qq = dq_[:, :D].view(SEQ, NH, DH)
        kk = dq_[:, D:2 * D].view(SEQ, NH, DH)
        vv = dq_[:, 2 * D:].view(SEQ, NH, DH)
        at = F.scaled_dot_product_attention(qq.transpose(0, 1).unsqueeze(0),
                                            kk.transpose(0, 1).unsqueeze(0),
                                            vv.transpose(0, 1).unsqueeze(0),
                                            is_causal=True, scale=SCALE)[0].transpose(0, 1)
        o = at.reshape(SEQ, D) @ Wor
        return (xr + o)

    out_t = torch_forward()
    loss = (out_t * 0.5).sum()

    def torch_bwd():
        for t in (xr, w1r, hr, Wqkvr, Wor, attr_, qr, kr, vr):
            if t.grad is not None:
                t.grad = None
        loss.backward(retain_graph=True)

    t_ours = bench(ours_attn_bwd)
    t_torch = bench(torch_bwd)

    # ---- MoE 段反向 ----
    moe_in = torch.randn(MROWS, D, device=dev) * 0.5
    w2 = torch.randn(D, device=dev)
    norm = torch.randn(MROWS, D, device=dev) * 0.5
    Wg = torch.randn(E, D, FF, device=dev) * 0.02
    Wu = torch.randn(E, D, FF, device=dev) * 0.02
    Wd = torch.randn(E, FF, D, device=dev) * 0.02
    gu = torch.randn(MROWS, FF, device=dev) * 0.5
    g_buf = torch.randn(MROWS, FF, device=dev) * 0.5
    u_buf = torch.randn(MROWS, FF, device=dev) * 0.5
    dout = torch.randn(MROWS, D, device=dev) * 0.5
    dWg = torch.zeros(E, D, FF, device=dev)
    dWu = torch.zeros(E, D, FF, device=dev)
    dWd = torch.zeros(E, FF, D, device=dev)
    dw2 = torch.zeros(D, device=dev)
    dg_buf = torch.zeros(ROWS, FF, device=dev)
    du_buf = torch.zeros(ROWS, FF, device=dev)
    dh_moe = torch.zeros(MROWS, D, device=dev)
    dmoe_in = torch.zeros(MROWS, D, device=dev)

    def ours_moe_bwd():
        for e in range(E):
            seg = slice(e * ROWS, (e + 1) * ROWS)
            dy = dout[seg]
            cb.gemm_tn(gu[seg], dy, out=dWd[e])
            dgu = cb.gemm_nt(dy, Wd[e])
            n = ROWS * FF
            cu.launch(mb, "silu_mul_bwd", ((n + 255) // 256, 1, 1), (256, 1, 1),
                      [P_(dgu), P_(g_buf[seg]), P_(u_buf[seg]),
                       P_(dg_buf), P_(du_buf), I(n)])
            cb.gemm_tn(norm[seg], dg_buf, out=dWg[e])
            cb.gemm_tn(norm[seg], du_buf, out=dWu[e])
            a1 = cb.gemm_nt(dg_buf, Wg[e])
            a2 = cb.gemm_nt(du_buf, Wu[e])
            dh_moe[seg] = a1 + a2
        cu.launch(mb, "rmsnorm_bwd", (MROWS, 1, 1), (256, 1, 1),
                  [P_(dh_moe), P_(moe_in), P_(w2), P_(dmoe_in), P_(dw2),
                   I(MROWS), I(D), Fv(1e-5)], shared=1024)

    # torch 侧 MoE —— 用 grouped bmm（PyTorch 的公平写法）
    # 输入堆叠成 [E, ROWS, D]，权重 [E, D, FF] / [E, FF, D]
    nn_st = moe_in.view(E, ROWS, D).clone().requires_grad_(True)
    w2r = w2.clone().requires_grad_(True)
    Wg_st = Wg.clone().requires_grad_(True)      # [E,D,FF]
    Wu_st = Wu.clone().requires_grad_(True)
    Wd_st = Wd.clone().requires_grad_(True)      # [E,FF,D]

    def torch_moe_forward():
        nz = nn_st * torch.rsqrt((nn_st * nn_st).mean(2, keepdim=True) + 1e-5) * w2r
        g = F.silu(torch.bmm(nz, Wg_st)) * torch.bmm(nz, Wu_st)     # [E,ROWS,FF]
        out = torch.bmm(g, Wd_st)                                   # [E,ROWS,D]
        return out.reshape(MROWS, D)

    out_m = torch_moe_forward()
    loss_m = (out_m * 0.5).sum()

    def torch_moe_bwd():
        for t in (nn_st, w2r, Wg_st, Wu_st, Wd_st):
            if t.grad is not None:
                t.grad = None
        loss_m.backward(retain_graph=True)

    t_ours_m = bench(ours_moe_bwd)
    t_torch_m = bench(torch_moe_bwd)

    print("%-26s %10s %10s %8s" % ("段", "ours ms", "torch ms", "比值"))
    print("-" * 60)
    print("%-26s %10.3f %10.3f %7.2fx" % ("attention 块 反向", t_ours, t_torch, t_ours / t_torch))
    print("%-26s %10.3f %10.3f %7.2fx" % ("MoE FFN 反向", t_ours_m, t_torch_m, t_ours_m / t_torch_m))
    tot_o, tot_t = t_ours + t_ours_m, t_torch + t_torch_m
    print("%-26s %10.3f %10.3f %7.2fx" % ("合计 反向", tot_o, tot_t, tot_o / tot_t))
    print("-" * 60)
    print("VERDICT bwd_ratio=%.2f" % (tot_o / tot_t))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

# -*- coding: utf-8 -*-
"""整层端到端基准：attention 块 + MoE FFN 块 的前向与反向，我们 vs PyTorch。

规模: seq=512, d=1024, nh=16, dh=64, ff=4096, E=8, 每专家 128 行（top-2 排序后均分）
说明: 路由本身（0.014 ms 级）不计入；MoE 用排序后的分组 GEMM（真实引擎做法）。

用法: python layer_bench.py <fwd.cu> <attn.cu> <bwd.cu>
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
SEQ, D, NH, DH, FF, E = 512, 1024, 16, 64, 4096, 8
ROWS = 128                      # 每专家行数
SCALE = 1.0 / (DH ** 0.5)
SMEM_ATTN = 32 * 1024
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


def main(fwd_src, attn_src, bwd_src):
    torch.cuda.init()
    cu = Cuda()
    cb = Cublas()

    def load(src):
        code = open(src, encoding="utf-8", errors="replace").read()
        code = code.replace("#include <cuda_runtime.h>", "").replace("#include <stddef.h>", "")
        return cu.compile(code, "sm_70")

    mf, ma, mb = load(fwd_src), load(attn_src), load(bwd_src)
    gy = (SEQ + BQ - 1) // BQ

    torch.manual_seed(0)
    x = torch.randn(SEQ, D, device=dev) * 0.5
    w1 = torch.randn(D, device=dev)
    Wqkv = torch.randn(D, 3 * D, device=dev) * 0.05
    Wo = torch.randn(D, D, device=dev) * 0.05
    cos = torch.rand(SEQ, DH // 2, device=dev) * 1.2
    sin = torch.rand(SEQ, DH // 2, device=dev) * 0.3

    # ---------------- attention 块：前向 ----------------
    h = torch.zeros(SEQ, D, device=dev)
    qkv = torch.zeros(SEQ, 3 * D, device=dev)
    o = torch.zeros(SEQ, D, device=dev)
    att = torch.zeros(SEQ, NH, DH, device=dev)
    Pt = torch.zeros(NH, SEQ, SEQ, device=dev)
    x2 = torch.zeros(SEQ, D, device=dev)
    cos_k = cos.contiguous()
    sin_k = sin.contiguous()

    def attn_blk_fwd2():
        cu.launch(mf, "rmsnorm_fwd", (SEQ, 1, 1), (256, 1, 1),
                  [P_(h), P_(x), P_(w1), I(SEQ), I(D), Fv(1e-5)], shared=1024)
        cb.sgemm(h, Wqkv, out=qkv)
        q = qkv[:, :D].contiguous().view(SEQ, NH, DH)
        k = qkv[:, D:2 * D].contiguous().view(SEQ, NH, DH)
        v = qkv[:, 2 * D:].contiguous().view(SEQ, NH, DH)
        cu.launch(mf, "rope_fwd", (SEQ, NH, 1), (128, 1, 1),
                  [P_(q), P_(k), I(SEQ), I(NH), I(DH), I(0), P_(cos_k), P_(sin_k), I(SEQ)])
        cu.launch(ma, "attn_fwd", (NH, gy, 1), (256, 1, 1),
                  [P_(q), P_(k), P_(v), P_(att), P_(Pt), I(SEQ), I(NH), I(DH), Fv(SCALE)],
                  shared=SMEM_ATTN)
        cb.sgemm(att.view(SEQ, D), Wo, out=o)
        x2.copy_(x + o)

    attn_blk_fwd2()

    # torch 侧同结构
    def torch_attn_blk():
        hh = x * torch.rsqrt((x * x).mean(1, keepdim=True) + 1e-5) * w1
        qq = (hh @ Wqkv)[:, :D].view(SEQ, NH, DH)
        kk = (hh @ Wqkv)[:, D:2 * D].view(SEQ, NH, DH)
        vv = (hh @ Wqkv)[:, 2 * D:].view(SEQ, NH, DH)

        def rot(t):
            ev = t[..., 0::2].clone()
            od = t[..., 1::2].clone()
            out = t.clone()
            out[..., 0::2] = ev * cos.unsqueeze(1) - od * sin.unsqueeze(1)
            out[..., 1::2] = ev * sin.unsqueeze(1) + od * cos.unsqueeze(1)
            return out

        qq, kk = rot(qq), rot(kk)
        at = F.scaled_dot_product_attention(qq.transpose(0, 1).unsqueeze(0),
                                            kk.transpose(0, 1).unsqueeze(0),
                                            vv.transpose(0, 1).unsqueeze(0),
                                            is_causal=True, scale=SCALE)[0].transpose(0, 1)
        oo = at.reshape(SEQ, D) @ Wo
        return x + oo

    # ---------------- MoE FFN：前向 ----------------
    MROWS = E * ROWS                      # top-2 汇总后的行数 = 1024
    moe_in = torch.randn(MROWS, D, device=dev) * 0.5
    norm = torch.zeros(MROWS, D, device=dev)
    moe_out = torch.zeros(MROWS, D, device=dev)
    w2 = torch.randn(D, device=dev)
    Wg = torch.randn(E, D, FF, device=dev) * 0.02
    Wu = torch.randn(E, D, FF, device=dev) * 0.02
    Wd = torch.randn(E, FF, D, device=dev) * 0.02
    gate_buf = torch.zeros(ROWS, FF, device=dev)
    up_buf = torch.zeros(ROWS, FF, device=dev)
    down_buf = torch.zeros(ROWS, D, device=dev)
    tmp_g = torch.zeros(ROWS, D, device=dev)
    tmp_u = torch.zeros(ROWS, D, device=dev)

    def moe_fwd():
        cu.launch(mf, "rmsnorm_fwd", (MROWS, 1, 1), (256, 1, 1),
                  [P_(norm), P_(moe_in), P_(w2), I(MROWS), I(D), Fv(1e-5)], shared=1024)
        moe_out.zero_()
        for e in range(E):
            src = norm[e * ROWS:(e + 1) * ROWS]
            tmp_g.copy_(src)
            tmp_u.copy_(src)
            cb.sgemm(tmp_g, Wg[e], out=gate_buf)
            cb.sgemm(tmp_u, Wu[e], out=up_buf)
            n = ROWS * FF
            cu.launch(mf, "silu_mul_fwd", ((n + 255) // 256, 1, 1), (256, 1, 1),
                      [P_(gate_buf), P_(up_buf), I(n)])
            cb.sgemm(gate_buf, Wd[e], out=down_buf)
            seg = moe_out[e * ROWS:(e + 1) * ROWS]
            cu.launch(mf, "moe_add_fwd", (ROWS, (D + 255) // 256, 1), (256, 1, 1),
                      [P_(seg), P_(down_buf), Fv(1.0), I(ROWS), I(D), I(256)])

    def torch_moe():
        nn = moe_in * torch.rsqrt((moe_in * moe_in).mean(1, keepdim=True) + 1e-5) * w2
        acc = torch.zeros(MROWS, D, device=dev)
        for e in range(E):
            src = nn[e * ROWS:(e + 1) * ROWS]
            g = F.silu(src @ Wg[e]) * (src @ Wu[e])
            acc[e * ROWS:(e + 1) * ROWS] = g @ Wd[e]
        return acc

    moe_fwd()

    t1 = bench(attn_blk_fwd2)
    t2 = bench(moe_fwd)
    t3 = bench(torch_attn_blk)
    t4 = bench(torch_moe)

    print("%-26s %10s %10s %8s" % ("段", "ours ms", "torch ms", "比值"))
    print("-" * 60)
    print("%-26s %10.3f %10.3f %7.2fx" % ("attention 块 前向", t1, t3, t1 / t3))
    print("%-26s %10.3f %10.3f %7.2fx" % ("MoE FFN 前向", t2, t4, t2 / t4))
    print("%-26s %10.3f %10.3f %7.2fx" % ("合计 前向", t1 + t2, t3 + t4, (t1 + t2) / (t3 + t4)))
    print("-" * 60)
    print("注: 仅前向。反向需按同样结构再搭一套（下一轮）。")
    print("VERDICT fwd_ratio=%.2f" % ((t1 + t2) / (t3 + t4)))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))

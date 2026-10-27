# -*- coding: utf-8 -*-
"""逐算子耗时分解：我们的 kernel vs PyTorch，在层规模上对比。

规模: seq=512, d=1024, nh=16, dh=64, ff=4096, E=8, top-2
用法: python op_bench.py <fwd.cu> <attn.cu>
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
SEQ, D, NH, DH, FF, E, TOPK = 512, 1024, 16, 64, 4096, 8, 2
SCALE = 1.0 / (DH ** 0.5)


def P_(t):
    return (t.data_ptr(), ctypes.c_void_p)


def I(v):
    return (int(v), ctypes.c_int)


def Fv(v):
    return (float(v), ctypes.c_float)


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True)
    en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def main(fwd_src, attn_src):
    torch.cuda.init()
    cu = Cuda()
    cb = Cublas()

    def load(src):
        code = open(src, encoding="utf-8", errors="replace").read()
        code = code.replace("#include <cuda_runtime.h>", "").replace("#include <stddef.h>", "")
        return cu.compile(code, "sm_70")

    mf = load(fwd_src)
    ma = load(attn_src)

    rows = []

    # ---------- rmsnorm ----------
    x = torch.randn(SEQ, D, device=dev)
    w = torch.randn(D, device=dev)
    out = torch.zeros_like(x)
    t_ours = bench(lambda: cu.launch(mf, "rmsnorm_fwd", (SEQ, 1, 1), (256, 1, 1),
                                     [P_(out), P_(x), P_(w), I(SEQ), I(D), Fv(1e-5)], shared=1024))
    t_ref = bench(lambda: x * torch.rsqrt((x * x).mean(1, keepdim=True) + 1e-5) * w)
    rows.append(("rmsnorm [512,1024]", t_ours, t_ref))

    # ---------- GEMM: qkv (512x1024 @ 1024x3072) ----------
    Wqkv = torch.randn(D, 3 * D, device=dev)
    t_ours = bench(lambda: cb.sgemm(x, Wqkv, out=torch.empty(SEQ, 3 * D, device=dev)))
    t_ref = bench(lambda: x @ Wqkv)
    rows.append(("GEMM qkv 512x1024x3072", t_ours, t_ref))

    # ---------- GEMM: o_proj (512x1024 @ 1024x1024) ----------
    Wo = torch.randn(D, D, device=dev)
    t_ours = bench(lambda: cb.sgemm(x, Wo, out=torch.empty(SEQ, D, device=dev)))
    t_ref = bench(lambda: x @ Wo)
    rows.append(("GEMM o 512x1024x1024", t_ours, t_ref))

    # ---------- GEMM: MoE 专家 (1024 行 x 1024 @ 1024x4096) ----------
    h = torch.randn(1024, D, device=dev)
    Wg = torch.randn(D, FF, device=dev)
    t_ours = bench(lambda: cb.sgemm(h, Wg, out=torch.empty(1024, FF, device=dev)))
    t_ref = bench(lambda: h @ Wg)
    rows.append(("GEMM moe-up 1024x1024x4096", t_ours, t_ref))

    # ---------- silu_mul (1024x4096) ----------
    a = torch.randn(1024, FF, device=dev)
    b = torch.randn(1024, FF, device=dev)
    n = a.numel()
    ac = a.clone()
    t_ours = bench(lambda: cu.launch(mf, "silu_mul_fwd", ((n + 255) // 256, 1, 1), (256, 1, 1),
                                     [P_(ac), P_(b), I(n)]))
    t_ref = bench(lambda: F.silu(a) * b)
    rows.append(("silu_mul 1024x4096", t_ours, t_ref))

    # ---------- moe_add (512x1024) ----------
    o0 = torch.randn(SEQ, D, device=dev)
    e1 = torch.randn(SEQ, D, device=dev)
    oc = o0.clone()
    t_ours = bench(lambda: cu.launch(mf, "moe_add_fwd", (SEQ, (D + 127) // 128, 1), (128, 1, 1),
                                     [P_(oc), P_(e1), Fv(0.5), I(SEQ), I(D), I(128)]))
    t_ref = bench(lambda: o0 + 0.5 * e1)
    rows.append(("moe_add 512x1024", t_ours, t_ref))

    # ---------- attention (tiled, seq=512) ----------
    aseq = 512
    q = torch.randn(aseq, NH, DH, device=dev) * 0.5
    k = torch.randn(aseq, NH, DH, device=dev) * 0.5
    v = torch.randn(aseq, NH, DH, device=dev) * 0.5
    o = torch.zeros(aseq, NH, DH, device=dev)
    pp = torch.zeros(NH, aseq, aseq, device=dev)
    smem = 32 * 1024
    gy = (aseq + 63) // 64
    t_ours = bench(lambda: cu.launch(ma, "attn_fwd", (NH, gy, 1), (256, 1, 1),
                                     [P_(q), P_(k), P_(v), P_(o), P_(pp),
                                      I(aseq), I(NH), I(DH), Fv(SCALE)], shared=smem))
    qh = q.transpose(0, 1).unsqueeze(0)
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)
    t_ref = bench(lambda: F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=SCALE))
    rows.append(("attention fwd seq=512 nh=16", t_ours, t_ref))

    # attention 反向
    dO = torch.randn(aseq, NH, DH, device=dev) * 0.5
    dQ = torch.zeros(aseq, NH, DH, device=dev)
    dK = torch.zeros(aseq, NH, DH, device=dev)
    dV = torch.zeros(aseq, NH, DH, device=dev)
    t_ours_b = bench(lambda: cu.launch(ma, "attn_bwd", (NH, gy, 1), (256, 1, 1),
                                       [P_(dO), P_(q), P_(k), P_(v), P_(pp),
                                        P_(dQ), P_(dK), P_(dV),
                                        I(aseq), I(NH), I(DH), Fv(SCALE)], shared=smem))

    def torch_attn_bwd():
        q2 = qh.clone().requires_grad_(True)
        k2 = kh.clone().requires_grad_(True)
        v2 = vh.clone().requires_grad_(True)
        oo = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True, scale=SCALE)
        oo.backward(dO.transpose(0, 1).unsqueeze(0))

    t_ref_b = bench(torch_attn_bwd)
    rows.append(("attention bwd seq=512 nh=16", t_ours_b, t_ref_b))
    print("%-30s %12s %12s %10s" % ("算子", "ours ms", "torch ms", "ours/torch"))
    print("-" * 70)
    for name, a_, b_ in rows:
        ratio = a_ / b_ if b_ > 0 else float("inf")
        print("%-30s %12.4f %12.4f %9.2fx" % (name, a_, b_, ratio))
    print("-" * 70)
    print("注: GEMM 走同一个 cuBLAS；elementwise 是我们的融合 kernel")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

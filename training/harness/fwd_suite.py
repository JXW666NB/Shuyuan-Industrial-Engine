# -*- coding: utf-8 -*-
"""前向算子判分台（与 tgai_cuda.cu 语义对齐）。

被测源码 device-only，导出 6 个 kernel：
  rmsnorm_fwd(out, x, w, seq, d, eps)                        grid=(seq,1,1) block=256 shared=1024B
  silu_mul_fwd(a, b, n)                                      grid=ceil(n/256) block=256（就地改 a）
  moe_add_fwd(out, e, weight, seq, d, blk)                   grid=(seq, ceil(d/blk),1) block=(blk,1,1)
  emb_gather_fwd(out, emb, ids, seq, d, vocab, unk)          grid=(seq,1,1) block=256
  rope_fwd(q, k, seq, nh, dh, cache_pos, cos, sin, max_seq)  grid=(seq,nh,1) block=128（就地）
  moe_route_fwd(logits,bias,n_experts,top_k,out_ids,out_w,out_stride)  grid=(seq,1,1) block=32

用法: python fwd_suite.py <src.cu>
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

dev = "cuda"


def P_(t):
    return (t.data_ptr(), ctypes.c_void_p)


def I(v):
    return (int(v), ctypes.c_int)


def Fv(v):
    return (float(v), ctypes.c_float)


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)


def t_rmsnorm(cu, m):
    seq, d, eps = 64, 256, 1e-5
    torch.manual_seed(0)
    x = torch.randn(seq, d, device=dev)
    w = torch.randn(d, device=dev)
    out = torch.zeros_like(x)
    cu.launch(m, "rmsnorm_fwd", (seq, 1, 1), (256, 1, 1),
              [P_(out), P_(x), P_(w), I(seq), I(d), Fv(eps)], shared=1024)
    cu.sync()
    ref = x * torch.rsqrt((x * x).mean(1, keepdim=True) + eps) * w
    return rel(out, ref), "seq=%d d=%d" % (seq, d)


def t_silu_mul(cu, m):
    n = 64 * 512
    torch.manual_seed(1)
    a0 = torch.randn(n, device=dev)
    b = torch.randn(n, device=dev)
    a = a0.clone()
    cu.launch(m, "silu_mul_fwd", ((n + 255) // 256, 1, 1), (256, 1, 1), [P_(a), P_(b), I(n)])
    cu.sync()
    ref = F.silu(a0) * b
    return rel(a, ref), "n=%d" % n


def t_moe_add(cu, m):
    seq, d, blk, wgt = 64, 256, 128, 0.37
    torch.manual_seed(2)
    out0 = torch.randn(seq, d, device=dev)
    e = torch.randn(seq, d, device=dev)
    out = out0.clone()
    cu.launch(m, "moe_add_fwd", (seq, (d + blk - 1) // blk, 1), (blk, 1, 1),
              [P_(out), P_(e), Fv(wgt), I(seq), I(d), I(blk)])
    cu.sync()
    return rel(out, out0 + wgt * e), "seq=%d d=%d" % (seq, d)


def t_emb(cu, m):
    seq, d, vocab, unk = 128, 128, 1000, 7
    torch.manual_seed(3)
    ids = torch.randint(0, vocab, (seq,), device=dev, dtype=torch.int32)
    ids[3] = -1
    ids[5] = vocab + 10
    emb = torch.randn(vocab, d, device=dev)
    out = torch.zeros(seq, d, device=dev)
    cu.launch(m, "emb_gather_fwd", (seq, 1, 1), (256, 1, 1),
              [P_(out), P_(emb), P_(ids), I(seq), I(d), I(vocab), I(unk)])
    cu.sync()
    safe = ids.clone()
    safe[(safe < 0) | (safe >= vocab)] = unk
    return rel(out, emb[safe.long()]), "seq=%d d=%d（含越界 id）" % (seq, d)


def t_rope(cu, m):
    seq, nh, dh, max_seq, pos0 = 32, 4, 64, 512, 5
    torch.manual_seed(4)
    q0 = torch.randn(seq, nh, dh, device=dev)
    k0 = torch.randn(seq, nh, dh, device=dev)
    cos = torch.rand(max_seq, dh // 2, device=dev) * 1.5
    sin = torch.rand(max_seq, dh // 2, device=dev) * 0.5

    q, k = q0.clone(), k0.clone()
    cu.launch(m, "rope_fwd", (seq, nh, 1), (128, 1, 1),
              [P_(q), P_(k), I(seq), I(nh), I(dh), I(pos0), P_(cos), P_(sin), I(max_seq)])
    cu.sync()

    def ref_rot(t):
        o = t.clone()
        for p in range(seq):
            c = cos[pos0 + p]
            s = sin[pos0 + p]
            blk = o[p]
            ev = blk[:, 0::2].clone()
            od = blk[:, 1::2].clone()
            blk[:, 0::2] = ev * c - od * s
            blk[:, 1::2] = ev * s + od * c
        return o

    return max(rel(q, ref_rot(q0)), rel(k, ref_rot(k0))), "seq=%d nh=%d dh=%d" % (seq, nh, dh)


def t_route(cu, m):
    N, E, K = 64, 8, 2
    stride = K
    torch.manual_seed(6)
    logits = torch.randn(N, E, device=dev)
    bias = torch.randn(E, device=dev) * 0.2
    ids = torch.zeros(N, stride, device=dev, dtype=torch.int32)
    w = torch.zeros(N, stride, device=dev)
    cu.launch(m, "moe_route_fwd", (N, 1, 1), (32, 1, 1),
              [P_(logits), P_(bias), I(E), I(K), P_(ids), P_(w), I(stride)])
    cu.sync()

    vals = logits + bias
    order = torch.argsort(-vals, dim=1, stable=True)
    ref_ids = order[:, :K]
    p = torch.softmax(vals.gather(1, ref_ids), dim=1)
    e1 = rel(ids.long().float(), ref_ids.float())
    e2 = rel(w, p)
    return max(e1, e2), "N=%d E=%d K=%d" % (N, E, K)


TESTS = [("rmsnorm", t_rmsnorm), ("silu_mul", t_silu_mul), ("moe_add", t_moe_add),
         ("emb_gather", t_emb), ("rope", t_rope), ("moe_route", t_route)]


def main(src):
    code = open(src, encoding="utf-8", errors="replace").read()
    code = code.replace("#include <cuda_runtime.h>", "").replace("#include <stddef.h>", "")
    torch.cuda.init()
    cu = Cuda()
    m = cu.compile(code, "sm_70")
    ok_all = True
    print("%-12s %-8s %-10s %s" % ("op", "status", "rel_err", "detail"))
    print("-" * 74)
    for name, fn in TESTS:
        try:
            e, detail = fn(cu, m)
            ok = e < 2e-4
            ok_all = ok_all and ok
            print("%-12s %-8s %-10.2e %s" % (name, "PASS" if ok else "FAIL", e, detail))
        except Exception as ex:
            ok_all = False
            print("%-12s %-8s %-10s %s: %s" % (name, "ERROR", "-", type(ex).__name__, str(ex)[:90]))
    print("-" * 74)
    print("SUMMARY ok=%s" % ok_all)
    return 0 if ok_all else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

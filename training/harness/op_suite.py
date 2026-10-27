# -*- coding: utf-8 -*-
"""TGAI GO 反向算子测试台：逐算子对拍 + 测速。

覆盖引擎 cuda/tgai_cuda.cu 里的前向算子对应的反向：
  rmsnorm_bwd / silu_mul_bwd / moe_add_bwd / emb_gather_bwd / rope_bwd / residual_bwd

被测源码必须 device-only（只含 __global__ / __device__），ABI 见各 OPS 定义。
用法:
    python op_suite.py <src.cu> <op> [op2 ...]
    python op_suite.py <src.cu> all
"""
import ctypes
import sys
import time

import torch
import torch.nn.functional as F

from tgai_nvrtc import Cuda

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

f32 = torch.float32
dev = "cuda"


def P(t):
    return (t.data_ptr(), ctypes.c_void_p)


def I(v):
    return (int(v), ctypes.c_int)


def Fv(v):
    return (float(v), ctypes.c_float)


def err(a, b):
    return (a - b).abs().max().item()


# ────────────────────────── 各算子 ──────────────────────────
def test_rmsnorm(cu, mod):
    seq, d, eps = 128, 256, 1e-5
    torch.manual_seed(0)
    x = torch.randn(seq, d, device=dev)
    w = torch.randn(d, device=dev)
    dy = torch.randn(seq, d, device=dev)

    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    ms = (xr * xr).mean(dim=1, keepdim=True)
    y = xr * torch.rsqrt(ms + eps) * wr
    (y * dy).sum().backward()

    dx = torch.zeros_like(x)
    dw = torch.zeros_like(w)
    cu.launch(mod, "rmsnorm_bwd", (seq, 1, 1), (256, 1, 1),
              [P(dy), P(x), P(w), P(dx), P(dw), I(seq), I(d), Fv(eps)],
              shared=256 * 4)
    cu.sync()
    e1, e2 = err(dx, xr.grad), err(dw, wr.grad)
    return e1, e2, "dx=%.2e dw=%.2e" % (e1, e2)


def test_silu_mul(cu, mod):
    n = 128 * 512
    torch.manual_seed(1)
    a = torch.randn(n, device=dev)
    b = torch.randn(n, device=dev)
    g = torch.randn(n, device=dev)

    ar = a.clone().requires_grad_(True)
    br = b.clone().requires_grad_(True)
    y = F.silu(ar) * br
    (y * g).sum().backward()

    da = torch.zeros_like(a)
    db = torch.zeros_like(b)
    cu.launch(mod, "silu_mul_bwd", ((n + 255) // 256, 1, 1), (256, 1, 1),
              [P(g), P(a), P(b), P(da), P(db), I(n)])
    cu.sync()
    e1, e2 = err(da, ar.grad), err(db, br.grad)
    return e1, e2, "da=%.2e db=%.2e" % (e1, e2)


def test_moe_add(cu, mod):
    seq, d = 128, 256
    torch.manual_seed(2)
    e = torch.randn(seq, d, device=dev)
    g = torch.randn(seq, d, device=dev)
    wgt = 0.37

    er = e.clone().requires_grad_(True)
    out = torch.zeros(seq, d, device=dev) + wgt * er
    (out * g).sum().backward()

    de = torch.zeros_like(e)
    dwgt = torch.zeros(1, device=dev)
    blk = 128
    cu.launch(mod, "moe_add_bwd", (seq, (d + blk - 1) // blk, 1), (blk, 1, 1),
              [P(g), P(e), P(de), P(dwgt), Fv(wgt), I(seq), I(d), I(blk)])
    cu.sync()
    e1 = err(de, er.grad)
    e2 = abs(dwgt.item() - (g * e).sum().item())
    return e1, e2, "de=%.2e dweight=%.2e" % (e1, e2)


def test_emb_gather(cu, mod):
    seq, d, vocab = 256, 128, 1000
    torch.manual_seed(3)
    ids = torch.randint(0, vocab, (seq,), device=dev, dtype=torch.int32)
    emb = torch.randn(vocab, d, device=dev)
    g = torch.randn(seq, d, device=dev)

    er = emb.clone().requires_grad_(True)
    out = er[ids.long()]
    (out * g).sum().backward()

    demb = torch.zeros_like(emb)
    cu.launch(mod, "emb_gather_bwd", (seq, 1, 1), (256, 1, 1),
              [P(g), P(ids), P(demb), I(seq), I(d)])
    cu.sync()
    e1 = err(demb, er.grad)
    return e1, 0.0, "demb=%.2e" % e1


def test_rope(cu, mod):
    seq, nh, dh, max_seq, cache_pos = 64, 4, 64, 512, 0
    torch.manual_seed(4)
    q = torch.randn(seq, nh, dh, device=dev)
    k = torch.randn(seq, nh, dh, device=dev)
    cos = torch.randn(max_seq, dh // 2, device=dev) * 0.1 + 1.0
    sin = torch.randn(max_seq, dh // 2, device=dev) * 0.1

    # 参考前向（与引擎一致：按 (2i,2i+1) 成对旋转）
    def rot(t, inverse=False):
        out = t.clone()
        for pos in range(seq):
            c = cos[pos]
            s = sin[pos]
            o = out[pos]                                   # [nh, dh]
            ev = o[:, 0::2].clone()
            od = o[:, 1::2].clone()
            if not inverse:
                o[:, 0::2] = ev * c - od * s
                o[:, 1::2] = ev * s + od * c
            else:
                o[:, 0::2] = ev * c + od * s
                o[:, 1::2] = -ev * s + od * c
        return out

    dq = torch.randn(seq, nh, dh, device=dev)
    dk = torch.randn(seq, nh, dh, device=dev)
    dq_ref = rot(dq, inverse=True)
    dk_ref = rot(dk, inverse=True)

    dq_c = dq.clone()
    dk_c = dk.clone()
    cu.launch(mod, "rope_bwd", (seq, nh, 1), (128, 1, 1),
              [P(dq_c), P(dk_c), I(seq), I(nh), I(dh), I(cache_pos), P(cos), P(sin), I(max_seq)])
    cu.sync()
    e1 = err(dq_c, dq_ref)
    e2 = err(dk_c, dk_ref)
    return e1, e2, "dq=%.2e dk=%.2e" % (e1, e2)


def test_residual(cu, mod):
    n = 4096
    torch.manual_seed(5)
    g = torch.randn(n, device=dev)
    dx = torch.zeros(n, device=dev)
    cu.launch(mod, "residual_bwd", ((n + 255) // 256, 1, 1), (256, 1, 1), [P(g), P(dx), I(n)])
    cu.sync()
    e1 = err(dx, g)
    return e1, 0.0, "dx=%.2e" % e1


def test_moe_route(cu, mod):
    N, E, K = 256, 8, 2
    stride = K                      # 测试里用紧凑布局（引擎实际 out_stride 可能更大）
    torch.manual_seed(6)
    logits = torch.randn(N, E, device=dev)
    bias = torch.randn(E, device=dev) * 0.1
    dW = torch.randn(N, stride, device=dev)

    # 参考：top-k（并列取小下标）+ 仅对选中项 softmax
    lg = logits.clone().requires_grad_(True)
    bs = bias.clone().requires_grad_(True)
    vals = lg + bs
    order = torch.argsort(-vals, dim=1, stable=True)
    ids = order[:, :K]
    sel = vals.gather(1, ids)
    p = torch.softmax(sel, dim=1)
    (p * dW[:, :K]).sum().backward()

    ids_c = ids.to(torch.int32).contiguous()
    p_c = p.detach().contiguous()
    dlogits = torch.zeros(N, E, device=dev)
    dbias = torch.zeros(E, device=dev)
    cu.launch(mod, "moe_route_bwd", (N, 1, 1), (32, 1, 1),
              [P(dW), P(ids_c), P(p_c), P(dlogits), P(dbias),
               I(E), I(K), I(stride)])
    cu.sync()
    e1, e2 = err(dlogits, lg.grad), err(dbias, bs.grad)
    return e1, e2, "dlogits=%.2e dbias=%.2e" % (e1, e2)


OPS = {
    "rmsnorm": test_rmsnorm,
    "silu_mul": test_silu_mul,
    "moe_add": test_moe_add,
    "emb_gather": test_emb_gather,
    "rope": test_rope,
    "residual": test_residual,
    "moe_route": test_moe_route,
}

TOL = {"rmsnorm": 2e-4, "silu_mul": 2e-4, "moe_add": 2e-4,
       "emb_gather": 2e-4, "rope": 2e-4, "residual": 1e-6, "moe_route": 2e-4}


def main(src_path, names):
    src = open(src_path, encoding="utf-8", errors="replace").read()
    src = src.replace("#include <cuda_runtime.h>", "").replace("#include <stddef.h>", "")
    torch.cuda.init()
    cu = Cuda()
    mod = cu.compile(src, "sm_70")

    ok_all = True
    print("%-12s %-10s %s" % ("op", "status", "detail"))
    print("-" * 62)
    for n in names:
        if n not in OPS:
            print("%-12s %-10s 未实现" % (n, "SKIP"))
            continue
        try:
            e1, e2, detail = OPS[n](cu, mod)
            ok = max(e1, e2) < TOL[n]
            ok_all = ok_all and ok
            print("%-12s %-10s %s" % (n, "PASS" if ok else "FAIL", detail))
        except Exception as ex:
            ok_all = False
            print("%-12s %-10s %s: %s" % (n, "ERROR", type(ex).__name__, str(ex)[:110]))
    print("-" * 62)
    print("SUMMARY ok=%s" % ok_all)
    return 0 if ok_all else 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    names = list(OPS) if sys.argv[2] == "all" else sys.argv[2:]
    sys.exit(main(sys.argv[1], names))

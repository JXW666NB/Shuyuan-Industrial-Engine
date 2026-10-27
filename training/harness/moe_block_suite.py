# -*- coding: utf-8 -*-
"""MoE FFN 块判分台：rmsnorm → top-1 路由 → 专家 SwiGLU → 残差，前向+反向。

被测源码 device-only，导出：
  extern "C" __global__ void moeblk_fwd(const float* x, const float* w1, const float* Wr,
                                        const float* Wg, const float* Wu, const float* Wd,
                                        float* out, float* cache,
                                        int seq, int d, int ff, int E, float eps);
  extern "C" __global__ void moeblk_bwd(const float* dOut, const float* x, const float* w1,
                                        const float* Wr, const float* Wg, const float* Wu,
                                        const float* Wd, const float* cache,
                                        float* dx, float* dw1, float* dWr,
                                        float* dWg, float* dWu, float* dWd,
                                        int seq, int d, int ff, int E, float eps);
启动: grid=(1,1,1) block=(1,1,1)（单线程串行即可，规模很小）
cache: 调用方分配 8192 个 float，布局由被测实现自定（前向写、反向读）
所有梯度缓冲由调用方清零，被测实现只累加。

用法:
    python moe_block_suite.py <src.cu> [seq d ff E]
输出最后一行:
    VERDICT ok=<bool> out=.. dx=.. dw1=.. dWr=.. dWg=.. dWu=.. dWd=..
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
FF = int(sys.argv[4]) if len(sys.argv) > 4 else 64
E = int(sys.argv[5]) if len(sys.argv) > 5 else 2
EPS = 1e-5
CACHE = 8192
dev = "cuda"


def P_(t):
    return (t.data_ptr(), ctypes.c_void_p)


def I(v):
    return (int(v), ctypes.c_int)


def Fv(v):
    return (float(v), ctypes.c_float)


def err(a, b):
    return (a - b).abs().max().item()


def main(src):
    code = open(src, encoding="utf-8", errors="replace").read()
    code = code.replace("#include <cuda_runtime.h>", "").replace("#include <stddef.h>", "")
    torch.cuda.init()
    cu = Cuda()
    mod = cu.compile(code, "sm_70")

    torch.manual_seed(7)
    x = torch.randn(SEQ, D, device=dev)
    w1 = torch.randn(D, device=dev)
    Wr = torch.randn(D, E, device=dev)
    Wg = torch.randn(E, D, FF, device=dev)
    Wu = torch.randn(E, D, FF, device=dev)
    Wd = torch.randn(E, FF, D, device=dev)
    dOut = torch.randn(SEQ, D, device=dev)

    # ---------- 参考 ----------
    xr = x.clone().requires_grad_(True)
    w1r = w1.clone().requires_grad_(True)
    Wrr = Wr.clone().requires_grad_(True)
    Wgr = Wg.clone().requires_grad_(True)
    Wur = Wu.clone().requires_grad_(True)
    Wdr = Wd.clone().requires_grad_(True)

    ms = (xr * xr).mean(dim=1, keepdim=True)
    h = xr * torch.rsqrt(ms + EPS) * w1r                     # [seq,d]
    logits = h @ Wrr                                         # [seq,E]
    e = logits.argmax(dim=1)                                 # [seq]
    gate = torch.sigmoid(logits.gather(1, e.unsqueeze(1))).squeeze(1)
    Wg_t, Wu_t, Wd_t = Wgr[e], Wur[e], Wdr[e]                # [seq,d,ff], [seq,ff,d]
    a = torch.einsum("td,tdf->tf", h, Wg_t)
    u = torch.einsum("td,tdf->tf", h, Wu_t)
    gu = F.silu(a) * u
    y = torch.einsum("tf,tfd->td", gu, Wd_t)
    out_ref = xr + gate.unsqueeze(1) * y
    (out_ref * dOut).sum().backward()

    # ---------- 被测 ----------
    out = torch.zeros(SEQ, D, device=dev)
    cache = torch.zeros(CACHE, device=dev)
    cu.launch(mod, "moeblk_fwd", (1, 1, 1), (1, 1, 1),
              [P_(x), P_(w1), P_(Wr), P_(Wg), P_(Wu), P_(Wd), P_(out), P_(cache),
               I(SEQ), I(D), I(FF), I(E), Fv(EPS)])
    cu.sync()

    dx = torch.zeros(SEQ, D, device=dev)
    dw1 = torch.zeros(D, device=dev)
    dWr = torch.zeros(D, E, device=dev)
    dWg = torch.zeros(E, D, FF, device=dev)
    dWu = torch.zeros(E, D, FF, device=dev)
    dWd = torch.zeros(E, FF, D, device=dev)
    cu.launch(mod, "moeblk_bwd", (1, 1, 1), (1, 1, 1),
              [P_(dOut), P_(x), P_(w1), P_(Wr), P_(Wg), P_(Wu), P_(Wd), P_(cache),
               P_(dx), P_(dw1), P_(dWr), P_(dWg), P_(dWu), P_(dWd),
               I(SEQ), I(D), I(FF), I(E), Fv(EPS)])
    cu.sync()

    checks = [
        ("out", out, out_ref.detach()),
        ("dx", dx, xr.grad),
        ("dw1", dw1, w1r.grad),
        ("dWr", dWr, Wrr.grad),
        ("dWg", dWg, Wgr.grad),
        ("dWu", dWu, Wur.grad),
        ("dWd", dWd, Wdr.grad),
    ]
    print("seq=%d d=%d ff=%d E=%d  (选中专家分布: %s)"
          % (SEQ, D, FF, E, e.cpu().tolist()))
    worst_rel = 0.0
    for name, a_, b_ in checks:
        absv = err(a_, b_)
        scale = b_.abs().max().item() + 1e-12
        relv = absv / scale
        worst_rel = max(worst_rel, relv)
        print("  %-4s abs=%.3e  rel=%.2e  (scale=%.3g)" % (name, absv, relv, scale))

    ok = worst_rel < 5e-4
    print("VERDICT ok=%s worst_rel=%.3e " % (ok, worst_rel)
          + " ".join("%s=%.1e" % (n, err(a_, b_) / (b_.abs().max().item() + 1e-12))
                     for n, a_, b_ in checks))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

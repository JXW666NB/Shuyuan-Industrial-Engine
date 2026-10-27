# -*- coding: utf-8 -*-
"""Attention 判分台：causal MHA 核心的前向 + 反向，与 PyTorch 逐元素对拍。

被测源码必须 device-only，导出：

  extern "C" __global__ void attn_fwd(const float* Q, const float* K, const float* V,
                                      float* O, float* P,
                                      int seq, int nh, int dh, float scale);
  extern "C" __global__ void attn_bwd(const float* dO, const float* Q, const float* K,
                                      const float* V, const float* P,
                                      float* dQ, float* dK, float* dV,
                                      int seq, int nh, int dh, float scale);

约定：
  Q,K,V,O,dQ,dK,dV 形状 [seq, nh, dh]，行优先；P 形状 [nh, seq, seq]
  启动: grid=(nh,1,1) block=(256,1,1) shared = seq*seq*4 字节
  前向必须同时写出 O 和 P（概率矩阵，mask 位置为 0）
  dQ/dK/dV 由调用方清零，逐元素直接写（不需要 atomic）

用法:
    python attn_suite.py <src.cu>
输出最后一行:
    VERDICT ok=<bool> O=.. P=.. dQ=.. dK=.. dV=..
"""
import ctypes
import sys

import torch

from tgai_nvrtc import Cuda

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 32
NH  = int(sys.argv[3]) if len(sys.argv) > 3 else 4
DH  = int(sys.argv[4]) if len(sys.argv) > 4 else 16
SCALE = 1.0 / (DH ** 0.5)
dev = "cuda"


def P_(t):
    return (t.data_ptr(), ctypes.c_void_p)


def I(v):
    return (int(v), ctypes.c_int)


def Fv(v):
    return (float(v), ctypes.c_float)


def err(a, b):
    return (a - b).abs().max().item()


def reference():
    torch.manual_seed(0)
    Q = torch.randn(SEQ, NH, DH, device=dev)
    K = torch.randn(SEQ, NH, DH, device=dev)
    V = torch.randn(SEQ, NH, DH, device=dev)
    dO = torch.randn(SEQ, NH, DH, device=dev)

    Qr = Q.clone().requires_grad_(True)
    Kr = K.clone().requires_grad_(True)
    Vr = V.clone().requires_grad_(True)

    scores = torch.einsum("thd,shd->hts", Qr, Kr) * SCALE      # [nh,seq,seq]
    mask = torch.tril(torch.ones(SEQ, SEQ, device=dev, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    P = torch.softmax(scores, dim=-1)
    O = torch.einsum("hts,shd->thd", P, Vr)
    (O * dO).sum().backward()
    return Q, K, V, dO, O.detach(), P.detach(), Qr.grad, Kr.grad, Vr.grad


def main(src):
    code = open(src, encoding="utf-8", errors="replace").read()
    code = code.replace("#include <cuda_runtime.h>", "").replace("#include <stddef.h>", "")
    torch.cuda.init()
    cu = Cuda()
    mod = cu.compile(code, "sm_70")

    Q, K, V, dO, O_ref, P_ref, dQ_ref, dK_ref, dV_ref = reference()

    O = torch.zeros(SEQ, NH, DH, device=dev)
    Pt = torch.zeros(NH, SEQ, SEQ, device=dev)
    smem = SEQ * SEQ * 4

    cu.launch(mod, "attn_fwd", (NH, 1, 1), (256, 1, 1),
              [P_(Q), P_(K), P_(V), P_(O), P_(Pt),
               I(SEQ), I(NH), I(DH), Fv(SCALE)], shared=smem)
    cu.sync()

    dQ = torch.zeros(SEQ, NH, DH, device=dev)
    dK = torch.zeros(SEQ, NH, DH, device=dev)
    dV = torch.zeros(SEQ, NH, DH, device=dev)
    cu.launch(mod, "attn_bwd", (NH, 1, 1), (256, 1, 1),
              [P_(dO), P_(Q), P_(K), P_(V), P_(Pt),
               P_(dQ), P_(dK), P_(dV),
               I(SEQ), I(NH), I(DH), Fv(SCALE)], shared=smem)
    cu.sync()

    eO, eP = err(O, O_ref), err(Pt, P_ref)
    eQ, eK, eV = err(dQ, dQ_ref), err(dK, dK_ref), err(dV, dV_ref)

    print("seq=%d nh=%d dh=%d scale=%.4f" % (SEQ, NH, DH, SCALE))
    print("  O   max_abs = %.3e" % eO)
    print("  P   max_abs = %.3e" % eP)
    print("  dQ  max_abs = %.3e" % eQ)
    print("  dK  max_abs = %.3e" % eK)
    print("  dV  max_abs = %.3e" % eV)

    worst = max(eO, eP, eQ, eK, eV)
    ok = worst < 2e-4
    print("VERDICT ok=%s O=%.3e P=%.3e dQ=%.3e dK=%.3e dV=%.3e" % (ok, eO, eP, eQ, eK, eV))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

# -*- coding: utf-8 -*-
"""softmax 交叉熵判分台（前向 loss + 反向 dlogits）。

被测源码 device-only，导出：
  extern "C" __global__ void ce_fwd(const float* logits, const int* targets, float* loss,
                                    int N, int V);
  extern "C" __global__ void ce_bwd(const float* logits, const int* targets, float* dlogits,
                                    int N, int V);
启动: grid=(1,1,1) block=(1,1,1)（可以串行）
语义: loss = mean_i( logsumexp(row_i) - row_i[target_i] )；dlogits = (softmax - onehot)/N
用法: python ce_suite.py <src.cu>
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


def main(src):
    code = open(src, encoding="utf-8", errors="replace").read()
    code = code.replace("#include <cuda_runtime.h>", "").replace("#include <stddef.h>", "")
    torch.cuda.init()
    cu = Cuda()
    mod = cu.compile(code, "sm_70")

    worst = 0.0
    for N, V in ((32, 128), (8, 64), (128, 256)):
        torch.manual_seed(N * 7 + V)
        logits = torch.randn(N, V, device=dev) * 2.0
        tgt = torch.randint(0, V, (N,), device=dev, dtype=torch.int32)

        lr = logits.clone().requires_grad_(True)
        loss_ref = F.cross_entropy(lr, tgt.long())
        loss_ref.backward()

        loss = torch.zeros(1, device=dev)
        dl = torch.zeros(N, V, device=dev)
        cu.launch(mod, "ce_fwd", (1, 1, 1), (1, 1, 1), [P_(logits), P_(tgt), P_(loss), I(N), I(V)])
        cu.launch(mod, "ce_bwd", (1, 1, 1), (1, 1, 1), [P_(logits), P_(tgt), P_(dl), I(N), I(V)])
        cu.sync()

        e_loss = abs(loss.item() - loss_ref.item()) / (abs(loss_ref.item()) + 1e-12)
        e_dl = (dl - lr.grad).abs().max().item() / (lr.grad.abs().max().item() + 1e-12)
        worst = max(worst, e_loss, e_dl)
        print("N=%-4d V=%-4d loss_rel=%.2e dlogits_rel=%.2e" % (N, V, e_loss, e_dl))

    ok = worst < 5e-4
    print("VERDICT ok=%s worst_rel=%.3e" % (ok, worst))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

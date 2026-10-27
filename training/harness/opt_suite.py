# -*- coding: utf-8 -*-
"""优化器判分台：Muon（含 Newton-Schulz 正交化）+ AdamW，逐元素对拍 PyTorch。

被测源码必须 device-only，导出两个 kernel：

  extern "C" __global__ void adamw_step(float* W, const float* G, float* m, float* v,
                                        float lr, float b1, float b2, float eps,
                                        float wd, int t, int n);
      grid=ceil(n/256) block=256
      语义（对齐 torch.optim.AdamW）:
        W *= (1 - lr*wd)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g^2
        mhat = m/(1-b1^t) ; vhat = v/(1-b2^t)
        W -= lr * mhat / (sqrt(vhat) + eps)

  extern "C" __global__ void muon_step(float* W, const float* G, float* B,
                                       float lr, float beta, float wd,
                                       int rows, int cols, int steps, float eps);
      grid=(1,1,1) block=(256,1,1)（整个更新在一个 block 内完成，可用静态 shared）
      语义（Muon，Keller Jordan 版）:
        B = beta*B + (1-beta)*G                     # 动量
        X = B / (||B||_F + eps)                     # 归一化
        if rows > cols: X = X^T                     # 转成 rows<=cols
        repeat steps 次:
            A  = X @ X^T                            # [r,r]
            Bq = b*A + c*(A@A)                      # b=-4.7750, c=2.0315
            X  = a*X + Bq@X                         # a=3.4445
        if rows > cols: X = X^T
        scale = sqrt(max(1, rows/cols))
        W -= lr * (X*scale + wd*W)

用法:
    python opt_suite.py <src.cu> [adamw] [muon]
输出:
    VERDICT ok=<bool> ...
"""
import ctypes
import sys

import torch

from tgai_nvrtc import Cuda

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

dev = "cuda"
f32 = torch.float32


def P(t):
    return (t.data_ptr(), ctypes.c_void_p)


def I(v):
    return (int(v), ctypes.c_int)


def Fv(v):
    return (float(v), ctypes.c_float)


def err(a, b):
    return (a - b).abs().max().item()


# ─────────── AdamW ───────────
def test_adamw(cu, mod):
    n, lr, b1, b2, eps, wd = 4096, 1e-3, 0.9, 0.999, 1e-8, 0.01
    torch.manual_seed(0)
    W0 = torch.randn(n, device=dev)
    G = torch.randn(n, device=dev)
    m0 = torch.randn(n, device=dev) * 0.01
    v0 = torch.rand(n, device=dev) * 0.01

    # 参考：torch.optim.AdamW 跑 3 步（每步换梯度）
    Wr = W0.clone().requires_grad_(True)
    opt = torch.optim.AdamW([Wr], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    opt.state[Wr]["step"] = torch.tensor(0.0)   # torch 内部会 +1 ⇒ 本步 t=1
    opt.state[Wr]["exp_avg"] = m0.clone()
    opt.state[Wr]["exp_avg_sq"] = v0.clone()
    Wr.grad = G.clone()
    opt.step()
    ref_W = Wr.detach().clone()
    ref_m = opt.state[Wr]["exp_avg"].clone()
    ref_v = opt.state[Wr]["exp_avg_sq"].clone()
    t = 1

    W = W0.clone()
    m = m0.clone()
    v = v0.clone()
    cu.launch(mod, "adamw_step", ((n + 255) // 256, 1, 1), (256, 1, 1),
              [P(W), P(G), P(m), P(v), Fv(lr), Fv(b1), Fv(b2), Fv(eps), Fv(wd), I(t), I(n)])
    cu.sync()
    e1, e2, e3 = err(W, ref_W), err(m, ref_m), err(v, ref_v)
    return max(e1, e2, e3), "W=%.2e m=%.2e v=%.2e" % (e1, e2, e3)


# ─────────── Muon ───────────
def _ns5(X, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    Y = X / (X.norm() + eps)
    tr = Y.size(0) > Y.size(1)
    if tr:
        Y = Y.T
    for _ in range(steps):
        A = Y @ Y.T
        Bq = b * A + c * (A @ A)
        Y = a * Y + Bq @ Y
    if tr:
        Y = Y.T
    return Y


def test_muon(cu, mod):
    worst = 0.0
    detail = []
    for rows, cols in ((32, 32), (64, 32)):
        lr, beta, wd, steps, eps = 0.02, 0.95, 0.0, 5, 1e-7
        torch.manual_seed(rows * 100 + cols)
        W0 = torch.randn(rows, cols, device=dev)
        G = torch.randn(rows, cols, device=dev)
        B0 = torch.randn(rows, cols, device=dev) * 0.01

        # 参考实现（严格按上面的语义）
        Br = beta * B0 + (1 - beta) * G
        Xo = _ns5(Br, steps, eps)
        scale = (max(1.0, rows / cols)) ** 0.5
        Wr = W0 - lr * (Xo * scale + wd * W0)

        W = W0.clone()
        B = B0.clone()
        cu.launch(mod, "muon_step", (1, 1, 1), (256, 1, 1),
                  [P(W), P(G), P(B), Fv(lr), Fv(beta), Fv(wd),
                   I(rows), I(cols), I(steps), Fv(eps)])
        cu.sync()
        eW, eB = err(W, Wr), err(B, Br)
        worst = max(worst, eW, eB)
        detail.append("%dx%d W=%.2e B=%.2e" % (rows, cols, eW, eB))
    return worst, " | ".join(detail)


TESTS = {"adamw": test_adamw, "muon": test_muon}


def main(src, names):
    code = open(src, encoding="utf-8", errors="replace").read()
    code = code.replace("#include <cuda_runtime.h>", "").replace("#include <stddef.h>", "")
    torch.cuda.init()
    cu = Cuda()
    mod = cu.compile(code, "sm_70")

    ok_all = True
    print("%-8s %-8s %s" % ("opt", "status", "detail"))
    print("-" * 70)
    for n in names:
        try:
            e, detail = TESTS[n](cu, mod)
            ok = e < 2e-4
            ok_all = ok_all and ok
            print("%-8s %-8s %s" % (n, "PASS" if ok else "FAIL", detail))
        except Exception as ex:
            ok_all = False
            print("%-8s %-8s %s: %s" % (n, "ERROR", type(ex).__name__, str(ex)[:110]))
    print("-" * 70)
    print("VERDICT ok=%s" % ok_all)
    return 0 if ok_all else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    ns = sys.argv[2:] if len(sys.argv) > 2 else list(TESTS)
    if "all" in ns:
        ns = list(TESTS)
    sys.exit(main(sys.argv[1], ns))

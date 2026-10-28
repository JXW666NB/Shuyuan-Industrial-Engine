# -*- coding: utf-8 -*-
"""独立判分：不写 P 的 attention（attn_fwd_nop / attn_bwd_recompute）。

刻意不复用 Grok 的脚本 —— 用我自己的接口契约和 torch 参考实现。
布局：[seq, nh, dh] fp32；grid=(nh, seq//64, 1)；block=256；shared=32KB。

用法：python attn_nop_check.py <cu路径> [seq] [nh] [dh]
"""
import ctypes
import sys
import time

import torch
import torch.nn.functional as F

from tgai_nvrtc import Cuda

dev = "cuda"
BQ = 64
SMEM = 32 * 1024


def P_(t):
    return (t.data_ptr(), ctypes.c_void_p)


def I(v):
    return (int(v), ctypes.c_int)


def Fv(v):
    return (float(v), ctypes.c_float)


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)


def ref(seq, nh, dh, seed=11):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = torch.randn(seq, nh, dh, generator=g).to(dev) * 0.5
    K = torch.randn(seq, nh, dh, generator=g).to(dev) * 0.5
    V = torch.randn(seq, nh, dh, generator=g).to(dev) * 0.5
    Q.requires_grad_(True); K.requires_grad_(True); V.requires_grad_(True)
    q = Q.permute(1, 0, 2)
    k = K.permute(1, 0, 2)
    v = V.permute(1, 0, 2)
    O = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # dO 必须按 [seq, nh, dh] 生成 —— 这是 kernel 的契约布局；
    # 参考实现里再 permute(1,0,2) 对齐到 SDPA 的 [nh, seq, dh]
    dO = torch.randn(seq, nh, dh, generator=g).to(dev)
    (O * dO.permute(1, 0, 2)).sum().backward()
    return Q.detach(), K.detach(), V.detach(), O.detach(), dO, \
        Q.grad.detach(), K.grad.detach(), V.grad.detach()


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def main(cu_path, seq=256, nh=8, dh=64):
    src = open(cu_path, encoding="utf-8").read()
    cu = Cuda()
    mod = cu.compile(src, "sm_70")
    scale = 1.0 / (dh ** 0.5)
    gy = seq // BQ

    Q, K, V, O_ref, dO, dQ_ref, dK_ref, dV_ref = ref(seq, nh, dh)
    O = torch.zeros(seq, nh, dh, device=dev)
    dQ = torch.zeros(seq, nh, dh, device=dev)
    dK = torch.zeros(seq, nh, dh, device=dev)
    dV = torch.zeros(seq, nh, dh, device=dev)

    def fwd():
        cu.launch(mod, "attn_fwd_nop", (nh, gy, 1), (256, 1, 1),
                  [P_(Q), P_(K), P_(V), P_(O), I(seq), I(nh), I(dh), Fv(scale)], shared=SMEM)

    def bwd():
        cu.launch(mod, "attn_bwd_recompute", (nh, gy, 1), (256, 1, 1),
                  [P_(dO), P_(Q), P_(K), P_(V), P_(dQ), P_(dK), P_(dV),
                   I(seq), I(nh), I(dh), Fv(scale)], shared=SMEM)

    fwd(); bwd(); cu.sync()

    rO, rQ, rK, rV = rel(O, O_ref.permute(1, 0, 2)), rel(dQ, dQ_ref), rel(dK, dK_ref), rel(dV, dV_ref)
    ok = max(rO, rQ, rK, rV) < 2e-4
    print("数值  seq=%d nh=%d dh=%d" % (seq, nh, dh))
    print("  O   rel=%.2e" % rO)
    print("  dQ  rel=%.2e" % rQ)
    print("  dK  rel=%.2e" % rK)
    print("  dV  rel=%.2e" % rV)
    print("  VERDICT ok=%s worst=%.2e" % (ok, max(rO, rQ, rK, rV)))

    # 速度：真实形状，各 3 轮取中位
    S, NH, DH = 512, 16, 64
    Q2, K2, V2, O2_ref, dO2, dQ2_ref, dK2_ref, dV2_ref = ref(S, NH, DH, seed=7)
    O2 = torch.zeros(S, NH, DH, device=dev)
    dQ2 = torch.zeros(S, NH, DH, device=dev)
    dK2 = torch.zeros(S, NH, DH, device=dev)
    dV2 = torch.zeros(S, NH, DH, device=dev)
    gy2 = S // BQ
    sc2 = 1.0 / (DH ** 0.5)

    def fwd2():
        cu.launch(mod, "attn_fwd_nop", (NH, gy2, 1), (256, 1, 1),
                  [P_(Q2), P_(K2), P_(V2), P_(O2), I(S), I(NH), I(DH), Fv(sc2)], shared=SMEM)

    def bwd2():
        cu.launch(mod, "attn_bwd_recompute", (NH, gy2, 1), (256, 1, 1),
                  [P_(dO2), P_(Q2), P_(K2), P_(V2), P_(dQ2), P_(dK2), P_(dV2),
                   I(S), I(NH), I(DH), Fv(sc2)], shared=SMEM)

    q2 = Q2.permute(1, 0, 2); k2 = K2.permute(1, 0, 2); v2 = V2.permute(1, 0, 2)
    Q3 = Q2.clone().requires_grad_(True); K3 = K2.clone().requires_grad_(True); V3 = V2.clone().requires_grad_(True)

    def tf():
        with torch.no_grad():
            F.scaled_dot_product_attention(Q3.permute(1, 0, 2), K3.permute(1, 0, 2), V3.permute(1, 0, 2), is_causal=True)

    o3 = F.scaled_dot_product_attention(Q3.permute(1, 0, 2), K3.permute(1, 0, 2), V3.permute(1, 0, 2), is_causal=True)

    def tb():
        torch.autograd.grad(o3, [Q3, K3, V3], grad_outputs=dO2.permute(1, 0, 2), retain_graph=True)

    ours_f = [bench(fwd2) for _ in range(3)]
    torch_f = [bench(tf) for _ in range(3)]
    ours_b = [bench(bwd2) for _ in range(3)]
    torch_b = [bench(tb) for _ in range(3)]
    med = lambda a: sorted(a)[1]
    print("速度  seq=%d nh=%d dh=%d" % (S, NH, DH))
    print("  fwd ours=%.4f ms  torch=%.4f ms  ratio=%.2fx  (ours %s)"
          % (med(ours_f), med(torch_f), med(ours_f) / med(torch_f),
             " / ".join("%.4f" % x for x in ours_f)))
    print("  bwd ours=%.4f ms  torch=%.4f ms  ratio=%.2fx  (ours %s)"
          % (med(ours_b), med(torch_b), med(ours_b) / med(torch_b),
             " / ".join("%.4f" % x for x in ours_b)))
    return 0 if ok else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    path = a[0]
    s = int(a[1]) if len(a) > 1 else 256
    nh = int(a[2]) if len(a) > 2 else 8
    dh = int(a[3]) if len(a) > 3 else 64
    sys.exit(main(path, s, nh, dh))

# -*- coding: utf-8 -*-
"""大尺寸 attention 判分台：tiled 版本（seq 128/256/512）。

被测源码 device-only，导出（签名与 attn_suite 一致，但必须能跑大 seq）：
  extern "C" __global__ void attn_fwd(const float* Q, const float* K, const float* V,
                                      float* O, float* P,
                                      int seq, int nh, int dh, float scale);
  extern "C" __global__ void attn_bwd(const float* dO, const float* Q, const float* K,
                                      const float* V, const float* P,
                                      float* dQ, float* dK, float* dV,
                                      int seq, int nh, int dh, float scale);

启动约定（与旧版不同，注意）：
  grid  = (nh, ceil(seq/BQ), 1)       —— 调用方按 BQ=64 计算，被测实现可自行解释 blockIdx
  block = (256, 1, 1)
  shared = 32KB（够放 64x64 的 score 瓦片）
  若实现需要别的 grid，可在 kernel 内用 gridDim 自适应；判分台只保证 blockIdx.y < ceil(seq/64)

  P 仍要求写出（[nh,seq,seq]），反向可复用；seq=512 时 P 为 16MB，可接受。

判据：
  正确性: rel < 2e-4（seq=128、256 各一次，对拍 torch）
  速度  : seq=512 nh=16，ours / torch(SDPA) < 3.0

用法: python attn_big_suite.py <src.cu>
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


def make(seq, nh, dh):
    s = 1.0 / (dh ** 0.5)
    torch.manual_seed(seq + nh)
    Q = torch.randn(seq, nh, dh, device=dev) * 0.5
    K = torch.randn(seq, nh, dh, device=dev) * 0.5
    V = torch.randn(seq, nh, dh, device=dev) * 0.5
    dO = torch.randn(seq, nh, dh, device=dev) * 0.5
    Qr = Q.clone().requires_grad_(True)
    Kr = K.clone().requires_grad_(True)
    Vr = V.clone().requires_grad_(True)
    qh = Qr.transpose(0, 1).unsqueeze(0)
    kh = Kr.transpose(0, 1).unsqueeze(0)
    vh = Vr.transpose(0, 1).unsqueeze(0)
    o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=s)
    (o * dO.transpose(0, 1).unsqueeze(0)).sum().backward()
    sc = (qh @ kh.transpose(-1, -2)) * s
    mask = torch.tril(torch.ones(seq, seq, device=dev, dtype=torch.bool))
    sc = sc.masked_fill(~mask, float("-inf"))
    Pref = torch.softmax(sc, dim=-1)
    return Q, K, V, dO, o[0].transpose(0, 1), Pref[0], Qr.grad, Kr.grad, Vr.grad, s


def run_candidate(cu, m, Q, K, V, dO, s):
    seq, nh, dh = Q.shape
    O = torch.zeros(seq, nh, dh, device=dev)
    Pt = torch.zeros(nh, seq, seq, device=dev)
    gy = (seq + BQ - 1) // BQ
    cu.launch(m, "attn_fwd", (nh, gy, 1), (256, 1, 1),
              [P_(Q), P_(K), P_(V), P_(O), P_(Pt), I(seq), I(nh), I(dh), Fv(s)], shared=SMEM)
    dQ = torch.zeros(seq, nh, dh, device=dev)
    dK = torch.zeros(seq, nh, dh, device=dev)
    dV = torch.zeros(seq, nh, dh, device=dev)
    cu.launch(m, "attn_bwd", (nh, gy, 1), (256, 1, 1),
              [P_(dO), P_(Q), P_(K), P_(V), P_(Pt), P_(dQ), P_(dK), P_(dV),
               I(seq), I(nh), I(dh), Fv(s)], shared=SMEM)
    cu.sync()
    return O, Pt, dQ, dK, dV


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


def main(src):
    code = open(src, encoding="utf-8", errors="replace").read()
    code = code.replace("#include <cuda_runtime.h>", "").replace("#include <stddef.h>", "")
    torch.cuda.init()
    cu = Cuda()
    m = cu.compile(code, "sm_70")

    worst = 0.0
    print("%-18s %-8s %-10s %s" % ("case", "status", "rel_err", "detail"))
    print("-" * 72)
    cases = ((128, 8, 64), (256, 8, 64))
    if len(sys.argv) > 2:
        cases = tuple((int(v), 8, 64) for v in sys.argv[2].split(","))
    for seq, nh, dh in cases:
        Q, K, V, dO, Oref, Pref, dQr, dKr, dVr, s = make(seq, nh, dh)
        try:
            O, Pt, dQ, dK, dV = run_candidate(cu, m, Q, K, V, dO, s)
            e = max(rel(O, Oref), rel(Pt, Pref), rel(dQ, dQr), rel(dK, dKr), rel(dV, dVr))
            ok = e < 2e-4
            worst = max(worst, e)
            print("%-18s %-8s %-10.2e O/P/dQ/dK/dV 全查" % ("seq=%d nh=%d" % (seq, nh),
                                                          "PASS" if ok else "FAIL", e))
        except Exception as ex:
            worst = 1e9
            print("%-18s %-8s %-10s %s: %s" % ("seq=%d nh=%d" % (seq, nh), "ERROR", "-",
                                               type(ex).__name__, str(ex)[:80]))

    # 速度: seq=512, nh=16（自证模式下跳过）
    if len(sys.argv) > 2:
        print("速度测试: SKIP（自证模式）")
        print("VERDICT ok=%s worst_rel=%.2e" % (worst < 2e-4, worst))
        return 0 if worst < 2e-4 else 1
    seq, nh, dh = 512, 16, 64
    s = 1.0 / (dh ** 0.5)
    Q = torch.randn(seq, nh, dh, device=dev) * 0.5
    K = torch.randn(seq, nh, dh, device=dev) * 0.5
    V = torch.randn(seq, nh, dh, device=dev) * 0.5
    O = torch.zeros(seq, nh, dh, device=dev)
    Pt = torch.zeros(nh, seq, seq, device=dev)
    gy = (seq + BQ - 1) // BQ
    t_ours = bench(lambda: cu.launch(m, "attn_fwd", (nh, gy, 1), (256, 1, 1),
                                     [P_(Q), P_(K), P_(V), P_(O), P_(Pt),
                                      I(seq), I(nh), I(dh), Fv(s)], shared=SMEM))

    dO = torch.randn(seq, nh, dh, device=dev) * 0.5
    dQ = torch.zeros(seq, nh, dh, device=dev)
    dK = torch.zeros(seq, nh, dh, device=dev)
    dV = torch.zeros(seq, nh, dh, device=dev)
    t_ours_b = bench(lambda: cu.launch(m, "attn_bwd", (nh, gy, 1), (256, 1, 1),
                                       [P_(dO), P_(Q), P_(K), P_(V), P_(Pt),
                                        P_(dQ), P_(dK), P_(dV),
                                        I(seq), I(nh), I(dh), Fv(s)], shared=SMEM))

    qh = Q.transpose(0, 1).unsqueeze(0)
    kh = K.transpose(0, 1).unsqueeze(0)
    vh = V.transpose(0, 1).unsqueeze(0)
    t_ref = bench(lambda: F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=s))

    def torch_bwd():
        q2 = qh.clone().requires_grad_(True)
        k2 = kh.clone().requires_grad_(True)
        v2 = vh.clone().requires_grad_(True)
        oo = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True, scale=s)
        oo.backward(dO.transpose(0, 1).unsqueeze(0))

    t_ref_b = bench(torch_bwd)

    print("-" * 72)
    print("速度 fwd seq=512 nh=16: ours=%.4f ms  torch=%.4f ms  ratio=%.2fx"
          % (t_ours, t_ref, t_ours / t_ref))
    print("速度 bwd seq=512 nh=16: ours=%.4f ms  torch=%.4f ms  ratio=%.2fx"
          % (t_ours_b, t_ref_b, t_ours_b / t_ref_b))
    ok = worst < 2e-4 and (t_ours / t_ref) < 2.0 and (t_ours_b / t_ref_b) < 2.0
    print("VERDICT ok=%s worst_rel=%.2e fwd_ratio=%.2fx bwd_ratio=%.2fx"
          % (ok, worst, t_ours / t_ref, t_ours_b / t_ref_b))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

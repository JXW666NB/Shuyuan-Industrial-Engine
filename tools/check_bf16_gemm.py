# -*- coding: utf-8 -*-
"""BF16 GEMM 验收（先对拍，后速度）。

参考：把 A、B 按与内核相同的 RTNE 收成 bf16 再抬回 fp32，做 fp32 GEMM，
和 _sgemm_bf16 比相对误差（max|diff| / max|ref|）。bf16 输入本身有舍入，
不能拿未截断的 fp32 结果去卡 1e-5。

用法：
    cd /mnt/workspace/tgai && PYTHONPATH=/mnt/workspace/tgai/python python3 tools/check_bf16_gemm.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# 门槛在 import 时读。这里不设 TG_BF16，先证明默认关。
os.environ.pop("TG_BF16", None)
os.environ.pop("TG_FP16", None)
os.environ.pop("TG_FP16_SPLIT", None)
os.environ.pop("TG_CUDA_GRAPH", None)

import tgai_go.train as T
import tgai_go.train.cuda as cu


def round_bf16(a):
    """与 cast_f32_to_bf16 同一条 RTNE，再左移 16 位回到 fp32。"""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    hi = ((u + bias) >> np.uint32(16)).astype(np.uint32)
    return (hi << np.uint32(16)).view(np.float32)


def rel(got, ref):
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    return float(np.max(np.abs(got - ref)) / max(float(np.max(np.abs(ref))), 1e-12))


def gemm_bf16(rt, A, B):
    """行优先 C = A@B，走 _sgemm_bf16 的列主序约定（和 sgemm 相同）。"""
    M, K = int(A.shape[0]), int(A.shape[1])
    N = int(B.shape[1])
    C = rt.empty((M, N), dtype=np.float32)
    rt._mark_submitted(A.ptr)
    rt._mark_submitted(B.ptr)
    rt._mark_submitted(C.ptr)
    rt._sgemm_bf16(cu.CUDA_OP_N, cu.CUDA_OP_N, N, M, K, B.ptr, N, A.ptr, K, C.ptr, N)
    return C


def bench(rt, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    rt.sync()
    return (time.perf_counter() - t0) / float(iters) * 1e3


def main():
    bad = 0
    T.set_default_device("cuda")
    rt = T._cuda()
    info = rt.device_info() if hasattr(rt, "device_info") else {}
    print("device %s" % (info.get("name") or "CUDA"))
    print("CUDA_R_16BF=%d CUDA_R_16F=%d COMPUTE_32F=%d" % (
        cu.CUDA_R_16BF, cu.CUDA_R_16F, cu.CUBLAS_COMPUTE_32F))
    print("default TG_BF16 on=%s MAX_K=%d MIN_OUTPUT=%d" % (
        rt._bf16_state()["on"], cu._BF16_MAX_K, cu._BF16_MIN_OUTPUT))
    if rt._bf16_state()["on"]:
        print("FAIL default switch is on")
        return 1

    rng = np.random.RandomState(20261030)

    # ---- cast 逐位 ----
    raw = rng.randn(10007).astype(np.float32)
    raw[0] = 0.0
    raw[1] = -0.0
    raw[2] = np.float32(1.0)
    raw[3] = np.float32(-2.5)
    raw[4] = np.float32(65504.0)
    raw[5] = np.float32(1e-20)
    src = rt.h2d(raw)
    got_u = np.asarray(rt.d2h(rt._cast_flat_ptr_bf16(src.ptr, int(src.n)))).reshape(-1)
    exp_u = round_bf16(raw).view(np.uint32) >> np.uint32(16)
    exp_u = exp_u.astype(np.uint16)
    mismatch = int(np.count_nonzero(got_u.view(np.uint16).reshape(-1) != exp_u))
    # d2h 的 dtype 就是 uint16，view 再比一次，避免 dtype 解释不一样。
    mismatch = int(np.count_nonzero(np.asarray(got_u, np.uint16) != exp_u))
    print("CAST n=%d bits_mismatch=%d %s" % (raw.size, mismatch, "PASS" if mismatch == 0 else "FAIL"))
    if mismatch:
        bad += 1
        print("  first got", got_u[:8], "exp", exp_u[:8])

    def check_shape(m, n, k, scale=0.02):
        A = (rng.randn(m, k).astype(np.float32) * np.float32(scale))
        B = (rng.randn(k, n).astype(np.float32) * np.float32(scale))
        ref = round_bf16(A) @ round_bf16(B)
        Ag, Bg = rt.h2d(A), rt.h2d(B)
        got = np.asarray(rt.d2h(gemm_bf16(rt, Ag, Bg)))
        # 同一套截断输入再走引擎 fp32，分开报告（判据以 numpy fp32 参考为准）。
        rt._bf16_state()["on"] = False
        via_fp32 = np.asarray(rt.d2h(rt.sgemm(rt.h2d(round_bf16(A)), rt.h2d(round_bf16(B)))))
        err = rel(got, ref)
        err_c = rel(got, via_fp32)
        nan = int(np.isnan(got).sum())
        ok = err <= 1e-5 and nan == 0
        print("GEMM MNK=%dx%dx%d rel_numpy=%.3e rel_cublas32=%.3e nan=%d %s" % (
            m, n, k, err, err_c, nan, "PASS" if ok else "FAIL"))
        return 0 if ok else 1

    print("--- correctness (before speed) ---")
    # 小形状先对拍，大形状是验收指定的两个。
    for shp in ((512, 512, 128), (4992, 256, 1280), (1280, 4992, 256)):
        bad += check_shape(*shp)

    # 分发：合格形状走 bf16，长 K / 捕获走 fp32。
    print("--- dispatch ---")
    m, n, k = 512, 640, 1280  # m*n=327680 >= 262144，K=1280
    A = rng.randn(m, k).astype(np.float32) * np.float32(0.02)
    B = rng.randn(k, n).astype(np.float32) * np.float32(0.02)
    Ag, Bg = rt.h2d(A), rt.h2d(B)
    direct = np.asarray(rt.d2h(gemm_bf16(rt, Ag, Bg)))
    st = rt._bf16_state()
    st["on"] = True
    dispatched = np.asarray(rt.d2h(rt.sgemm(Ag, Bg)))
    same = bool(np.array_equal(direct, dispatched))
    print("DISPATCH eligible bitwise_eq_direct=%s %s" % (same, "PASS" if same else "FAIL"))
    if not same:
        bad += 1

    # K 超门槛必须和 fp32 逐位相同。
    m2, n2, k2 = 512, 512, 2000
    A2 = rng.randn(m2, k2).astype(np.float32) * np.float32(0.02)
    B2 = rng.randn(k2, n2).astype(np.float32) * np.float32(0.02)
    st["on"] = False
    fp32_long = np.asarray(rt.d2h(rt.sgemm(rt.h2d(A2), rt.h2d(B2))))
    st["on"] = True
    via_long = np.asarray(rt.d2h(rt.sgemm(rt.h2d(A2), rt.h2d(B2))))
    same_l = bool(np.array_equal(fp32_long, via_long))
    print("DISPATCH longK=%d bitwise_eq_fp32=%s %s" % (k2, same_l, "PASS" if same_l else "FAIL"))
    if not same_l:
        bad += 1

    # 捕获期间不启用：结果必须和 fp32 逐位相同，且不能等于（或至少不走）bf16 直接结果的差异路径。
    Ag2, Bg2 = rt.h2d(A), rt.h2d(B)
    st["on"] = False
    fp32_e = np.asarray(rt.d2h(rt.sgemm(Ag2, Bg2)))
    st["on"] = True
    rt._capturing = True
    try:
        cap = np.asarray(rt.d2h(rt.sgemm(Ag2, Bg2)))
    finally:
        rt._capturing = False
    same_c = bool(np.array_equal(cap, fp32_e))
    print("DISPATCH capture bitwise_eq_fp32=%s %s" % (same_c, "PASS" if same_c else "FAIL"))
    if not same_c:
        bad += 1

    if bad:
        print("CORRECTNESS_FAIL %d" % bad)
        return 1

    print("--- speed (after correctness) ---")
    st["on"] = False
    specs = ((4992, 256, 1280), (1280, 4992, 256))
    for m, n, k in specs:
        A = rng.randn(m, k).astype(np.float32) * np.float32(0.02)
        B = rng.randn(k, n).astype(np.float32) * np.float32(0.02)
        Ag, Bg = rt.h2d(A), rt.h2d(B)
        C = rt.empty((m, n), dtype=np.float32)

        def run_fp32(Ag=Ag, Bg=Bg, C=C, m=m, n=n, k=k):
            rt._sgemm(cu.CUDA_OP_N, cu.CUDA_OP_N, n, m, k, Bg.ptr, n, Ag.ptr, k, C.ptr, n)

        def run_bf16(Ag=Ag, Bg=Bg, C=C, m=m, n=n, k=k):
            rt._sgemm_bf16(cu.CUDA_OP_N, cu.CUDA_OP_N, n, m, k, Bg.ptr, n, Ag.ptr, k, C.ptr, n)

        ms32 = bench(rt, run_fp32, 10, 50)
        ms16 = bench(rt, run_bf16, 10, 50)
        ratio = ms32 / ms16 if ms16 > 0 else float("inf")
        print("SPEED MNK=%dx%dx%d fp32=%.4f ms bf16=%.4f ms ratio=%.3fx %s" % (
            m, n, k, ms32, ms16, ratio, "PASS" if ms16 < ms32 else "FAIL"))
        if not (ms16 < ms32):
            bad += 1

    print("RESULT %s" % ("PASS" if bad == 0 else "FAIL"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

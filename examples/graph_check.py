# -*- coding: utf-8 -*-
"""CUDA Graph 验收：先 A@B 两次，再同一份权重上 graph 与普通模式对一步。"""
import os
import sys

import numpy as np

from tgai_go.train.cuda import runtime


def rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b)) / denom)


def check(name, got, ref, tol):
    err = rel(got, ref)
    ok = np.isfinite(got).all() and np.isfinite(ref).all() and err <= tol
    print("%s rel=%g %s" % (name, err, "OK" if ok else "FAIL"), flush=True)
    return ok


def test_kernels():
    rt = runtime()
    print("struct_bytes", __import__("ctypes").sizeof(
        __import__("tgai_go.train.cuda", fromlist=["_CudaKernelNodeParams"])._CudaKernelNodeParams),
          flush=True)
    if not rt.graph_arm():
        print("ARM_FAIL", rt.graph_stats(), flush=True)
        return False
    # 两个不同输出的逐元素加，用来确认捕获没有把参数模板串成同一份。
    n = 1024
    a = rt.h2d(np.arange(n, dtype=np.float32))
    b = rt.h2d(np.ones(n, dtype=np.float32))
    c = rt.h2d(np.full(n, 3.0, np.float32))
    held = []
    for _ in range(2):
        held.append(rt.add(a, b))
        held.append(rt.add(a, c))
    del held
    rt.sync()
    if not rt.graph_begin():
        print("BEGIN_FAIL", rt.graph_stats(), flush=True)
        return False
    x = rt.add(a, b)
    y = rt.add(a, c)
    if not rt.graph_end():
        print("END_FAIL", rt.graph_stats(), flush=True)
        return False
    st = rt.graph_stats()
    print("ewise_graph", st, flush=True)
    if not rt.graph_launch():
        print("LAUNCH_FAIL", rt.graph_stats(), flush=True)
        return False
    rt.sync()
    ref_x = np.arange(n, dtype=np.float32) + 1.0
    ref_y = np.arange(n, dtype=np.float32) + 3.0
    ok = check("ewise_x", rt.d2h(x), ref_x, 1e-6)
    ok = check("ewise_y", rt.d2h(y), ref_y, 1e-6) and ok

    # A@B 两次。第二次重放前改 A，确认图读的是活数据而不是烤死的结果。
    rng = np.random.RandomState(0)
    ah = rng.randn(64, 32).astype(np.float32)
    bh = rng.randn(32, 48).astype(np.float32)
    A = rt.h2d(ah)
    B = rt.h2d(bh)
    held = []
    for _ in range(2):
        held.append(rt.sgemm(A, B))
        held.append(rt.sgemm(A, B))
    del held
    rt.sync()
    if not rt.graph_begin():
        print("SGEMM_BEGIN_FAIL", rt.graph_stats(), flush=True)
        return False
    C = rt.sgemm(A, B)
    D = rt.sgemm(A, B)
    if not rt.graph_end():
        print("SGEMM_END_FAIL", rt.graph_stats(), flush=True)
        return False
    print("sgemm_graph", rt.graph_stats(), flush=True)
    if not rt.graph_launch():
        print("SGEMM_LAUNCH_FAIL", rt.graph_stats(), flush=True)
        return False
    rt.sync()
    ref = ah @ bh
    ok = check("sgemm_C", rt.d2h(C), ref, 1e-6) and ok
    ok = check("sgemm_D", rt.d2h(D), ref, 1e-6) and ok
    ah2 = rng.randn(64, 32).astype(np.float32)
    hold = np.ascontiguousarray(ah2, np.float32)
    rt._activate()
    rt.cu.cuMemcpyHtoDAsync(
        __import__("ctypes").c_uint64(A.ptr),
        hold.ctypes.data_as(__import__("ctypes").c_void_p),
        __import__("ctypes").c_size_t(hold.nbytes),
        rt._stream)
    A.host_hold = hold
    rt.sync()
    if not rt.graph_launch():
        print("SGEMM_RELAUNCH_FAIL", rt.graph_stats(), flush=True)
        return False
    rt.sync()
    ref2 = ah2 @ bh
    ok = check("sgemm_C_updated", rt.d2h(C), ref2, 1e-6) and ok
    ok = check("sgemm_D_updated", rt.d2h(D), ref2, 1e-6) and ok
    return ok


def _copy_into(rt, gpu, arr):
    import ctypes
    hold = np.ascontiguousarray(arr, dtype=np.float32)
    rt._activate()
    rt._chk(rt.cu.cuMemcpyHtoDAsync(
        ctypes.c_uint64(gpu.ptr),
        hold.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(int(hold.nbytes)),
        rt._stream), "cuMemcpyHtoDAsync(restore)")
    gpu.host_hold = hold


def _restore(rt, params, snap, opt):
    for k, arr in snap.items():
        _copy_into(rt, params[k]._gpu, arr)
    for buf in list(opt.m) + list(opt.v):
        rt._memset32(buf, 0)
    rt.sync()
    opt.t = 0
    for p in opt.params:
        p.grad = None


def _one(exam, params, x, y, cfg, opt, numpy_target):
    logits = exam.forward(params, x, cfg)
    tgt = y.reshape(-1)
    flat = logits.reshape(cfg["S"] * x.shape[0], cfg["V"])
    if numpy_target:
        loss = exam.F.cross_entropy(flat, tgt, ignore_index=-100)
    else:
        loss = exam.F.cross_entropy(flat, exam.T.tensor(tgt), ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss


def test_step():
    import importlib
    exam = importlib.import_module("exam2_chat10m_tgai")
    exam.T.set_default_device("cuda")
    rt = exam.T._cuda()
    if not rt.graph_arm():
        print("STEP_ARM_FAIL", rt.graph_stats(), flush=True)
        return False
    text, src = exam.load_corpus()
    _toks, stoi = exam.build_vocab(text)
    x, y = exam.build_examples(text, stoi, exam.CFG["MAXLEN"])
    if x is None:
        print("NO_CORPUS", src, flush=True)
        return False
    cfg = dict(exam.CFG, V=len(_toks), S=int(x.shape[1]))
    params = exam.make_model(cfg, np.random.RandomState(0))
    opt = exam.T.optim.AdamW(list(params.values()), lr=2e-3)
    snap = {k: np.array(v.to_numpy(), copy=True) for k, v in params.items()}
    loss_e = _one(exam, params, x, y, cfg, opt, False)
    v_e = float(np.asarray(loss_e.to_numpy()).reshape(-1)[0])
    keys = [
        "token_embedding.weight",
        "blocks.0.attn.q_proj.weight",
        "blocks.0.moe.experts.0.w1.weight",
    ]
    grads = {k: np.array(params[k].grad.to_numpy(), copy=True) for k in keys}
    print("eager_loss", v_e, flush=True)
    _restore(rt, params, snap, opt)
    for i in range(3):
        _one(exam, params, x, y, cfg, opt, False)
        print("warmup", i, "pool", rt.pool_stats()["misses"], flush=True)
    _restore(rt, params, snap, opt)
    k0 = rt.graph_stats()["kernel_launches"]
    if not rt.graph_begin():
        print("STEP_BEGIN_FAIL", rt.graph_stats(), flush=True)
        return False
    try:
        loss_g = _one(exam, params, x, y, cfg, opt, True)
    except Exception as ex:
        rt.graph_abort(ex)
        print("STEP_CAPTURE_EXC", ex, flush=True)
        return False
    if not rt.graph_end():
        print("STEP_END_FAIL", rt.graph_stats(), flush=True)
        return False
    st = rt.graph_stats()
    print("step_graph", st, "capture_kernel_delta", st["kernel_launches"] - k0, flush=True)
    # 捕获不执行。用图里烤好的 bc 重放一次，loss / 梯度才是这一步的。
    if not rt.graph_launch():
        print("STEP_REPLAY_FAIL", rt.graph_stats(), flush=True)
        return False
    v_g = float(np.asarray(loss_g.to_numpy()).reshape(-1)[0])
    print("graph_loss", v_g, flush=True)
    ok = check("loss", np.array([v_g]), np.array([v_e]), 1e-5)
    for k in keys:
        g = params[k].grad
        if g is None:
            print("missing_grad", k, flush=True)
            return False
        ok = check("grad " + k, g.to_numpy(), grads[k], 1e-5) and ok
    if not (np.isfinite(v_g) and np.isfinite(v_e)):
        print("NAN", flush=True)
        return False
    return ok


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = True
    if which in ("all", "gemm"):
        ok = test_kernels() and ok
        print("KERNEL_TEST", "OK" if ok else "FAIL", flush=True)
        if which == "gemm":
            return 0 if ok else 1
    if which in ("all", "step"):
        step_ok = test_step()
        print("STEP_TEST", "OK" if step_ok else "FAIL", flush=True)
        ok = step_ok and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

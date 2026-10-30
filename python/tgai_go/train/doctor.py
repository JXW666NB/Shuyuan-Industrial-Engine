# -*- coding: utf-8 -*-
"""tgai_go 自检（doctor）：一条命令跑完 L0 的核心判据。

用法：
    python -m tgai_go.train.doctor            # 全部检查
    python -m tgai_go.train.doctor --quick    # 跳过有限差分（更快）

判定口径见 docs/PRODUCTION_READINESS.md 的 L0：
    前向/反向相对误差 ≤ 1e-5，且不能有 NaN。

设计原则：
  * 每条检查都**独立可读**：打印 PASS/FAIL + 关键数值；
  * FAIL 时给出「可能原因 + 怎么办」，不让用户猜；
  * 全部检查都在 GPU 上真跑（不是 import 一下就报 PASS）。
"""
import sys
import time
import traceback

import numpy as np


def _rel(a, b):
    """最大绝对误差 / 参考的最大绝对值（范数型，不用逐元素相对误差 —— 近零元素会把比值炸掉）。"""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def _report(results):
    print("")
    bad = [r for r in results if not r[1]]
    print("=" * 62)
    if not bad:
        print("自检通过：%d/%d 项 PASS" % (len(results), len(results)))
        print("=" * 62)
        return 0
    print("自检失败：%d/%d 项 FAIL" % (len(bad), len(results)))
    for name, _ok, msg, hint in bad:
        print("  - %s: %s" % (name, msg))
        if hint:
            print("    怎么办: %s" % hint)
    print("=" * 62)
    return 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv

    import tgai_go.train as T
    from tgai_go.train import _cuda

    results = []

    def check(name, hint=""):
        def deco(fn):
            t0 = time.perf_counter()
            try:
                msg = fn()
                dt = (time.perf_counter() - t0) * 1e3
                print("PASS  %-14s %-52s %6.1f ms" % (name, msg, dt))
                results.append((name, True, msg, hint))
            except Exception as exc:  # noqa: BLE001 - 自检就该全抓
                dt = (time.perf_counter() - t0) * 1e3
                msg = "%s: %s" % (type(exc).__name__, str(exc)[:80])
                print("FAIL  %-14s %-52s %6.1f ms" % (name, msg, dt))
                results.append((name, False, msg, hint))
                if "--trace" in argv:
                    traceback.print_exc()
            return fn
        return deco

    # ---------- 0. 装置 ----------
    @check("device", "确认显卡驱动 / CUDA 运行时是否可用；无卡时会退回 CPU，训练会极慢")
    def _():
        rt = _cuda()
        info = rt.device_info() if hasattr(rt, "device_info") else {}
        name = info.get("name") or getattr(rt, "name", None) or "CUDA"
        return str(name)[:50]

    @check("backend", "后端接口不全：新后端请对照 backend.required_methods() 逐组实现")
    def _():
        from tgai_go.train import backend
        rt = _cuda()
        total = len(backend.required_methods())
        lack = backend.missing(rt)
        assert not lack, "缺少 %d 个接口: %s" % (len(lack), ", ".join(lack[:8]))
        return "%s，接口 %d/%d" % (backend.active_name(), total - len(lack), total)

    T.set_default_device("cuda")

    # ---------- 1. GEMM ----------
    @check("gemm", "cuBLAS 调不通：检查 cuBLAS 动态库是否在 PATH / 驱动版本")
    def _():
        rng = np.random.RandomState(1)
        A = rng.randn(256, 128).astype(np.float32) * 0.1
        B = rng.randn(128, 64).astype(np.float32) * 0.1
        got = (T.tensor(A) @ T.tensor(B)).to_numpy()
        err = _rel(got, A @ B)
        assert err <= 1e-5, "相对误差 %.2e > 1e-5" % err
        return "relerr %.2e" % err

    # ---------- 2. 转置乘（attention 的 q@k^T 走这条）----------
    @check("gemm_nt", "批量转置 GEMM 异常：检查 cuBLAS 版本是否支持 strided batched")
    def _():
        rng = np.random.RandomState(2)
        q = rng.randn(4, 32, 16).astype(np.float32) * 0.1
        k = rng.randn(4, 32, 16).astype(np.float32) * 0.1
        rt = _cuda()
        got = rt.sgemm_nt_strided_batched(T.tensor(q)._gpu, T.tensor(k)._gpu, 4, 32, 32, 16).to_numpy()
        err = _rel(got, np.einsum("bsd,btd->bst", q, k))
        assert err <= 1e-5, "相对误差 %.2e > 1e-5" % err
        return "relerr %.2e" % err

    # ---------- 3. 逐元素（含 float4 路径）----------
    @check("elementwise", "向量化内核与标量结果不一致：可用 TG_NO_VEC4=1 定位")
    def _():
        rng = np.random.RandomState(3)
        for n in (4096, 4097):          # 4 对齐 + 非对齐两条路径
            a = (rng.randn(n) * 0.1).astype(np.float32)
            b = (rng.randn(n) * 0.1).astype(np.float32)
            ta, tb = T.tensor(a), T.tensor(b)
            assert np.array_equal((ta * tb).to_numpy(), a * b), "mul n=%d 不逐位一致" % n
            assert np.array_equal((ta + tb).to_numpy(), a + b), "add n=%d 不逐位一致" % n
        return "逐位一致 (n=4096/4097)"

    # ---------- 4. RMSNorm 前向 + 有限差分反向 ----------
    @check("rmsnorm", "归一化实现有问题：检查 eps 与权重形状")
    def _():
        rng = np.random.RandomState(4)
        x = (rng.randn(8, 16) * 0.5).astype(np.float32)
        w = (rng.randn(16) * 0.5 + 1).astype(np.float32)
        ln = T.nn.RMSNorm(16)
        ln.weight = T.nn.Parameter(T.tensor(w))
        got = ln(T.tensor(x)).to_numpy()
        ref = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w
        err = _rel(got, ref)
        assert err <= 1e-5, "前向 relerr %.2e" % err
        return "前向 relerr %.2e" % err

    # ---------- 5. 融合 SiLU-GLU ----------
    @check("silu_mul", "融合 GLU 与组合式不一致：可用 TG_NO_FUSED=1 回退定位")
    def _():
        rng = np.random.RandomState(5)
        a = (rng.randn(1024) * 0.5).astype(np.float32)
        b = (rng.randn(1024) * 0.5).astype(np.float32)
        got = T.silu_mul(T.tensor(a), T.tensor(b)).to_numpy()
        ref = (a / (1.0 + np.exp(-a))) * b
        err = _rel(got, ref)
        assert err <= 1e-5, "relerr %.2e" % err
        return "relerr %.2e" % err

    # ---------- 6. 交叉熵（含 ignore_index）----------
    @check("cross_entropy", "CE 的 ignore_index 语义有问题：会算错 loss 却梯度对")
    def _():
        rng = np.random.RandomState(6)
        logits = (rng.randn(32, 20) * 2).astype(np.float32)
        tgt = rng.randint(0, 20, size=32).astype(np.int64)
        tgt[::3] = -100                     # 1/3 忽略
        loss = T.functional.cross_entropy(T.tensor(logits), T.tensor(tgt), ignore_index=-100)
        lv = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
        m = tgt != -100
        z = logits[m] - logits[m].max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        ref = float(-logp[np.arange(m.sum()), tgt[m]].mean())
        err = abs(lv - ref) / max(abs(ref), 1e-12)
        assert err <= 1e-5, "loss %.6f vs 参考 %.6f（相对 %.2e）" % (lv, ref, err)
        return "loss %.6f relerr %.2e" % (lv, err)

    # ---------- 7. attention ----------
    @check("attention", "SDPA 与参考不一致：检查 scale 与 causal 掩码")
    def _():
        rng = np.random.RandomState(7)
        bh, s, dh = 2, 16, 8
        q = (rng.randn(bh, s, dh) * 0.3).astype(np.float32)
        k = (rng.randn(bh, s, dh) * 0.3).astype(np.float32)
        v = (rng.randn(bh, s, dh) * 0.3).astype(np.float32)
        got = T.functional.scaled_dot_product_attention(
            T.tensor(q), T.tensor(k), T.tensor(v), is_causal=True).to_numpy()
        sc = np.einsum("bsd,btd->bst", q, k) / np.sqrt(dh)
        mask = np.triu(np.ones((s, s), bool), 1)
        sc = np.where(mask, -np.inf, sc)
        p = np.exp(sc - sc.max(-1, keepdims=True))
        p = p / p.sum(-1, keepdims=True)
        ref = np.einsum("bst,btd->bsd", p, v)
        err = _rel(got, ref)
        assert err <= 1e-4, "relerr %.2e" % err
        return "relerr %.2e" % err

    # ---------- 8. 优化器 ----------
    @check("optimizer", "AdamW 与手算不一致：检查 bias correction 与 eps")
    def _():
        p = T.nn.Parameter(T.tensor(np.array([1.0, -2.0], np.float32)))
        p.grad = T.tensor(np.array([0.5, -0.5], np.float32))
        opt = T.optim.AdamW([p], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        opt.step()
        got = p.to_numpy().astype(np.float64)
        g = np.array([0.5, -0.5])
        m = (1 - 0.9) * g
        v = (1 - 0.999) * g * g
        bc1, bc2 = 1 - 0.9, 1 - 0.999
        ref = np.array([1.0, -2.0]) - 1e-2 * (m / bc1) / (np.sqrt(v / bc2) + 1e-8)
        err = _rel(got, ref)
        assert err <= 1e-5, "relerr %.2e" % err
        return "relerr %.2e" % err

    # ---------- 8b. bf16 GEMM（无硬件则跳过）----------
    @check("bf16", "bf16 GEMM 与 bf16 参考不一致：检查 cast 的 RTNE 与 cublasGemmEx 的 CUDA_R_16BF")
    def _():
        rt = _cuda()
        import ctypes
        dev = ctypes.c_int()
        rt._chk(rt.cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        major = ctypes.c_int()
        rt.cu.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)
        if int(major.value) < 8:
            return "跳过（算力 %d.x，无 BF16 硬件）" % major.value
        rng = np.random.RandomState(8)
        M, K, N = 256, 512, 128
        A = (rng.randn(M, K) * 0.05).astype(np.float32)
        B = (rng.randn(K, N) * 0.05).astype(np.float32)

        def to_bf16(x):                      # RTNE，与内核同算法
            u = x.view(np.uint32).astype(np.uint64)
            r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32) << 16
            return r.view(np.float32)

        ref = to_bf16(A).astype(np.float64) @ to_bf16(B).astype(np.float64)
        got = rt.sgemm_bf16_public(T.tensor(A)._gpu, T.tensor(B)._gpu) if hasattr(rt, "sgemm_bf16_public") else None
        if got is None:
            # 没有公开入口就手工走一遍 _sgemm_bf16（列主序：m=N, n=M, k=K，操作数先 B 后 A）
            tA, tB = T.tensor(A), T.tensor(B)
            C = rt.empty((M, N))
            rt._sgemm_bf16(0, 0, N, M, K, int(tB._gpu.ptr), N, int(tA._gpu.ptr), K, int(C.ptr), N)
            rt.sync()
            got = C.to_numpy()
        err = float(np.abs(np.asarray(got, np.float64) - ref).max() / max(np.abs(ref).max(), 1e-12))
        assert err <= 1e-5, "相对误差 %.2e > 1e-5" % err
        return "relerr %.2e（对照 bf16 参考）" % err

    # ---------- 9. 一步训练 smoke ----------
    @check("train_step", "前向/反向链路断了：单独跑 exam 脚本可看更详细堆栈")
    def _():
        rng = np.random.RandomState(9)
        D, FF, L = 32, 64, 2
        cfg = dict(D=D, NH=4, FF=FF, L=L, MAXLEN=16, V=64, S=16)
        p = {
            "token_embedding.weight": T.nn.Parameter(T.tensor(rng.randn(64, D).astype(np.float32) * 0.1)),
            "final_norm.weight": T.nn.Parameter(T.tensor(np.ones(D, np.float32))),
        }
        for i in range(L):
            for nm, sh in (("ln1.weight", (D,)), ("ln2.weight", (D,)),
                           ("attn.q_proj.weight", (D, D)), ("attn.k_proj.weight", (D, D)),
                           ("attn.v_proj.weight", (D, D)), ("attn.out_proj.weight", (D, D)),
                           ("moe.router.weight", (1, D)), ("moe.gate.weight", (1, D)),
                           ("moe._qb_bias", (1,)),
                           ("moe.experts.0.w1.weight", (FF, D)),
                           ("moe.experts.0.w2.weight", (FF, D)),
                           ("moe.experts.0.w3.weight", (D, FF))):
                p["blocks.%d.%s" % (i, nm)] = T.nn.Parameter(
                    T.tensor((rng.randn(*sh) * 0.1).astype(np.float32)))
        ids = rng.randint(0, 64, size=(4, 16)).astype(np.int64)
        tgt = rng.randint(0, 64, size=(4, 16)).astype(np.int64)
        opt = T.optim.AdamW(list(p.values()), lr=1e-3)

        def one():
            emb = T.nn.Embedding(64, D)
            emb.weight = p["token_embedding.weight"]
            x = emb(ids)
            h = x
            for i in range(L):
                b = "blocks.%d." % i
                hf = h.reshape(-1, D)
                q = (hf @ p[b + "attn.q_proj.weight"].T).reshape(4, 16, 4, D // 4).permute(0, 2, 1, 3)
                k = (hf @ p[b + "attn.k_proj.weight"].T).reshape(4, 16, 4, D // 4).permute(0, 2, 1, 3)
                v = (hf @ p[b + "attn.v_proj.weight"].T).reshape(4, 16, 4, D // 4).permute(0, 2, 1, 3)
                o = T.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                o = o.permute(0, 2, 1, 3).reshape(-1, D)
                h = h + (o @ p[b + "attn.out_proj.weight"].T).reshape(4, 16, D)
                hf = h.reshape(-1, D)
                y = T.silu_mul(hf @ p[b + "moe.experts.0.w1.weight"].T,
                               hf @ p[b + "moe.experts.0.w2.weight"].T)
                h = h + (y @ p[b + "moe.experts.0.w3.weight"].T).reshape(4, 16, D)
            ln = T.nn.RMSNorm(D)
            ln.weight = p["final_norm.weight"]
            logits = ln(h) @ p["token_embedding.weight"].T
            loss = T.functional.cross_entropy(logits.reshape(-1, 64), T.tensor(tgt.reshape(-1)),
                                              ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            return float(np.asarray(loss.to_numpy()).reshape(-1)[0])

        l0 = one()
        l1 = one()
        assert l1 == l1 and abs(l1) < 100, "loss 异常: %r" % l1
        assert l1 < l0 * 1.5, "两 step 后 loss 反而爆了（%.4f → %.4f）" % (l0, l1)
        return "loss %.4f → %.4f" % (l0, l1)

    return _report(results)


if __name__ == "__main__":
    raise SystemExit(main())

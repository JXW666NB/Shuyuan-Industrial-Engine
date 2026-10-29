# -*- coding: utf-8 -*-
"""CUDA cat 对拍：任意轴，前向和反向都要和 CPU 逐位一致。

用法：python cat_parity.py
判据：前向、反向的 max|cuda-cpu| 都必须是 0。形状不一致要报错，不能写出错的结果。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "python"))

import tgai_go.train as T  # noqa: E402


def _arr(t):
    if t is None:
        return None
    if getattr(t, "device", "cpu") == "cuda" or hasattr(t, "to_numpy"):
        return np.asarray(t.to_numpy(), dtype=np.float32)
    return np.asarray(t, dtype=np.float32)


def _tensor(data, device, requires_grad=False):
    return T.Tensor(np.asarray(data, dtype=np.float32), device=device,
                    requires_grad=requires_grad)


def _run(arrays, axis, grad, device, requires=None):
    if requires is None:
        requires = [True] * len(arrays)
    ts = [_tensor(a, device, requires_grad=bool(r)) for a, r in zip(arrays, requires)]
    out = T.cat(ts, dim=axis)
    y = _arr(out)
    if grad is None:
        return y, [None] * len(ts)
    out.backward(np.asarray(grad, dtype=np.float32))
    return y, [_arr(t.grad) for t in ts]


def _max_abs(a, b):
    if a is None and b is None:
        return 0.0
    if a is None or b is None or a.shape != b.shape:
        return float("inf")
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _expect(fn, exc, needle):
    try:
        fn()
    except exc as ex:
        text = str(ex)
        if needle not in text:
            raise SystemExit("报错里没有 %r：%s" % (needle, text))
        return text.splitlines()[0]
    raise SystemExit("应该抛 %s（含 %r），但是没抛" % (exc.__name__, needle))


def main():
    rng = np.random.RandomState(0)
    fwd_worst = 0.0
    bwd_worst = 0.0
    n_fwd = 0
    n_bwd = 0
    cases = 0

    specs = [
        ("axis0-x2", 0, [(2, 3, 4), (5, 3, 4)]),
        ("axis0-x3", 0, [(1, 2), (3, 2), (4, 2)]),
        ("axis1-x2", 1, [(2, 3, 4), (2, 5, 4)]),
        ("axis1-x3", 1, [(2, 3, 4), (2, 1, 4), (2, 5, 4)]),
        ("axis2-x2", 2, [(2, 3, 4), (2, 3, 5)]),
        ("axis2-x3", 2, [(2, 3, 1), (2, 3, 4), (2, 3, 2)]),
        ("axis-1", -1, [(2, 3, 4), (2, 3, 8), (2, 3, 1)]),
        ("axis-2", -2, [(4, 2, 3), (4, 5, 3)]),
        ("1d-x3", 0, [(3,), (4,), (5,)]),
        ("mla-last", -1, [(2, 4, 16, 16), (2, 4, 16, 8)]),
        ("gqa-axis0", 0, [(1, 8, 4), (1, 8, 4), (1, 8, 4)]),
        ("zero-mid", 1, [(2, 3, 4), (2, 0, 4), (2, 5, 4)]),
        ("zero-lead", 0, [(0, 3, 4), (2, 3, 4)]),
        ("single", 1, [(3, 5)]),
        ("4d-axis1", 1, [(2, 1, 3, 4), (2, 6, 3, 4), (2, 2, 3, 4)]),
    ]

    for name, axis, shapes in specs:
        arrays = [rng.randn(*sh).astype(np.float32) for sh in shapes]
        cpu_y, _ = _run(arrays, axis, None, "cpu", requires=[False] * len(arrays))
        cuda_y, _ = _run(arrays, axis, None, "cuda", requires=[False] * len(arrays))
        d = _max_abs(cpu_y, cuda_y)
        fwd_worst = max(fwd_worst, d)
        n_fwd += int(cpu_y.size)
        g = rng.randn(*cpu_y.shape).astype(np.float32)
        _, cpu_g = _run(arrays, axis, g, "cpu")
        _, cuda_g = _run(arrays, axis, g, "cuda")
        gd = 0.0
        for a, b in zip(cpu_g, cuda_g):
            gd = max(gd, _max_abs(a, b))
            n_bwd += 0 if a is None else int(a.size)
        bwd_worst = max(bwd_worst, gd)
        cases += 1
        if d != 0.0 or gd != 0.0:
            raise SystemExit("%s 不是逐位：fwd %.3e bwd %.3e" % (name, d, gd))
        print("  %-12s fwd=%.3e bwd=%.3e shape=%s" % (name, d, gd, cpu_y.shape))

    # 非连续输入：CPU 走 numpy 的步长，CUDA 先收成逻辑顺序。两边的值必须一样。
    base = rng.randn(4, 6).astype(np.float32)
    arrays = [base.T, rng.randn(6, 2).astype(np.float32)]  # (6, 4) 非连续 + (6, 2)
    cpu_y, cpu_g = _run(arrays, 1, rng.randn(6, 6).astype(np.float32), "cpu")
    cuda_y, cuda_g = _run(arrays, 1, None, "cuda", requires=[False, False])
    # 反向单独喂同一份梯度
    g = rng.randn(*cpu_y.shape).astype(np.float32)
    _, cpu_g = _run(arrays, 1, g, "cpu")
    _, cuda_g = _run(arrays, 1, g, "cuda")
    d = _max_abs(cpu_y, cuda_y)
    gd = max(_max_abs(a, b) for a, b in zip(cpu_g, cuda_g))
    fwd_worst = max(fwd_worst, d)
    bwd_worst = max(bwd_worst, gd)
    cases += 1
    if d != 0.0 or gd != 0.0:
        raise SystemExit("非连续输入不是逐位：fwd %.3e bwd %.3e" % (d, gd))
    print("  %-12s fwd=%.3e bwd=%.3e shape=%s" % ("noncontig", d, gd, cpu_y.shape))

    # lazy .T：2D 转置不物化，cat 必须按逻辑形状读，梯度再转回去。
    x = rng.randn(3, 4).astype(np.float32)
    z = rng.randn(4, 2).astype(np.float32)
    g = rng.randn(4, 5).astype(np.float32)

    def _lazy(device):
        a = _tensor(x, device, requires_grad=True)
        b = _tensor(z, device, requires_grad=True)
        y = T.cat([a.T, b], dim=1)
        y.backward(g)
        return _arr(y), _arr(a.grad), _arr(b.grad)

    cy, cga, cgb = _lazy("cpu")
    gy, gga, ggb = _lazy("cuda")
    d = max(_max_abs(cy, gy), _max_abs(cga, gga), _max_abs(cgb, ggb))
    fwd_worst = max(fwd_worst, _max_abs(cy, gy))
    bwd_worst = max(bwd_worst, _max_abs(cga, gga), _max_abs(cgb, ggb))
    cases += 1
    if d != 0.0:
        raise SystemExit("lazy .T 不是逐位：%.3e" % d)
    print("  %-12s fwd=%.3e bwd=%.3e shape=%s" % ("lazy-T", _max_abs(cy, gy), max(_max_abs(cga, gga), _max_abs(cgb, ggb)), cy.shape))

    # 同一个张量拼两次：两段梯度要加回同一块。
    x = rng.randn(2, 3).astype(np.float32)
    g = rng.randn(4, 3).astype(np.float32)

    def _twice(device):
        a = _tensor(x, device, requires_grad=True)
        y = T.cat([a, a], dim=0)
        y.backward(g)
        return _arr(a.grad)

    d = _max_abs(_twice("cpu"), _twice("cuda"))
    bwd_worst = max(bwd_worst, d)
    cases += 1
    if d != 0.0:
        raise SystemExit("重复张量的梯度不是逐位：%.3e" % d)
    print("  %-12s bwd=%.3e" % ("dup-leaf", d))

    # 只有一部分需要梯度。另一块的 .grad 必须还是 None。
    a = rng.randn(2, 3).astype(np.float32)
    b = rng.randn(4, 3).astype(np.float32)
    g = rng.randn(6, 3).astype(np.float32)
    _, cpu_g = _run([a, b], 0, g, "cpu", requires=[True, False])
    _, cuda_g = _run([a, b], 0, g, "cuda", requires=[True, False])
    if cpu_g[1] is not None or cuda_g[1] is not None:
        raise SystemExit("不需要梯度的输入被写了 .grad：%s %s" % (cpu_g[1], cuda_g[1]))
    d = _max_abs(cpu_g[0], cuda_g[0])
    bwd_worst = max(bwd_worst, d)
    cases += 1
    if d != 0.0:
        raise SystemExit("部分 requires_grad 不是逐位：%.3e" % d)
    print("  %-12s bwd=%.3e" % ("partial-rg", d))

    msgs = []
    msgs.append(_expect(
        lambda: T.cat([
            _tensor(np.zeros((2, 3), np.float32), "cuda"),
            _tensor(np.zeros((2, 4), np.float32), "cuda"),
        ], dim=0),
        ValueError, "形状必须一致"))
    msgs.append(_expect(
        lambda: T.cat([
            _tensor(np.zeros((2, 3), np.float32), "cuda"),
            _tensor(np.zeros((4, 3), np.float32), "cuda"),
        ], dim=2),
        ValueError, "越界"))
    msgs.append(_expect(
        lambda: T.cat([
            _tensor(np.zeros((2, 3), np.float32), "cpu"),
            _tensor(np.zeros((2, 3), np.float32), "cuda"),
        ], dim=0),
        RuntimeError, "不能混用"))
    msgs.append(_expect(
        lambda: T.cat([
            _tensor(np.zeros((2, 3), np.float32), "cuda"),
            _tensor(np.zeros((2,), np.float32), "cuda"),
        ], dim=0),
        ValueError, "维度数必须相同"))
    print("  报错 %d 条都按预期：" % len(msgs))
    for m in msgs:
        print("    %s" % m)

    print("[cat 对拍] cases=%d 元素 fwd=%d bwd=%d" % (cases, n_fwd, n_bwd))
    print("[cat 对拍] forward max_abs=%.3e backward max_abs=%.3e" % (fwd_worst, bwd_worst))
    ok = fwd_worst == 0.0 and bwd_worst == 0.0
    print("  VERDICT ok=%s（必须逐位为 0）" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

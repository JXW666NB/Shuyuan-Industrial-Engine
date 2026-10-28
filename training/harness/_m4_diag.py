# -*- coding: utf-8 -*-
"""Temporary M4 numeric probe. Not part of the harness."""
import sys
import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def ns_as_is(g, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = np.array(g, dtype=np.float32, copy=True)
    print("  start", x.dtype)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    nrm = np.linalg.norm(x)
    print("  norm", type(nrm).__name__, getattr(nrm, "dtype", None), float(nrm))
    x = x / (nrm + eps)
    print("  after div", x.dtype)
    for i in range(steps):
        s = x @ x.T
        y = b * s + c * (s @ s)
        x = a * x + y @ x
        if i == 0:
            print("  iter0 s", s.dtype, "y", y.dtype, "x", x.dtype)
    if transposed:
        x = x.T
    return np.asarray(x, np.float32)


def ns_f32(g, steps=5, eps=1e-7):
    a = np.float32(3.4445)
    b = np.float32(-4.7750)
    c = np.float32(2.0315)
    eps = np.float32(eps)
    x = np.array(g, dtype=np.float32, copy=True)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T.copy()
        transposed = True
    # left-fold sum of squares in float32, then sqrt
    ssq = np.float32(0)
    flat = np.ascontiguousarray(x, dtype=np.float32).ravel()
    for v in flat:
        ssq = np.float32(ssq + np.float32(v * v))
    nrm = np.sqrt(ssq)
    x = x / np.float32(nrm + eps)
    for _ in range(steps):
        s = x @ x.T
        y = b * s + c * (s @ s)
        x = a * x + y @ x
    if transposed:
        x = x.T.copy()
    return np.asarray(x, np.float32)


def main():
    print("numpy", np.__version__)
    rng = np.random.RandomState(0)
    mats = {
        "8x8": rng.randn(8, 8).astype(np.float32),
        "4x8": rng.randn(4, 8).astype(np.float32),
        "8x4": rng.randn(8, 4).astype(np.float32),
    }
    T.set_device("cuda")
    rt = T._cuda()
    for name, g in mats.items():
        print("==", name)
        cpu = ns_as_is(g)
        f32 = ns_f32(g)
        gpu = rt.d2h(T._newton_schulz5_gpu(rt, rt.h2d(g)))
        print("  asis vs f32", rel(cpu, f32))
        print("  gpu vs asis", rel(gpu, cpu))
        print("  gpu vs f32 ", rel(gpu, f32))

    print("== gemm modes vs numpy @")
    A = mats["4x8"]
    B = mats["4x8"].T.copy()  # 8x4, so A@B is 4x4? wait A is 4x8, B should be 8xN
    B = rng.randn(8, 4).astype(np.float32)
    ref = A @ B
    ga, gb = rt.h2d(A), rt.h2d(B)
    print("  cublas", rel(rt.d2h(rt.sgemm(ga, gb)), ref))
    for mode in range(13):
        got = rt.d2h(rt.sgemm_mode(ga, gb, mode))
        print("  mode %2d" % mode, rel(got, ref), "maxabs", float(np.max(np.abs(got - ref))))

    print("== gemm_nt vs A@A.T")
    A = mats["4x8"]
    ref = A @ A.T
    ga = rt.h2d(A)
    print("  cublas nt", rel(rt.d2h(rt.gemm_nt(ga, ga)), ref))

    print("== norm")
    x = mats["4x8"]
    gx = rt.h2d(x)
    ssq = rt.read_scalar(rt.sumsq_sequential(gx))
    print("  gpu sumsq", ssq, "np.sum", float((x * x).sum()), "f64", float((x.astype(np.float64) ** 2).sum()))
    print("  gpu nrm", float(np.sqrt(np.float32(ssq))), "linalg", float(np.linalg.norm(x)))

    # actual first-step gradient from the harness
    print("== harness grad NS")
    T.set_device("cpu")
    np.random.seed(555)
    lin1 = T.nn.Linear(8, 8)
    lin2 = T.nn.Linear(8, 4)
    g = np.random.RandomState(11)
    X = (g.randn(12, 8) * 0.5).astype(np.float32)
    tgt = g.randint(0, 4, size=12).astype(np.int64)
    x = T.Tensor(X)
    y = lin2(T.functional.silu(lin1(x)))
    loss = T.functional.cross_entropy(y, tgt)
    lin1.zero_grad(); lin2.zero_grad()
    loss.backward()
    for label, w in (("lin1", lin1.weight), ("lin2", lin2.weight)):
        grad = np.array(w.grad, np.float32)
        print(label, "grad", grad.shape, grad.dtype)
        cpu = ns_as_is(grad)
        f32 = ns_f32(grad)
        T.set_device("cuda")
        rt = T._cuda()
        gpu = rt.d2h(T._newton_schulz5_gpu(rt, rt.h2d(grad)))
        T.set_device("cpu")
        print(" ", label, "gpu vs asis", rel(gpu, cpu), "gpu vs f32", rel(gpu, f32), "asis vs f32", rel(cpu, f32))


if __name__ == "__main__":
    main()

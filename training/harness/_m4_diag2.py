# -*- coding: utf-8 -*-
import sys
import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12)), float(np.max(np.abs(a - b)))


def left_sumsq(x):
    s = np.float32(0)
    for v in np.ascontiguousarray(x, np.float32).ravel():
        s = np.float32(s + np.float32(v) * np.float32(v))
    return s


def main():
    print("numpy", np.__version__, "blas", np.__config__.show())
    T.set_device("cuda")
    rt = T._cuda()
    rng = np.random.RandomState(0)

    shapes = [(8, 8), (4, 8), (8, 4), (4, 4)]
    print("== mode1 vs numpy @ and @.T")
    for shp in shapes:
        A = rng.randn(*shp).astype(np.float32)
        K = shp[1]
        B = rng.randn(K, 6).astype(np.float32)
        ref = A @ B
        got = rt.d2h(rt.sgemm_mode(rt.h2d(A), rt.h2d(B), 1))
        print("  gemm", shp, "@", B.shape, rel(got, ref))
        refT = A @ A.T
        AT = np.ascontiguousarray(A.T)
        gotT = rt.d2h(rt.sgemm_mode(rt.h2d(A), rt.h2d(AT), 1))
        print("  nt  ", shp, rel(gotT, refT))

    # actual grads
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
    lin1.zero_grad()
    lin2.zero_grad()
    loss.backward()

    print("== norms on real grads")
    T.set_device("cuda")
    rt = T._cuda()
    for label, w in (("lin1", lin1.weight), ("lin2", lin2.weight)):
        grad = np.array(w.grad, np.float32)
        n1 = np.linalg.norm(grad)
        ssq = left_sumsq(grad)
        n2 = np.sqrt(ssq)
        n3 = np.sqrt(np.sum(grad * grad))
        n4 = np.sqrt(np.dot(grad.ravel(), grad.ravel()))
        gpu = rt.read_scalar(rt.sumsq_sequential(rt.h2d(grad)))
        print(label, "shape", grad.shape)
        print("  linalg", float(n1), getattr(n1, "dtype", None))
        print("  left  ", float(n2), "sumsq", float(ssq))
        print("  npsum ", float(n3), getattr(n3, "dtype", None), "dot", float(n4), getattr(n4, "dtype", None))
        print("  gpu   ", float(np.sqrt(np.float32(gpu))), "sumsq", gpu)
        # one matmul of the normalized matrix
        x0 = grad / (n1 + np.float32(1e-7))
        x0 = np.asarray(x0, np.float32)
        ref = x0 @ x0.T
        XT = np.ascontiguousarray(x0.T)
        got = rt.d2h(rt.sgemm_mode(rt.h2d(np.ascontiguousarray(x0)), rt.h2d(XT), 1))
        cub = rt.d2h(rt.gemm_nt(rt.h2d(np.ascontiguousarray(x0)), rt.h2d(np.ascontiguousarray(x0))))
        print("  X@XT mode1", rel(got, ref), "cublas", rel(cub, ref))

    print("== momentum dtype")
    buf = np.zeros_like(lin1.weight.grad)
    mom = 0.95
    g = lin1.weight.grad
    nxt = mom * buf + g
    print("  buf*mom+g", nxt.dtype)
    u = g + mom * nxt
    print("  nesterov u", u.dtype)
    # explicit f32
    momf = np.float32(0.95)
    nxtf = momf * buf + g
    print("  explicit", nxtf.dtype, "match", np.array_equal(np.asarray(nxt, np.float32), nxtf))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import numpy as np
from _m4_deriv import exp_d


def silu_d(v):
    v = np.asarray(v, np.float32)
    neg = np.float32(v * np.float32(-1))
    s, ds_dneg = exp_d(neg)
    ds = np.float32(ds_dneg * np.float32(-1))
    den = np.float32(s + np.float32(1))
    inv = np.float32(np.float32(1) / den)
    y = np.float32(v * inv)
    dinv = np.float32(np.float32(np.float32(-inv) * inv) * ds)
    dy = np.float32(inv + np.float32(v * dinv))
    return y, dy


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 5)).astype(np.float32)
    B = rng.standard_normal((5, 3)).astype(np.float32)
    Y = A @ B

    def loss_of(M):
        s = silu_d(M)[0]
        z = np.float32((s * s).mean())
        return float(z) * 1.7 + float(M.sum()) * 0.3

    # autograd-like: dloss/dY then chain to A is not needed; compare dloss/dY vs FD of loss wrt Y
    # test A FDs A, which includes matmul. Do the full thing numerically on A.
    eps = 1e-3

    def full(Aval):
        M = Aval @ B
        s = silu_d(M)[0]
        z = float((s * s).mean())
        return z * 1.7 + float(M.sum()) * 0.3

    # analytic dloss/dM
    M = A @ B
    s, ds = silu_d(M)
    # z = mean(s^2), dz/ds = 2s / n
    n = s.size
    dloss_ds = 1.7 * (2 * s / n)
    dloss_dM = dloss_ds * ds + 0.3
    # dloss/dA = dloss_dM @ B.T
    g_auto = dloss_dM.astype(np.float64) @ B.astype(np.float64).T

    g = np.zeros_like(A, np.float64)
    flat = A.reshape(-1)
    gf = g.reshape(-1)
    base = A.copy()
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        lp = full(A)
        flat[i] = old - eps
        lm = full(A)
        flat[i] = old
        gf[i] = (lp - lm) / (2 * eps)
    err = np.abs(g_auto - g).max() / (np.abs(g).max() + 1e-12)
    print("silu chain FD", err)


if __name__ == "__main__":
    main()

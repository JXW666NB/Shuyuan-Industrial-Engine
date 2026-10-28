# -*- coding: utf-8 -*-
"""Does a smooth Taylor exp let test A's finite difference pass?"""
import numpy as np

def taylor_exp(x, n=16):
    x = np.asarray(x, np.float32)
    # c_n .. c_0 with c_k = 1/k!
    c = np.float32(1.0)
    coeffs = [c]
    for k in range(1, n + 1):
        c = np.float32(c / np.float32(k))
        coeffs.append(c)
    coeffs = list(reversed(coeffs))
    p = np.float32(coeffs[0]) + np.zeros_like(x, np.float32)
    for ck in coeffs[1:]:
        p = np.float32(np.float32(p * x) + ck)
    return p


def silu_from_exp(v, expfn):
    v = np.asarray(v, np.float32)
    s = expfn(np.float32(v * np.float32(-1)))
    inv = np.float32(np.float32(1) / np.float32(s + np.float32(1)))
    y = np.float32(v * inv)
    # derivative assuming d(exp)/dx = exp
    ds = np.float32(s * np.float32(-1))
    dinv = np.float32(np.float32(np.float32(-inv) * inv) * ds)
    dy = np.float32(inv + np.float32(v * dinv))
    return y, dy


def trial(expfn, label):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 5)).astype(np.float32)
    B = rng.standard_normal((5, 3)).astype(np.float32)
    eps = 1e-3

    def full(Aval):
        M = np.float32(Aval @ B)
        s = silu_from_exp(M, expfn)[0]
        z = float((s * s).mean())
        return z * 1.7 + float(M.sum()) * 0.3

    M = np.float32(A @ B)
    s, ds = silu_from_exp(M, expfn)
    n = s.size
    dloss_ds = np.float32(1.7) * np.float32(np.float32(2) * s / np.float32(n))
    dloss_dM = np.float32(dloss_ds * ds) + np.float32(0.3)
    g_auto = dloss_dM.astype(np.float64) @ B.astype(np.float64).T
    g = np.zeros_like(A, np.float64)
    flat = A.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        lp = full(A)
        flat[i] = old - eps
        lm = full(A)
        flat[i] = old
        g.reshape(-1)[i] = (lp - lm) / (2 * eps)
    err = np.abs(g_auto - g).max() / (np.abs(g).max() + 1e-12)
    print(label, float(err))


def np_exp(x):
    return np.exp(np.asarray(x, np.float32)).astype(np.float32)


if __name__ == "__main__":
    trial(np_exp, "np.exp")
    for n in (8, 10, 12, 14, 16, 18):
        trial(lambda x, n=n: taylor_exp(x, n), "taylor%d" % n)

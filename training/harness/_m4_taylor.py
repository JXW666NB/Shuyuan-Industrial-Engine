# -*- coding: utf-8 -*-
import numpy as np

# 1/k! for k=14..0
coeffs = [np.float32(1.0)]
for k in range(1, 15):
    coeffs.append(np.float32(coeffs[-1] / np.float32(k)))
coeffs = list(reversed(coeffs))  # c14 .. c0
print("ncoeff", len(coeffs), "c14", float(coeffs[0]))


def exp_smooth(x):
    x = np.asarray(x, np.float32)
    p = np.broadcast_to(coeffs[0], np.shape(x)).astype(np.float32).copy()
    dp = np.zeros(np.shape(x), np.float32)
    for c in coeffs[1:]:
        dp = np.float32(np.float32(dp * x) + p)
        p = np.float32(np.float32(p * x) + c)
    return p, dp


def main():
    xs = np.linspace(-2.2, 2.2, 401).astype(np.float32)
    y, dy = exp_smooth(xs)
    ref = np.exp(xs.astype(np.float64))
    print("vs real", float(np.max(np.abs(y.astype(np.float64) - ref) / ref)))
    # FD of exp
    e = np.float32(1e-3)
    errs = []
    for v in xs[::10]:
        yp = float(exp_smooth(v + e)[0])
        ym = float(exp_smooth(v - e)[0])
        fd = (yp - ym) / (2e-3)
        an = float(exp_smooth(v)[1])
        errs.append(abs(fd - an) / (abs(fd) + 1e-12))
    print("exp FD", max(errs))


if __name__ == "__main__":
    main()

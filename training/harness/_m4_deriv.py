# -*- coding: utf-8 -*-
import numpy as np

F = lambda u: np.array([u], np.uint32).view(np.float32)[0]
LOG2E = F(0x3FB8AA3B)
LN2 = F(0x3F317218)
C = (F(0x39500D01), F(0x3AB60B61), F(0x3C088889), F(0x3D2AAAAB), F(0x3E2AAAAB),
     np.float32(0.5), np.float32(1.0), np.float32(1.0))


def exp_d(x):
    x = np.asarray(x, np.float32)
    x = np.clip(x, np.float32(-80), np.float32(80))
    nf = np.rint(np.float32(x * LOG2E)).astype(np.float32)
    t = np.float32(x - np.float32(nf * LN2))
    p = np.broadcast_to(C[0], x.shape).astype(np.float32).copy()
    dp = np.zeros(x.shape, np.float32)
    for c in C[1:]:
        dp = np.float32(np.float32(dp * t) + p)
        p = np.float32(np.float32(p * t) + c)
    n = nf.astype(np.int32)
    one_bits = np.float32(1.0).view(np.int32)
    scale = (one_bits + n * np.int32(8388608)).view(np.uint32).view(np.float32)
    bits = p.view(np.uint32).view(np.int32)
    y = (bits + n * np.int32(8388608)).view(np.uint32).view(np.float32)
    dy = np.float32(dp * scale)
    return y, dy


def main():
    rng = np.random.default_rng(0)
    xs = rng.standard_normal(40).astype(np.float32)
    errs = []
    e = np.float32(1e-3)
    for v in xs:
        v = np.float32(v)
        yp = float(np.asarray(exp_d(v + e)[0]).reshape(-1)[0])
        ym = float(np.asarray(exp_d(v - e)[0]).reshape(-1)[0])
        fd = (yp - ym) / (2 * float(e))
        an = float(np.asarray(exp_d(v)[1]).reshape(-1)[0])
        errs.append(abs(fd - an) / (abs(fd) + 1e-12))
    print("exp deriv maxrel", max(errs))


if __name__ == "__main__":
    main()

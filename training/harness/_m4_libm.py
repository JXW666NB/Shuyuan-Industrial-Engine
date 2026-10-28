# -*- coding: utf-8 -*-
import sys
import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = np.max(np.abs(a - b))
    return float(d / (np.max(np.abs(b)) + 1e-12)), float(d), int(np.sum(a != b))


def main():
    T.set_device("cuda")
    rt = T._cuda()
    rng = np.random.RandomState(0)
    x = rng.randn(1000).astype(np.float32)
    gx = rt.h2d(x)
    print("exp", rel(rt.d2h(rt.exp(gx)), np.exp(x)))
    pos = np.abs(x) + np.float32(0.1)
    print("log", rel(rt.d2h(rt.log(rt.h2d(pos))), np.log(pos)))
    print("recip", rel(rt.d2h(rt.recip(rt.h2d(pos))), np.float32(1.0) / pos))
    # silu chain pieces
    neg = (-x).astype(np.float32)
    e = np.exp(neg)
    den = (e + np.float32(1.0)).astype(np.float32)
    inv = (np.float32(1.0) / den).astype(np.float32)
    chain = (x * inv).astype(np.float32)
    ge = rt.exp(rt.h2d(neg))
    # den via add broadcast? use numpy den uploaded to compare recip only
    print("exp neg", rel(rt.d2h(ge), e))
    print("recip den", rel(rt.d2h(rt.recip(rt.h2d(den))), inv))


if __name__ == "__main__":
    main()

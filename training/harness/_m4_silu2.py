# -*- coding: utf-8 -*-
import sys
import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12)), int(np.sum(np.asarray(a, np.float32) != np.asarray(b, np.float32)))


def main():
    T.set_device("cuda")
    rt = T._cuda()
    rng = np.random.RandomState(11)
    x = (rng.randn(12, 8) * 0.8).astype(np.float32)
    # numpy chain, same ops as CPU silu
    s = np.exp(x * np.float32(-1.0))
    chain = (x * ((s + np.float32(1.0)) ** np.float32(-1.0))).astype(np.float32)
    gx = rt.h2d(x)
    neg = rt.scale(gx, -1.0)
    e = rt.exp(neg)
    one = rt.h2d(np.ones_like(x))
    den = rt.add(e, one)
    inv = rt.recip(den)
    got = rt.mul(gx, inv)
    print("gpu chain vs numpy", rel(rt.d2h(got), chain))
    # also the literal CPU expression via Tensor on CPU
    T.set_device("cpu")
    t = T.Tensor(x)
    cpu = T.functional.silu(t).to_numpy()
    print("numpy expr vs functional", rel(chain, cpu))


if __name__ == "__main__":
    main()

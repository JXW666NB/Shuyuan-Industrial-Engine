# -*- coding: utf-8 -*-
import numpy as np

rng = np.random.RandomState(0)
rows = rng.randn(20, 4).astype(np.float32)
# also exp-shifted like CE
rows2 = np.exp(rows - rows.max(axis=1, keepdims=True)).astype(np.float32)


def left(r):
    s = np.float32(0)
    for v in r:
        s = np.float32(s + v)
    return s


def pair(r):
    return np.float32(np.float32(r[0] + r[1]) + np.float32(r[2] + r[3]))


def pair2(r):
    return np.float32(np.float32(np.float32(r[0] + r[1]) + r[2]) + r[3])


def tree(r):
    a = np.float32(r[0] + r[1])
    b = np.float32(r[2] + r[3])
    return np.float32(a + b)


bad = {"left": 0, "pair": 0, "pair2": 0, "tree": 0}
for name, block in (("plain", rows), ("exp", rows2)):
    for r in block:
        ref = np.sum(r)
        for k, fn in (("left", left), ("pair", pair), ("pair2", pair2), ("tree", tree)):
            if fn(r) != ref:
                bad[k] += 1
    print(name, "mismatches", bad)
    bad = {k: 0 for k in bad}

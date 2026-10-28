# -*- coding: utf-8 -*-
import numpy as np

rng = np.random.RandomState(0)
data = np.abs(rng.randn(20).astype(np.float32)) + np.float32(0.2)
g = rng.randn(20).astype(np.float32)
p = -1.0
expr = g * p * (data ** (p - 1))
print("expr dtype", expr.dtype)
cast = np.asarray(expr, np.float32)
# candidates
inv2 = np.float32(1.0) / np.float32(data * data)
a = np.float32(np.float32(g * np.float32(-1.0)) * inv2)
b = np.float32(-(g * (np.float32(1.0) / np.float32(data * data))))
c = np.float32(g * (data ** np.float32(-2)))
d = np.float32(np.float32(-g) * (data ** np.float32(-2)))
for name, v in (("a", a), ("b", b), ("c", c), ("d", d)):
    print(name, int(np.sum(v != cast)), "dtype", v.dtype)

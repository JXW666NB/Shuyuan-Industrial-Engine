# -*- coding: utf-8 -*-
import numpy as np

rng = np.random.RandomState(0)
data = np.abs(rng.randn(50).astype(np.float32)) + np.float32(0.2)
ref = data ** np.float32(-2)
cands = {
    "1/(x*x)": np.float32(1.0) / np.float32(data * data),
    "(1/x)*(1/x)": np.float32(np.float32(1.0) / data) * np.float32(np.float32(1.0) / data),
    "1/(x*x) py": (np.float32(1.0) / (data * data)).astype(np.float32),
    "recip sq": np.reciprocal(data * data),
    "pow -2.0": data ** np.float64(-2.0),
}
for name, v in cands.items():
    v = np.asarray(v, np.float32)
    print(name, int(np.sum(v != ref)))

# -*- coding: utf-8 -*-
import numpy as np

rng = np.random.RandomState(1)
data = np.abs(rng.randn(200).astype(np.float32)) + np.float32(0.05)
ref = data ** np.float32(-2)
d = data.astype(np.float64)
cands = {
    "exp log": np.float32(np.exp(np.log(d) * -2.0)),
    "1/x/x": np.float32(1.0 / d / d),
    "1/(x*x)": np.float32(1.0 / (d * d)),
    "recip64": np.float32(np.reciprocal(d * d)),
}
for name, v in cands.items():
    print(name, int(np.sum(np.asarray(v, np.float32) != ref)))

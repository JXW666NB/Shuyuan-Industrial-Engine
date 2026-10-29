import inspect
import numpy as np
import tgai_go.train as T
F = T.functional
for name in ("cross_entropy", "_cross_entropy_cpu", "linear_cross_entropy"):
    print("%-22s %s" % (name, type(getattr(type(F), name, None)).__name__))
x = T.Tensor(np.random.randn(4, 8).astype(np.float32))
w = T.Tensor(np.random.randn(8, 4).astype(np.float32))
t = np.array([1, 2, 3, 4], dtype=np.int64)
print("CPU CE:", float(np.asarray(F.cross_entropy(x, t).to_numpy())))
print("CPU 分块 CE:", float(np.asarray(F.linear_cross_entropy(x, w, t, chunk=2).to_numpy())))

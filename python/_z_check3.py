import numpy as np
import tgai_go.train as T
F = T.functional
for name in ("cross_entropy", "_cross_entropy_cpu", "linear_cross_entropy"):
    print("%-22s %s" % (name, type(getattr(type(F), name, None)).__name__))
rs = np.random.RandomState(0)
x = T.Tensor(rs.randn(4, 8).astype(np.float32))
w = T.Tensor(rs.randn(4, 8).astype(np.float32))   # [V=4, D=8]
t = np.array([0, 1, 2, 3], dtype=np.int64)
a = float(np.asarray(F.cross_entropy(x @ w.T, t).to_numpy()))
b = float(np.asarray(F.linear_cross_entropy(x, w, t, chunk=2).to_numpy()))
print("CPU 普通 CE = %.8f | 分块 CE = %.8f | 差 = %.2e" % (a, b, abs(a - b)))

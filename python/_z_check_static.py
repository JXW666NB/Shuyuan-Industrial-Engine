import inspect
import tgai_go.train as T
F = T.functional
for name in ("cross_entropy", "_cross_entropy_cpu", "linear_cross_entropy"):
    fn = getattr(type(F), name, None)
    print("%-22s %s" % (name, type(fn).__name__))
print("F.cross_entropy 签名:", inspect.signature(F.cross_entropy))
import numpy as np
x = T.Tensor(np.random.randn(4, 8).astype(np.float32))
t = np.array([1, 2, 3, 4], dtype=np.int64)
print("CPU CE 结果:", float(np.asarray(F.cross_entropy(x, t).to_numpy())))
print("分块 CE 结果:", float(np.asarray(F.linear_cross_entropy(x, T.Tensor(np.random.randn(8, 4).astype(np.float32)), t, chunk=2).to_numpy())))

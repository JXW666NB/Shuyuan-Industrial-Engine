import numpy as np
import tgai_go.train as T

x = T.Tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
print("x.shape", x.shape)
try:
    print("x[..., :2].shape", x[..., :2].shape, x[..., :2].to_numpy().reshape(2,3,2)[0,0])
except Exception as e:
    print("切片失败:", type(e).__name__, e)
try:
    h = x[..., :2]; t = x[..., 2:]
    c = T.cat([T.Tensor(np.asarray(-h.to_numpy())), t], dim=-1)
    print("cat(-h, t) shape", c.shape)
except Exception as e:
    print("cat 失败:", type(e).__name__, e)
print("neg:", np.asarray((-x).to_numpy()).reshape(-1)[:3])
print("mul+add:", np.asarray((x * 2.0 + 1.0).to_numpy()).reshape(-1)[:3])

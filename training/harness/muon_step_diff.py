# -*- coding: utf-8 -*-
"""定位 Muon 的 CPU/GPU 分歧：同一步，逐项比 buf / u / o / param。"""
import sys

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

D, H = 8, 8
SEED = 555
LR, MOM, WD = 0.02, 0.95, 0.0


def build(device):
    if device == "cuda":
        T.set_device("cuda")
    else:
        T.set_device("cpu")
    np.random.seed(SEED)
    lin = T.nn.Linear(D, H)
    g = np.random.RandomState(11)
    x = (g.randn(12, D) * 0.5).astype(np.float32)
    tgt = g.randint(0, H, size=12).astype(np.int64)
    y = lin(T.Tensor(x))
    loss = T.functional.cross_entropy(y, tgt)
    lin.zero_grad()
    loss.backward()
    return lin, loss


def npv(x):
    if x is None:
        return None
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy(), np.float64)
    if hasattr(x, "ptr"):          # GpuBuf
        return np.asarray(T._cuda().d2h(x), np.float64)
    return np.asarray(x, np.float64)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


# 两边必须拿到同一份初始权重和同一份梯度
lin_c, loss_c = build("cpu")
w0 = np.array(lin_c.weight.to_numpy(), np.float64).copy()
g_c = npv(lin_c.weight.grad).copy()
print("CPU 侧: loss=%.8f  |grad|max=%.6e" % (float(npv(loss_c)), np.abs(g_c).max()))

lin_g, loss_g = build("cuda")
# 强制同一份权重与梯度
lin_g.weight = T.Parameter(w0.astype(np.float32))
# 重建前向后梯度会变，所以直接比较 optim 内部逻辑：用同一份 g 手工跑两侧 step

# ---- CPU: 走真实 Muon.step（float32，才是真值）----
T.set_device("cpu")
p_cpu = T.Parameter(w0.astype(np.float32))
p_cpu.grad = g_c.astype(np.float32).copy()
opt_c = T.optim.Muon([p_cpu], lr=LR, momentum=MOM, nesterov=True, weight_decay=WD)
opt_c.t = 1
opt_c.step()
w_c = npv(p_cpu)
buf_c = npv(opt_c.buf[0])
u_c = None
o_c = None
rows, cols = w0.shape
scale = float(np.sqrt(max(1.0, rows / cols)))

# ---- GPU: 走真实 _step_cuda 路径 ----
T.set_device("cuda")
rt = T._cuda()
p_gpu = T.Parameter(w0.astype(np.float32))
p_gpu.grad = T.Tensor(g_c.astype(np.float32))
opt = T.optim.Muon([p_gpu], lr=LR, momentum=MOM, nesterov=True, weight_decay=WD)
opt.t = 1
opt._step_cuda(0, p_gpu)

buf_g = npv(opt.buf[0])
w_g = npv(p_gpu._gpu if hasattr(p_gpu, "_gpu") else p_gpu)

print()
print("buf   cpu vs gpu: %.3e" % rel(buf_g, buf_c))
print("param cpu vs gpu: %.3e" % rel(w_g, w_c))

# GPU 内部的 u / o：手工再算一遍设备上的
p2 = T.Parameter(w0.astype(np.float32))
p2.grad = T.Tensor(g_c.astype(np.float32))
buf2 = rt.zeros(w0.shape)
u2 = rt.empty(w0.shape)
rt.muon_momentum(p2._gpu, p2.grad._gpu, buf2, u2, np.float32(MOM), 0.0, False, True)
u_g = npv(u2)
o_g = npv(T._newton_schulz5_gpu(rt, u2))
print("u     gpu(仅参考) |u|max=%.3e" % np.abs(u_g).max())
print("o     gpu(仅参考) |o|max=%.3e" % np.abs(o_g).max())
print()
print("scale: cpu=%.10f" % scale)
print("param 绝对差最大值: %.3e" % np.max(np.abs(w_g - w_c)))

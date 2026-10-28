# -*- coding: utf-8 -*-
"""本地 CPU 冒烟：名字、平局、空专家、重复命中、有限差分。不进判分。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python")))
import tgai_go.train as T


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def main():
    T.set_device("cpu")
    T.manual_seed(0)
    m = T.nn.MoE(4, 6, 3, top_k=2, gate="softmax")
    names = [n for n, _ in m.named_parameters()]
    assert names[0] == "router.weight", names
    assert "experts.0.gate.weight" in names, names
    assert "experts.1.up.weight" in names
    assert "experts.2.down.weight" in names
    assert m.router.weight.shape == (3, 4)
    assert m.experts[0].gate.weight.shape == (6, 4)
    assert m.experts[0].up.weight.shape == (6, 4)
    assert m.experts[0].down.weight.shape == (4, 6)
    print("names", names)

    # 平局：专家 0 和 1 的 router 行完全相同，且比 2 大。top_k=1 必须选 0。
    T.manual_seed(1)
    m = T.nn.MoE(4, 4, 3, top_k=1, gate="sigmoid")
    sd = m.state_dict()
    rw = sd["router.weight"].copy()
    rw[0] = np.array([1, 0, 0, 0], np.float32)
    rw[1] = rw[0].copy()
    rw[2] = np.array([-5, 0, 0, 0], np.float32)
    sd["router.weight"] = rw
    # 两个专家输出故意差很多
    sd["experts.0.down.weight"] = np.ones((4, 4), np.float32)
    sd["experts.1.down.weight"] = np.full((4, 4), 3.0, np.float32)
    sd["experts.0.gate.weight"] = np.zeros((4, 4), np.float32)
    sd["experts.0.gate.weight"][0, 0] = 1.0
    sd["experts.1.gate.weight"] = sd["experts.0.gate.weight"].copy()
    sd["experts.0.up.weight"] = np.zeros((4, 4), np.float32)
    sd["experts.0.up.weight"][0, 0] = 1.0
    sd["experts.1.up.weight"] = sd["experts.0.up.weight"].copy()
    m.load_state_dict(sd)
    x = T.tensor(np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0.5, 0, 0, 0]], np.float32), requires_grad=True)
    y = m(x)
    # 专家 2 空，1 因平局落选
    (y * T.tensor(np.arange(12, dtype=np.float32).reshape(3, 4) * 0.1 + 0.3)).sum().backward()
    assert m.experts[1].gate.weight.grad is None
    assert m.experts[2].down.weight.grad is None
    assert m.experts[0].gate.weight.grad is not None
    # 未选中的 router 行必须是 0
    rg = np.asarray(m.router.weight.grad)
    print("sigmoid router grad rows", np.max(np.abs(rg), axis=1))
    assert np.max(np.abs(rg[1])) == 0.0
    assert np.max(np.abs(rg[2])) == 0.0
    assert np.max(np.abs(rg[0])) > 0.0
    print("tie/empty/repeat ok", y.shape, "y0", np.asarray(y.to_numpy())[0, :2])

    # 有限差分：内部权重，不改路由
    T.manual_seed(2)
    m = T.nn.MoE(4, 4, 3, top_k=1, gate="sigmoid")
    sd = m.state_dict()
    rw = np.zeros((3, 4), np.float32)
    rw[0, 0] = 4.0
    rw[1, 0] = -4.0
    rw[2, 0] = -4.0
    sd["router.weight"] = rw
    m.load_state_dict(sd)
    xnp = np.array([[0.4, -0.2, 0.1, 0.3],
                    [0.2, 0.2, -0.4, 0.1],
                    [-0.3, 0.5, 0.2, -0.1]], np.float32)

    def loss_of(sd_):
        mm = T.nn.MoE(4, 4, 3, top_k=1, gate="sigmoid")
        mm.load_state_dict(sd_)
        yy = mm(T.tensor(xnp))
        return float(np.sum(yy.to_numpy()))

    base = loss_of(sd)
    eps = np.float32(1e-3)
    key = "experts.0.down.weight"
    sd_p = {k: v.copy() for k, v in sd.items()}
    sd_m = {k: v.copy() for k, v in sd.items()}
    sd_p[key] = sd_p[key].copy(); sd_m[key] = sd_m[key].copy()
    sd_p[key][0, 0] = np.float32(sd_p[key][0, 0] + eps)
    sd_m[key][0, 0] = np.float32(sd_m[key][0, 0] - eps)
    fd = (loss_of(sd_p) - loss_of(sd_m)) / float(2 * eps)
    m.load_state_dict(sd)
    xx = T.tensor(xnp, requires_grad=True)
    yy = m(xx)
    yy.sum().backward()
    ag = float(m.experts[0].down.weight.grad[0, 0])
    print("fd down", fd, "analytic", ag, "abs", abs(fd - ag), "base", base)

    # 输入梯度
    sd_p = {k: v.copy() for k, v in sd.items()}
    # 扰动 x
    def loss_x(x_):
        mm = T.nn.MoE(4, 4, 3, top_k=1, gate="sigmoid")
        mm.load_state_dict(sd)
        return float(np.sum(mm(T.tensor(x_)).to_numpy()))
    xp = xnp.copy(); xm = xnp.copy()
    xp[0, 0] = np.float32(xp[0, 0] + eps)
    xm[0, 0] = np.float32(xm[0, 0] - eps)
    fdx = (loss_x(xp) - loss_x(xm)) / float(2 * eps)
    agx = float(xx.grad.to_numpy()[0, 0] if hasattr(xx.grad, "to_numpy") else xx.grad[0, 0])
    print("fd x", fdx, "analytic", agx, "abs", abs(fdx - agx))

    # softmax top2：专家 2 空，0 和 1 被所有 token 命中
    T.manual_seed(3)
    m = T.nn.MoE(4, 4, 3, top_k=2, gate="softmax")
    sd = m.state_dict()
    rw = np.zeros((3, 4), np.float32)
    rw[0] = 1.0
    rw[1] = 0.5
    rw[2] = -8.0
    sd["router.weight"] = rw
    m.load_state_dict(sd)
    x3 = T.tensor(np.random.RandomState(0).randn(2, 2, 4).astype(np.float32) * 0.3, requires_grad=True)
    y3 = m(x3)
    assert y3.shape == (2, 2, 4)
    (y3 * y3).sum().backward()
    assert m.experts[2].gate.weight.grad is None
    assert m.experts[0].up.weight.grad is not None
    assert m.experts[1].down.weight.grad is not None
    print("softmax 3d", y3.shape, "empty2", m.experts[2].up.weight.grad)
    print("SMOKE_OK")


if __name__ == "__main__":
    main()

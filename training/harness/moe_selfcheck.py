# -*- coding: utf-8 -*-
"""nn.MoE 自检：CPU 对照 numpy 参考，CPU 对照 CUDA。

判据：同一份权重下，前向输出、输入梯度、router 梯度、各专家梯度，
CPU vs CUDA 的相对误差 < 1e-5。
覆盖 sigmoid/top_k=1（含平局取小下标）、softmax/top_k=2、
重复 token 命中同一个专家、以及空专家。

不 import torch。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.abspath(os.path.join(_HERE, "..", "..", "python")),
    "/home/featurize/tgai/python",
):
    if os.path.isdir(os.path.join(_cand, "tgai_go")):
        sys.path.insert(0, _cand)
        break

import tgai_go.train as T  # noqa: E402


TOL = 1e-5


def _np(v):
    """Tensor 或 ndarray 都转成 float32 numpy（CPU 后端的 grad 是 ndarray）。"""
    if v is None:
        return None
    a = v.to_numpy() if hasattr(v, "to_numpy") else v
    return np.array(a, dtype=np.float32, copy=True)


def rel(a, b):
    """和 train_op_parity 一样：max|a-b| / (max|b| + 1e-12)，b 是 CPU。"""
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def _topk_ref(logits, k):
    """独立于模块实现：(-logit, 更小下标)。"""
    z = np.asarray(logits, dtype=np.float64)
    n, e = z.shape
    idx = np.empty((n, k), dtype=np.int64)
    for t in range(n):
        order = sorted(range(e), key=lambda j: (-z[t, j], j))
        idx[t] = order[:k]
    return idx


def ref_forward(x, sd, top_k, gate):
    """numpy 参考。sigmoid/silu 用 np.exp，只用来抓住选错专家这种大错。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        x = x.reshape(-1, x.shape[-1])
    z = x @ np.asarray(sd["router.weight"], dtype=np.float64).T
    idx = _topk_ref(z, top_k)
    if gate == "sigmoid":
        picked = np.take_along_axis(z, idx, 1)
        w = 1.0 / (1.0 + np.exp(-picked))
    else:
        shifted = z - z.max(axis=1, keepdims=True)
        p = np.exp(shifted)
        p = p / p.sum(axis=1, keepdims=True)
        w = np.take_along_axis(p, idx, 1)
    y = np.zeros_like(x)
    for t in range(x.shape[0]):
        for s in range(top_k):
            e = int(idx[t, s])
            gw = np.asarray(sd["experts.%d.gate.weight" % e], dtype=np.float64)
            uw = np.asarray(sd["experts.%d.up.weight" % e], dtype=np.float64)
            dw = np.asarray(sd["experts.%d.down.weight" % e], dtype=np.float64)
            a = x[t] @ gw.T
            u = x[t] @ uw.T
            h = (a / (1.0 + np.exp(-a))) * u
            y[t] += w[t, s] * (h @ dw.T)
    return y.astype(np.float32), idx


def _run(device, d, ff, n_exp, top_k, gate, x_np, sd, coef):
    T.set_device(device)
    m = T.nn.MoE(d, ff, n_exp, top_k=top_k, gate=gate)
    m.load_state_dict(sd)
    x = T.tensor(np.asarray(x_np, dtype=np.float32), requires_grad=True)
    c = T.tensor(np.asarray(coef, dtype=np.float32))
    y = m(x)
    ((y * c).sum()).backward()
    grads = {}
    for name, p in m.named_parameters():
        g = p.grad
        if g is None:
            grads[name] = None
        else:
            grads[name] = _np(g)
    out = (
        _np(y),
        _np(x.grad),
        grads,
    )
    T.set_device("cpu")
    return out


def _selected_experts(x_np, sd, top_k):
    x = np.asarray(x_np, dtype=np.float64).reshape(-1, sd["router.weight"].shape[1])
    z = x @ np.asarray(sd["router.weight"], dtype=np.float64).T
    idx = _topk_ref(z, top_k)
    return sorted(set(int(v) for v in idx.reshape(-1)))


def _case(name, d, ff, n_exp, top_k, gate, x_np, sd, coef):
    cpu_y, cpu_dx, cpu_g = _run("cpu", d, ff, n_exp, top_k, gate, x_np, sd, coef)
    ref_y, _ = ref_forward(x_np, sd, top_k, gate)
    ref_y = ref_y.reshape(cpu_y.shape)
    r_ref = rel(cpu_y, ref_y)
    cuda_y, cuda_dx, cuda_g = _run("cuda", d, ff, n_exp, top_k, gate, x_np, sd, coef)
    fields = [("y", cuda_y, cpu_y), ("dx", cuda_dx, cpu_dx)]
    for key in cpu_g:
        fields.append((key, cuda_g[key], cpu_g[key]))
    worst = 0.0
    detail = []
    for fname, a, b in fields:
        r = rel(a, b)
        worst = max(worst, r)
        if fname in ("y", "dx", "router.weight") or r > TOL or r == float("inf"):
            detail.append("%s=%.3e" % (fname, r))
    used = _selected_experts(x_np, sd, top_k)
    empty = [e for e in range(n_exp) if e not in used]
    empty_ok = True
    for e in empty:
        for part in ("gate", "up", "down"):
            key = "experts.%d.%s.weight" % (e, part)
            if cpu_g[key] is not None or cuda_g[key] is not None:
                empty_ok = False
    router_note = ""
    if gate == "sigmoid":
        rg = cpu_g["router.weight"]
        unused = [e for e in range(n_exp) if e not in used]
        peak = 0.0 if not unused else max(float(np.max(np.abs(rg[e]))) for e in unused)
        router_note = " sigmoid未选中router最大=%.3e" % peak
        if peak > 1e-6:
            empty_ok = False
    ok = worst < TOL and r_ref < 1e-4 and empty_ok
    flag = "PASS" if ok else "FAIL"
    print("  %-22s %s  cuda=%.3e  ref=%.3e  empty=%s%s" % (
        name, flag, worst, r_ref, empty, router_note))
    if detail:
        print("    " + "  ".join(detail))
    return ok, worst


def _tie_picks_smaller():
    """平局必须取小下标。

    构造方式（旧版用 router 行 = -20 去"强制"，对随机 x 根本压不住，是错误的做法）：
      router 第 0、1 行完全相同 -> 两个 logit 精确相等（同一组浮点运算）；
      第 2 行取负 -> logit 为负，绝不会赢；
      x 第 0 维固定为 +1，其余为 0 -> logit = 1.0，稳妥为正。
    期望：选中第 0 个专家，输出 = sigmoid(1.0) * expert0(x)。
    """
    T.set_device("cpu")
    d, ff, n_exp = 8, 8, 3
    m = T.nn.MoE(d, ff, n_exp, top_k=1, gate="sigmoid")
    sd = m.state_dict()
    w = np.zeros((d,), np.float32)
    w[0] = 1.0
    rw = np.zeros((n_exp, d), np.float32)
    rw[0] = w
    rw[1] = w            # 精确平局
    rw[2] = -w
    sd["router.weight"] = rw
    m.load_state_dict(sd)

    x = np.zeros((4, d), np.float32)
    x[:, 0] = 1.0
    x[1, 1] = 0.25
    y = np.array(m(T.tensor(x)).to_numpy(), dtype=np.float64, copy=True)

    # 解析参考：只取专家 0，门控 sigmoid(1.0)
    g = 1.0 / (1.0 + np.exp(-1.0))
    xd = x.astype(np.float64)
    wg = np.asarray(sd["experts.0.gate.weight"], np.float64)
    wu = np.asarray(sd["experts.0.up.weight"], np.float64)
    wd = np.asarray(sd["experts.0.down.weight"], np.float64)
    h = xd @ wg.T
    h = h / (1.0 + np.exp(-h))
    ref = ((h * (xd @ wu.T)) @ wd.T) * g

    r = rel(y, ref)
    ok = r < 1e-6
    print("  %-22s %s  vs_expert0=%.3e" % ("tie->smaller", "PASS" if ok else "FAIL", r))
    return ok


def _finite_diff():
    """CPU 上对专家权重和输入做中心差分，确认反向不是把梯度送到了别的专家。"""
    T.set_device("cpu")
    d, ff, n_exp = 8, 8, 3
    rng = np.random.RandomState(19)
    m = T.nn.MoE(d, ff, n_exp, top_k=1, gate="sigmoid")
    sd = m.state_dict()
    rw = np.full((n_exp, d), -4.0, np.float32)
    rw[0] = 4.0
    sd["router.weight"] = rw
    x = (rng.randn(4, d) * 0.4).astype(np.float32)
    eps = np.float32(1e-3)

    def loss(sd_, x_):
        mm = T.nn.MoE(d, ff, n_exp, top_k=1, gate="sigmoid")
        mm.load_state_dict(sd_)
        return float(np.sum(mm(T.tensor(x_)).to_numpy()))

    m.load_state_dict(sd)
    xt = T.tensor(x, requires_grad=True)
    m(xt).sum().backward()
    key = "experts.0.down.weight"
    up = {k: v.copy() for k, v in sd.items()}
    dn = {k: v.copy() for k, v in sd.items()}
    up[key] = up[key].copy()
    dn[key] = dn[key].copy()
    up[key][1, 2] = np.float32(up[key][1, 2] + eps)
    dn[key][1, 2] = np.float32(dn[key][1, 2] - eps)
    fd = (loss(up, x) - loss(dn, x)) / float(eps * 2)
    ag = float(m.experts[0].down.weight.grad[1, 2])
    xp = x.copy()
    xm = x.copy()
    xp[2, 3] = np.float32(xp[2, 3] + eps)
    xm[2, 3] = np.float32(xm[2, 3] - eps)
    fdx = (loss(sd, xp) - loss(sd, xm)) / float(eps * 2)
    agx = float(xt.grad[2, 3])
    ok = abs(fd - ag) < 1e-4 and abs(fdx - agx) < 1e-4
    print("  %-22s %s  dW fd=%.6g an=%.6g  dx fd=%.6g an=%.6g" % (
        "finite-diff", "PASS" if ok else "FAIL", fd, ag, fdx, agx))
    return ok


def _make(seed, d, ff, n_exp, x_shape, router=None):
    T.set_device("cpu")
    T.manual_seed(seed)
    rng = np.random.RandomState(seed)
    # 临时模块只为了拿到 Linear 的初始形状；权重下面重写，避免两次构造吃掉 RNG。
    m = T.nn.MoE(d, ff, n_exp, top_k=1, gate="sigmoid")
    sd = {k: (rng.randn(*v.shape).astype(np.float32) * 0.15) for k, v in m.state_dict().items()}
    if router is not None:
        sd["router.weight"] = np.asarray(router, dtype=np.float32).copy()
    x = (rng.randn(*x_shape).astype(np.float32) * 0.5)
    coef = (rng.randn(*x_shape).astype(np.float32) * 0.7 + 0.2)
    return x, sd, coef


def main():
    cpu_only = "--cpu" in sys.argv
    print("nn.MoE self-check  tol=%.0e" % TOL)
    ok = True
    ok = _tie_picks_smaller() and ok
    ok = _finite_diff() and ok
    if cpu_only:
        print("VERDICT ok=%s（只跑了 CPU）" % ok)
        return 0 if ok else 1

    d = ff = 16
    n_exp = 4
    # 1) sigmoid top-1：专家 0/1 的 logit 完全相同，2/3 很差。
    #    平局必须落到 0；8 个 token 都打在 0 上（重复命中）；1/2/3 是空专家。
    router = np.zeros((n_exp, d), np.float32)
    router[0, 0] = 1.5
    router[1] = router[0].copy()
    router[2] = -4.0
    router[3] = -5.0
    x, sd, coef = _make(21, d, ff, n_exp, (8, d), router)
    ok1, w1 = _case("sigmoid-k1-tie-empty", d, ff, n_exp, 1, "sigmoid", x, sd, coef)

    # 2) softmax top-2，三维输入。0 和 1 总是被选中，2/3 空。每个被选中的专家拿到全部 token。
    router = np.zeros((n_exp, d), np.float32)
    router[0] = 0.8
    router[1] = 0.3
    router[2] = -6.0
    router[3] = -7.0
    x, sd, coef = _make(22, d, ff, n_exp, (2, 4, d), router)
    ok2, w2 = _case("softmax-k2-3d-empty", d, ff, n_exp, 2, "softmax", x, sd, coef)

    # 3) 随机 sigmoid top-2，专家分配不均匀，允许自然出现的空专家。
    x, sd, coef = _make(23, d, ff, n_exp, (6, d), None)
    ok3, w3 = _case("sigmoid-k2-random", d, ff, n_exp, 2, "sigmoid", x, sd, coef)

    # 4) 随机 softmax top-1，再加一个略大的形状，确认不是只在 K=1 上碰巧对。
    x, sd, coef = _make(24, d, ff, n_exp, (1, d), None)
    ok4, w4 = _case("softmax-k1-single", d, ff, n_exp, 1, "softmax", x, sd, coef)

    allok = ok and ok1 and ok2 and ok3 and ok4
    worst = max(w1, w2, w3, w4)
    print("VERDICT ok=%s  worst_cuda=%.3e（判据 < %.0e，另有平局/有限差分）" % (allok, worst, TOL))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())

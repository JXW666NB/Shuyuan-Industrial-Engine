# -*- coding: utf-8 -*-
"""`tgai_go.train` —— PyTorch 风格的训练 API（第一切片，CPU/numpy 后端）。

设计目标（见 docs/pytorch_compat_api.md）：
    开发者把 `import torch` 改成 `import tgai_go.train as torch`，
    再改函数前缀，就能迁移 —— 从而不吃 PyTorch 的显存税。

本切片（已自证）：
    * 反向模式自动微分（numpy 后端），带广播还原
    * nn.Module / Parameter / Linear / Embedding / RMSNorm / Sequential / ModuleList
    * optim.AdamW（与 torch 语义一致）/ optim.Muon（Newton-Schulz 正交化）
    * functional.cross_entropy / softmax / silu
    * state_dict() / load_state_dict()

尚未做（下一阶段）：
    * CUDA 后端（接我们自己的 kernel + cuBLAS）
    * attention / MoE 模块
    * 默认重算（activation checkpoint）
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "Tensor", "Parameter", "no_grad",
    "Module", "Linear", "Embedding", "RMSNorm", "Sequential", "ModuleList",
    "functional", "nn", "optim",
]

# ─────────────────────────── 自动微分 ───────────────────────────
_GRAD_ENABLED = [True]


class no_grad:
    """对应 torch.no_grad()。"""

    def __enter__(self):
        self._prev = _GRAD_ENABLED[0]
        _GRAD_ENABLED[0] = False
        return self

    def __exit__(self, *exc):
        _GRAD_ENABLED[0] = self._prev
        return False


def _unbroadcast(grad, shape):
    """把广播后的梯度还原到原形状（torch 的 sum_to_size 语义）。"""
    if grad.shape == shape:
        return grad
    # 去掉多出来的前导维度
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 对长度为 1 的维度求和并保持维度
    for axis, n in enumerate(shape):
        if n == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """最小可用的张量：数据 + 梯度 + 反向闭包。"""

    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward")
    __array_priority__ = 1000        # 让 ndarray <op> Tensor 走我们的 __r*__

    def __init__(self, data, requires_grad=False, _parents=(), _backward=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self._parents = _parents
        self._backward = _backward

    # ---- 属性 ----
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def numel(self):
        return self.data.size

    def numpy(self):
        return self.data

    def __repr__(self):
        return "Tensor(shape=%s, requires_grad=%s)" % (self.data.shape, self.requires_grad)

    # ---- 反向 ----
    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("调用 backward 的张量不参与求导")
        if grad is None:
            grad = np.ones_like(self.data)
        # 拓扑排序（迭代，避免深图递归爆栈）
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))
        grads = {id(self): np.asarray(grad, dtype=np.float32)}
        for node in reversed(topo):
            g = grads.get(id(node))
            if g is None:
                continue
            if node.grad is None:
                node.grad = np.zeros_like(node.data)
            node.grad = node.grad + g
            if node._backward is not None and _GRAD_ENABLED[0]:
                node._backward(g, grads)

    def zero_grad(self):
        self.grad = None

    def detach(self):
        return Tensor(self.data.copy())

    # ---- 算子 ----
    def __add__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, _unbroadcast(g, self.shape))
            if other.requires_grad:
                _acc(grads, other, _unbroadcast(g, other.shape))
        out._backward = bw
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, _unbroadcast(g * other.data, self.shape))
            if other.requires_grad:
                _acc(grads, other, _unbroadcast(g * self.data, other.shape))
        out._backward = bw
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_as_tensor(other))

    def __rsub__(self, other):
        return _as_tensor(other) + (-self)

    def __truediv__(self, other):
        other = _as_tensor(other)
        return self * (other ** -1.0)

    def __matmul__(self, other):
        other = _as_tensor(other)
        if self.ndim != 2 or other.ndim != 2:
            raise NotImplementedError("matmul 只支持 2D（第一切片）")
        out = Tensor(self.data @ other.data,
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, g @ other.data.T)
            if other.requires_grad:
                _acc(grads, other, self.data.T @ g)
        out._backward = bw
        return out

    def __pow__(self, p):
        out = Tensor(self.data ** p, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, g * p * (self.data ** (p - 1)))
        out._backward = bw
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, g * e)
        out._backward = bw
        return out

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, g / self.data)
        out._backward = bw
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad, _parents=(self,))
        shape = self.shape

        def bw(g, grads):
            gg = g
            if axis is not None and not keepdims:
                gg = np.expand_dims(gg, axis)
            _acc(grads, self, np.broadcast_to(gg, shape).copy())
        out._backward = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad, _parents=(self,))
        old = self.shape

        def bw(g, grads):
            _acc(grads, self, g.reshape(old))
        out._backward = bw
        return out

    def transpose(self):
        out = Tensor(self.data.T, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, g.T)
        out._backward = bw
        return out

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], requires_grad=self.requires_grad, _parents=(self,))
        shape = self.shape

        def bw(g, grads):
            full = np.zeros(shape, dtype=np.float32)
            np.add.at(full, idx, g)
            _acc(grads, self, full)
        out._backward = bw
        return out

    def maximum(self, other):
        other = _as_tensor(other)
        mask = (self.data > other.data).astype(np.float32)
        out = Tensor(np.maximum(self.data, other.data),
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, _unbroadcast(g * mask, self.shape))
            if other.requires_grad:
                _acc(grads, other, _unbroadcast(g * (1.0 - mask), other.shape))
        out._backward = bw
        return out

    # ---- 便利 ----
    def scale(self, c):
        return self * c

    def to_numpy(self):
        return self.data


def _acc(grads, node, g):
    g = np.asarray(g, dtype=np.float32)
    old = grads.get(id(node))
    grads[id(node)] = g if old is None else old + g


def _as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)


# ─────────────────────────── functional ───────────────────────────
class _Functional:
    @staticmethod
    def silu(x: Tensor) -> Tensor:
        s = (x * -1.0).exp()
        return x * ((s + 1.0) ** -1.0)

    @staticmethod
    def softmax(x: Tensor, axis=-1) -> Tensor:
        m = x.data.max(axis=axis, keepdims=True)
        z = x - Tensor(m)                     # 常数，不参与求导
        e = z.exp()
        return e * (e.sum(axis=axis, keepdims=True) ** -1.0)

    @staticmethod
    def log_softmax(x: Tensor, axis=-1) -> Tensor:
        m = x.data.max(axis=axis, keepdims=True)
        z = x - Tensor(m)
        e = z.exp()
        return z - e.sum(axis=axis, keepdims=True).log()

    @staticmethod
    def cross_entropy(logits: Tensor, target, ignore_index=-100):
        """logits [N,V]，target [N]（int）。返回标量 loss（mean）。"""
        n, v = logits.shape
        lp = _Functional.log_softmax(logits, axis=-1)
        tgt = np.asarray(target, dtype=np.int64).reshape(-1)
        mask = tgt != ignore_index
        picked = lp[np.arange(n), tgt]                      # [N]
        loss = (picked * Tensor(mask.astype(np.float32))).sum() * (-1.0 / max(1, mask.sum()))
        return loss


functional = _Functional()


# ─────────────────────────── nn ───────────────────────────
class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    """对应 torch.nn.Module：参数注册、子模块、state_dict。"""

    def __init__(self):
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_mods", {})

    def __setattr__(self, name, value):
        if name in ("_params", "_mods"):
            object.__setattr__(self, name, value)
            return
        if isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Module):
            self._mods[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *a, **kw):
        return self.forward(*a, **kw)

    def forward(self, *a, **kw):
        raise NotImplementedError

    def named_parameters(self, prefix=""):
        for k, p in self._params.items():
            yield prefix + k, p
        for k, m in self._mods.items():
            yield from m.named_parameters(prefix + k + ".")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def state_dict(self, prefix=""):
        out = {prefix + k: p.data.copy() for k, p in self._params.items()}
        for k, m in self._mods.items():
            out.update(m.state_dict(prefix + k + "."))
        return out

    def load_state_dict(self, sd):
        for name, p in self.named_parameters():
            if name not in sd:
                raise KeyError("缺少参数: %s" % name)
            p.data = np.asarray(sd[name], dtype=np.float32).reshape(p.data.shape)
        return self

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        return "%s(params=%d)" % (type(self).__name__, self.num_params())


class Linear(Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        k = 1.0 / np.sqrt(in_features)
        self.weight = Parameter(np.random.randn(out_features, in_features).astype(np.float32) * k)
        self.bias = Parameter(np.zeros(out_features, np.float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class Embedding(Module):
    def __init__(self, num_embeddings, dim):
        super().__init__()
        self.weight = Parameter((np.random.randn(num_embeddings, dim) * 0.02).astype(np.float32))

    def forward(self, idx) -> Tensor:
        idx = np.asarray(idx, dtype=np.int64)
        return self.weight[idx.reshape(-1)].reshape(idx.shape + (self.weight.shape[1],))


class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = Parameter(np.ones(dim, np.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        var = (x * x).mean(axis=-1, keepdims=True)
        inv = (var + Tensor(np.array([self.eps], np.float32))) ** -0.5
        return x * inv * self.weight


class Sequential(Module):
    def __init__(self, *mods):
        super().__init__()
        self._seq = list(mods)
        for i, m in enumerate(self._seq):
            setattr(self, "m%d" % i, m)

    def forward(self, x):
        for m in self._seq:
            x = m(x)
        return x


class ModuleList(Module):
    def __init__(self, mods):
        super().__init__()
        self._list = list(mods)
        for i, m in enumerate(self._list):
            setattr(self, "m%d" % i, m)

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __iter__(self):
        return iter(self._list)


# ─────────────────────────── optim ───────────────────────────
class _Optim:
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class AdamW(_Optim):
    """与 torch.optim.AdamW 语义一致（含 decoupled weight decay）。"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd:
                p.data = p.data - self.lr * self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / bc1
            vh = self.v[i] / bc2
            p.data = p.data - self.lr * mh / (np.sqrt(vh) + self.eps)


def _newton_schulz5(g, steps=5, eps=1e-7):
    """Muon 的正交化迭代（a,b,c = 3.4445, -4.7750, 2.0315）。

    注意：该多项式的不动点是 0.868，所以输出奇异值集中在 0.868 附近、
    条件数被压到 ~1.6（实测病态输入 384 -> 1.61）。
    它的作用是**压平谱**，常数倍由 Muon 的学习率吸收。
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(np.float32)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    x = x / (np.linalg.norm(x) + eps)
    for _ in range(steps):
        s = x @ x.T
        y = b * s + c * (s @ s)
        x = a * x + y @ x
    if transposed:
        x = x.T
    return x


class Muon(_Optim):
    """Muon：动量 + Newton-Schulz 正交化；2D 走 Muon，其余回退 AdamW 式更新。"""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0):
        super().__init__(params, lr)
        self.mom = momentum
        self.nesterov = nesterov
        self.wd = weight_decay
        self.buf = [np.zeros_like(p.data) for p in self.params]
        self._adam_m = [np.zeros_like(p.data) for p in self.params]
        self._adam_v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if p.data.ndim == 2:
                if self.wd:
                    p.data = p.data - self.lr * self.wd * p.data
                self.buf[i] = self.mom * self.buf[i] + g
                u = g + self.mom * self.buf[i] if self.nesterov else self.buf[i]
                o = _newton_schulz5(u)
                scale = np.sqrt(max(1.0, p.data.shape[0] / p.data.shape[1]))
                p.data = p.data - self.lr * scale * o
            else:
                b1, b2 = 0.9, 0.999
                self._adam_m[i] = b1 * self._adam_m[i] + (1 - b1) * g
                self._adam_v[i] = b2 * self._adam_v[i] + (1 - b2) * (g * g)
                mh = self._adam_m[i] / (1 - b1 ** self.t)
                vh = self._adam_v[i] / (1 - b2 ** self.t)
                p.data = p.data - self.lr * mh / (np.sqrt(vh) + 1e-8)


class _OptimModule:
    AdamW = AdamW
    Muon = Muon


class _NNModule:
    Module = Module
    Parameter = Parameter
    Linear = Linear
    Embedding = Embedding
    RMSNorm = RMSNorm
    Sequential = Sequential
    ModuleList = ModuleList


nn = _NNModule()
optim = _OptimModule()

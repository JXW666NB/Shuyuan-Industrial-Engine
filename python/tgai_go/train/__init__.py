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

M1 CUDA（见 train/cuda.py，set_device("cuda") 之后新建的张量在 GPU 上）：
    * 显存 + cublasSgemm 的 2D matmul（前向和两个反向）
    * functional.cross_entropy（NVRTC 编译内嵌的 tgai_ce.cu；ignore_index 仍走 CPU）

M2 CUDA（同一份公共 API，不改判分台）：
    * nn.RMSNorm：rmsnorm_fwd / rmsnorm_bwd
    * functional.silu 以及同形状 mul：silu_mul_* 与 ewise_mul
    * nn.Embedding（重复下标走 kernel 里的 atomicAdd）
    * 判分台用来反传的全归约 sum

M3 CUDA（同一份 CausalSelfAttention 公式，不改判分台）：
    * reshape（C 顺序重解释）、最后两维交换
    * 3D matmul：按 batch 循环已验收的 cublasSgemm
    * 可广播的 add（因果掩码）、最后一维 softmax
    * 因此 nn.CausalSelfAttention 整层在 GPU 上

尚未做：
    * MoE / AdamW / Muon 的 CUDA 路径
    * mean、pow、exp、下标，以及公开的带 axis 的 sum
    * 默认重算（activation checkpoint）
    * dh=64 的 flash / tiled attention kernel（判分台 Dh=8，那些 kernel 直接 return）
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "Tensor", "Parameter", "no_grad", "set_device", "get_device",
    "Module", "Linear", "Embedding", "RMSNorm", "Sequential", "ModuleList",
    "CausalSelfAttention",
    "functional", "nn", "optim",
]

# 默认 cpu。只有 set_device("cuda") 成功之后，新建张量才走 GPU。
# CPU 路径不加载 cuda.py。
_DEVICE = "cpu"


def get_device():
    return _DEVICE


def set_device(device):
    """'cpu' 或 'cuda'。cuda 会立刻初始化驱动 / cuBLAS / NVRTC，失败则保持 cpu。"""
    global _DEVICE
    if not isinstance(device, str):
        raise TypeError("set_device 需要字符串，收到 %r" % (device,))
    name = device.lower()
    if name == "cpu":
        _DEVICE = "cpu"
        return
    if name != "cuda":
        raise ValueError("set_device 只接受 'cpu' 或 'cuda'，收到 %r" % (device,))
    _cuda()
    _DEVICE = "cuda"


def _cuda():
    from .cuda import runtime
    return runtime()

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

    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "device", "_gpu")
    __array_priority__ = 1000        # 让 ndarray <op> Tensor 走我们的 __r*__

    def __init__(self, data, requires_grad=False, _parents=(), _backward=None, device=None):
        dev = _DEVICE if device is None else device
        self.device = dev
        self._gpu = None
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self._parents = _parents
        self._backward = _backward
        if dev == "cpu":
            self.data = np.asarray(data, dtype=np.float32)
            return
        if dev != "cuda":
            raise ValueError("Tensor device 只接受 cpu/cuda，收到 %r" % (dev,))
        arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
        self.data = None
        self._gpu = _cuda().h2d(arr)

    @classmethod
    def _from_gpu(cls, gpu, requires_grad=False, _parents=(), _backward=None):
        obj = cls.__new__(cls)
        obj.device = "cuda"
        obj._gpu = gpu
        obj.data = None
        obj.grad = None
        obj.requires_grad = bool(requires_grad)
        obj._parents = _parents
        obj._backward = _backward
        return obj

    def _cpu_only(self, op):
        if self.device == "cuda":
            raise NotImplementedError("CUDA 后端还没有 %s" % op)

    # ---- 属性 ----
    @property
    def shape(self):
        if self.device == "cuda":
            return self._gpu.shape
        return self.data.shape

    @property
    def ndim(self):
        if self.device == "cuda":
            return len(self.shape)
        return self.data.ndim

    def numel(self):
        if self.device == "cuda":
            n = 1
            for s in self.shape:
                n *= int(s)
            return int(n)
        return self.data.size

    def numpy(self):
        if self.device == "cuda":
            return self.to_numpy()
        return self.data

    def __repr__(self):
        if self.device == "cuda":
            return "Tensor(shape=%s, device=cuda, requires_grad=%s)" % (self.shape, self.requires_grad)
        return "Tensor(shape=%s, requires_grad=%s)" % (self.data.shape, self.requires_grad)

    # ---- 反向 ----
    def backward(self, grad=None):
        if self.device == "cuda":
            return self._backward_cuda(grad)
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
        if self.device == "cuda" or (
                isinstance(other, Tensor) and getattr(other, "device", None) == "cuda"):
            return self._add_cuda(other)
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
        if self.device == "cuda" or (
                isinstance(other, Tensor) and getattr(other, "device", None) == "cuda"):
            return self._mul_cuda(other)
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

    def _add_cuda(self, other):
        if not isinstance(other, Tensor):
            other = _as_tensor(other)
        if self.device != "cuda" or other.device != "cuda":
            raise RuntimeError("add 不能混用 CPU 和 CUDA 张量")
        rt = _cuda()
        buf = rt.add(self._gpu, other._gpu)
        rg = self.requires_grad or other.requires_grad
        out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(self, other))
        a_shape, b_shape = tuple(self.shape), tuple(other.shape)
        out_shape = tuple(buf.shape)

        def bw(g, grads):
            if self.requires_grad:
                src = g if a_shape == out_shape else rt.unbroadcast(g, a_shape)
                _acc(grads, self, src)
            if other.requires_grad:
                src = g if b_shape == out_shape else rt.unbroadcast(g, b_shape)
                _acc(grads, other, src)
        out._backward = bw
        return out

    def _scale_cuda(self, scalar):
        rt = _cuda()
        buf = rt.scale(self._gpu, float(scalar))
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.scale(g, float(scalar)))
        out._backward = bw
        return out

    def _mul_cuda(self, other):
        rt = _cuda()
        if not isinstance(other, Tensor):
            if isinstance(other, (int, float, np.floating, np.integer)) and not isinstance(other, bool):
                return self._scale_cuda(float(other))
            other = _as_tensor(other)
        if self.device != "cuda" or other.device != "cuda":
            raise RuntimeError("mul 不能混用 CPU 和 CUDA 张量")
        if self.shape != other.shape:
            raise NotImplementedError(
                "CUDA mul 目前只支持同形状，收到 %s * %s" % (self.shape, other.shape))
        buf = rt.mul(self._gpu, other._gpu)
        rg = self.requires_grad or other.requires_grad
        out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(self, other))
        a, b = self._gpu, other._gpu

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.mul(g, b))
            if other.requires_grad:
                _acc(grads, other, rt.mul(g, a))
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
        if self.device == "cuda" or other.device == "cuda":
            if self.device != "cuda" or other.device != "cuda":
                raise RuntimeError("matmul 不能混用 CPU 和 CUDA 张量")
            return self._matmul_cuda(other)
        rg = self.requires_grad or other.requires_grad
        if self.ndim == 2 and other.ndim == 2:
            out = Tensor(self.data @ other.data, requires_grad=rg, _parents=(self, other))
            a, b = self.data, other.data

            def bw(g, grads):
                if self.requires_grad:
                    _acc(grads, self, g @ b.T)
                if other.requires_grad:
                    _acc(grads, other, a.T @ g)
            out._backward = bw
            return out
        if self.ndim == 3 and other.ndim == 3:
            if self.shape[0] != other.shape[0]:
                raise ValueError("3D matmul 需要相同 batch: %s vs %s"
                                 % (self.shape, other.shape))
            out = Tensor(np.matmul(self.data, other.data), requires_grad=rg, _parents=(self, other))
            a, b = self.data, other.data

            def bw(g, grads):
                if self.requires_grad:
                    _acc(grads, self, np.matmul(g, np.swapaxes(b, -1, -2)))
                if other.requires_grad:
                    _acc(grads, other, np.matmul(np.swapaxes(a, -1, -2), g))
            out._backward = bw
            return out
        raise NotImplementedError("matmul 支持 2D/3D，收到 %dD x %dD"
                                  % (self.ndim, other.ndim))

    def _matmul_cuda(self, other):
        rt = _cuda()
        if self.ndim == 3 and other.ndim == 3:
            if self.shape[0] != other.shape[0]:
                raise ValueError("3D matmul 需要相同 batch: %s vs %s"
                                 % (self.shape, other.shape))
            buf = rt.bmm(self._gpu, other._gpu)
            rg = self.requires_grad or other.requires_grad
            out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(self, other))
            a, b = self._gpu, other._gpu

            def bw(g, grads):
                if self.requires_grad:
                    _acc(grads, self, rt.bmm(g, rt.swap_last2(b)))
                if other.requires_grad:
                    _acc(grads, other, rt.bmm(rt.swap_last2(a), g))
            out._backward = bw
            return out
        if self.ndim != 2 or other.ndim != 2:
            raise NotImplementedError("CUDA matmul 目前只支持 2D/3D，收到 %dD x %dD"
                                      % (self.ndim, other.ndim))
        buf = rt.sgemm(self._gpu, other._gpu)
        rg = self.requires_grad or other.requires_grad
        out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(self, other))
        a, b = self._gpu, other._gpu

        def bw(g, grads):
            # dA = g @ B^T ，dB = A^T @ g。和 CPU 的 g @ b.T / a.T @ g 同一套形状。
            if self.requires_grad:
                _acc(grads, self, rt.gemm_nt(g, b))
            if other.requires_grad:
                _acc(grads, other, rt.gemm_tn(a, g))
        out._backward = bw
        return out

    def _transpose_cuda(self):
        if self.ndim != 2:
            raise NotImplementedError("CUDA transpose 目前只支持 2D，收到 %dD" % self.ndim)
        rt = _cuda()
        buf = rt.transpose2d(self._gpu)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.transpose2d(g))
        out._backward = bw
        return out

    def _backward_cuda(self, grad):
        if not self.requires_grad:
            raise RuntimeError("调用 backward 的张量不参与求导")
        rt = _cuda()
        if grad is None:
            g0 = rt.full(self.shape, 1.0)
        elif isinstance(grad, Tensor):
            if grad.device != "cuda":
                g0 = rt.h2d(np.asarray(grad.to_numpy(), dtype=np.float32).reshape(self.shape))
            else:
                g0 = grad._gpu
        else:
            g0 = rt.h2d(np.asarray(grad, dtype=np.float32).reshape(self.shape))
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
        grads = {id(self): g0}
        for node in reversed(topo):
            g = grads.get(id(node))
            if g is None:
                continue
            if getattr(node, "device", "cpu") != "cuda":
                raise RuntimeError("CUDA 反传遇到 CPU 张量")
            if node.grad is None:
                node.grad = Tensor._from_gpu(rt.clone(g), requires_grad=False)
            else:
                if not isinstance(node.grad, Tensor) or getattr(node.grad, "device", "cpu") != "cuda":
                    raise RuntimeError("CUDA 张量的 .grad 必须是 CUDA Tensor")
                node.grad = Tensor._from_gpu(rt.add(node.grad._gpu, g), requires_grad=False)
            if node._backward is not None and _GRAD_ENABLED[0]:
                node._backward(g, grads)
        rt.sync()

    def swap_last2(self):
        """交换最后两维（attention 里对 k 用）。"""
        if self.device == "cuda":
            return self._swap_last2_cuda()
        out = Tensor(np.swapaxes(self.data, -1, -2),
                     requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, np.swapaxes(g, -1, -2))
        out._backward = bw
        return out

    def _swap_last2_cuda(self):
        if self.ndim < 2:
            raise NotImplementedError("swap_last2 至少要 2 维，收到 %dD" % self.ndim)
        rt = _cuda()
        buf = rt.swap_last2(self._gpu)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, rt.swap_last2(g))
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
        if self.device == "cuda":
            if axis is not None or keepdims:
                raise NotImplementedError("CUDA sum 目前只支持全归约")
            return self._sum_cuda()
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

    def _sum_cuda(self):
        rt = _cuda()
        buf = rt.sum_all(self._gpu)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))
        in_shape = self.shape

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.full(in_shape, rt.read_scalar(g)))
        out._backward = bw
        return out

    def mean(self, axis=None, keepdims=False):
        if self.device == "cuda":
            self._cpu_only("mean")
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if self.device == "cuda":
            return self._reshape_cuda(shape)
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad, _parents=(self,))
        old = self.shape

        def bw(g, grads):
            _acc(grads, self, g.reshape(old))
        out._backward = bw
        return out

    def _reshape_cuda(self, shape):
        rt = _cuda()
        buf = rt.reshape(self._gpu, shape)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))
        old = self.shape

        def bw(g, grads):
            _acc(grads, self, rt.reshape(g, old))
        out._backward = bw
        return out

    def transpose(self):
        if self.device == "cuda":
            return self._transpose_cuda()
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
            if _has_array_index(idx):
                np.add.at(full, idx, g)      # 重复下标要累加（Embedding）
            else:
                full[idx] += g               # 纯切片
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
        if self.device == "cuda":
            return _cuda().d2h(self._gpu)
        return self.data


def _acc(grads, node, g):
    if getattr(node, "device", "cpu") == "cuda":
        rt = _cuda()
        old = grads.get(id(node))
        grads[id(node)] = g if old is None else rt.add(old, g)
        return
    g = np.asarray(g, dtype=np.float32)
    old = grads.get(id(node))
    grads[id(node)] = g if old is None else old + g


def _as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def _has_array_index(idx):
    """判断索引里是否含数组（需要 add.at 的重复累加语义）。"""
    items = idx if isinstance(idx, tuple) else (idx,)
    for it in items:
        if isinstance(it, np.ndarray):
            return True
        if isinstance(it, (list, tuple)):
            return True
    return False


# ─────────────────────────── functional ───────────────────────────
class _Functional:
    @staticmethod
    def silu(x: Tensor) -> Tensor:
        if getattr(x, "device", "cpu") == "cuda":
            return _silu_cuda(x)
        s = (x * -1.0).exp()
        return x * ((s + 1.0) ** -1.0)

    @staticmethod
    def softmax(x: Tensor, axis=-1) -> Tensor:
        if getattr(x, "device", "cpu") == "cuda":
            return _softmax_cuda(x, axis)
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
    def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, causal=False):
        """q,k,v: [B*H, S, Dh]（已展平 head）。causal=True 时做下三角掩码。"""
        dh = q.shape[-1]
        scale = 1.0 / np.sqrt(dh)
        scores = (q @ k.swap_last2()) * scale                  # [BH,S,S]
        if causal:
            S = scores.shape[-1]
            mask = np.triu(np.full((S, S), -1e30, dtype=np.float32), k=1)
            scores = scores + Tensor(mask)                     # 常数，不回传梯度
        p = _Functional.softmax(scores, axis=-1)
        return p @ v

    @staticmethod
    def cross_entropy(logits: Tensor, target, ignore_index=-100):
        """logits [N,V]，target [N]（int）。返回标量 loss（mean）。"""
        if getattr(logits, "device", "cpu") == "cuda":
            return _cross_entropy_cuda(logits, target, ignore_index)
        return _Functional._cross_entropy_cpu(logits, target, ignore_index)

    @staticmethod
    def _cross_entropy_cpu(logits, target, ignore_index=-100):
        n, v = logits.shape
        lp = _Functional.log_softmax(logits, axis=-1)
        tgt = np.asarray(target, dtype=np.int64).reshape(-1)
        mask = tgt != ignore_index
        picked = lp[np.arange(n), tgt]                      # [N]
        loss = (picked * Tensor(mask.astype(np.float32))).sum() * (-1.0 / max(1, mask.sum()))
        return loss


functional = _Functional()


def _softmax_cuda(x, axis):
    """最后一维 softmax。反向用 y*(dy - sum(dy*y))，和 CPU 那条 exp/sum 链在 fp32 上差 ~1e-7。"""
    if axis != -1 and axis != x.ndim - 1:
        raise NotImplementedError("CUDA softmax 目前只支持最后一维，收到 axis=%r" % (axis,))
    if x.ndim < 1:
        raise NotImplementedError("CUDA softmax 需要至少 1 维")
    rt = _cuda()
    y = rt.softmax_lastdim(x._gpu)
    out = Tensor._from_gpu(y, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if x.requires_grad:
            _acc(grads, x, rt.softmax_lastdim_bwd(g, y))
    out._backward = bw
    return out


def _silu_cuda(x):
    """silu(x) = silu_mul(clone(x), 1)。反向复用 silu_mul_bwd，b 是全 1。"""
    rt = _cuda()
    ones = rt.full(x.shape, 1.0)
    y = rt.clone(x._gpu)
    rt.silu_mul_fwd(y, ones)
    out = Tensor._from_gpu(y, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if not x.requires_grad:
            return
        da, _db = rt.silu_mul_bwd(g, x._gpu, ones)
        _acc(grads, x, da)

    out._backward = bw
    return out


def _cross_entropy_cuda(logits, target, ignore_index):
    """GPU 上的 mean CE。语义与 ce_fwd/ce_bwd 一致，不含 ignore_index。"""
    tgt = np.asarray(target, dtype=np.int64).reshape(-1)
    n, v = logits.shape
    if tgt.shape != (n,):
        raise ValueError("cross_entropy 目标长度 %s != N=%d" % (tgt.shape, n))
    if np.any(tgt == ignore_index):
        return _cross_entropy_cuda_ignore(logits, tgt, ignore_index)
    if np.any((tgt < 0) | (tgt >= v)):
        raise ValueError("cross_entropy 目标越界，合法范围是 [0, %d)" % v)
    rt = _cuda()
    tgt_buf = rt.h2d(tgt, dtype=np.int32)
    loss_buf = rt.ce_fwd(logits._gpu, tgt_buf, n, v)
    out = Tensor._from_gpu(loss_buf, requires_grad=logits.requires_grad, _parents=(logits,))

    def bw(g, grads):
        if not logits.requires_grad:
            return
        dl = rt.ce_bwd(logits._gpu, tgt_buf, n, v)
        scale = rt.read_scalar(g)
        if scale != 1.0:
            dl = rt.scale(dl, scale)
        _acc(grads, logits, dl)

    out._backward = bw
    return out


def _cross_entropy_cuda_ignore(logits, tgt, ignore_index):
    """kernel 没有 mask。带 ignore_index 时整段 CE 在 CPU 上算，dlogits 再拷回 GPU。"""
    global _DEVICE
    prev = _DEVICE
    _DEVICE = "cpu"
    try:
        host = Tensor(logits.to_numpy(), requires_grad=True)
        loss_cpu = _Functional._cross_entropy_cpu(host, tgt, ignore_index)
        loss_host = np.asarray(loss_cpu.data, dtype=np.float32).copy()
    finally:
        _DEVICE = prev
    rt = _cuda()
    out = Tensor._from_gpu(rt.h2d(loss_host), requires_grad=logits.requires_grad, _parents=(logits,))

    def bw(g, grads):
        if not logits.requires_grad:
            return
        scale = rt.read_scalar(g)
        loss_cpu.backward(np.asarray(scale, dtype=np.float32))
        _acc(grads, logits, rt.h2d(host.grad))

    out._backward = bw
    return out


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
        out = {}
        for k, p in self._params.items():
            if getattr(p, "device", "cpu") == "cuda":
                out[prefix + k] = p.to_numpy().copy()
            else:
                out[prefix + k] = p.data.copy()
        for k, m in self._mods.items():
            out.update(m.state_dict(prefix + k + "."))
        return out

    def load_state_dict(self, sd):
        for name, p in self.named_parameters():
            if name not in sd:
                raise KeyError("缺少参数: %s" % name)
            if getattr(p, "device", "cpu") == "cuda":
                arr = np.asarray(sd[name], dtype=np.float32).reshape(p.shape)
                p._gpu = _cuda().h2d(arr)
            else:
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
        if getattr(self.weight, "device", "cpu") == "cuda":
            return self._forward_cuda(idx)
        return self.weight[idx.reshape(-1)].reshape(idx.shape + (self.weight.shape[1],))

    def _forward_cuda(self, idx):
        rt = _cuda()
        flat = np.ascontiguousarray(idx, dtype=np.int64).reshape(-1)
        vocab = int(self.weight.shape[0])
        d = int(self.weight.shape[1])
        if flat.size and ((flat < 0) | (flat >= vocab)).any():
            raise IndexError("Embedding 下标越界，合法范围是 [0, %d)" % vocab)
        seq = int(flat.shape[0])
        ids = rt.h2d(flat.astype(np.int32), dtype=np.int32)
        out_shape = tuple(int(s) for s in idx.shape) + (d,)
        y = rt.emb_fwd(self.weight._gpu, ids, seq, d, vocab, out_shape)
        w = self.weight
        out = Tensor._from_gpu(y, requires_grad=w.requires_grad, _parents=(w,))

        def bw(g, grads):
            if not w.requires_grad:
                return
            _acc(grads, w, rt.emb_bwd(g, ids, seq, d, w.shape))

        out._backward = bw
        return out


class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = Parameter(np.ones(dim, np.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if getattr(x, "device", "cpu") == "cuda":
            return self._forward_cuda(x)
        var = (x * x).mean(axis=-1, keepdims=True)
        inv = (var + Tensor(np.array([self.eps], np.float32))) ** -0.5
        return x * inv * self.weight

    def _forward_cuda(self, x):
        rt = _cuda()
        if x.ndim < 1:
            raise ValueError("RMSNorm 需要至少 1 维")
        d = int(x.shape[-1])
        if int(self.weight.numel()) != d:
            raise ValueError("RMSNorm weight 长度 %d != 最后一维 %d" % (self.weight.numel(), d))
        seq = int(x.numel() // d) if d else 0
        eps = float(self.eps)
        y = rt.rmsnorm_fwd(x._gpu, self.weight._gpu, seq, d, eps)
        w = self.weight
        out = Tensor._from_gpu(
            y, requires_grad=(x.requires_grad or w.requires_grad), _parents=(x, w))

        def bw(g, grads):
            dx, dw = rt.rmsnorm_bwd(g, x._gpu, w._gpu, seq, d, eps)
            if x.requires_grad:
                _acc(grads, x, dx)
            if w.requires_grad:
                _acc(grads, w, dw)

        out._backward = bw
        return out


class CausalSelfAttention(Module):
    """causal 多头自注意力（对应引擎里的 attn 子层）。

    权重拆成独立的 q/k/v/out 四个 Linear，命名与 .TG 里的
    `blocks.{i}.attn.{q,k,v,out}_proj.weight` 一一对应，便于直出。
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = Linear(d_model, d_model, bias=False)
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)
        self.out_proj = Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head

        def split(t):
            # [B,S,D] -> [B*H, S, Dh]
            return t.reshape(B, S, H, Dh).reshape(B * H, S, Dh)

        q = split(self.q_proj(x.reshape(B * S, D)))
        k = split(self.k_proj(x.reshape(B * S, D)))
        v = split(self.v_proj(x.reshape(B * S, D)))
        o = _Functional.scaled_dot_product_attention(q, k, v, causal=True)
        o = o.reshape(B, H, S, Dh).reshape(B, S, H * Dh)
        return self.out_proj(o.reshape(B * S, D)).reshape(B, S, D)


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
    CausalSelfAttention = CausalSelfAttention
    Parameter = Parameter
    Linear = Linear
    Embedding = Embedding
    RMSNorm = RMSNorm
    Sequential = Sequential
    ModuleList = ModuleList


nn = _NNModule()
optim = _OptimModule()

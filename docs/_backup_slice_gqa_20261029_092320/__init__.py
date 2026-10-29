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

M5 CPU（同一份公共 API，浮点一律 fp32）：
    * 工厂：tensor / zeros / ones / full / randn / rand / arange / from_numpy / manual_seed
    * dtype（float32 / int64 等）、device、cuda.is_available
    * permute / transpose(dim0, dim1) / contiguous / view / size / dim
    * 模块级 exp / log / sqrt / abs / clamp / cat / stack / matmul / bmm / einsum
      / max / min / topk / where / save / load
    * matmul 支持任意前导维，因此 nn.Linear 接受 ND 输入

M5 CUDA（判分台 `--cuda` 那 6 个洞；值在 GPU 上算，主机只算形状/步长/偏差校正这几个标量）：
    * 广播 add 的反向：按轴求和，不再把梯度拉回主机
    * 最后两维 matmul + 前导维广播（3D x 2D 的 Linear）。2D、同 batch 的 3D 仍走原来的 sgemm/bmm
    * permute，以及 transpose(dim0, dim1) 的非 2D 路径（反向是逆置换）
    * 最后一维 log_softmax
    * AdamW.step：参数和动量都在设备上。CUDA 张量的 data 是 None，不是「没有梯度」

M4 CUDA（优化器更新留在显存上。主机只算偏差校正、学习率这些标量）：
    * AdamW：weight decay、一阶/二阶动量、偏差校正后的参数更新，一次 kernel 原地写完
    * Muon：动量和 Nesterov 在 kernel 里。Newton-Schulz 的 X@X.T、(X@X.T)@(X@X.T)、
      Y@X 都在设备上做，累加顺序是顺序 FMA（sgemm_mode 1），和 NumPy float32 的
      `@` 逐位相同。cublasSgemm_v2（pedantic）单次相对误差约 3e-8，五轮迭代后
      参数误差超过 1e-7，所以这里不用它。转置守卫在设备上做 transpose2d。
      ||X|| 的平方和在设备上顺序累加，只把这一个标量降回主机做 float32 sqrt

MoE（nn.MoE，CPU / CUDA 同一份模块，激活不整段拉回主机）：
    * gate="sigmoid"：top-k（logit 更大者优先，平局取更小下标），权重 = sigmoid(logit)。
      top_k=1 与 tgai_moeblk.cu 的路由一致：argmax、平局取小下标、门控 = sigmoid。
    * gate="softmax"：先对全部 E 个 logit 做 softmax，再取同一个 top-k，权重就是这些
      softmax 值，不再在 top-k 内归一。反向走标准 softmax VJP，未选中的 logit
      会通过分母拿到梯度；sigmoid 则只有被选中的专家行有路由梯度。
    * 专家：h = silu(x @ gate.T) * (x @ up.T)，y = h @ down.T，再按门控加权求和。
      参数名 router.weight [E,d]、experts.{e}.gate/up.weight [ff,d]、
      experts.{e}.down.weight [d,ff]。
    * CUDA 只把每个专家的 token 计数（E+1 个 int32）读回主机，用来决定发不发 GEMM。
      空专家不发 GEMM，梯度保持 None（等于没贡献）。

尚未做：
    * M5 里仍只在 CPU 上的算子：exp / log / abs / clamp / cat / stack / einsum
      / max / min / topk / where（CUDA 上仍是 NotImplementedError）
    * 默认重算（activation checkpoint）
    * 默认 SDPA 仍物化 P。dh=64 且 causal 的低显存路径要显式 lowmem=True
      （functional.sdpa_lowmem / scaled_dot_product_attention(..., lowmem=True)），
      其它 dh 或非 causal 回退到物化实现。判分台 Dh=8，不会走到这条路径
    * 与 PyTorch 逐位相同的随机数、float64 存储、torch.save 的 zip 格式
"""
from __future__ import annotations

import numpy as np

# 模块最后会把 max 绑成 torch.max。这两处要的是内置 max，先留住。
_builtin_max = max


def _f32(x):
    """numpy 2 把 Python float 乘进 float32 数组之前做的那一次舍入。"""
    return float(np.float32(x))


def _f32bits(u):
    return np.array([u], np.uint32).view(np.float32)[0]


_LOG2E32 = _f32bits(0x3FB8AA3B)
_LN232 = _f32bits(0x3F317218)
_EXP_C = (
    _f32bits(0x39500D01),
    _f32bits(0x3AB60B61),
    _f32bits(0x3C088889),
    _f32bits(0x3D2AAAAB),
    _f32bits(0x3E2AAAAB),
    np.float32(0.5),
    np.float32(1.0),
    np.float32(1.0),
)
# 1/12! .. 1/0!，|x|<=2.5 时用这条光滑泰勒，有限差分才稳。
_EXP_TAYLOR = (
    _f32bits(0x310F76C8),
    _f32bits(0x32D7322C),
    _f32bits(0x3493F27E),
    _f32bits(0x3638EF1D),
    _f32bits(0x37D00D01),
    _f32bits(0x39500D01),
    _f32bits(0x3AB60B61),
    _f32bits(0x3C088889),
    _f32bits(0x3D2AAAAB),
    _f32bits(0x3E2AAAAB),
    np.float32(0.5),
    np.float32(1.0),
    np.float32(1.0),
)
_EXP_SMOOTH = _f32bits(0x40200000)  # 2.5


def _exp32_taylor(x):
    x = np.asarray(x, np.float32)
    p = np.broadcast_to(_EXP_TAYLOR[0], np.shape(x)).astype(np.float32).copy()
    for c in _EXP_TAYLOR[1:]:
        p = np.float32(np.float32(p * x) + c)
    return p


def _exp32_wide(x):
    x = np.asarray(x, np.float32)
    x = np.clip(x, np.float32(-80.0), np.float32(80.0))
    nf = np.rint(np.float32(x * _LOG2E32)).astype(np.float32)
    t = np.float32(x - np.float32(nf * _LN232))
    p = np.broadcast_to(_EXP_C[0], np.shape(x)).astype(np.float32).copy()
    for c in _EXP_C[1:]:
        p = np.float32(np.float32(p * t) + c)
    bits = p.view(np.uint32).view(np.int32)
    n = nf.astype(np.int32)
    return (bits + n * np.int32(8388608)).view(np.uint32).view(np.float32)


def _exp32(x):
    """与 CUDA exp32_bits 逐位相同。|x|<=2.5 用泰勒，外面用换底。"""
    x = np.asarray(x, np.float32)
    smooth = np.abs(x) <= _EXP_SMOOTH
    # 两边都算，再用 where 选。NaN 的比较是 False，走 wide。
    return np.where(smooth, _exp32_taylor(x), _exp32_wide(x)).astype(np.float32)


def _log32(x):
    """与 CUDA log32_bits 逐位相同。"""
    x = np.asarray(x, np.float32)
    bits = x.view(np.uint32)
    e = ((bits >> 23) & np.uint32(255)).astype(np.int32) - np.int32(127)
    m = ((bits & np.uint32(0x7FFFFF)) | np.uint32(0x3F800000)).view(np.float32)
    num = np.float32(m - np.float32(1.0))
    den = np.float32(m + np.float32(1.0))
    z = np.float32(num / den)
    z2 = np.float32(z * z)
    z3 = np.float32(z2 * z)
    z5 = np.float32(z3 * z2)
    z7 = np.float32(z5 * z2)
    z9 = np.float32(z7 * z2)
    third, fifth = _f32bits(0x3EAAAAAB), _f32bits(0x3E4CCCCD)
    seventh, ninth = _f32bits(0x3E124925), _f32bits(0x3DE38E39)
    s = z
    s = np.float32(s + np.float32(z3 * third))
    s = np.float32(s + np.float32(z5 * fifth))
    s = np.float32(s + np.float32(z7 * seventh))
    s = np.float32(s + np.float32(z9 * ninth))
    ln_m = np.float32(np.float32(2.0) * s)
    return np.float32(ln_m + np.float32(e.astype(np.float32) * _LN232))


def _exp32_deriv(x):
    """exp32 的值，以及它对输入的导数。和 CUDA exp32_pair 同一顺序。"""
    x = np.asarray(x, np.float32)
    x = np.clip(x, np.float32(-80.0), np.float32(80.0))
    nf = np.rint(np.float32(x * _LOG2E32)).astype(np.float32)
    t = np.float32(x - np.float32(nf * _LN232))
    p = np.broadcast_to(_EXP_C[0], np.shape(x)).astype(np.float32).copy()
    dp = np.zeros(np.shape(x), np.float32)
    for c in _EXP_C[1:]:
        dp = np.float32(np.float32(dp * t) + p)
        p = np.float32(np.float32(p * t) + c)
    n = nf.astype(np.int32)
    one_bits = np.float32(1.0).view(np.int32)
    scale = (one_bits + n * np.int32(8388608)).view(np.uint32).view(np.float32)
    y = (p.view(np.uint32).view(np.int32) + n * np.int32(8388608)).view(np.uint32).view(np.float32)
    dy = np.float32(dp * scale)
    return y, dy


def _silu32_pair(v):
    """与 CUDA silu32_pair 同一顺序。返回 (y, dy/dv)。"""
    v = np.asarray(v, np.float32)
    neg = np.float32(v * np.float32(-1.0))
    s, ds_dneg = _exp32_deriv(neg)
    # 用 exp 的数学导数 s，而不是多项式的分段导数。有限差分对这个更稳，
    # GPU 的 silu32_pair 用同一条。
    ds = np.float32(s * np.float32(-1.0))
    den = np.float32(s + np.float32(1.0))
    inv = np.float32(np.float32(1.0) / den)
    y = np.float32(v * inv)
    ninv = np.float32(np.float32(np.float32(-1.0) * inv) * inv)
    dinv = np.float32(ninv * ds)
    d = np.float32(inv + np.float32(v * dinv))
    return y, d


def _ce32(logits, tgt):
    """与 CUDA ce_fwd / ce_bwd 同一套 exp32、log32 和左折叠。"""
    logits = np.ascontiguousarray(logits, np.float32)
    tgt = np.asarray(tgt, np.int64).reshape(-1)
    n, v = logits.shape
    inv_n = np.float32(np.float32(1.0) / np.float32(n))
    acc = np.float32(0.0)
    dl = np.empty((n, v), np.float32)
    one = np.float32(1.0)
    for i in range(int(n)):
        row = logits[i]
        m = np.float32(row[0])
        for j in range(1, int(v)):
            m = np.float32(np.maximum(m, np.float32(row[j])))
        ex = np.empty(int(v), np.float32)
        s = np.float32(0.0)
        for j in range(int(v)):
            e = np.float32(_exp32(np.float32(np.float32(row[j]) - m)))
            ex[j] = e
            s = np.float32(s + e)
        inv_sum = np.float32(one / s)
        term = np.float32(np.float32(m + np.float32(_log32(s))) - np.float32(row[int(tgt[i])]))
        acc = np.float32(acc + term)
        ti = int(tgt[i])
        for j in range(int(v)):
            p = np.float32(ex[j] * inv_sum)
            if j == ti:
                p = np.float32(p - one)
            dl[i, j] = np.float32(p * inv_n)
    loss = np.float32(acc / np.float32(n))
    return loss, dl


def _cross_entropy_fused(logits, tgt):
    loss_v, dl = _ce32(logits.data, tgt)
    out = Tensor(np.asarray(loss_v, np.float32), requires_grad=logits.requires_grad, _parents=(logits,))

    def bw(g, grads):
        if not logits.requires_grad:
            return
        scale = np.asarray(g, np.float32).reshape(-1)[0]
        grad = dl if scale == np.float32(1.0) else np.float32(np.float32(scale) * dl)
        _acc(grads, logits, grad)

    out._backward = bw
    return out

__all__ = [
    "Tensor", "Parameter", "no_grad", "set_device", "get_device",
    "Module", "Linear", "Embedding", "RMSNorm", "Sequential", "ModuleList",
    "CausalSelfAttention", "MoE",
    "functional", "nn", "optim",
    "tensor", "zeros", "ones", "full", "randn", "rand", "arange", "from_numpy",
    "manual_seed", "cat", "stack", "matmul", "bmm", "einsum", "permute",
    "exp", "log", "sqrt", "abs", "clamp", "max", "min", "topk", "where",
    "save", "load",
    "dtype", "float32", "float64", "int32", "int64", "long",
    "device", "cuda",
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
    """加载 train/cuda.py。导入会临时占用包属性名 cuda，这里把公开名字空间放回去。"""
    import importlib
    import sys
    pkg = sys.modules[__name__]
    saved = pkg.__dict__.get("cuda")
    mod = importlib.import_module(__name__ + ".cuda")
    if saved is not None and saved is not mod:
        pkg.__dict__["cuda"] = saved
    return mod.runtime()


class DType:
    """对应 torch.dtype。浮点运算仍一律存成 float32，见模块说明。"""

    def __init__(self, name, numpy_dtype, floating):
        self.name = name
        self.numpy_dtype = np.dtype(numpy_dtype)
        self.is_floating_point = bool(floating)

    def __repr__(self):
        return "torch." + self.name

    def __eq__(self, other):
        if isinstance(other, DType):
            return self.numpy_dtype == other.numpy_dtype and self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash((self.name, str(self.numpy_dtype)))


dtype = DType
float32 = DType("float32", np.float32, True)
float64 = DType("float64", np.float64, True)
int32 = DType("int32", np.int32, False)
int64 = DType("int64", np.int64, False)
long = int64
_DTYPES = (float32, float64, int32, int64)


def _as_dtype(dt):
    if dt is None:
        return None
    if isinstance(dt, DType):
        return dt
    if dt is int:
        return int64
    if dt is float:
        return float64
    try:
        np_dt = np.dtype(dt)
    except TypeError:
        raise TypeError("不认识的 dtype: %r" % (dt,))
    for item in _DTYPES:
        if item.numpy_dtype == np_dt:
            return item
    return DType(str(np_dt), np_dt, np.issubdtype(np_dt, np.floating))


def _is_float_dtype(dt):
    wrapped = _as_dtype(dt)
    return wrapped is None or wrapped.is_floating_point


def _numpy_dtype(dt):
    return _as_dtype(dt).numpy_dtype


def _wrap_dtype(np_dt):
    np_dt = np.dtype(np_dt)
    for item in _DTYPES:
        if item.numpy_dtype == np_dt:
            return item
    return DType(str(np_dt), np_dt, np.issubdtype(np_dt, np.floating))


class Device:
    """对应 torch.device。index 只做记录；真正建张量时只用 type（cpu / cuda）。"""

    def __init__(self, type, index=None):
        if isinstance(type, Device):
            if index is not None:
                raise RuntimeError("device() 不能同时从已有 device 再给 index")
            self.type = type.type
            self.index = type.index
            return
        text = str(type).strip()
        if ":" in text:
            name, idx_s = text.split(":", 1)
            parsed = int(idx_s)
            if index is not None and int(index) != parsed:
                raise RuntimeError("device 字符串里的 index 和参数冲突")
            self.type = name
            self.index = parsed
        else:
            self.type = text
            self.index = None if index is None else int(index)

    def __repr__(self):
        if self.index is None:
            return "device(type=%r)" % self.type
        return "device(type=%r, index=%s)" % (self.type, self.index)

    def __str__(self):
        if self.index is None:
            return self.type
        return "%s:%s" % (self.type, self.index)

    def __eq__(self, other):
        if isinstance(other, Device):
            return self.type == other.type and self.index == other.index
        if isinstance(other, str):
            return str(self) == other or self.type == other
        return NotImplemented


device = Device


def _device_arg(dev):
    if dev is None:
        return None
    if isinstance(dev, Device):
        return dev.type
    if isinstance(dev, str):
        if ":" in dev:
            return dev.split(":", 1)[0]
        return dev
    raise TypeError("device 需要 str 或 torch.device，收到 %r" % (dev,))


class _CudaAPI:
    """对应 torch.cuda 的只读查询。不在 import 时加载驱动。"""

    @staticmethod
    def is_available():
        try:
            _cuda()
        except Exception:
            return False
        return True

    @staticmethod
    def device_count():
        return 1 if _CudaAPI.is_available() else 0

    @staticmethod
    def current_device():
        if not _CudaAPI.is_available():
            raise RuntimeError("CUDA 不可用")
        return 0


cuda = _CudaAPI()

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

    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "device", "_gpu",
                 "_lazy_T", "_lazy_shape")
    __array_priority__ = 1000        # 让 ndarray <op> Tensor 走我们的 __r*__

    def __init__(self, data, requires_grad=False, _parents=(), _backward=None, device=None, dtype=None):
        if device is not None and not isinstance(device, str):
            device = _device_arg(device)
        dev = _DEVICE if device is None else device
        self.device = dev
        self._gpu = None
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self._parents = _parents
        self._backward = _backward
        self._lazy_T = None
        self._lazy_shape = None
        if dev == "cpu":
            # 整数/布尔只当显式 dtype 给出时才保留。默认路径与原来一样，强制 float32。
            if dtype is not None and not _is_float_dtype(dtype):
                if requires_grad:
                    raise RuntimeError("只有浮点张量可以 requires_grad")
                self.data = np.asarray(data, dtype=_numpy_dtype(dtype))
                return
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
        obj._lazy_T = None
        obj._lazy_shape = None
        return obj

    def _cpu_only(self, op):
        if self.device == "cuda":
            raise NotImplementedError("CUDA 后端还没有 %s" % op)

    # ---- 属性 ----
    @property
    def shape(self):
        lazy = getattr(self, "_lazy_shape", None)
        if lazy is not None:
            return lazy
        if self.device == "cuda":
            return self._gpu.shape
        return self.data.shape

    @property
    def ndim(self):
        if self.device == "cuda":
            return len(self.shape)
        return self.data.ndim

    @property
    def dtype(self):
        if self.data is None:
            return float32
        return _wrap_dtype(self.data.dtype)

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
        # CPU：和原来一样拷贝 data。CUDA：data 是 None，经 to_numpy 拉回主机。
        if self.device == "cuda":
            return Tensor(self.to_numpy().copy())
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
        # 任意前导维（nn.Linear 的 ND 输入走这里）。2D x 2D 和 3D x 3D 仍走上面的原路径。
        if self.ndim >= 1 and other.ndim >= 1:
            out = Tensor(np.matmul(self.data, other.data), requires_grad=rg, _parents=(self, other))
            a_data, b_data = self.data, other.data

            def bw(g, grads):
                da, db = _matmul_grad_numpy(g, a_data, b_data)
                if self.requires_grad:
                    _acc(grads, self, da)
                if other.requires_grad:
                    _acc(grads, other, db)
            out._backward = bw
            return out
        raise NotImplementedError("matmul 支持 2D/3D 以及带前导维的矩阵乘，收到 %dD x %dD"
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
        src = getattr(other, "_lazy_T", None)
        if src is not None and self.ndim == 2 and getattr(src, "ndim", 0) == 2:
            M, K = int(self.shape[0]), int(self.shape[1])
            N, K2 = int(src.shape[0]), int(src.shape[1])
            if K != K2:
                raise ValueError("matmul K 不匹配: %s @ %s^T" % (self.shape, src.shape))
            if not rt._numpy_exact_gemm(M, N, K):
                buf = rt.gemm_nt(self._gpu, src._gpu)
                rg = self.requires_grad or src.requires_grad
                out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(self, src) if rg else ())
                a, w = self._gpu, src._gpu

                def bw(g, grads, _a=a, _w=w, _self=self, _src=src):
                    # dA = g @ W，dW = g^T @ A。不再经过物化的 W^T。
                    if _self.requires_grad:
                        _acc(grads, _self, rt.sgemm(g, _w))
                    if _src.requires_grad:
                        _acc(grads, _src, rt.gemm_tn(g, _a))
                out._backward = bw
                return out
            buf_t = rt.transpose2d(src._gpu)
            other = Tensor._from_gpu(
                buf_t, requires_grad=src.requires_grad,
                _parents=(src,) if src.requires_grad else ())

            def bw_t(g, grads, _src=src):
                if _src.requires_grad:
                    _acc(grads, _src, rt.transpose2d(g))
            other._backward = bw_t
        if self.ndim != 2 or other.ndim != 2:
            if self.ndim >= 2 and other.ndim >= 2:
                return self._matmul_cuda_nd(other)
            raise NotImplementedError("CUDA matmul 目前只支持至少 2D，收到 %dD x %dD"
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

    def _materialize_lazy_T(self):
        """把惰性转置还原成一个真实节点（带标准 transpose 反向）。

        `.T` 默认是惰性的（只为 2D matmul 省一次拷贝）。N 维 matmul 读的是原始缓冲区，
        不认识惰性状态，所以这条路径必须先实体化 —— 否则形状对不上
        （实测症状：`linear_nd` 报 matmul K 不匹配 (2,3,16) @ (8,16)）。
        """
        rt = _cuda()
        base = self._lazy_T
        buf = rt.transpose2d(base._gpu)
        out = Tensor._from_gpu(buf, requires_grad=base.requires_grad, _parents=(base,))

        def bw(g, grads):
            if base.requires_grad:
                _acc(grads, base, rt.transpose2d(g))

        out._backward = bw
        return out

    def _matmul_cuda_nd(self, other):
        """最后两维 matmul，前导维广播。反向先乘再 unbroadcast 回原形状。"""
        rt = _cuda()
        # 惰性转置在这条路径上必须先落地（原缓冲区形状是转置前的）
        if getattr(self, "_lazy_T", None) is not None:
            self = self._materialize_lazy_T()
        if getattr(other, "_lazy_T", None) is not None:
            other = other._materialize_lazy_T()
        buf = rt.matmul_general(self._gpu, other._gpu)
        rg = self.requires_grad or other.requires_grad
        out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(self, other))
        a, b = self._gpu, other._gpu
        a_shape = tuple(int(s) for s in self.shape)
        b_shape = tuple(int(s) for s in other.shape)

        def bw(g, grads):
            if self.requires_grad:
                da = rt.matmul_general(g, rt.swap_last2(b))
                if tuple(int(s) for s in da.shape) != a_shape:
                    da = rt.unbroadcast(da, a_shape)
                _acc(grads, self, da)
            if other.requires_grad:
                db = rt.matmul_general(rt.swap_last2(a), g)
                if tuple(int(s) for s in db.shape) != b_shape:
                    db = rt.unbroadcast(db, b_shape)
                _acc(grads, other, db)
        out._backward = bw
        return out

    def _transpose_cuda(self):
        if self.ndim != 2:
            raise NotImplementedError("CUDA transpose 目前只支持 2D，收到 %dD" % self.ndim)
        # 不物化。matmul 看见 _lazy_T 就走 gemm_nt，省掉词表那么大的转置。
        # 真要读数值时 to_numpy 再转。小矩阵的精确 GEMM 会在 matmul 里物化。
        rt = _cuda()
        out = Tensor._from_gpu(self._gpu, requires_grad=self.requires_grad, _parents=(self,))
        out._lazy_T = self
        out._lazy_shape = (int(self.shape[1]), int(self.shape[0]))

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
            # 只给叶子张量留 .grad，和 PyTorch 一样。中间激活的梯度以前每次都 clone
            # 一份，seq=8192 上这 100 多次拷贝既占时间也把显存顶过 13 GiB。
            # 叶子的 g 是父算子刚写出来的新缓冲，反向不再改它，可以直接挂上，不必再拷。
            if node._backward is None:
                if node.grad is None:
                    node.grad = Tensor._from_gpu(g, requires_grad=False)
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
        if self.device == "cuda":
            self._cpu_only("exp")
        e = np.exp(self.data)
        out = Tensor(e, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, g * e)
        out._backward = bw
        return out

    def log(self):
        if self.device == "cuda":
            self._cpu_only("log")
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
        new = self.data.reshape(shape)
        if not np.issubdtype(self.data.dtype, np.floating):
            return Tensor(new, dtype=self.dtype, requires_grad=False)
        out = Tensor(new, requires_grad=self.requires_grad, _parents=(self,))
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

    def transpose(self, dim0=None, dim1=None):
        # 不传参数：原来的完整转置（.T 走这里）。2D 上与 transpose(0, 1) 相同。
        if dim0 is None and dim1 is None:
            if self.device == "cuda":
                return self._transpose_cuda()
            out = Tensor(self.data.T, requires_grad=self.requires_grad, _parents=(self,))

            def bw(g, grads):
                _acc(grads, self, g.T)
            out._backward = bw
            return out
        if dim0 is None or dim1 is None:
            raise TypeError("transpose() 要么不传参数，要么同时给 dim0 和 dim1")
        if self.device == "cuda":
            nd = self.ndim
            a = int(dim0) + (nd if int(dim0) < 0 else 0)
            b = int(dim1) + (nd if int(dim1) < 0 else 0)
            if nd == 2 and {a, b} == {0, 1}:
                return self._transpose_cuda()
            # 更高维走 permute。下面和 CPU 共用轴交换。
        nd = self.ndim
        a = int(dim0)
        b = int(dim1)
        if a < 0:
            a += nd
        if b < 0:
            b += nd
        axes = list(range(nd))
        axes[a], axes[b] = axes[b], axes[a]
        return self.permute(*axes)

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        nd = self.ndim
        axes = []
        for d in dims:
            d = int(d)
            if d < 0:
                d += nd
            axes.append(d)
        if sorted(axes) != list(range(nd)):
            raise ValueError("permute 必须是 0..%d 的一个排列，收到 %s" % (nd - 1, axes))
        if self.device == "cuda":
            return self._permute_cuda(tuple(axes))
        moved = np.transpose(self.data, axes)
        if not np.issubdtype(self.data.dtype, np.floating):
            return Tensor(moved, dtype=self.dtype, requires_grad=False)
        out = Tensor(moved, requires_grad=self.requires_grad, _parents=(self,))
        inv = tuple(int(i) for i in np.argsort(axes))

        def bw(g, grads):
            _acc(grads, self, np.transpose(g, inv))
        out._backward = bw
        return out

    def _permute_cuda(self, axes):
        rt = _cuda()
        buf = rt.permute(self._gpu, axes)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))
        inv = [0] * len(axes)
        for i, src in enumerate(axes):
            inv[src] = i
        inv = tuple(inv)

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.permute(g, inv))
        out._backward = bw
        return out

    def contiguous(self):
        if self.device == "cuda":
            return self
        if self.data.flags.c_contiguous:
            return self
        packed = np.ascontiguousarray(self.data)
        if not np.issubdtype(self.data.dtype, np.floating):
            return Tensor(packed, dtype=self.dtype, requires_grad=False)
        out = Tensor(packed, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, np.asarray(g, dtype=np.float32).reshape(self.shape))
        out._backward = bw
        return out

    def view(self, *shape):
        # 近似：非连续时先拷贝。PyTorch 的 view 在非连续上会直接报错。
        if self.device == "cuda":
            return self.reshape(*shape)
        if self.data.flags.c_contiguous:
            return self.reshape(*shape)
        return self.contiguous().reshape(*shape)

    def size(self, dim=None):
        if dim is None:
            return self.shape
        d = int(dim)
        if d < 0:
            d += self.ndim
        return int(self.shape[d])

    def dim(self):
        return int(self.ndim)

    def abs(self):
        if self.device == "cuda":
            self._cpu_only("abs")
        out = Tensor(np.abs(self.data), requires_grad=self.requires_grad, _parents=(self,))
        sign = np.sign(self.data).astype(np.float32)

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, g * sign)
        out._backward = bw
        return out

    def sqrt(self):
        return self ** 0.5

    def clamp(self, min=None, max=None):
        if self.device == "cuda":
            self._cpu_only("clamp")
        lo = _scalar_or_none(min, "clamp.min")
        hi = _scalar_or_none(max, "clamp.max")
        y = self.data
        if lo is not None:
            y = np.maximum(y, np.float32(lo))
        if hi is not None:
            y = np.minimum(y, np.float32(hi))
        out = Tensor(y, requires_grad=self.requires_grad, _parents=(self,))
        src = self.data

        def bw(g, grads):
            if not self.requires_grad:
                return
            mask = np.ones(src.shape, dtype=np.float32)
            if lo is not None:
                mask = np.where(src < lo, np.float32(0), mask)
            if hi is not None:
                mask = np.where(src > hi, np.float32(0), mask)
            _acc(grads, self, g * mask)
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
            src = getattr(self, "_lazy_T", None)
            if src is not None:
                return _cuda().d2h(_cuda().transpose2d(src._gpu))
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


def _matmul_grad_numpy(g, a, b):
    """np.matmul 的反向。返回与 a、b 同形状的 (da, db)。"""
    a_shape = a.shape
    b_shape = b.shape
    a_1 = a.ndim == 1
    b_1 = b.ndim == 1
    g = np.asarray(g, dtype=np.float32)
    if a_1:
        a = a.reshape(1, a.shape[0])
    if b_1:
        b = b.reshape(b.shape[0], 1)
    if a_1 and b_1:
        g2 = g.reshape(1, 1)
    elif a_1:
        g2 = np.expand_dims(g, axis=-2)
    elif b_1:
        g2 = np.expand_dims(g, axis=-1)
    else:
        g2 = g
    da = np.matmul(g2, np.swapaxes(b, -1, -2))
    db = np.matmul(np.swapaxes(a, -1, -2), g2)
    if a_1:
        da = np.squeeze(da, axis=-2)
    if b_1:
        db = np.squeeze(db, axis=-1)
    da = _unbroadcast(np.asarray(da, dtype=np.float32), a_shape)
    db = _unbroadcast(np.asarray(db, dtype=np.float32), b_shape)
    return da, db


def _scalar_or_none(v, name):
    if v is None:
        return None
    if isinstance(v, Tensor):
        if v.numel() != 1:
            raise NotImplementedError("%s 目前只支持标量" % name)
        return float(np.asarray(v.to_numpy()).reshape(-1)[0])
    if isinstance(v, (bool, np.bool_)):
        raise TypeError("%s 不接受布尔值" % name)
    return float(v)


def _as_index_array(target):
    if isinstance(target, Tensor):
        target = target.to_numpy()
    return np.asarray(target, dtype=np.int64)


def _normalize_size(size):
    if len(size) == 1 and isinstance(size[0], (tuple, list, np.ndarray)):
        return tuple(int(x) for x in size[0])
    return tuple(int(x) for x in size)


def _infer_dtype(data):
    if isinstance(data, Tensor):
        return data.dtype
    if isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.floating):
            return float32
        if data.dtype == np.int32:
            return int32
        if np.issubdtype(data.dtype, np.integer):
            return int64
        raise TypeError("tensor 暂不支持 dtype=%s" % data.dtype)
    arr = np.asarray(data)
    if arr.dtype == np.bool_:
        raise TypeError("bool 张量尚未实现")
    if np.issubdtype(arr.dtype, np.floating):
        return float32
    if np.issubdtype(arr.dtype, np.integer):
        return int64
    return float32


def tensor(data, dtype=None, device=None, requires_grad=False):
    if isinstance(data, Tensor):
        data = data.to_numpy()
    if dtype is None:
        dtype = _infer_dtype(data)
    else:
        dtype = _as_dtype(dtype)
    dev = _device_arg(device)
    if _is_float_dtype(dtype):
        return Tensor(data, requires_grad=requires_grad, device=dev)
    if requires_grad:
        raise RuntimeError("只有浮点张量可以 requires_grad")
    return Tensor(data, requires_grad=False, device=dev, dtype=dtype)


def _constant(shape, value, dt, device, requires_grad):
    dev = _device_arg(device)
    if dt is None or _is_float_dtype(dt):
        data = np.full(shape, value, dtype=np.float32)
        return Tensor(data, requires_grad=requires_grad, device=dev)
    if requires_grad:
        raise RuntimeError("只有浮点张量可以 requires_grad")
    data = np.full(shape, value, dtype=_numpy_dtype(dt))
    return Tensor(data, requires_grad=False, device=dev, dtype=_as_dtype(dt))


def zeros(*size, dtype=None, device=None, requires_grad=False):
    return _constant(_normalize_size(size), 0, dtype, device, requires_grad)


def ones(*size, dtype=None, device=None, requires_grad=False):
    return _constant(_normalize_size(size), 1, dtype, device, requires_grad)


def full(size, fill_value, dtype=None, device=None, requires_grad=False):
    if isinstance(size, (int, np.integer)) and not isinstance(size, bool):
        shape = (int(size),)
    else:
        shape = tuple(int(x) for x in size)
    return _constant(shape, fill_value, dtype, device, requires_grad)


def randn(*size, dtype=None, device=None, requires_grad=False):
    if dtype is not None and not _is_float_dtype(dtype):
        raise TypeError("randn 只支持浮点 dtype")
    data = np.random.standard_normal(_normalize_size(size)).astype(np.float32)
    return Tensor(data, requires_grad=requires_grad, device=_device_arg(device))


def rand(*size, dtype=None, device=None, requires_grad=False):
    if dtype is not None and not _is_float_dtype(dtype):
        raise TypeError("rand 只支持浮点 dtype")
    data = np.random.random(_normalize_size(size)).astype(np.float32)
    return Tensor(data, requires_grad=requires_grad, device=_device_arg(device))


def arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False):
    if end is None:
        end = start
        start = 0
    if dtype is None:
        vals = (start, end, step)
        integral = all(
            isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
            for v in vals
        )
        dtype = int64 if integral else float32
    else:
        dtype = _as_dtype(dtype)
    dev = _device_arg(device)
    if _is_float_dtype(dtype):
        data = np.arange(start, end, step, dtype=np.float32)
        return Tensor(data, requires_grad=requires_grad, device=dev)
    if requires_grad:
        raise RuntimeError("只有浮点张量可以 requires_grad")
    data = np.arange(start, end, step, dtype=_numpy_dtype(dtype))
    return Tensor(data, requires_grad=False, device=dev, dtype=dtype)


def manual_seed(seed):
    """种子打进全局 numpy RNG。nn.Linear / Embedding 的初始化也吃这一个 RNG。

    近似：没有 torch.Generator，也没有单独的 CUDA 随机流。
    两次 randn 在同一种子下一致，但数值不会和 PyTorch 逐位相同。
    """
    np.random.seed(int(seed))
    return int(seed)


def from_numpy(ndarray):
    """近似：总是拷贝，不和调用方共享内存。浮点收成 float32。"""
    if not isinstance(ndarray, np.ndarray):
        raise TypeError("from_numpy 需要 numpy.ndarray，收到 %s" % type(ndarray).__name__)
    if ndarray.dtype == np.bool_:
        raise TypeError("bool 张量尚未实现")
    if np.issubdtype(ndarray.dtype, np.floating):
        return Tensor(np.asarray(ndarray, dtype=np.float32).copy())
    if ndarray.dtype == np.int32:
        return Tensor(ndarray.copy(), dtype=int32)
    if np.issubdtype(ndarray.dtype, np.integer):
        return Tensor(np.asarray(ndarray, dtype=np.int64).copy(), dtype=int64)
    raise TypeError("from_numpy 暂不支持 dtype=%s" % ndarray.dtype)


def cat(tensors, dim=0):
    ts = [_as_tensor(t) for t in tensors]
    if not ts:
        raise ValueError("cat 至少需要一个张量")
    if any(getattr(t, "device", "cpu") == "cuda" for t in ts):
        raise NotImplementedError("CUDA cat 尚未实现")
    axis = int(dim)
    if axis < 0:
        axis += ts[0].ndim
    data = np.concatenate([t.data for t in ts], axis=axis)
    floating = all(np.issubdtype(t.data.dtype, np.floating) for t in ts)
    if not floating:
        return Tensor(data, dtype=ts[0].dtype, requires_grad=False)
    rg = any(t.requires_grad for t in ts)
    out = Tensor(data, requires_grad=rg, _parents=tuple(ts))
    sizes = [int(t.shape[axis]) for t in ts]

    def bw(g, grads):
        parts = np.split(np.asarray(g, dtype=np.float32), np.cumsum(sizes)[:-1], axis=axis)
        for t, part in zip(ts, parts):
            if t.requires_grad:
                _acc(grads, t, part)
    out._backward = bw
    return out


def stack(tensors, dim=0):
    ts = [_as_tensor(t) for t in tensors]
    if not ts:
        raise ValueError("stack 至少需要一个张量")
    if any(getattr(t, "device", "cpu") == "cuda" for t in ts):
        raise NotImplementedError("CUDA stack 尚未实现")
    axis = int(dim)
    data = np.stack([t.data for t in ts], axis=axis)
    floating = all(np.issubdtype(t.data.dtype, np.floating) for t in ts)
    if not floating:
        return Tensor(data, dtype=ts[0].dtype, requires_grad=False)
    rg = any(t.requires_grad for t in ts)
    out = Tensor(data, requires_grad=rg, _parents=tuple(ts))

    def bw(g, grads):
        parts = np.split(np.asarray(g, dtype=np.float32), len(ts), axis=axis)
        for t, part in zip(ts, parts):
            if t.requires_grad:
                _acc(grads, t, np.squeeze(part, axis=axis))
    out._backward = bw
    return out


def matmul(input, other):
    return _as_tensor(input) @ _as_tensor(other)


def bmm(input, mat2):
    a = _as_tensor(input)
    b = _as_tensor(mat2)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("bmm 需要两个 3D 张量，收到 %s x %s" % (a.shape, b.shape))
    return a @ b


def exp(input):
    return _as_tensor(input).exp()


def log(input):
    return _as_tensor(input).log()


def sqrt(input):
    return _as_tensor(input).sqrt()


def _api_abs(input):
    return _as_tensor(input).abs()


def _api_clamp(input, min=None, max=None):
    return _as_tensor(input).clamp(min=min, max=max)


def _reduce_values_grad(data, grad, axis, keepdim, pick):
    idx = pick(data, axis=axis)
    full = np.zeros(data.shape, dtype=np.float32)
    g = np.asarray(grad, dtype=np.float32)
    idx_exp = np.expand_dims(idx, axis)
    if not keepdim:
        g = np.expand_dims(g, axis)
    np.put_along_axis(full, idx_exp, g, axis=axis)
    return full


def _reduce_pair(t, dim, keepdim, value_fn, index_fn):
    if getattr(t, "device", "cpu") == "cuda":
        raise NotImplementedError("CUDA max/min 尚未实现")
    floating = np.issubdtype(t.data.dtype, np.floating)
    if dim is None:
        flat = t.data.reshape(-1)
        flat_i = int(index_fn(flat))
        raw = flat[flat_i]
        if not floating:
            return Tensor(np.array(raw), dtype=t.dtype, requires_grad=False)
        out = Tensor(np.array(raw, dtype=np.float32), requires_grad=t.requires_grad, _parents=(t,))

        def bw(g, grads):
            if not t.requires_grad:
                return
            full = np.zeros(t.shape, dtype=np.float32)
            full.reshape(-1)[flat_i] = np.asarray(g, dtype=np.float32).reshape(-1)[0]
            _acc(grads, t, full)
        out._backward = bw
        return out
    axis = int(dim)
    if axis < 0:
        axis += t.ndim
    values = value_fn(t.data, axis=axis, keepdims=keepdim)
    indices = index_fn(t.data, axis=axis)
    if keepdim:
        indices = np.expand_dims(indices, axis)
    it = Tensor(indices, dtype=int64)
    if not floating:
        return Tensor(values, dtype=t.dtype, requires_grad=False), it
    vt = Tensor(values, requires_grad=t.requires_grad, _parents=(t,))

    def bw(g, grads):
        if t.requires_grad:
            _acc(grads, t, _reduce_values_grad(t.data, g, axis, keepdim, index_fn))
    vt._backward = bw
    return vt, it


def _api_max(input, dim=None, keepdim=False):
    return _reduce_pair(_as_tensor(input), dim, keepdim, np.max, np.argmax)


def _api_min(input, dim=None, keepdim=False):
    return _reduce_pair(_as_tensor(input), dim, keepdim, np.min, np.argmin)


def topk(input, k, dim=-1, largest=True, sorted=True):
    """近似：并列时名次跟 numpy.argpartition，不保证和 PyTorch 的稳定排序一致。"""
    t = _as_tensor(input)
    if getattr(t, "device", "cpu") == "cuda":
        raise NotImplementedError("CUDA topk 尚未实现")
    k = int(k)
    axis = int(dim)
    if axis < 0:
        axis += t.ndim
    n = int(t.shape[axis])
    if k < 0 or k > n:
        raise ValueError("topk 的 k=%d 超出维度长度 %d" % (k, n))
    data = t.data
    if k == 0:
        shape = list(data.shape)
        shape[axis] = 0
        idx = np.zeros(shape, dtype=np.int64)
        values = np.zeros(shape, dtype=np.float32 if np.issubdtype(data.dtype, np.floating) else data.dtype)
    else:
        if largest:
            part = np.argpartition(data, n - k, axis=axis)
            idx = np.take(part, np.arange(n - k, n), axis=axis)
        else:
            part = np.argpartition(data, k - 1, axis=axis)
            idx = np.take(part, np.arange(0, k), axis=axis)
        values = np.take_along_axis(data, idx, axis=axis)
        if sorted:
            key = -values if largest else values
            order = np.argsort(key, axis=axis)
            idx = np.take_along_axis(idx, order, axis=axis)
            values = np.take_along_axis(values, order, axis=axis)
    floating = np.issubdtype(data.dtype, np.floating)
    saved_idx = np.array(idx, dtype=np.int64, copy=True)
    it = Tensor(saved_idx, dtype=int64)
    if not floating:
        return Tensor(values, dtype=t.dtype, requires_grad=False), it
    vt = Tensor(np.asarray(values, dtype=np.float32), requires_grad=t.requires_grad, _parents=(t,))

    def bw(g, grads):
        if not t.requires_grad:
            return
        full = np.zeros(data.shape, dtype=np.float32)
        if k:
            np.put_along_axis(full, saved_idx, np.asarray(g, dtype=np.float32), axis=axis)
        _acc(grads, t, full)
    vt._backward = bw
    return vt, it


def where(condition, x=None, y=None):
    if isinstance(condition, Tensor):
        cond = condition.to_numpy()
    else:
        cond = np.asarray(condition)
    if x is None and y is None:
        return tuple(Tensor(np.asarray(a, dtype=np.int64), dtype=int64) for a in np.nonzero(cond))
    if x is None or y is None:
        raise TypeError("where 需要只给 condition，或者同时给 x 和 y")
    xt = _as_tensor(x)
    yt = _as_tensor(y)
    if getattr(xt, "device", "cpu") == "cuda" or getattr(yt, "device", "cpu") == "cuda":
        raise NotImplementedError("CUDA where 尚未实现")
    mask = np.asarray(cond, dtype=bool)
    floating = np.issubdtype(xt.data.dtype, np.floating) or np.issubdtype(yt.data.dtype, np.floating)
    if not floating:
        return Tensor(np.where(mask, xt.data, yt.data), dtype=xt.dtype, requires_grad=False)
    data = np.where(mask, xt.data, yt.data).astype(np.float32)
    rg = xt.requires_grad or yt.requires_grad
    out = Tensor(data, requires_grad=rg, _parents=(xt, yt))

    def bw(g, grads):
        gg = np.asarray(g, dtype=np.float32)
        if xt.requires_grad:
            _acc(grads, xt, _unbroadcast(np.where(mask, gg, np.float32(0)), xt.shape))
        if yt.requires_grad:
            _acc(grads, yt, _unbroadcast(np.where(mask, np.float32(0), gg), yt.shape))
    out._backward = bw
    return out


def _split_einsum(equation, operands):
    eq = equation.replace(" ", "")
    if "..." in eq:
        raise NotImplementedError("einsum 暂不支持省略号")
    if "->" in eq:
        lhs, rhs = eq.split("->")
    else:
        lhs = eq
        counts = {}
        for ch in lhs:
            if ch.isalpha():
                counts[ch] = counts.get(ch, 0) + 1
        rhs = "".join(ch for ch in sorted(counts) if counts[ch] == 1)
    labels = lhs.split(",") if lhs else []
    if len(labels) != len(operands):
        raise ValueError("einsum 操作数个数 %d 和方程 %r 不一致" % (len(operands), equation))
    for lab in labels + [rhs]:
        letters = [c for c in lab if c.isalpha()]
        if len(letters) != len(set(letters)):
            raise NotImplementedError("einsum 暂不支持重复字母: %r" % equation)
        if any(not c.isalpha() for c in lab):
            raise NotImplementedError("einsum 暂不支持字母以外的下标: %r" % equation)
    return labels, rhs


def _einsum_vjp(labels, rhs, operands, grad, index):
    target = labels[index]
    others_l = [labels[j] for j in range(len(operands)) if j != index]
    others_a = [operands[j].data for j in range(len(operands)) if j != index]
    provided = set(rhs)
    for lab in others_l:
        provided.update(lab)
    missing = [c for c in target if c not in provided]
    known = "".join(c for c in target if c in provided)
    g = np.asarray(grad, dtype=np.float32)
    if not missing:
        if rhs == "" and not others_l:
            return np.broadcast_to(g.reshape(()), operands[index].shape).astype(np.float32).copy()
        if rhs == "":
            gi = np.einsum(",".join(others_l) + "->" + target, *others_a) * g.reshape(())
        else:
            gi = np.einsum(",".join([rhs] + others_l) + "->" + target, g, *others_a)
        return np.asarray(gi, dtype=np.float32).reshape(operands[index].shape)
    if known == "" and not others_l and rhs == "":
        return np.broadcast_to(g.reshape(()), operands[index].shape).astype(np.float32).copy()
    if rhs == "":
        if known == "":
            partial = g.reshape(())
        else:
            partial = np.asarray(np.einsum(",".join(others_l) + "->" + known, *others_a), dtype=np.float32)
            partial = partial * g.reshape(())
    elif known == "":
        partial = np.asarray(np.einsum(",".join([rhs] + others_l) + "->", g, *others_a), dtype=np.float32)
    else:
        partial = np.asarray(
            np.einsum(",".join([rhs] + others_l) + "->" + known, g, *others_a), dtype=np.float32)
    sizes = {c: n for c, n in zip(target, operands[index].shape)}
    ones_labels = "".join(missing)
    ones = np.ones(tuple(sizes[c] for c in ones_labels), dtype=np.float32)
    if known == "":
        gi = np.einsum(ones_labels + "->" + target, ones) * np.asarray(partial, dtype=np.float32).reshape(())
    else:
        gi = np.einsum(known + "," + ones_labels + "->" + target, np.asarray(partial, dtype=np.float32), ones)
    return np.asarray(gi, dtype=np.float32).reshape(operands[index].shape)


def einsum(equation, *operands):
    """近似：不支持省略号和重复字母。浮点结果收成 float32。"""
    ops = [_as_tensor(o) for o in operands]
    if any(getattr(o, "device", "cpu") == "cuda" for o in ops):
        raise NotImplementedError("CUDA einsum 尚未实现")
    if not ops or not all(np.issubdtype(o.data.dtype, np.floating) for o in ops):
        raise NotImplementedError("einsum 目前只支持浮点张量")
    labels, rhs = _split_einsum(equation, ops)
    for o, lab in zip(ops, labels):
        if o.ndim != len(lab):
            raise ValueError("einsum %r 的字母数和形状 %s 不一致" % (lab, o.shape))
    out_data = np.einsum(",".join(labels) + "->" + rhs, *[o.data for o in ops])
    rg = any(o.requires_grad for o in ops)
    out = Tensor(np.asarray(out_data, dtype=np.float32), requires_grad=rg, _parents=tuple(ops))

    def bw(g, grads):
        for i, o in enumerate(ops):
            if o.requires_grad:
                _acc(grads, o, _einsum_vjp(labels, rhs, ops, g, i))
    out._backward = bw
    return out


def _pack(obj):
    if isinstance(obj, Tensor):
        return {
            "__tgai__": "tensor",
            "data": np.array(obj.to_numpy(), copy=True),
            "requires_grad": bool(obj.requires_grad),
        }
    if isinstance(obj, Module):
        return {"__tgai__": "state_dict", "data": obj.state_dict()}
    if isinstance(obj, dict):
        return {"__tgai__": "dict", "data": {str(k): _pack(v) for k, v in obj.items()}}
    if isinstance(obj, tuple):
        return {"__tgai__": "tuple", "data": [_pack(v) for v in obj]}
    if isinstance(obj, list):
        return {"__tgai__": "list", "data": [_pack(v) for v in obj]}
    if isinstance(obj, np.ndarray):
        return {"__tgai__": "ndarray", "data": np.array(obj, copy=True)}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError("save 不支持 %s。这不是 torch.save 的 zip 格式。" % type(obj).__name__)


def _unpack(obj):
    if not isinstance(obj, dict) or "__tgai__" not in obj:
        return obj
    kind = obj["__tgai__"]
    if kind == "tensor":
        data = obj["data"]
        if np.issubdtype(data.dtype, np.floating):
            return Tensor(data, requires_grad=bool(obj.get("requires_grad", False)))
        return Tensor(data, dtype=_wrap_dtype(data.dtype), requires_grad=False)
    if kind == "ndarray":
        return obj["data"]
    if kind == "dict":
        return {k: _unpack(v) for k, v in obj["data"].items()}
    if kind == "list":
        return [_unpack(v) for v in obj["data"]]
    if kind == "tuple":
        return tuple(_unpack(v) for v in obj["data"])
    if kind == "state_dict":
        return {k: np.array(v, copy=True) for k, v in obj["data"].items()}
    raise TypeError("未知存档类型 %s" % kind)


def _move_loaded(obj, dev):
    if isinstance(obj, Tensor) and obj.data is not None and np.issubdtype(obj.data.dtype, np.floating):
        return Tensor(obj.to_numpy(), requires_grad=obj.requires_grad, device=dev)
    if isinstance(obj, dict):
        return {k: _move_loaded(v, dev) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_loaded(v, dev) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_loaded(v, dev) for v in obj)
    return obj


def save(obj, f):
    """近似：pickle 一份带标记的 numpy，不是 torch.save 的 zip，两边不能互读。"""
    import pickle
    payload = {"__tgai_save__": 1, "obj": _pack(obj)}
    if hasattr(f, "write"):
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    with open(f, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load(f, map_location=None):
    import pickle
    if hasattr(f, "read"):
        payload = pickle.load(f)
    else:
        with open(f, "rb") as fh:
            payload = pickle.load(fh)
    if not isinstance(payload, dict) or payload.get("__tgai_save__") != 1:
        raise TypeError("不是 tgai_go.train.save 写出的文件（不能读 torch.save 的 zip）")
    obj = _unpack(payload["obj"])
    dest = None
    if isinstance(map_location, Device):
        dest = map_location.type
    elif isinstance(map_location, str):
        dest = map_location.split(":", 1)[0]
    if dest not in (None, "cpu"):
        obj = _move_loaded(obj, dest)
    return obj


# ─────────────────────────── functional ───────────────────────────
class _Functional:
    @staticmethod
    def silu(x: Tensor) -> Tensor:
        if getattr(x, "device", "cpu") == "cuda":
            return _silu_cuda(x)
        y, d = _silu32_pair(x.data)
        out = Tensor(y, requires_grad=x.requires_grad, _parents=(x,))

        def bw(g, grads):
            if x.requires_grad:
                _acc(grads, x, np.float32(np.asarray(g, np.float32) * d))
        out._backward = bw
        return out

    @staticmethod
    def softmax(x: Tensor, axis=-1, dim=None) -> Tensor:
        if dim is not None:
            axis = dim
        if getattr(x, "device", "cpu") == "cuda":
            return _softmax_cuda(x, axis)
        m = x.data.max(axis=axis, keepdims=True)
        z = x - Tensor(m)                     # 常数，不参与求导
        e = z.exp()
        return e * (e.sum(axis=axis, keepdims=True) ** -1.0)

    @staticmethod
    def log_softmax(x: Tensor, axis=-1, dim=None) -> Tensor:
        if dim is not None:
            axis = dim
        if getattr(x, "device", "cpu") == "cuda":
            return _log_softmax_cuda(x, axis)
        m = x.data.max(axis=axis, keepdims=True)
        z = x - Tensor(m)
        e = z.exp()
        return z - e.sum(axis=axis, keepdims=True).log()

    @staticmethod
    def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, causal=False, is_causal=None,
                                     scale=None, lowmem=False):
        """q,k,v: [B*H, S, Dh]（已展平 head）。causal=True 时做下三角掩码。

        lowmem=True 且设备是 CUDA、dh==64、causal 时，走不物化 P 的内核。
        其它情况（含 dh!=64、非 causal、CPU）回退到下面的物化实现。默认 lowmem=False。
        """
        if is_causal is not None:
            causal = bool(is_causal)
        dh = int(q.shape[-1])
        if scale is None:
            scale = 1.0 / np.sqrt(dh)
        else:
            scale = float(scale)
        if lowmem and _sdpa_lowmem_ok(q, k, v, causal, dh):
            return _sdpa_lowmem(q, k, v, scale)
        return _sdpa_materialized(q, k, v, causal, scale)

    @staticmethod
    def sdpa_lowmem(q: Tensor, k: Tensor, v: Tensor, causal=True, scale=None):
        """不物化 P 的 SDPA。dh!=64、非 causal 或非 CUDA 时回退到物化实现。"""
        return _Functional.scaled_dot_product_attention(
            q, k, v, causal=causal, scale=scale, lowmem=True)

    @staticmethod
    def sdpa_lowmem_ready():
        """提前编译 tgai_attn_nop.cu。计时前调用，避免把 NVRTC 算进 s/step。"""
        return _cuda().ensure_attn_nop()

    @staticmethod
    def cross_entropy(logits: Tensor, target, ignore_index=-100):
        """logits [N,V]，target [N]（int）。返回标量 loss（mean）。"""
        if getattr(logits, "device", "cpu") == "cuda":
            return _cross_entropy_cuda(logits, target, ignore_index)
        return _Functional._cross_entropy_cpu(logits, target, ignore_index)

    @staticmethod
    def _cross_entropy_cpu(logits, target, ignore_index=-100):
        n, v = logits.shape
        tgt = _as_index_array(target).reshape(-1)
        if np.any(tgt == ignore_index):
            lp = _Functional.log_softmax(logits, axis=-1)
            mask = tgt != ignore_index
            picked = lp[np.arange(n), tgt]
            return (picked * Tensor(mask.astype(np.float32))).sum() * (
                -1.0 / _builtin_max(1, mask.sum()))
        return _cross_entropy_fused(logits, tgt)


functional = _Functional()


def _sdpa_materialized(q, k, v, causal, scale):
    """物化 scores/P 的 SDPA。默认路径，也是 lowmem 不合格时的回退。"""
    scores = (q @ k.swap_last2()) * scale                  # [BH,S,S]
    if causal:
        S = scores.shape[-1]
        mask = np.triu(np.full((S, S), -1e30, dtype=np.float32), k=1)
        scores = scores + Tensor(mask)                     # 常数，不回传梯度
    p = _Functional.softmax(scores, axis=-1)
    return p @ v


def _sdpa_lowmem_ok(q, k, v, causal, dh):
    """内核契约：CUDA、causal、dh==64、三维同形状 [BH, S, Dh]。"""
    if not causal or dh != 64:
        return False
    if getattr(q, "device", "cpu") != "cuda":
        return False
    if getattr(k, "device", "cpu") != "cuda" or getattr(v, "device", "cpu") != "cuda":
        return False
    if q.ndim != 3 or tuple(k.shape) != tuple(q.shape) or tuple(v.shape) != tuple(q.shape):
        return False
    bh, seq, _dh = q.shape
    return int(bh) > 0 and int(seq) > 0


def _sdpa_lowmem(q, k, v, scale):
    """[BH, S, Dh] -> 显式 permute 成 [S, BH, Dh] 再调内核，结果 permute 回来。

    B=1 时 BH 就是 H，布局正好是内核要的 [S, H, Dh]。B>1 时各头本来就不互相看，
    把 BH 当成 nh 同样成立。反向不读 P：前向记下每行 m/l，attn_bwd_flash 每个
    Q 块只扫一遍 K/V。
    """
    rt = _cuda()
    rt.ensure_attn_nop()
    # 走运行时 permute，不把这次拷贝接进 autograd，避免和自定义 VJP 重复反传。
    q_sk = rt.permute(q._gpu, (1, 0, 2))
    k_sk = rt.permute(k._gpu, (1, 0, 2))
    v_sk = rt.permute(v._gpu, (1, 0, 2))
    o_sk, m_sk, l_sk = rt.attn_nop_fwd_stats(q_sk, k_sk, v_sk, scale)
    o_hs = rt.permute(o_sk, (1, 0, 2))
    need = bool(q.requires_grad or k.requires_grad or v.requires_grad)
    out = Tensor._from_gpu(o_hs, requires_grad=need, _parents=(q, k, v) if need else ())
    if not need:
        return out

    def bw(g, grads):
        g_sk = rt.permute(g, (1, 0, 2))
        dq_sk, dk_sk, dv_sk = rt.attn_nop_bwd_flash(
            g_sk, q_sk, k_sk, v_sk, o_sk, m_sk, l_sk, scale)
        if q.requires_grad:
            _acc(grads, q, rt.permute(dq_sk, (1, 0, 2)))
        if k.requires_grad:
            _acc(grads, k, rt.permute(dk_sk, (1, 0, 2)))
        if v.requires_grad:
            _acc(grads, v, rt.permute(dv_sk, (1, 0, 2)))

    out._backward = bw
    return out


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


def _log_softmax_cuda(x, axis):
    """最后一维 log_softmax。反向是 g - softmax(x) * sum(g)。"""
    if axis != -1 and axis != x.ndim - 1:
        raise NotImplementedError("CUDA log_softmax 目前只支持最后一维，收到 axis=%r" % (axis,))
    if x.ndim < 1:
        raise NotImplementedError("CUDA log_softmax 需要至少 1 维")
    rt = _cuda()
    y = rt.log_softmax_lastdim(x._gpu)
    out = Tensor._from_gpu(y, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if x.requires_grad:
            _acc(grads, x, rt.log_softmax_lastdim_bwd(g, y))
    out._backward = bw
    return out


def _cross_entropy_cuda(logits, target, ignore_index):
    """GPU 上的 mean CE。语义与 ce_fwd/ce_bwd 一致，不含 ignore_index。"""
    tgt = _as_index_array(target).reshape(-1)
    n, v = logits.shape
    if tgt.shape != (n,):
        raise ValueError("cross_entropy 目标长度 %s != N=%d" % (tgt.shape, n))
    if np.any(tgt == ignore_index):
        return _cross_entropy_cuda_ignore(logits, tgt, ignore_index)
    if np.any((tgt < 0) | (tgt >= v)):
        raise ValueError("cross_entropy 目标越界，合法范围是 [0, %d)" % v)
    rt = _cuda()
    tgt_buf = rt.h2d(tgt, dtype=np.int32, cache_slot="ce_tgt")
    loss_buf = rt.ce_fwd(logits._gpu, tgt_buf, n, v)
    out = Tensor._from_gpu(loss_buf, requires_grad=logits.requires_grad, _parents=(logits,))

    def bw(g, grads):
        if not logits.requires_grad:
            return
        # 根梯度几乎总是主机填好的 1。先读已知值，再发射 ce_bwd，
        # 这样就算真要回读，也不会把 ce_bwd 自己等进去。
        scale = rt.read_scalar(g)
        dl = rt.ce_bwd(logits._gpu, tgt_buf, n, v)
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
        ids = rt.h2d(flat.astype(np.int32), dtype=np.int32, cache_slot="emb_ids")
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


def _sigmoid32(v):
    """与 moe_sigmoid 同一套 exp32：1 / (1 + exp32(-v))。"""
    v = np.asarray(v, dtype=np.float32)
    neg = np.multiply(v, np.float32(-1.0), dtype=np.float32)
    e = np.asarray(_exp32(neg), dtype=np.float32)
    den = np.add(np.float32(1.0), e, dtype=np.float32)
    return np.divide(np.float32(1.0), den, dtype=np.float32)


def _sigmoid_deriv32(y):
    """moeblk 用的 g*(1-g)，不是 exp32 多项式的导数。CPU / CUDA 同一式子。"""
    y = np.asarray(y, dtype=np.float32)
    neg = np.multiply(y, np.float32(-1.0), dtype=np.float32)
    one_m = np.add(np.float32(1.0), neg, dtype=np.float32)
    return np.multiply(y, one_m, dtype=np.float32)


def _sigmoid_cuda(x):
    rt = _cuda()
    buf = rt.moe_sigmoid(x._gpu)
    out = Tensor._from_gpu(buf, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if not x.requires_grad:
            return
        one = rt.full(tuple(int(s) for s in buf.shape), 1.0)
        deriv = rt.mul(buf, rt.add(one, rt.scale(buf, -1.0)))
        _acc(grads, x, rt.mul(g, deriv))

    out._backward = bw
    return out


def _sigmoid_tensor(x):
    if getattr(x, "device", "cpu") == "cuda":
        return _sigmoid_cuda(x)
    y = _sigmoid32(x.data)
    out = Tensor(y, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if x.requires_grad:
            _acc(grads, x, np.asarray(g, dtype=np.float32) * _sigmoid_deriv32(y))

    out._backward = bw
    return out


def _moe_topk_idx(logits, k):
    """logit 更大者优先，平局取更小下标。和 moe_topk kernel 同一套扫描。"""
    z = np.asarray(logits, dtype=np.float32)
    n, e = int(z.shape[0]), int(z.shape[1])
    k = int(k)
    idx = np.empty((n, k), dtype=np.int64)
    for t in range(n):
        row = z[t]
        chosen = []
        for s in range(k):
            best_e = -1
            best_v = np.float32(0.0)
            for j in range(e):
                used = False
                for prev in chosen:
                    if prev == j:
                        used = True
                        break
                if used:
                    continue
                v = np.float32(row[j])
                if best_e < 0 or v > best_v or (v == best_v and j < best_e):
                    best_e = j
                    best_v = v
            chosen.append(best_e)
            idx[t, s] = best_e
    return idx


def _moe_swiglu(rows, expert):
    gate = rows @ expert.gate.weight.T
    up = rows @ expert.up.weight.T
    hidden = functional.silu(gate) * up
    return hidden @ expert.down.weight.T


def _moe_scatter_cpu(n, d, tok, rows):
    tok = np.asarray(tok, dtype=np.int64)
    buf = np.zeros((int(n), int(d)), dtype=np.float32)
    if tok.size:
        np.add.at(buf, tok, np.asarray(rows.data, dtype=np.float32))
    out = Tensor(buf, requires_grad=rows.requires_grad, _parents=(rows,))

    def bw(g, grads):
        if rows.requires_grad:
            _acc(grads, rows, np.asarray(g, dtype=np.float32)[tok])

    out._backward = bw
    return out


def _moe_take_cuda(src, idx_buf, n, e, k):
    rt = _cuda()
    buf = rt.moe_gather_pairs(src._gpu, idx_buf, n, e, k)
    out = Tensor._from_gpu(buf, requires_grad=src.requires_grad, _parents=(src,))

    def bw(g, grads):
        if src.requires_grad:
            _acc(grads, src, rt.moe_scatter_pairs(g, idx_buf, n, e, k))

    out._backward = bw
    return out


def _moe_gather_rows_cuda(x, tok, begin, m, d):
    rt = _cuda()
    buf = rt.moe_gather_rows(x._gpu, tok, begin, m, d)
    out = Tensor._from_gpu(buf, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if x.requires_grad:
            _acc(grads, x, rt.moe_scatter_rows(g, tok, begin, m, int(x.shape[0]), d))

    out._backward = bw
    return out


def _moe_scatter_rows_cuda(n, d, tok, begin, m, rows):
    rt = _cuda()
    buf = rt.moe_scatter_rows(rows._gpu, tok, begin, m, n, d)
    out = Tensor._from_gpu(buf, requires_grad=rows.requires_grad, _parents=(rows,))

    def bw(g, grads):
        if rows.requires_grad:
            _acc(grads, rows, rt.moe_gather_rows(g, tok, begin, m, d))

    out._backward = bw
    return out


def _moe_gather_slots_cuda(w, tok, slot, begin, m, k):
    rt = _cuda()
    n = int(w.shape[0])
    buf = rt.moe_gather_slots(w._gpu, tok, slot, begin, m, k)
    out = Tensor._from_gpu(buf, requires_grad=w.requires_grad, _parents=(w,))

    def bw(g, grads):
        if w.requires_grad:
            _acc(grads, w, rt.moe_scatter_slots(g, tok, slot, begin, m, n, k))

    out._backward = bw
    return out


def _moe_row_scale_cuda(y, w):
    rt = _cuda()
    buf = rt.moe_row_scale(y._gpu, w._gpu)
    rg = y.requires_grad or w.requires_grad
    out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(y, w))

    def bw(g, grads):
        if y.requires_grad:
            _acc(grads, y, rt.moe_row_scale(g, w._gpu))
        if w.requires_grad:
            _acc(grads, w, rt.moe_row_dot(g, y._gpu))

    out._backward = bw
    return out


class _MoEExpert(Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = Linear(d_model, d_ff, bias=False)
        self.up = Linear(d_model, d_ff, bias=False)
        self.down = Linear(d_ff, d_model, bias=False)


class _ExpertList(Module):
    """名字用 "0","1"，这样 state_dict 是 experts.0.gate.weight，不是 experts.m0。"""

    def __init__(self, experts):
        super().__init__()
        self._n = len(experts)
        for i, expert in enumerate(experts):
            setattr(self, str(i), expert)

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        i = int(i)
        if i < 0:
            i += self._n
        return getattr(self, str(i))

    def __iter__(self):
        for i in range(self._n):
            yield self[i]


def _moe_grouped_cuda(x, w, tok_buf, slot_buf, off, experts, n, d, k):
    """8 个专家的 GEMM 合成一次 cublasSgemmGroupedBatched。

    token 数不必相同，也不 pad：每组 group_size=1。小形状（自检）返回 None，
    仍走逐专家的原路径，保证和 CPU 的逐位/1e-5 判据不变。
    反向里 pad 不存在；没选中的专家不写梯度。
    """
    counts = [int(off[e + 1]) - int(off[e]) for e in range(len(experts))]
    active = [e for e, c in enumerate(counts) if c > 0]
    if not active or _builtin_max(counts) <= 64:
        return None
    rt = _cuda()
    rows = [rt.moe_gather_rows(x._gpu, tok_buf, int(off[e]), counts[e], d) for e in active]
    wg = [experts[e].gate.weight._gpu for e in active]
    wu = [experts[e].up.weight._gpu for e in active]
    wd = [experts[e].down.weight._gpu for e in active]
    gate = rt.grouped_nt(rows, wg)
    up = rt.grouped_nt(rows, wu)
    silu = []
    for g in gate:
        s = rt.clone(g)
        ones = rt.full(g.shape, 1.0)
        rt.silu_mul_fwd(s, ones)
        s.keepalive = ones
        silu.append(s)
    hidden = [rt.mul(s, u) for s, u in zip(silu, up)]
    y_rows = rt.grouped_nt(hidden, wd)
    wws = []
    out_buf = rt.zeros((n, d), dtype=np.float32)
    for e, y in zip(active, y_rows):
        ww = rt.moe_gather_slots(w._gpu, tok_buf, slot_buf, int(off[e]), counts[e], k)
        wws.append(ww)
        scaled = rt.moe_row_scale(y, ww)
        rt.moe_scatter_rows_add(scaled, tok_buf, int(off[e]), counts[e], out_buf)
    parents = [x, w]
    for e in active:
        parents.append(experts[e].gate.weight)
        parents.append(experts[e].up.weight)
        parents.append(experts[e].down.weight)
    need = any(getattr(p, "requires_grad", False) for p in parents)
    out = Tensor._from_gpu(out_buf, requires_grad=need, _parents=tuple(parents) if need else ())
    if not need:
        return out
    saved = (rows, gate, up, silu, hidden, y_rows, wws)

    def bw(g, grads, _saved=saved, _active=tuple(active), _counts=tuple(counts)):
        rows_s, gate_s, up_s, silu_s, hidden_s, y_s, ww_s = _saved
        d_scaled = [rt.moe_gather_rows(g, tok_buf, int(off[e]), _counts[e], d) for e in _active]
        d_y = [rt.moe_row_scale(ds, ww) for ds, ww in zip(d_scaled, ww_s)]
        d_ww = [rt.moe_row_dot(ds, y) for ds, y in zip(d_scaled, y_s)]
        d_hidden = rt.grouped_nn(d_y, wd)
        d_wd = rt.grouped_tn(d_y, hidden_s)
        d_s = [rt.mul(dh, u) for dh, u in zip(d_hidden, up_s)]
        d_up = [rt.mul(dh, s) for dh, s in zip(d_hidden, silu_s)]
        d_gate = []
        for ds, gpre in zip(d_s, gate_s):
            ones = rt.full(gpre.shape, 1.0)
            da, _db = rt.silu_mul_bwd(ds, gpre, ones)
            da.keepalive = ones
            d_gate.append(da)
        d_wg = rt.grouped_tn(d_gate, rows_s)
        d_wu = rt.grouped_tn(d_up, rows_s)
        d_rows_g = rt.grouped_nn(d_gate, wg)
        d_rows_u = rt.grouped_nn(d_up, wu)
        hold = (d_scaled, d_y, d_hidden, d_s, d_up, d_gate, d_rows_g, d_rows_u)
        if x.requires_grad:
            dx = rt.zeros((n, d), dtype=np.float32)
            for e, a, b in zip(_active, d_rows_g, d_rows_u):
                both = rt.add(a, b)
                rt.moe_scatter_rows_add(both, tok_buf, int(off[e]), _counts[e], dx)
            dx.keepalive = hold
            _acc(grads, x, dx)
        elif d_wg:
            d_wg[0].keepalive = hold
        for e, dw in zip(_active, d_wg):
            if experts[e].gate.weight.requires_grad:
                _acc(grads, experts[e].gate.weight, dw)
        for e, dw in zip(_active, d_wu):
            if experts[e].up.weight.requires_grad:
                _acc(grads, experts[e].up.weight, dw)
        for e, dw in zip(_active, d_wd):
            if experts[e].down.weight.requires_grad:
                _acc(grads, experts[e].down.weight, dw)
        if w.requires_grad:
            acc_w = None
            for e, dw in zip(_active, d_ww):
                part = rt.moe_scatter_slots(dw, tok_buf, slot_buf, int(off[e]), _counts[e], n, k)
                acc_w = part if acc_w is None else rt.add(acc_w, part)
            if acc_w is not None:
                _acc(grads, w, acc_w)

    out._backward = bw
    out_buf.keepalive = saved
    return out


class MoE(Module):
    """带两种门控的 MoE。专家按 token 分组，空专家不参与 GEMM。

    前向（每个 token）：
        z = x @ router.weight.T
        S = topk(z)                         # 更大的 logit 优先，平局取更小下标
        sigmoid:  y = Σ_{e in S} sigmoid(z_e) * Expert_e(x)
        softmax:  p = softmax(z);  y = Σ_{e in S} p_e * Expert_e(x)
        Expert:   h = silu(x @ gate.T) * (x @ up.T);  out = h @ down.T
    """

    def __init__(self, d_model, d_ff, n_experts, top_k=2, gate="softmax"):
        super().__init__()
        d_model = int(d_model)
        d_ff = int(d_ff)
        n_experts = int(n_experts)
        top_k = int(top_k)
        if d_model < 1 or d_ff < 1:
            raise ValueError("MoE 的 d_model / d_ff 必须为正")
        if n_experts < 1:
            raise ValueError("MoE 至少要 1 个专家")
        if top_k < 1 or top_k > n_experts:
            raise ValueError("top_k=%d 必须落在 [1, %d]" % (top_k, n_experts))
        if gate not in ("softmax", "sigmoid"):
            raise ValueError("gate 只接受 softmax / sigmoid，收到 %r" % (gate,))
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = gate
        self.router = Linear(d_model, n_experts, bias=False)
        self.experts = _ExpertList([
            _MoEExpert(d_model, d_ff) for _ in range(n_experts)
        ])

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = _as_tensor(x)
        if x.ndim < 1:
            raise ValueError("MoE 输入至少 1 维")
        if int(x.shape[-1]) != self.d_model:
            raise ValueError(
                "MoE 最后一维 %d != d_model %d" % (int(x.shape[-1]), self.d_model))
        dev_x = getattr(x, "device", "cpu")
        dev_w = getattr(self.router.weight, "device", "cpu")
        if dev_x != dev_w:
            raise RuntimeError("MoE 参数在 %s，输入在 %s" % (dev_w, dev_x))
        n = 1
        for s in x.shape[:-1]:
            n *= int(s)
        if n > 2147483647 // self.top_k:
            raise ValueError("MoE token 数过大")
        flat = x.reshape((n, self.d_model))
        y = flat * 0.0 if n == 0 else self._forward_flat(flat)
        return y.reshape(tuple(int(s) for s in x.shape))

    def _forward_flat(self, x):
        logits = x @ self.router.weight.T
        if getattr(x, "device", "cpu") == "cuda":
            return self._combine_cuda(x, logits)
        return self._combine_cpu(x, logits)

    def _weights_cpu(self, logits, idx):
        n, k = int(idx.shape[0]), int(idx.shape[1])
        rows = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], (n, k)).copy()
        cols = np.asarray(idx, dtype=np.int64)
        if self.gate == "sigmoid":
            return _sigmoid_tensor(logits[rows, cols])
        return functional.softmax(logits, axis=-1)[rows, cols]

    def _combine_cpu(self, x, logits):
        n = int(x.shape[0])
        d = int(x.shape[1])
        idx = _moe_topk_idx(logits.data, self.top_k)
        w = self._weights_cpu(logits, idx)
        acc = None
        for e, expert in enumerate(self.experts):
            tok = []
            slot = []
            for t in range(n):
                row = idx[t]
                for s in range(self.top_k):
                    if int(row[s]) == e:
                        tok.append(t)
                        slot.append(s)
            if not tok:
                continue
            tok_a = np.asarray(tok, dtype=np.int64)
            slot_a = np.asarray(slot, dtype=np.int64)
            y = _moe_swiglu(x[tok_a], expert)
            y = y * w[tok_a, slot_a].reshape((-1, 1))
            part = _moe_scatter_cpu(n, d, tok_a, y)
            acc = part if acc is None else acc + part
        if acc is None:
            return x * 0.0
        return acc

    def _combine_cuda(self, x, logits):
        rt = _cuda()
        n = int(x.shape[0])
        d = int(x.shape[1])
        k = self.top_k
        n_exp = self.n_experts
        idx_buf, val_buf = rt.moe_topk(logits._gpu, k)
        if self.gate == "sigmoid":
            sel = Tensor._from_gpu(
                val_buf, requires_grad=logits.requires_grad, _parents=(logits,))

            def bw_sel(g, grads, _idx=idx_buf, _rt=rt, _logits=logits):
                if _logits.requires_grad:
                    _acc(grads, _logits, _rt.moe_scatter_pairs(g, _idx, n, n_exp, k))

            sel._backward = bw_sel
            w = _sigmoid_tensor(sel)
        else:
            w = _moe_take_cuda(functional.softmax(logits, axis=-1), idx_buf, n, n_exp, k)
        off_buf, tok_buf, slot_buf = rt.moe_compact(idx_buf, n, n_exp, k)
        # 只有 E+1 个计数回到主机，用来跳过空专家。激活留在设备上。
        off = np.asarray(rt.d2h(off_buf), dtype=np.int32).reshape(-1)
        grouped = _moe_grouped_cuda(
            x, w, tok_buf, slot_buf, off, self.experts, n, d, k)
        if grouped is not None:
            return grouped
        acc = None
        for e, expert in enumerate(self.experts):
            begin = int(off[e])
            count = int(off[e + 1]) - begin
            if count <= 0:
                continue
            rows = _moe_gather_rows_cuda(x, tok_buf, begin, count, d)
            y = _moe_swiglu(rows, expert)
            ww = _moe_gather_slots_cuda(w, tok_buf, slot_buf, begin, count, k)
            y = _moe_row_scale_cuda(y, ww)
            part = _moe_scatter_rows_cuda(n, d, tok_buf, begin, count, y)
            acc = part if acc is None else acc + part
        if acc is None:
            return x * 0.0
        return acc


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
        self.m = []
        self.v = []
        for p in self.params:
            if getattr(p, "device", "cpu") == "cuda":
                # 值在 _gpu 上，p.data 是 None。动量也放设备上，别 zeros_like(None)。
                rt = _cuda()
                self.m.append(rt.zeros(tuple(int(s) for s in p.shape)))
                self.v.append(rt.zeros(tuple(int(s) for s in p.shape)))
            else:
                self.m.append(np.zeros_like(p.data))
                self.v.append(np.zeros_like(p.data))

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        # 这些标量和参数无关。按 numpy 的标量提升收成 float32，再整步留在 GPU。
        b1 = _f32(self.b1)
        omb1 = _f32(1.0 - self.b1)
        b2 = _f32(self.b2)
        omb2 = _f32(1.0 - self.b2)
        lr = _f32(self.lr)
        wd_lr = _f32(self.lr * self.wd) if self.wd else 0.0
        eps = _f32(self.eps)
        bc1s = _f32(bc1)
        bc2s = _f32(bc2)
        do_wd = bool(self.wd)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if getattr(p, "device", "cpu") == "cuda":
                # 崩在 float * p.data：data 是 None，不是梯度缺失。
                # Linear 的 weight 参加了前向，grad 是 CUDA Tensor。
                # PyTorch 的 AdamW 跳过 grad is None；这里同样跳过，更新本身在 GPU 上做。
                g = p.grad
                if not isinstance(g, Tensor) or getattr(g, "device", "cpu") != "cuda":
                    raise TypeError("CUDA AdamW 需要 CUDA Tensor 梯度，收到 %s" % type(g).__name__)
                _cuda().adamw(
                    p._gpu, g._gpu, self.m[i], self.v[i],
                    b1, omb1, b2, omb2, lr, wd_lr, eps, bc1s, bc2s, do_wd,
                )
                continue
            g = p.grad
            if self.wd:
                p.data = p.data - self.lr * self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / bc1
            vh = self.v[i] / bc2
            p.data = p.data - self.lr * mh / (np.sqrt(vh) + self.eps)


def _sumsq32(x):
    """按地址顺序的 float32 平方和，和 GPU sumsq_sequential 同一套舍入。"""
    acc = np.float32(0.0)
    flat = np.ascontiguousarray(x, dtype=np.float32).ravel()
    for v in flat:
        vv = np.float32(v)
        acc = np.float32(acc + np.float32(vv * vv))
    return acc


def _newton_schulz5(g, steps=5, eps=1e-7):
    """Muon 的正交化迭代（a,b,c = 3.4445, -4.7750, 2.0315）。

    注意：该多项式的不动点是 0.868，所以输出奇异值集中在 0.868 附近、
    条件数被压到 ~1.6（实测病态输入 384 -> 1.61）。
    它的作用是**压平谱**，常数倍由 Muon 的学习率吸收。

    ||X|| 用顺序 float32 平方和，和 GPU 上只降下来的那个标量同一套舍入。
    np.linalg.norm 在 n=64 时和这个差约 1 ulp，Newton-Schulz 会把它放大过 1e-7。
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = np.array(g, dtype=np.float32, copy=True)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = np.ascontiguousarray(x.T)
        transposed = True
    nrm = np.sqrt(_sumsq32(x))
    denom = np.float32(nrm + np.float32(eps))
    x = np.float32(x / denom)
    for _ in range(steps):
        s = x @ x.T
        y = b * s + c * (s @ s)
        x = a * x + y @ x
    if transposed:
        x = np.ascontiguousarray(x.T)
    return np.asarray(x, np.float32)


def _newton_schulz5_gpu(rt, x, steps=5, eps=1e-7):
    """与 _newton_schulz5 同一公式，值留在显存。

    X@X.T、(X@X.T) 的平方、Y@X 走设备上的顺序 FMA（sgemm_mode 1）。
    这和 NumPy 1.26 / OpenBLAS 的 float32 `@` 在这些形状上逐位相同。
    cublasSgemm_v2 即使开了 PEDANTIC，单次也有约 3e-8 的相对误差，
    五轮多项式迭代后参数对不上 1e-7，所以这里不用 cuBLAS。
    行数大于列数时先 transpose2d，迭代完再转回去。
    ||X||^2 在设备上按地址顺序累加，只把这一个标量降回主机做 float32 sqrt，
    再把除数送回设备做真除法。不把矩阵本身拉回主机。
    """
    transposed = False
    if int(x.shape[0]) > int(x.shape[1]):
        x = rt.transpose2d(x)
        transposed = True
    sumsq = rt.read_scalar(rt.sumsq_sequential(x))
    nrm = np.sqrt(np.float32(sumsq))
    denom = np.float32(nrm + np.float32(eps))
    x = rt.div_scalar(x, float(denom))
    a = _f32(3.4445)
    b = _f32(-4.7750)
    c = _f32(2.0315)
    for _ in range(int(steps)):
        # X @ X.T：先把 X 转成连续的 X.T，再做顺序 FMA 的普通乘。
        s = rt.sgemm_mode(x, rt.transpose2d(x), 1)
        ss = rt.sgemm_mode(s, s, 1)
        y = rt.add(rt.scale(s, b), rt.scale(ss, c))
        x = rt.add(rt.scale(x, a), rt.sgemm_mode(y, x, 1))
    if transposed:
        x = rt.transpose2d(x)
    return x


class Muon(_Optim):
    """Muon：动量 + Newton-Schulz 正交化；2D 走 Muon，其余回退 AdamW 式更新。"""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0):
        super().__init__(params, lr)
        self.mom = momentum
        self.nesterov = nesterov
        self.wd = weight_decay
        self.buf = []
        self._adam_m = []
        self._adam_v = []
        rt = None
        for p in self.params:
            if getattr(p, "device", "cpu") == "cuda":
                # p.data 是 None。动量放设备上，别 zeros_like(None)。
                if rt is None:
                    rt = _cuda()
                shp = tuple(int(s) for s in p.shape)
                self.buf.append(rt.zeros(shp))
                self._adam_m.append(rt.zeros(shp))
                self._adam_v.append(rt.zeros(shp))
            else:
                self.buf.append(np.zeros_like(p.data))
                self._adam_m.append(np.zeros_like(p.data))
                self._adam_v.append(np.zeros_like(p.data))
        self.t = 0

    def _step_cuda(self, i, p):
        rt = _cuda()
        g = p.grad
        if not isinstance(g, Tensor) or getattr(g, "device", "cpu") != "cuda":
            raise TypeError("CUDA Muon 需要 CUDA Tensor 梯度，收到 %s" % type(g).__name__)
        if len(p.shape) != 2:
            b1, b2, eps = 0.9, 0.999, 1e-8
            bc1 = 1.0 - b1 ** self.t
            bc2 = 1.0 - b2 ** self.t
            rt.adamw(
                p._gpu, g._gpu, self._adam_m[i], self._adam_v[i],
                _f32(b1), _f32(1.0 - b1), _f32(b2), _f32(1.0 - b2),
                _f32(self.lr), 0.0, _f32(eps), _f32(bc1), _f32(bc2), False,
            )
            return
        rows = int(p.shape[0])
        cols = int(p.shape[1])
        u = rt.empty(p.shape, dtype=np.float32)
        rt.muon_momentum(
            p._gpu, g._gpu, self.buf[i], u,
            _f32(self.mom),
            _f32(self.lr * self.wd) if self.wd else 0.0,
            bool(self.wd), bool(self.nesterov),
        )
        o = _newton_schulz5_gpu(rt, u)
        scale = float(np.sqrt(_builtin_max(1.0, rows / cols)))
        rt.sub_scale_(p._gpu, o, _f32(self.lr * scale))

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if getattr(p, "device", "cpu") == "cuda":
                self._step_cuda(i, p)
                continue
            g = p.grad
            if p.data.ndim == 2:
                if self.wd:
                    p.data = p.data - self.lr * self.wd * p.data
                self.buf[i] = self.mom * self.buf[i] + g
                u = g + self.mom * self.buf[i] if self.nesterov else self.buf[i]
                o = _newton_schulz5(u)
                scale = np.sqrt(_builtin_max(1.0, p.data.shape[0] / p.data.shape[1]))
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
    MoE = MoE
    Parameter = Parameter
    Linear = Linear
    Embedding = Embedding
    RMSNorm = RMSNorm
    Sequential = Sequential
    ModuleList = ModuleList
    functional = functional


nn = _NNModule()
optim = _OptimModule()

def permute(input, dims):
    """对应 torch.permute(input, dims)。dims 是轴的排列。"""
    if not isinstance(dims, (tuple, list)):
        raise TypeError("permute 的 dims 需要 tuple 或 list，收到 %s" % type(dims).__name__)
    return _as_tensor(input).permute(*dims)


# 公开名字放在文件末尾，避免盖住上面 Muon / cross_entropy 里的内置 max。
abs = _api_abs
clamp = _api_clamp
max = _api_max
min = _api_min

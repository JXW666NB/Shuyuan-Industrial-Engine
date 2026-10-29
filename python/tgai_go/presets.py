# -*- coding: utf-8 -*-
"""C 编排的 llama / qwen2 / gpt2 前向。

同一份 ``native/libtgai_llama_fwd.so``。架构差异放在 ``TgArch`` 里：
llama 的 ``qkv_bias=0``，qwen2 的 ``qkv_bias=1``，gpt2 的 ``bias_all=1``
再加上 LayerNorm / GELU-tanh / 学习式位置。一次 ``forward`` 只跨进 C 一次。
"""
import ctypes
import os

import numpy as np

_LIB = None
_PLAN_SIZE = None

# 与 native/llama_fwd.h 的枚举一致。0 留给“没填”，C 会拒绝。
TG_NORM_RMS = 1
TG_NORM_LN = 2
TG_POS_ROPE = 1
TG_POS_WPE = 2
TG_FFN_SWIGLU = 1
TG_FFN_GELU = 2


def _u64(value):
    if value is None:
        return 0
    raw = getattr(value, "value", value)
    if raw is None:
        return 0
    return int(raw) & ((1 << 64) - 1)


def _so_path():
    env = os.environ.get("TGAI_LLAMA_FWD_SO")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(here, "..", "..", "native", "libtgai_llama_fwd.so"))
    return cand


def _load():
    global _LIB, _PLAN_SIZE
    if _LIB is not None:
        return _LIB
    path = _so_path()
    if not os.path.isfile(path):
        raise RuntimeError(
            "找不到 libtgai_llama_fwd.so（%s）。在 native/ 下跑 build_llama_fwd.sh" % path)
    lib = ctypes.CDLL(path)
    lib.tgai_llama_plan_size.restype = ctypes.c_int
    lib.tgai_llama_plan_size.argtypes = []
    lib.tgai_llama_last_ffi_calls.restype = ctypes.c_int
    lib.tgai_llama_last_ffi_calls.argtypes = []
    lib.tgai_llama_last_error.restype = ctypes.c_char_p
    lib.tgai_llama_last_error.argtypes = []
    lib.tgai_llama_forward.restype = ctypes.c_int
    lib.tgai_llama_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ]
    got = int(lib.tgai_llama_plan_size())
    if got != ctypes.sizeof(_Plan):
        raise RuntimeError(
            "TgaiLlamaPlan 大小不一致：C=%d Python=%d" % (got, ctypes.sizeof(_Plan)))
    _PLAN_SIZE = got
    _LIB = lib
    return lib


class _Arch(ctypes.Structure):
    """与 native/llama_fwd.h 的 TgArch 逐字段对齐。"""

    _fields_ = [
        ("qkv_bias", ctypes.c_int32),
        ("norm", ctypes.c_int32),
        ("pos", ctypes.c_int32),
        ("ffn", ctypes.c_int32),
        ("n_kv", ctypes.c_int32),
        ("bias_all", ctypes.c_int32),
    ]


class _Plan(ctypes.Structure):
    _fields_ = [
        ("cu_launch", ctypes.c_uint64),
        ("cu_h2d", ctypes.c_uint64),
        ("cu_d2h", ctypes.c_uint64),
        ("cu_d2d", ctypes.c_uint64),
        ("cu_alloc", ctypes.c_uint64),
        ("cu_free", ctypes.c_uint64),
        ("cu_memset32", ctypes.c_uint64),
        ("cu_stream_sync", ctypes.c_uint64),
        ("sgemm", ctypes.c_uint64),
        ("cublas", ctypes.c_uint64),
        ("k_emb", ctypes.c_uint64),
        ("k_rms", ctypes.c_uint64),
        ("k_perm", ctypes.c_uint64),
        ("k_add", ctypes.c_uint64),
        ("k_mul", ctypes.c_uint64),
        ("k_scale", ctypes.c_uint64),
        ("k_bcast", ctypes.c_uint64),
        ("k_softmax", ctypes.c_uint64),
        ("k_swap", ctypes.c_uint64),
        ("k_silu", ctypes.c_uint64),
        ("k_gather", ctypes.c_uint64),
        ("k_bcast_to", ctypes.c_uint64),
        ("k_exp", ctypes.c_uint64),
        ("k_recip", ctypes.c_uint64),
        ("k_pow", ctypes.c_uint64),
        ("k_sum", ctypes.c_uint64),
        ("d", ctypes.c_int32),
        ("L", ctypes.c_int32),
        ("nh", ctypes.c_int32),
        ("ff", ctypes.c_int32),
        ("V", ctypes.c_int32),
        ("rope_seq", ctypes.c_int32),
        ("n_w", ctypes.c_int32),
        ("arch", _Arch),
        ("w", ctypes.c_uint64),
    ]


def _rotate_perm(head_dim):
    """与 llama_ref._rotate_perm 相同：x @ R == [-x2, x1]。"""
    half = int(head_dim) // 2
    R = np.zeros((head_dim, head_dim), np.float32)
    for j in range(half):
        R[j + half, j] = np.float32(-1.0)
    for j in range(half, head_dim):
        R[j - half, j] = np.float32(1.0)
    return R


def _sym(fn):
    return _u64(ctypes.cast(fn, ctypes.c_void_p))


def _vec(weights, key, n):
    arr = np.ascontiguousarray(weights[key], dtype=np.float32)
    if arr.shape != (int(n),):
        raise ValueError("%s 需要形状 (%d,)，收到 %s" % (key, int(n), arr.shape))
    return arr


def _mat(weights, key, rows, cols):
    arr = np.ascontiguousarray(weights[key], dtype=np.float32)
    if arr.shape != (int(rows), int(cols)):
        raise ValueError("%s 需要形状 (%d, %d)，收到 %s" % (key, int(rows), int(cols), arr.shape))
    return arr


class _LlamaC(object):
    """presets.llama / qwen2 / gpt2 的返回值。forward(ids) -> float32 [B*S, V]。"""

    def __init__(self, cfg, arch_name):
        from tgai_go.train.cuda import CudaRuntime

        lib = _load()
        rt = CudaRuntime()
        rt.ensure()
        self._lib = lib
        self._rt = rt
        self._cfg = cfg
        if arch_name not in ("llama", "qwen2", "gpt2"):
            raise ValueError("未知架构 %r" % (arch_name,))
        self._arch_name = arch_name
        qkv_bias = 1 if arch_name == "qwen2" else 0
        bias_all = 1 if arch_name == "gpt2" else 0
        d = int(cfg["d"])
        nh = int(cfg["nh"])
        if d % nh:
            raise ValueError("d 必须能被 nh 整除")
        dh = d // nh
        weights = cfg["weights"]
        keep = []

        def up(arr):
            buf = rt.h2d(np.ascontiguousarray(arr, dtype=np.float32))
            keep.append(buf)
            return int(buf.ptr)

        if arch_name == "gpt2":
            n_kv = nh
            wpe = _mat(weights, "wpe.weight", int(cfg["seq"]), d)
            ptrs = [
                up(_mat(weights, "wte.weight", int(cfg["V"]), d)),
                up(_vec(weights, "ln_f.weight", d)),
                up(_mat(weights, "wte.weight" if cfg.get("tie") else "lm_head", int(cfg["V"]), d)),
                up(wpe),
                up(_vec(weights, "ln_f.bias", d)),
            ]
            for i in range(int(cfg["L"])):
                p = "blocks.%d." % i
                qkv_w = _mat(weights, p + "attn.qkv.weight", 3 * d, d)
                qkv_b = _vec(weights, p + "attn.qkv.bias", 3 * d)
                ptrs.append(up(_vec(weights, p + "ln_1.weight", d)))
                ptrs.append(up(_vec(weights, p + "ln_1.bias", d)))
                ptrs.append(up(qkv_w[0:d]))
                ptrs.append(up(qkv_w[d:2 * d]))
                ptrs.append(up(qkv_w[2 * d:3 * d]))
                ptrs.append(up(_mat(weights, p + "attn.proj.weight", d, d)))
                ptrs.append(up(_vec(weights, p + "ln_2.weight", d)))
                ptrs.append(up(_vec(weights, p + "ln_2.bias", d)))
                ptrs.append(up(_mat(weights, p + "mlp.fc.weight", int(cfg["ff"]), d)))
                ptrs.append(up(_mat(weights, p + "mlp.proj.weight", d, int(cfg["ff"]))))
                ptrs.append(up(qkv_b[0:d]))
                ptrs.append(up(qkv_b[d:2 * d]))
                ptrs.append(up(qkv_b[2 * d:3 * d]))
                ptrs.append(up(_vec(weights, p + "attn.proj.bias", d)))
                ptrs.append(up(_vec(weights, p + "mlp.fc.bias", int(cfg["ff"]))))
                ptrs.append(up(_vec(weights, p + "mlp.proj.bias", d)))
            rope_seq = int(wpe.shape[0])
        else:
            n_kv = int(cfg["n_kv"])
            ptrs = [
                up(weights["tok_emb.weight"]),
                up(weights["final_norm.weight"]),
                up(weights["lm_head.weight"]),
                up(cfg["cos"]),
                up(cfg["sin"]),
                up(_rotate_perm(dh)),
            ]
            kv_dim = n_kv * dh
            for i in range(int(cfg["L"])):
                p = "blocks.%d." % i
                for name in (
                    "attn_norm.weight",
                    "attn.q.weight",
                    "attn.k.weight",
                    "attn.v.weight",
                    "attn.o.weight",
                    "ffn_norm.weight",
                    "ffn.gate.weight",
                    "ffn.up.weight",
                    "ffn.down.weight",
                ):
                    ptrs.append(up(weights[p + name]))
                if qkv_bias:
                    ptrs.append(up(_vec(weights, p + "attn.q.bias", d)))
                    ptrs.append(up(_vec(weights, p + "attn.k.bias", kv_dim)))
                    ptrs.append(up(_vec(weights, p + "attn.v.bias", kv_dim)))
            rope_seq = int(np.asarray(cfg["cos"]).shape[0])
        rt._activate()
        rt._chk(rt.cu.cuStreamSynchronize(ctypes.c_void_p(0)), "cuStreamSynchronize(weights)")

        table = (ctypes.c_uint64 * len(ptrs))(*[_u64(p) for p in ptrs])
        plan = _Plan()
        cu = rt.cu
        plan.cu_launch = _sym(cu.cuLaunchKernel)
        plan.cu_h2d = _sym(cu.cuMemcpyHtoD)
        plan.cu_d2h = _sym(cu.cuMemcpyDtoH)
        plan.cu_d2d = _sym(cu.cuMemcpyDtoD)
        plan.cu_alloc = _sym(cu.cuMemAlloc)
        plan.cu_free = _sym(cu.cuMemFree)
        plan.cu_memset32 = _sym(cu.cuMemsetD32Async)
        plan.cu_stream_sync = _sym(cu.cuStreamSynchronize)
        plan.sgemm = _sym(rt.cublas.cublasSgemm_v2)
        plan.cublas = _u64(rt.handle)
        plan.k_emb = _u64(rt._fn("emb_gather_fwd"))
        plan.k_rms = _u64(rt._fn("rmsnorm_fwd"))
        plan.k_perm = _u64(rt._fn("permute8"))
        plan.k_add = _u64(rt._fn("ewise_add"))
        plan.k_mul = _u64(rt._fn("ewise_mul"))
        plan.k_scale = _u64(rt._fn("ewise_scale"))
        plan.k_bcast = _u64(rt._fn("bcast_add4"))
        plan.k_softmax = _u64(rt._fn("softmax_lastdim"))
        plan.k_swap = _u64(rt._fn("swap_last2"))
        plan.k_silu = _u64(rt._fn("silu_mul_fwd"))
        plan.k_gather = _u64(rt._fn("index_gather8"))
        plan.k_bcast_to = _u64(rt._fn("broadcast_to8"))
        plan.k_exp = _u64(rt._fn("ewise_exp"))
        plan.k_recip = _u64(rt._fn("ewise_recip"))
        plan.k_pow = _u64(rt._fn("ewise_pow"))
        plan.k_sum = _u64(rt._fn("sum_axis8"))
        plan.d = d
        plan.L = int(cfg["L"])
        plan.nh = nh
        plan.ff = int(cfg["ff"])
        plan.V = int(cfg["V"])
        plan.rope_seq = rope_seq
        plan.n_w = len(ptrs)
        # 嵌套 Structure 读出来是副本，必须整份写回。
        arch = _Arch()
        arch.qkv_bias = qkv_bias
        arch.bias_all = bias_all
        arch.n_kv = n_kv
        if arch_name == "gpt2":
            arch.norm = TG_NORM_LN
            arch.pos = TG_POS_WPE
            arch.ffn = TG_FFN_GELU
        else:
            arch.norm = TG_NORM_RMS
            arch.pos = TG_POS_ROPE
            arch.ffn = TG_FFN_SWIGLU
        plan.arch = arch
        plan.w = ctypes.addressof(table)
        self._keep = keep
        self._table = table
        self._plan = plan

    def forward(self, ids):
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        if ids.ndim != 2:
            raise ValueError("ids 需要 [B,S]，收到 %s" % (ids.shape,))
        b = int(ids.shape[0])
        s = int(ids.shape[1])
        v = int(self._cfg["V"])
        out = np.empty((b * s, v), dtype=np.float32)
        self._rt._activate()
        rc = self._lib.tgai_llama_forward(
            ctypes.byref(self._plan),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int32(b),
            ctypes.c_int32(s),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            err = self._lib.tgai_llama_last_error()
            msg = err.decode("utf-8", "replace") if err else ("rc=%d" % rc)
            raise RuntimeError("%s C forward 失败：%s" % (self._arch_name, msg))
        return out


def llama(cfg):
    """用 cfg（llama_ref.make_cfg 的 dict）建 llama 前向。不看 cfg 里的 qkv_bias。"""
    return _LlamaC(cfg, "llama")


def qwen2(cfg):
    """同一份 C。唯一差别是 TgArch.qkv_bias=1，Q/K/V 投影后加 bias。"""
    return _LlamaC(cfg, "qwen2")


def gpt2(cfg):
    """同一份 C。LayerNorm + GELU-tanh + 学习式位置 + 全部线性 bias。"""
    return _LlamaC(cfg, "gpt2")


def last_ffi_calls():
    """最近一次 ``forward`` 里 Python 进入 C 的次数。"""
    lib = _load()
    return int(lib.tgai_llama_last_ffi_calls())

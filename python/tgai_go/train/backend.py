# -*- coding: utf-8 -*-
"""后端接口与工厂（P0 后端抽象）。

## 为什么要这一层

`tgai_go.train.__init__` 里有几百处 `rt.xxx(...)`，直接绑死在 `cuda.py` 的具体类上。
要接 Vulkan / ROCm，不可能把每个调用点都改一遍。

这一层提供**唯一的取后端入口**：

```python
from tgai_go.train import backend
rt = backend.get()          # 默认按 TG_BACKEND / 自动选择
backend.register("vulkan", make_vulkan)   # 新后端从这里登记
```

`tgai_go.train._cuda()` 也走这里，所以**现有 79 个调用点一个都不用改**。

## 接口到底有多少？

不是 179 个。实测（从 `__init__.py` 里 `rt.xxx(` 的全部调用点提取）训练路径只用到 **79 个**，
其中 11 个是 MoE 专用。做最小可用后端（dense 模型、不用 MoE/融合核）只需 **45 个**。

分组见下面的 `GROUPS`。新后端按组实现，`missing(inst)` 会告诉你还差哪些。

## 语义契约（每条都必须一致，否则数值对不上）

* 所有缓冲区是**连续行优先**的 float32（除非显式声明 dtype）；
* `sync()` 前所有操作**异步**，`sync()` 之后主机可见；
* `h2d/d2h` 是同步拷贝；
* `*_strided_batched` 的 batch 维在**最外层**（stride = 内层元素数）；
* `gemm_nt(A,B)` 算 `A[M,K] @ B[N,K]^T`；`gemm_tn(A,B)` 算 `A[K,M]^T @ B[K,N]`；
* 反向算子的入参梯度布局与前向输出一致。
"""
from __future__ import annotations

import os
import sys

# ---- 接口清单（按组，便于新后端逐组实现）------------------------------------

GROUPS = {
    "内存/拷贝": [
        "empty", "zeros", "full", "clone", "h2d", "d2h", "sync",
    ],
    "形状/布局": [
        "reshape", "permute", "swap_last2", "transpose2d", "broadcast_to",
    ],
    "归约": [
        "sum_all", "sum_axis", "sumsq_sequential", "read_scalar",
    ],
    "GEMM": [
        "sgemm", "gemm_nt", "gemm_tn", "bmm", "sgemm_mode",
        "sgemm_strided_batched", "sgemm_nt_strided_batched", "sgemm_tn_strided_batched",
        "matmul_general",
    ],
    "逐元素": [
        "add", "add_", "mul", "scale", "scale_inplace", "exp",
        "div_scalar", "pow_scalar", "sub_scale_", "unbroadcast",
    ],
    "归一化": [
        "rmsnorm_fwd", "rmsnorm_bwd",
    ],
    "注意力": [
        "softmax_lastdim", "softmax_lastdim_bwd",
        "log_softmax_lastdim", "log_softmax_lastdim_bwd",
        "attn_nop_fwd_stats", "attn_nop_bwd_flash", "ensure_attn_nop",
    ],
    "融合核（可回退）": [
        "silu_mul_fwd", "silu_mul_bwd", "silu_fwd_inplace", "silu_bwd_da",
        "fused_rope_", "fused_rope_bwd",
        "attn_mask_softmax_", "attn_mask_softmax_bwd",
    ],
    "嵌入/损失": [
        "emb_fwd", "emb_bwd", "ce_fwd", "ce_bwd",
        "index_gather", "index_scatter_add",
    ],
    "优化器": [
        "adamw", "muon_momentum",
    ],
    "MoE（可选）": [
        "moe_topk", "moe_sigmoid", "moe_compact", "moe_gather_pairs",
        "moe_gather_rows", "moe_gather_slots", "moe_scatter_pairs",
        "moe_scatter_rows", "moe_scatter_rows_add", "moe_scatter_slots",
        "moe_row_dot", "moe_row_scale",
        "grouped_nn", "grouped_nt", "grouped_tn",
        "cat_extract", "cat_place",
    ],
    "内部/调试": [
        "_fn", "_numpy_exact_gemm",
    ],
}

OPTIONAL_GROUPS = ("MoE（可选）",)


def required_methods(include_optional: bool = True):
    """返回接口方法名列表。"""
    out = []
    for name, items in GROUPS.items():
        if not include_optional and name in OPTIONAL_GROUPS:
            continue
        out.extend(items)
    return out


def missing(inst, include_optional: bool = False):
    """检查一个后端实例缺哪些方法（新后端接入时先跑这个）。"""
    return [m for m in required_methods(include_optional) if not callable(getattr(inst, m, None))]


# ---- 工厂 ---------------------------------------------------------------

_REGISTRY = {}          # name -> factory() -> 实例
_CACHE = {}             # name -> 实例
_ACTIVE = None          # 当前实例
_ACTIVE_NAME = None
_TOKEN = 0              # 每次 register/reset 自增，供热路径做廉价失效判断

#: 推荐实现顺序（覆盖面优先）
PREFERRED = ("cuda", "rocm", "vulkan", "cpu")


def _cuda_factory():
    """默认后端：延迟导入 cuda.py，避开包属性名冲突。"""
    import importlib
    pkg = sys.modules["tgai_go.train"]
    saved = pkg.__dict__.get("cuda")
    mod = importlib.import_module("tgai_go.train.cuda")
    if saved is not None and saved is not mod:
        pkg.__dict__["cuda"] = saved
    return mod.runtime()


def cache_token():
    """热路径用的失效令牌：变了就说明缓存的实例该重建。"""
    return _TOKEN


def register(name, factory, replace=False):
    """登记一个后端工厂。factory 必须无参并返回实例（或抛异常表示不可用）。"""
    name = str(name).strip().lower()
    if not name:
        raise ValueError("后端名不能为空")
    if name in _REGISTRY and not replace:
        raise ValueError("后端 %r 已登记；要替换请传 replace=True" % name)
    global _TOKEN
    _REGISTRY[name] = factory
    _TOKEN += 1


def available():
    """已登记的后端名（不代表当前环境真能用）。"""
    return sorted(_REGISTRY)


def _want_name(name=None):
    if name:
        return str(name).strip().lower()
    env = os.environ.get("TG_BACKEND", "").strip().lower()
    if env:
        return env
    return _ACTIVE_NAME or "cuda"


def get(name=None, *, check=True):
    """取后端实例（带缓存）。name 为空时按 TG_BACKEND → 当前 → cuda。"""
    global _ACTIVE, _ACTIVE_NAME
    want = _want_name(name)
    if want in _CACHE:
        _ACTIVE, _ACTIVE_NAME = _CACHE[want], want
        return _ACTIVE
    if want not in _REGISTRY:
        if want == "cuda":
            register("cuda", _cuda_factory)
        else:
            raise NotImplementedError(
                "后端 %r 尚未登记。已登记：%s。\n"
                "要接新后端：tgai_go.train.backend.register(%r, factory)，"
                "接口清单见 backend.required_methods()。" % (
                    want, ", ".join(available()) or "(空)", want))
    inst = _REGISTRY[want]()          # 这里的异常就是"后端不可用"的真相
    _CACHE[want] = inst
    _ACTIVE, _ACTIVE_NAME = inst, want
    if check:
        lack = missing(inst)
        if lack:
            raise RuntimeError(
                "后端 %r 缺少 %d 个接口方法：%s\n"
                "（MoE 相关可选，详见 tgai_go.train.backend.GROUPS）"
                % (want, len(lack), ", ".join(lack[:10]) + ("..." if len(lack) > 10 else "")))
    return inst


def active_name():
    return _ACTIVE_NAME


def reset(name=None):
    """清缓存（测试用；也用于切换后端后强制重建实例）。"""
    global _ACTIVE, _ACTIVE_NAME
    global _TOKEN
    _CACHE.clear()
    _ACTIVE, _ACTIVE_NAME = None, (str(name).strip().lower() if name else None)
    _TOKEN += 1


register("cuda", _cuda_factory)

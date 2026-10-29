# -*- coding: utf-8 -*-
"""Qwen2 预设（B3）——**复用 llama 核心，只开一个开关**。

Qwen2 相对 llama 的差异（HuggingFace `Qwen2ForCausalLM`）：
- Q/K/V 投影**带 bias**（这是唯一的结构差异）
- GQA 可配（`n_kv_heads`）
- RMSNorm / RoPE / SwiGLU / untied 输出头 —— 与 llama 相同
真实 Qwen2 的 `tie_word_embeddings=False`，即 untied，与本实现一致。

**本文件不写任何模型逻辑**：全部走 `llama_ref.Llama`，只是把 `qkv_bias` 打开。
这就是"一份 C 核心、多套架构"在 Python 侧的形态；C 侧对应的是架构描述符里的一个字段。
"""
try:
    from . import llama_ref
    from .llama_ref import Llama as _Llama      # 包内（相对）
except Exception:
    import llama_ref
    from llama_ref import Llama as _Llama       # 源码树（绝对）


def make_cfg(d=32, L=2, nh=4, n_kv=2, ff=64, V=64, seq=16, theta=10000.0, seed=0):
    """与 llama 同签名，唯一差别是 qkv_bias=True。"""
    return llama_ref.make_cfg(d=d, L=L, nh=nh, n_kv=n_kv, ff=ff, V=V, seq=seq,
                              theta=theta, seed=seed, qkv_bias=True)


def build(cfg):
    return llama_ref.Llama(cfg, cfg["weights"])

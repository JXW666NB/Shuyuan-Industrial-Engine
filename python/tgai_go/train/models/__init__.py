# -*- coding: utf-8 -*-
"""预设架构的"薄定义"（随包分发）。

每个文件只写**架构差异**，算子全部来自 `tgai_go.train`：
- `llama_ref`：RMSNorm / RoPE / SwiGLU / GQA
- `gpt2_ref`：LayerNorm / GELU(tanh) / 学习式位置 / 全 bias
- `qwen2_ref`：复用 llama，只把 `qkv_bias` 打开

与 `training/harness/` 下的同名文件**逐字节一致**（那边是对拍用的开发副本）。
放在包里的原因：一键流水线（`tgai_go.train.pipeline`）要能在 `pip install` 之后工作，
不能依赖仓库里的 `training/` 目录。
"""
from . import llama_ref, gpt2_ref, qwen2_ref   # noqa: F401

__all__ = ["llama_ref", "gpt2_ref", "qwen2_ref"]

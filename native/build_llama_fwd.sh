#!/bin/bash
# Linux x86_64。不链 libcuda / libcublas：驱动和 cuBLAS 符号由 Python 侧
# 已经加载好的运行时传进来（同一 primary context、PEDANTIC 句柄）。
# 一份 llama_fwd.c 同时服务 presets.llama（qkv_bias=0）和 presets.qwen2（qkv_bias=1）。
# 差异在 TgArch，不另编一份算子。
set -euo pipefail
cd "$(dirname "$0")"
gcc -shared -fPIC -O2 -std=c11 -Wall -Wextra -Wno-unused-function \
    -o libtgai_llama_fwd.so llama_fwd.c -lm
echo "built $(pwd)/libtgai_llama_fwd.so"

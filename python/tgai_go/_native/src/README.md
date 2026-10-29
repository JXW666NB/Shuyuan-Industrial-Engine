# 纯 C 预设核心（源码）

本目录是**随 wheel 分发的 C 源码**，与 `native/` 同源。

| 文件 | 作用 |
|---|---|
| `llama_fwd.c` / `llama_fwd.h` | 架构描述符（`TgArch`）驱动的 C 前向：llama / qwen2 / gpt2 共用同一份实现 |
| `build_llama_fwd.sh` | Linux 下的编译脚本（产出 `libtgai_llama_fwd.so`） |

**为什么随包分发源码**：
1. 用户能看到"预设架构真的是一份 C 核心 + 一张描述符表"，而不是黑盒；
2. 需要自己重编（换编译器/换平台/打补丁）时不必再找仓库；
3. 与 `tgai_go/_native/*.dll|*.so` 里的预编译产物对应。

**描述符字段**（`TgArch`）：`qkv_bias` / `norm`（rmsnorm|ln） / `pos`（rope|wpe） /
`ffn`（swiglu|gelu） / `n_kv`（GQA 比例） —— 加一套 llama 系架构只需要填字段，不改 C 逻辑。

# TGAI GO

**推理 + 训练一体的自研引擎** —— `pip install` 后即可加载 `.TG` 模型推理，无需编译；
也能**不用 PyTorch** 直接训练（CUDA 后端），训完直出 `.TG` 交给自家引擎加载。
硬依赖只有 `numpy`。

```python
import tgai_go

m = tgai_go.load("v27_merged_bf16.TG")
print(m.generate("什么是机器学习？", max_tokens=128))
```

---

## 特性

| 特性 | 说明 |
|---|---|
| **零编译安装** | 原生引擎（C + OpenMP）以预编译库随包分发，`pip install tgai-go` 即可用 |
| **无外部依赖** | 引擎只依赖系统 `kernel32`（静态链接 CRT）；Python 侧只要 `numpy` |
| **CPU 推理** | 纯 CPU，**4 线程实测约 11 tok/s**（28 层 MoE，5.6 GB BF16 权重）—— 已达内存带宽的约 92% |
| **多轮对话** | 内置 KV cache 复用，跨轮只算新增 token |
| **配置随模型走** | `.TG` 可内嵌对话模板（九标记 / ChatML / 纯文本），库自动读取 |
| **流式输出** | `for chunk in m.stream(...)` 逐段拿文本 |
| **CPU 友好** | 不需要独显；核显机器、笔记本、服务器都能跑 |
| **可训练** | `tgai_go.train` 提供 CUDA 训练后端，**不依赖 PyTorch**，常用写法只改 `import` 即可迁移（见下） |

---

## 训练（CUDA 后端，1.4.0 起）

从 1.4.0 开始，本库不止能推理，还能**训练**：`tgai_go.train` 是一个自研的自动微分 + CUDA 后端，
显存、cuBLAS、NVRTC 全部由本包直接驱动，**不引入 PyTorch**。

### 安装

```bash
pip install tgai-go              # 推理 + CPU 训练
pip install "tgai-go[cuda]"      # 追加 CUDA 训练的运行时（cuBLAS / NVRTC）
```

> **Linux 必看**：`/usr/lib` 里通常**没有** cuBLAS，必须装 `[cuda]` extra（它拉
> `nvidia-cublas-cu12` + `nvidia-cuda-nvrtc-cu12`）。不装时推理不受影响，只会在初始化
> CUDA 训练后端时报 `找不到 libcublas.so.12`。
> **实测**：干净 venv 里 `pip install "tgai-go[cuda]"` 后，11/11 迁移一致性用例全过（无 PyTorch、无 PYTHONPATH）。

### 只改 import 即可迁移

```python
# 原来
import torch
import torch.nn as nn
# 迁移后 —— 只改这一行，其余代码不动
import tgai_go.train as torch
import tgai_go.train as nn           # nn 也在这个名字空间里
m = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 4))
x = torch.randn(8, 16)
y = m(x)
loss = torch.nn.functional.cross_entropy(y, targets)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
opt.zero_grad(); loss.backward(); opt.step()
```

仓库里的 `training/harness/pytorch_conformance.py` 用 **11 段标准 PyTorch 写法**做了回归，
**CPU 与 CUDA 两个后端都是 11/11 通过**。

### 实测（RTX 4090 单卡，2026-10）

| 场景 | 对比 PyTorch | 备注 |
|---|---|---|
| seq=2048 / MoE（E=8 top-2, 138.4M） | **0.116 s vs 0.104 s（1.12×）** | 短上下文已接近追平 |
| seq=4096 / 4 层 MoE（453.0M） | 1.034 s vs 0.181 s（5.7×） | 长上下文差距在 attention 重算 |
| seq=8192 / 1 层 MoE（138.4M） | 0.987 s vs 0.189 s（5.2×） | 峰值 14.2 GiB（PyTorch 7.1 GB） |

过拟合验证：4 层 / 453.0M 的 V2.7 形状 MoE，40 步 loss **10.47 → 1.51**（PyTorch 同配置 10.45 → 1.80）。

### 已支持 / 边界（重要）

| 项 | 状态 |
|---|---|
| `nn.Linear / Embedding / RMSNorm / CausalSelfAttention / Sequential` | ✅ |
| **`nn.MoE`**（`top_k` 可配，`gate="softmax"` 或 `"sigmoid"`） | ✅ 参数名与 `.TG` 对齐，便于直出 |
| `optim.AdamW`（GPU 上更新） | ✅ 与 CPU 后端相对误差 **9.1e-08** |
| `optim.Muon` | ⚠️ 相对误差 **7.0e-07**（跨 BLAS 累加顺序差异，未达 1e-7 的内部判据） |
| 长上下文 `functional.scaled_dot_product_attention(..., lowmem=True)` | ✅ 不物化 P，seq=8192 可跑 |
| dtype | 仅 `float32` / `int64`（**暂无 bf16/fp16 训练**） |
| `torch.save/load` | 不是 torch 的 zip 格式（用 `state_dict` + `numpy.save`） |
| 比较运算 `>` `<` | 未实现 |

> **口径**：11/11 证明的是"这 11 种常用写法可以零改动迁移"，**不等于**任意 PyTorch 代码都能跑。
> 边界见上表，遇到不支持的地方会明确抛异常，不会静默算错。

---

## 安装

```bash
pip install tgai-go
```

> 平台：Windows x64 已实测；Linux/macOS 需自行编译原生库（见文末）。

---

## 模型从哪来

本库是**推理引擎**，不含模型权重。TGAI 系列模型请在魔搭社区下载：

| 模型 | 格式 | 大小 | 说明 |
|---|---|---|---|
| **TGAI NB V2.7** | `.tg` | 5.6 GB | **最新**（域隔离 MoE + 9 标记结构化对话）|
| TGAI NB V2.6 | `.gguf` | 5.5 GB | 需用转换工具转成 `.tg` |
| TGAI NB V2.6 Q4_K_M | `.gguf` | 1.7 GB | 量化版（低配/手机）|

```bash
pip install modelscope
modelscope download --model JXW7777/TGAI_NB TGAI_NB_V2.7.tg --local_dir .
```

**推荐直接用 V2.7 的 `.tg` 文件** —— 它是本引擎的原生格式，
对话模板、停止符、采样默认值全部内嵌，加载后引擎自动读取，无需手工配置。

> 📦 模型主页：<https://modelscope.cn/models/JXW7777/TGAI_NB>

---

> 📖 **第一次使用？看 [使用手册.md](使用手册.md)** —— 3 步跑通，含必踩的坑。

## 快速开始

### 单轮生成

```python
import tgai_go

m = tgai_go.load("v27.TG")

print(m.generate("什么是人工智能？"))
print(m.generate("写一句关于春天的话。", max_tokens=64, stop=["。"]))
```

### 多轮对话（自动带历史）

```python
with tgai_go.load("v27.TG") as m:
    s = m.session(system="你是 TGAI，一个中文助手")
    print(s.chat("我叫小明，今年8岁。"))
    print(s.chat("我叫什么名字？"))     # → "你叫小明，今年8岁。"
    print(s.chat("我几岁了？"))         # → "8岁"
```

### 流式输出

```python
with tgai_go.load("v27.TG") as m:
    for chunk in m.stream("讲个短故事", max_tokens=200):
        print(chunk, end="", flush=True)
```

### 性能调节

```python
m = tgai_go.load("v27.TG", threads=8, preload=True)
print(m.threads)          # 8

# 实测（Ryzen 7 7730U, 8C/16T, V2.7 28层 MoE）：
#   4 线程  → 约 11 tok/s   ← 默认值，已饱和
#   8 线程  → 约 11 tok/s   （无增益）
#  16 线程  → 约 10 tok/s   （超线程反而略慢）
#  说明：瓶颈是内存带宽，不是算力 —— 每 token 要把 1.7 GB 权重过一遍
#        （V2.7 BF16）。想更快只能降精度，加线程没用。
```

---

## API

### `tgai_go.load(path, *, dll=None, extend_ctx=0, threads=None, preload=False)`

| 参数 | 说明 |
|---|---|
| `path` | `.TG` 模型路径 |
| `dll` | 指定引擎库（默认用包内 `_native/`） |
| `extend_ctx` | 扩展上下文（NTK-by-parts / YaRN），0 = 用模型自带 `max_seq` |
| `threads` | CPU 线程数；`None` = 引擎默认，`0` = 全部逻辑核 |
| `preload` | `True` = 预解码全部权重（首次推理不卡顿，但占更多内存） |

返回 `Model` 对象，支持 `with` 语句自动释放。

### `Model`

| 成员 | 说明 |
|---|---|
| `generate(prompt, **kw)` | 单轮生成，返回字符串 |
| `stream(prompt, **kw)` | 流式生成，逐段 yield |
| `session(system="")` | 开多轮会话，返回 `ChatSession` |
| `encode(text)` / `decode(id)` | 分词 / 反分词 |
| `set_threads(n)` / `threads` | 线程数 |
| `preload()` | 预加载权重 |
| `n_layers` / `d_model` / `vocab_size` / `max_seq_len` | 模型信息 |
| `chat_meta` | 模型内嵌的对话配置 |
| `close()` | 释放（`with` 会自动调） |

### `ChatSession`

| 成员 | 说明 |
|---|---|
| `chat(msg, **kw)` | 一轮对话，返回完整回答 |
| `stream(msg, **kw)` | 流式一轮 |
| `chat_with_prompt(msg)` | 返回 `(回答, 实际 prompt)`，排查格式用 |
| `history` | `[(q, a), ...]`，可直接改 |
| `clear()` | 清历史 + 重置 KV cache |
| `max_turns` | 进 prompt 的历史轮数（默认 8） |

### 生成参数（`**kw`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `max_tokens` | 256 | 最大生成长度 |
| `temperature` | 0.6 | 随机性（0 = 贪心） |
| `top_k` | 40 | 前 K 个采样 |
| `top_p` | 0.9 | 核采样 |
| `min_p` | 0.0 | 最小概率阈值 |
| `min_tokens` | 4 | 至少生成 N 个 token 才允许停止（防空回复） |
| `repeat_penalty` | 1.05 | 重复惩罚 |
| `frequency_penalty` | 0.0 | 频率惩罚 |
| `dry_multiplier` | 0.8 | DRY 防复读强度 |
| `stop` | `None` | **自定义停止串**，如 `stop=["。", "<用户>"]` |
| `seed` | -1 | 随机种子（-1 = 随机） |

**参数优先级**：函数参数 > `.TG` 内嵌默认值 > 库内置默认值

---

## `.TG` 模型内嵌对话配置

`.TG` 文件可以自带对话模板，让库知道"怎么跟这个模型说话"。
第三方模型只要打包时带上配置，**不用改库的代码**。

```bash
# 查看配置
python -m tgai_go.tools.write_chat_meta model.TG --show

# 写内置预设: v27（TGAI 九标记） / chatml（Qwen2 等） / plain（纯文本）
python -m tgai_go.tools.write_chat_meta model.TG --preset v27
```

配置内容：

```
render_mode    markers | chatml | plain
chat_template  {SYS}{HIST}<用户>\n{Q}\n<TGAI>{TAIL}\n
template_turn  <用户>\n{Q}\n<TGAI><回答>\n{A}
special_tokens <用户> <TGAI> <思考> <回答> ...
turn_markers   <用户> <TGAI> <系统> ...     （生成到它即停）
stop_strings   ...                          （停止串）
gen_defaults   temperature=0.6 top_k=40 ...
```

只读查看（不加载权重，很快）：

```python
import tgai_go
print(tgai_go.chat_meta_of("v27.TG"))
```

---

## 内存

| 模型 | 权重（BF16） | 运行峰值 |
|---|---|---|
| TGAI V2.7（28 层 MoE） | 5.6 GB | 约 3.6 GB（开启 BF16 直算）|

`preload=True` 会额外占用（用于缓存解码后的权重），内存紧张时不要开。

---

## 常见问题

**Q: 生成结果是空的？**
调大 `min_tokens`（默认 4）。某些模型的首选 token 是停止符，不禁就会立即停。

**Q: 一直复读不停止？**
加自定义停止串：`m.generate(q, stop=["<用户>", "\n\n"])`。
或降低 `temperature`、调高 `dry_multiplier`。

**Q: 多轮对话记不住前文？**
检查 `.TG` 是否带对话配置（`tgai_go.chat_meta_of(path)`）。
没有配置时库会按 `arch_type` 猜，可能不匹配。

**Q: 加载很慢？**
首次加载要 mmap 权重。加 `preload=True` 把首次推理的卡顿提前。

**Q: 支持 GPU 吗？**
当前包是纯 CPU 引擎。GPU 桥接版本仍在验证（实测核显不如 CPU，
原因见主仓文档）。

---

## 从源码构建原生库（Linux/macOS）

```bash
git clone <repo>
cd "TGAI GO"
# Linux
gcc -O3 -fopenmp -mavx2 -shared -fPIC -o cpu/build/libtgai_engine.so \
    src/tgai_engine.c
# macOS
clang -O3 -Xpreprocessor -fopenmp -shared -o cpu/build/libtgai_engine.dylib \
    src/tgai_engine.c
```

然后把产物放到 `tgai_go/_native/`。

---

## License

MIT

---

## 相关

- `.TG` 格式：128 字节头 + 分词器段 + 权重表 + 权重数据（+ 可选对话配置段）
- 支持的架构：TGAI MoE / Llama / Mistral（按 `arch_type` 字段区分）
- 量化：FP32 / FP16 / BF16 / INT8 / Q8_0 / Q4_0 / Q4K

---

## 预设架构 + 纯 C（1.4.1 起）

**一个 C 核心 + 一张描述符表**，多套架构共用同一份实现 —— 不是每套架构复制一份算子。

```python
from tgai_go import presets

m = presets.llama(cfg)      # RMSNorm / RoPE / SwiGLU / GQA
m = presets.qwen2(cfg)      # 同一份 C，仅 qkv_bias=1
m = presets.gpt2(cfg)       # LayerNorm / GELU / 学习式位置 / 全 bias
y = m.forward(ids)          # 一次前向只跨 Python↔C 一次
```

描述符（`TgArch`）字段决定差异，C 里用 `if/switch` 选分支：

| 字段 | 取值 | 影响 |
|---|---|---|
| `qkv_bias` | 0/1 | Q/K/V 投影后是否加 bias（llama=0，qwen2=1） |
| `norm` | `rmsnorm` / `ln` | 归一化方式（GPT-2 用 LayerNorm，带 beta） |
| `pos` | `rope` / `wpe` | 位置编码（GPT-2 用学习式位置表） |
| `ffn` | `swiglu` / `gelu` | FFN 类型 |
| `n_kv` | 整数 | GQA 比例（repeat 次数 = n_heads / n_kv） |

**C 源码随包分发**：`tgai_go/_native/src/llama_fwd.c`（约 40 KB）与 `llama_fwd.h`。
需要换编译器或打补丁时直接改它，编译脚本 `build_llama_fwd.sh` 也在同目录。

### 实测（与 PyTorch 参考实现同权重、同输入对拍）

| 架构 | logits 相对误差 | 梯度最差 |
|---|---|---|
| llama | 8.664e-06 | 9.228e-06 |
| qwen2 | 4.403e-06 | 8.382e-06 |
| gpt2 | 1.857e-07 | 4.321e-07 |

判据是 logits < 1e-4、梯度 < 1e-3。C 路径与 Python 参考的误差同为 1e-6 量级，
且连跑两次逐位一致。

### 一键流水线（给定语料即可训练）

```python
from tgai_go.train.pipeline import run
run(corpus_dir="我的语料/", out_dir="run/", arch="llama", steps=200)
```

- **分词表不用你提供**：引擎会用你的语料自训一个 BPE（`tgai_go.train.bpe`）；
- 产出 `tokenizer.json` / `ckpt.npz` / `meta.json`；
- 实测（小配置）：自训词表 132、loss 5.3091 → 0.1755、重载后 logits 逐位一致。

### 导出 `.TG` 给引擎加载

```python
from tgai_go.train.export import save_tg
save_tg("model.TG", params, n_layers=..., d_model=..., n_heads=..., d_ff=...,
        vocab_tokens=tokens, merges=merges, rope_theta=10000.0)
```

实测：导出的 `.TG` 能被 `tgai_go.load()` 加载并在 GPU 引擎上稳定生成（5/5，退出码 0）。

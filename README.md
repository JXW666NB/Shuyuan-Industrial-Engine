# 树缘智能 · 工业 AI 引擎

<p align="center">
  <strong>推理 + 训练一体，自研内核，零框架依赖</strong><br>
  <sub>不装 PyTorch 也能训模型 · 一份 C 核跑多套架构 · 训完直出 <code>.TG</code> 交给自家引擎加载</sub>
</p>

<p align="center">
  <a href="https://github.com/JXW666NB/Shuyuan-Industrial-Engine"><img src="https://img.shields.io/badge/GitHub-Shuyuan--Industrial--Engine-black" alt="GitHub"></a>
  <img src="https://img.shields.io/badge/version-1.4.1-blue" alt="version">
  <img src="https://img.shields.io/badge/language-C%20%2B%20Python-green" alt="language">
  <img src="https://img.shields.io/badge/CUDA-supported-brightgreen" alt="cuda">
  <img src="https://img.shields.io/badge/deps-numpy%20only-orange" alt="deps">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20ESP32-red" alt="platform">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="license">
</p>

---

## 这是什么

一套**自研的推理 + 训练引擎**。同一份模型代码，CPU 与 CUDA 都能跑；训练不需要 PyTorch，
装完只有 `numpy` 一个硬依赖。

两条线共用一套内核：

| 方向 | 用途 |
|---|---|
| **推理** | 把模型转成 `.TG`，下发到工控机 / 边缘盒子 / 采集端，零外部依赖跑起来 |
| **训练** | 用 CUDA 后端训自己的模型，训完直出 `.TG`，闭环交给上面的推理侧 |

> 仓库里的 `tgai_*` 文件名、`.TG` 格式与 `pip install tgai-go` 包名是既有产品命名，保留不变。

### 工业场景仍然是第一约束

| 现场约束 | 我们的应对 |
|---|---|
| **数据不出厂** | 全离线；语料、权重、日志都留在本地（训练也不需要联网或外部框架） |
| **设备碎片化** | 同一份源码覆盖 工控机(x86) → 边缘盒子(ARM) → ESP32 采集端 |
| **无运行时环境** | 静态链接 CRT，Windows 侧实测仅依赖 `kernel32.dll` |
| **算力参差** | 纯标量 / AVX2 / CUDA / Vulkan 多后端，同一模型按设备选精度（FP32→Q4K） |

> 细节见归档的旧版说明：`docs/README_legacy_inference.md`。

---

## 三条命令看懂全部

```bash
pip install tgai-go          # 硬依赖只有 numpy；CUDA 训练另装 [cuda] extra
```

```python
# ① 推理
import tgai_go
m = tgai_go.load("model.TG")
print(m.generate("你好，介绍一下你自己"))

# ② 训练（PyTorch 风格，只改导入）
import tgai_go.train as torch
from tgai_go.train import nn, optim
lin = nn.Linear(64, 128)                     # 与 torch.nn 同名同义
loss.backward(); opt.step()                  # 与 torch 写法一致

# ③ 预设架构训练：给语料就行，分词表引擎自训
from tgai_go.train.pipeline import run
r = run(corpus_dir="我的语料/", out_dir="run/", arch="gpt2", steps=200)
print(r["first_loss"], "->", r["final_loss"])
```

---

## 实测数字（RTX 4090 单卡，与 PyTorch 同结构同参数量）

### 训练性能与显存

| 项 | TGAI GO | PyTorch | 结果 |
|---|---|---|---|
| seq=2048 单步 | **0.036 s** | 0.096 s | **快 2.7 倍** |
| seq=8192 单步 | **0.144 s** | 0.189 s | **快 1.3 倍** |
| 峰值显存 seq=2048 | **3.418 GiB** | 2.966 GiB | 1.15× |
| 峰值显存 seq=8192 | **6.775 GiB** | 7.101 GiB | **0.95×（更省）** |

模型：MoE d=1024 / nh=16 / dh=64 / ff=4096 / E=8 top-2 / V=32779。

### 能不能替代 PyTorch？——过拟合对照

同一份训练循环，一份用 `tgai_go.train`(CUDA)、一份用真 `torch`，**同一份初始化**：

| 项 | TGAI CUDA | PyTorch |
|---|---|---|
| loss 首 → 末（固定 32 样本过拟合） | 4.161366 → **0.008179** | 4.161367 → 0.008179 |
| **400 步逐 step 最大相对差** | **5.68e-05** | — |

短程 20 步时差 **2.3e-07**（fp32 舍入量级）。→ 在覆盖的算子范围内可以替代。

### 依赖对比

| 环境 | 体积 |
|---|---|
| TGAI GO（CPU 推理/训练） | **101 MB**（`numpy` + 本包） |
| TGAI GO（CUDA 训练） | 1.2 GB（多两个 nvidia 库） |
| PyTorch | 1.5 GB **+** nvidia 库 2.9 GB |

---

## 预设架构：一份 C 核心 + 一张描述符表

各架构只写"薄"的定义，**算子实现只有一份**。C 里由结构体 `TgArch` 驱动：

| 字段 | 取值 | 对应架构 |
|---|---|---|
| `qkv_bias` | 0/1 | qwen2 = 1 |
| `norm` | rmsnorm / **ln** | GPT-2 用 LayerNorm（带 beta） |
| `pos` | rope / **wpe** | GPT-2 用学习式位置 |
| `ffn` | swiglu / **gelu** | GPT-2 用 GELU(tanh) |
| `n_kv` | 整数 | GQA 比例 |

加一套 llama 系架构 = 填字段，**不改 C 逻辑、不复制算子**。

| 架构 | Python 与 PyTorch 对拍 | C 核心 |
|---|---|---|
| llama（RMSNorm/RoPE/SwiGLU/GQA） | 8.664e-06 | ✅ 8.664e-06 |
| qwen2（llama + QKV bias） | 4.403e-06 | ✅ 4.403e-06（同一份 C） |
| gpt2（LayerNorm/GELU/学习式位置/全 bias） | 1.857e-07 · **真实 GPT-2 权重 8.254e-07** | ✅ 2.698e-07 |
| TGAI V3（非 CSA：Block AttnRes + SiTU-MoE + 共享专家） | 前向 4.793e-05 · 反向 1.267e-04 | ⏳ |
| DeepSeek MLA（隐空间注意力） | CPU 1.944e-07 · CUDA 3.092e-07 | ⏳ |

判据：前向 logits < 1e-4、反向梯度 < 1e-3。**每个判分台先自证**（同侧两次必须恰好 0）。

C 源码随包分发在 `tgai_go/_native/src/llama_fwd.c`（约 40 KB）。

---

## 从 PyTorch 迁移

**只需要改导入**，其余写法不变：

```python
# 原来
import torch
from torch import nn, optim
import torch.nn.functional as F

# 迁移后 —— 只改这一行
import tgai_go.train as torch
from tgai_go.train import nn, optim
from tgai_go.train import functional as F
```

同名列：`nn.Linear / Embedding / RMSNorm / LayerNorm / MoE`、`optim.AdamW / Muon`、
`F.cross_entropy / softmax / scaled_dot_product_attention`、`backward() / zero_grad() / step()`。

迁移一致性判分台：**13/13 PASS**（CPU 与 CUDA 各一遍），覆盖切片、`repeat_interleave`、
交叉熵、SDPA、优化器步进等。

### 已知差异（会咬人的地方）

1. **给 CUDA 参数赋初值必须用 `load_state_dict()`** —— `p.data = ...` 在 CUDA 上不生效（值在设备侧）；
2. 尚未提供的 API：`nn.ReLU / MSELoss / SGD / torch.utils.data` 等，用到要自己补；
3. `nn.Linear` 默认 `bias=True`（1.4.1 起与 PyTorch 对齐；更早版本是 `False`）。

---

## 一键流水线

```python
from tgai_go.train.pipeline import run
run(corpus_dir="语料目录/", out_dir="run/", arch="llama", steps=200)
```

- **不用准备分词表**：引擎会用你的语料自训一个 BPE；
- 产出 `tokenizer.json` / `ckpt.npz` / `meta.json`；
- 实测（小配置）：自训词表 132、loss 5.3091 → 0.1755、**重载后 logits 逐位一致**；
- 支持 `arch="llama" | "qwen2" | "gpt2"`（同一份流程，只换一个参数）。

一句话版本（15 行）：`examples/train_preset.py`；迁移示例：`examples/torch_style_overfit.py`。

---

## 训练完直出 `.TG`

```python
from tgai_go.train.export import save_tg
save_tg("model.TG", params, n_layers=..., d_model=..., n_heads=..., d_ff=...,
        vocab_tokens=tokens, merges=merges, rope_theta=10000.0)
```

实测：导出的 `.TG` 能被 `tgai_go.load()` 加载，并在 GPU 引擎上稳定生成（5/5，退出码 0）；
导出**字节确定**（同输入两次 sha256 一致）。

---

## 目录结构

```
python/tgai_go/            推理库 + 训练后端
  _native/src/             纯 C 预设核心源码（随包分发）
  train/                   训练后端（numpy autograd + CUDA）
    bpe.py                 自训 BPE（随包，不依赖仓库路径）
    pipeline.py            一键流水线
    export.py              导出 .TG
    models/                预设架构的"薄定义"
examples/                  可运行示例（迁移对照 / 预设训练）
training/harness/          全部判分台（先自证，再考核）
docs/                      设计文档、验收记录、归档
native/                    C 核心源码（与 _native/src 同源）
```

---

## 文档

| 文档 | 内容 |
|---|---|
| `docs/goal_final_report.md` | 逐项判据、实测数字、复跑命令、**未达成项清单** |
| `training/ACCEPTANCE_TRAIN_CUDA.md` | 全部里程碑的实测记录与踩坑 |
| `docs/why_tgai_go.md` | 对外"为什么用它"的证据包 |
| `docs/v3_architecture_facts.md` | 从 `model_v3.py` 源码读出的 V3 架构事实 |
| `docs/README_legacy_inference.md` | 1.4.0 时期的推理引擎 README（归档） |

---

## 许可

MIT

# TGAI GO 训练 · 一页 A4 入门

> 目标读者：想用自己的模型/语料在 TGAI GO 上训练的人。
> 全程只有三件事：**跑自检 → 准备语料 → 开训**。

---

## 0. 环境要求

| 项 | 要求 |
|---|---|
| 系统 | Windows 10/11 x64 或 Linux x64 |
| 显卡 | NVIDIA（CUDA 11.8+ 驱动）。无卡会自动退回 CPU（能跑但很慢） |
| Python | 3.9+（**不需要装 PyTorch**） |
| 依赖 | `numpy`（唯一必需）；`paramiko` 只有远端脚本才用 |

安装：把 `python/tgai_go` 放进 `PYTHONPATH`，或 `pip install -e python/`。

---

## 1. 先跑自检（30 秒）

```bash
python -m tgai_go.train.doctor
```

期望输出结尾：

```
自检通过：10/10 项 PASS
```

它真跑 GPU，逐项检查：GEMM、转置 GEMM、逐元素（含 float4）、RMSNorm、
SiLU-GLU、带 `ignore_index` 的交叉熵、causal attention、AdamW、一步训练。
**任何一项 FAIL 都会打印「怎么办」**，先修环境再往下走。

---

## 2. 准备语料

每行一个 JSON，字段 `text`，内部用九标记格式：

```json
{"text": "<用户>\n你是谁研发的<TGAI><回答>\n我是由JXW开发的TGAI。"}
```

规则：
- `<TGAI>` 前**不要**换行（引擎就是这样拼的）；
- 回答长度 **≤ 40 字符**（超出的会被跳过，用于字符级小模型）；
- 一行一条对话。

用环境变量指定语料（**会替换默认语料**，不是追加）：

```bash
# Windows
set TG_CORPUS=D:\data\my_corpus.jsonl
# Linux
export TG_CORPUS=/data/my_corpus.jsonl
```

⚠️ 两个常见坑（都已在代码里改成硬报错/告警）：
- 语料解析不出任何一条 → **直接报错**，不会静默用内置样例；
- 超过 `TG_MAX_CHARS`（默认 200 万字符）→ 打印「扫过多少 / 用了多少」，
  要全量设 `TG_MAX_CHARS=0`。

---

## 3. 开训

```bash
python examples/exam2_chat10m_tgai.py --device cuda --steps 600 --batch 64 --out my_model
```

关键参数：

| 参数 | 含义 | 建议 |
|---|---|---|
| `--batch` | 每步取多少条对话 | **语料大于几千条时必须给**（否则整语料当一批，显存爆炸） |
| `--steps` | 训练步数 | 小语料 600 步足够；语料大按 `epoch × ceil(条数/batch)` 算 |
| `--lr` | 学习率 | 默认 2e-3（AdamW） |
| `--out` | 输出名 | 产物：`my_model.TG`（引擎格式）+ 词表/元信息 |

产物直接能被 C 引擎加载（`.TG` 单文件：权重 + 词表 + 对话模板）。

---

## 4. 可选开关（出问题时逐个关掉定位）

| 环境变量 | 作用 |
|---|---|
| `TG_NO_FUSED=1` | 关掉融合内核（RoPE/GLU），回退组合实现 |
| `TG_NO_VEC4=1` | 关掉 float4 向量化 |
| `TG_LAUNCH_FAST=0` | 关掉 launch 参数模板缓存 |
| `TG_CUDA_GRAPH=1` | 开 CUDA Graph 重放（**只在固定 batch 时可用**，与 `--batch` 互斥） |
| `TG_FP16=1` | fp16 GEMM（实验性：精度达标时几乎没有速度收益） |

---

## 5. 换形状 / 换模型

模型结构在脚本顶部的 `CFG` 里：

```python
CFG = dict(D=256, NH=8, L=8, FF=1280, MAXLEN=64)
```

已实测可用范围（4090）：`D 128~512`、`NH 4~8`（`D/NH` 需整除，16/32/64 都测过）、
`FF 512~2048`、`L 2~8`、`S 32~128`。改完直接重训，不需要改引擎。

---

## 6. 从 PyTorch 迁过来

同一份代码，只换导入：

```python
import tgai_go.train as T      # 原来是 import torch as T
```

其余写法照旧（`T.tensor` / `T.nn.RMSNorm` / `T.optim.AdamW` / `T.functional.*`）。

---

## 7. 出问题怎么查

1. `python -m tgai_go.train.doctor --trace` —— 带完整堆栈的自检；
2. 训练脚本会打印 `语料=… 词表 … 对话 N 条 序列 S`，先确认这三个数对；
3. 数值不对时按上表逐个关融合开关，二分定位。

# 为什么用 TGAI GO 而不用 PyTorch（训练侧）

> 本文只写**实测数字**与**可复现命令**。凡是还没做到或还没验证的，单列在「已知短板」，不美化。
> 测试机：云端单卡 RTX 4090（47.4 GB 可用）、Ubuntu、驱动 610.57。
> 仓库版本：`tgai_go` 1.4.0（含 1.4.0 之后的性能与预设改动）。

---

## 一句话结论

**同样的 MoE 模型、同一份数据、同一份训练循环**，TGAI GO 在短上下文比 PyTorch **快 2.7 倍**、
长上下文快约 1.3 倍；显存 2048 是 1.14 倍、**8192 反而更省（0.94×）**；
依赖只是 `numpy`，不需要装 PyTorch；脚本迁移只需改一行 `import`。

---

## 1. 速度：两个上下文都不慢于 PyTorch

基准：MoE `d=1024 / nh=16 / dh=64 / ff=4096 / E=8 top-2 / V=32779`、1 层、**两边参数量相同**、
同一份语料、同一组超参。脚本：`training/harness/v27_moe_bench.py`。

| 项 | TGAI GO | PyTorch | 比值 |
|---|---|---|---|
| seq=2048 单步 | **0.035 s** | 0.096 s | **0.365×（快 2.7 倍）** |
| seq=8192 单步 | **0.141 s** | 0.189 s | **0.75×（快约 1.3 倍）** |
| seq=2048 分配器峰值显存 | **3.386 GiB** | 2.966 GiB | **1.14×** |
| seq=8192 分配器峰值显存 | **6.650 GiB** | 7.101 GiB | **0.94×（更省）** |
| seq=2048 吞吐 | 56 238 tokens/s | 约 21 300 tokens/s | 2.6× |
| seq=8192 吞吐 | 47 941 tokens/s | 约 43 300 tokens/s | 1.1× |

复现：

```bash
# 我们
PYTHONPATH=python python3 training/harness/v27_moe_bench.py --backend tgai --layers 1 --steps 3 --seq 2048
PYTHONPATH=python python3 training/harness/v27_moe_bench.py --backend tgai --layers 1 --steps 3 --seq 8192
# 参照
PYTHONPATH=python python3 training/harness/v27_moe_bench.py --backend torch --layers 1 --steps 3 --seq 2048
```

> 从 97× 落后打到反超的过程与每次归因都在 `training/ACCEPTANCE_TRAIN_CUDA.md`。

---

## 2. 依赖：只装 numpy，不装 PyTorch

实测（干净 venv，`pip install "tgai-go[cuda]"` 之后 `pip list`）：

| 环境 | 依赖包 | venv 体积 |
|---|---|---|
| **TGAI GO（CPU 训练）** | `numpy`、`tgai-go` | **101 MB** |
| **TGAI GO（CUDA 训练）** | `numpy`、`nvidia-cublas-cu12`、`nvidia-cuda-nvrtc-cu12`、`tgai-go` | **1.2 GB**（含 nvidia 库） |
| PyTorch（仅 torch 本体） | — | **1.5 GB** |
| PyTorch（+ CUDA 运行库） | — | 再多 **2.8 GB**（`nvidia/*`） |

wheel 元数据可直接核对：`Requires-Dist` 只有 `numpy>=1.20`，**没有任何 torch 依赖**。

```bash
python -c "import zipfile;z=zipfile.ZipFile('tgai_go-1.4.0-py3-none-any.whl');\
m=z.read('tgai_go-1.4.0.dist-info/METADATA').decode();\
print([l for l in m.splitlines() if l.startswith('Requires-Dist')])"
```

> Windows 上若已装 CUDA Toolkit（如 V100 上的 12.4），连 `[cuda]` extra 都不用装 —— 直接找 `cublas64_12.dll`。

---

## 3. 迁移：只改一行 import

`training/harness/pytorch_conformance.py` 里 13 段标准 PyTorch 写法，**只把 `import torch` 换成
`import tgai_go.train as torch`**，其余一字不改：

| 后端 | 结果 |
|---|---|
| CPU | **13/13 PASS** |
| CUDA | **13/13 PASS** |

覆盖：张量基础、N 维 Linear、Sequential、functional 名称、cross_entropy、no_grad、优化器 step、
state_dict、reshape/permute/transpose、随机种子可复现、rmsnorm+attention、**切片**、**repeat_interleave**。

```bash
PYTHONPATH=python python3 training/harness/pytorch_conformance.py          # CPU
PYTHONPATH=python python3 training/harness/pytorch_conformance.py --cuda   # GPU
```

---

## 4. 同一份模型代码，CPU 与 GPU 都能跑

上面 13 个用例**是同一批脚本**，分别跑在 CPU 与 CUDA 两个后端上，都是 13/13。
不需要 `device=` 之外任何改动，也没有"CPU 一套、GPU 一套"的分叉。

```python
import tgai_go.train as torch
torch.set_device("cuda")     # 或 "cpu"，其余代码完全一样
```

---

## 5. 预设架构（纯 C 编排，进行中）

- **llama**：Python 版与 PyTorch 对拍 **logits 8.664e-06 / 梯度 9.228e-06**（判据 1e-4 / 1e-3）
- **C 核心（llama 前向）**：层循环在 C 内，**一次 forward 只跨 Python↔C 一次**
  （原 Python 路径一次前向约 96 次算子调用），logits 与 Python 版逐位一致。
- 源码：`native/llama_fwd.c` + `python/tgai_go/presets.py`，编译：`bash native/build_llama_fwd.sh`

```bash
PYTHONPATH=python python3 training/harness/llama_parity.py --cuda        # Python 版
PYTHONPATH=python python3 training/harness/llama_parity.py --c-forward   # C 版（只比前向）
```

---

## 6. 已知短板（不美化）

| 项 | 现状 | 影响 |
|---|---|---|
| `.TG` 直出 | **未实现**：训练侧只有 `state_dict`（`save`），没有 `.TG` 写入器 | 训完还不能直接喂给自家推理引擎，需另做导出 |
| 精度 | 只支持 **float32 / int64** | 没有 bf16/fp16 训练，大模型更吃显存 |
| Muon 端到端 | 与 CPU 后端差 **7.0e-07**（AdamW 9.1e-08） | 同输入下 step 逐位一致（0），差值来自 NS5 放大前向差异；见验收档 |
| `torch.save` 格式 | 不支持，用 `state_dict` + numpy | 不能直接读 PyTorch 的 `.pt` |
| 预设覆盖 | llama ✅；GPT-2 / Qwen2 / TGAI V3 / DeepSeek ⏳ | 架构描述符表尚未铺开 |

---

## 7. 怎么向用户一句话解释

> "你想训一个中文 MoE 或 llama，脚本几乎不用改（改一行 import），不用装 PyTorch（只要 numpy），
> 短上下文比 PyTorch 快 2.7 倍；**显存目前还比它多一倍**，我们在补。"

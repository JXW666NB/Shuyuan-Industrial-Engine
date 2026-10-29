# TGAI GO 训练引擎 —— 目标达成报告

> 目标：① 性能追平/超越 PyTorch（A1–A4）② 真纯 C 的预设架构（B1–B7）。
> 本文只写**实测数字**与**可复跑命令**；未达成项单独列在 §4，并说明缺什么。
>
> 记录时间：2026-10-29。远端基准机：RTX 4090 单卡；构建/验证机：本机 + V100。

---

## 1. A 段：性能（全部达成）

| 判据 | 目标 | **实测** | 判定 |
|---|---|---|---|
| **A1** seq=2048 单步 | ≤ 0.90× PyTorch | **0.036 s vs 0.096 s = 0.375×**（快 2.7 倍） | ✅ |
| **A2** seq=8192 单步 | ≤ 1.00× PyTorch | **0.144 s vs 0.189 s = 0.762×**（快 1.3 倍） | ✅ |
| **A3** 峰值显存 seq=2048 | ≤ 1.2× | **3.418 GiB vs 2.966 GiB = 1.152×** | ✅ |
| **A3** 峰值显存 seq=8192 | ≤ 1.2× | **6.775 GiB vs 7.101 GiB = 0.954×（比 PyTorch 更省）** | ✅ |

> 注：上表是 `nn.Linear` 默认 `bias` 改为 PyTorch 一致的 `True` **之后**的复测值，
> 比改之前（0.035 s / 0.141 s / 3.386 / 6.650 GiB）慢约 2%、多约 30-120 MiB —— 多出的 bias 参数与加法。
> 四条判据仍全部满足。

口径：我们取分配器高水位（`CudaRuntime.pool_stats()['peak_bytes']`，脚本 `a3_mem_parity.py`）；
PyTorch 取 `torch.cuda.max_memory_allocated()`。**不用 nvidia-smi 采样**（实测同一份代码能读出 2.90 和 6.17 GiB）。

关键手段：C 侧图执行器（一步前向 ≈1 次 Python↔C 往返）、显存池、lazy-transpose、低显存 attention。

**A3 的三处改法**（收益 9.811 → 6.650 GiB）：
1. 大词表 `ce_bwd` 原地写进 logits 缓冲；
2. 连续 `reshape` 只改形状元数据、不 memcpy；
3. MoE 的 silu 不再配全 1 向量、也不再算用不到的 `db`；分组反向用完立刻还池。

### A4 证据包（四项全部达成）

| 项 | 判据 | **实测** |
|---|---|---|
| ① 依赖树 | 安装体积、是否含 torch | CPU venv **101 MB**（只有 `numpy` + `tgai-go`）；CUDA venv 1.2 GB（+2 个 nvidia 库）。wheel `Requires-Dist` **只有 numpy**，**不含 torch**。对照：torch 1.5 GB + nvidia 2.9 GB |
| ② 只改 import 迁移 | 同一份脚本 | **13/13 PASS**（CPU）、**13/13 PASS**（CUDA），覆盖 slice / repeat_interleave / CE / SDPA / 优化器步进等 |
| ③ 训完直出 `.TG` 被自家引擎加载 | 加载 + 生成 | 导出字节确定（sha256 两次一致）、`load` 5/5、**GPU 引擎生成 5/5、退出码 0**（随机权重与全零权重各 5/5） |
| ④ CPU/GPU 同一份模型代码 | 同一个模型类 | `llama/gpt2/qwen2/V3` 的参考实现在 CPU 与 CUDA 上是**同一份 Python**；判分台 `llama_parity --cpu` 与 `--cuda` 共用同一套代码路径 |

---

## 2. B 段：纯 C 预设架构

### 2.1 一份 C 核心 + 一张描述符表（复用性硬要求）

`native/llama_fwd.c` 由结构体 `TgArch` 驱动，**三套架构共用同一份 C**，差异只在字段：

| 字段 | 取值 | 影响的架构 |
|---|---|---|
| `qkv_bias` | 0/1 | qwen2=1 |
| `norm` | rmsnorm / **ln** | GPT-2 用 LayerNorm（带 beta） |
| `pos` | rope / **wpe** | GPT-2 用学习式位置 |
| `ffn` | swiglu / **gelu** | GPT-2 用 GELU(tanh) |
| `n_kv` | 整数 | GQA 比例 |

LayerNorm 与 GELU 作为**共享原语**实现（用既有的 `sum_axis8`/`ewise_pow`/`ewise_exp`/`broadcast_to8` 拼装），
**没有为任何架构复制一份算子**。加一套 llama 系架构 = 填字段，不改 C 逻辑。

### 2.2 逐架构数值判据

| 架构 | Python 前向 | **C 前向** | 反向梯度 | 判定 |
|---|---|---|---|---|
| **B1 llama** | CPU 4.261e-07；CUDA **8.664e-06** | **8.664e-06** | 9.228e-06（CUDA） | ✅ |
| **B2 GPT-2** | CPU 1.905e-07；CUDA **1.857e-07** | **2.698e-07** | 4.321e-07 | ✅（除"真实权重"项，见 §4） |
| **B3 Qwen2** | CPU 4.645e-07；CUDA **4.403e-06** | **4.403e-06** | 8.382e-06 | ✅ |
| **B4 V3（非 CSA）** | 前向 **4.793e-05**（对官方 `model_v3.py`） | 未做（见 §4） | **1.267e-04** | 前向/反向 ✅ |
| **B4 V3 1/8 规模** | 相对误差 **1.062e-05** | — | — | ✅ |

判据：前向 logits < 1e-4、反向梯度 < 1e-3。每一条判分台都先自证（同侧两次必须**恰好 0**）：
`gpt2 --selftest` 0、`v3 --selftest` 0、`v3 反向 --selftest` 0、`c_arch --selftest` 三架构 0。

### 2.3 B4 的"1/8 规模 + 参数对齐"

| 配置 | 参数量 |
|---|---|
| `save_1_5b_exp` 全量 | **1.5295 B**（官方模型数出；与文档 1.529 B 一致） |
| 1/8 配置（V=96000 d=512 L=8 h=8 ff=832 n_kv=2 E=12 act=2 +1共享，CSA 关） | **0.1874 B**（比值 **0.979**） |

同进程、同权重、同输入，官方 vs 我们：logits 相对误差 **1.062e-05**（判据 <1e-4），前向 1.0 s（CPU numpy）。

### 2.4 B6 一键流水线

```python
from tgai_go.train.pipeline import run
run(corpus_dir="我的语料/", out_dir="run/", arch="llama", steps=200)
```

实测：自训词表 **132** token / **74** merges（**用户不提供分词表**）；loss **5.3091 → 0.1755**；
产出 `tokenizer.json`/`ckpt.npz`/`meta.json`；**重载后 logits 逐位一致**。

### 2.5 B7 打包

wheel `tgai_go-1.4.1`：**C 源码随包**（`python/tgai_go/_native/src/llama_fwd.c` 41 KB）、README 增「预设架构 + 纯 C」章节。
真实安装态验证：`__version__`=1.4.1 与 metadata 一致、BPE 取自包内 `tgai_go.train.bpe`、`save_tg` 可导入。

---

## 3. 复跑命令（全部可复现）

```bash
# ── 判分台自证与主判据（4090，CPU/GPU 都行）──
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/pytorch_conformance.py          # 13/13
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/pytorch_conformance.py --cuda   # 13/13
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/llama_parity.py --cuda          # 8.664e-06
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/llama_parity.py --arch qwen2 --cuda  # 4.403e-06
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/gpt2_parity.py --cuda           # 1.857e-07
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/c_arch_parity.py --compare      # 三架构全 PASS
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/v3_parity.py --compare          # 4.793e-05
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/v3_grad_parity.py --compare     # 1.267e-04
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/pipeline_smoke.py --check       # loss 5.31->0.18

# ── 性能与显存 ──
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/v27_moe_bench.py --backend tgai --layers 1 --steps 3 --seq 2048
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/v27_moe_bench.py --backend tgai --layers 1 --steps 3 --seq 8192
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/a3_mem_parity.py --backend tgai --seq 8192

# ── C 引擎构建（本机，MSVC+nvcc 都在）──
cd "D:\tgai_nlp\TGAI GO" && build_cuda_v100.bat
```

---

## 4. **未达成项**（不掩饰）

| 项 | 状态 | 缺什么 | 判断 |
|---|---|---|---|
| **B1/B3 的 C 侧反向** | ❌ 未做 | C 核心目前只有前向图；反向仍在 Python（CUDA 后端） | 目标原文 B1 只要求"与 PyTorch 对拍前向+反向"，已在 Python 侧满足；C 侧反向是加分项，工作量大（要在 C 里搭反向图） |
| **B2 加载真实 GPT-2 权重** | ❌ 未做 | 需要下载官方 GPT-2 权重 + 其 BPE（网络/磁盘），并把 `wte` 与 `lm_head` 的 tie 关系对齐 | 结构已支持（`tie=True` 可切），只差权重文件 |
| **B4 的 CSA** | ❌ 未做 | 压缩稀疏注意力是独立大模块：分段池化 + 块对角掩码 + 局部窗口 + 压缩后的卷积 | 我判断在剩余轮次内做不完，硬塞会得到半成品 |
| **B4 的 C 侧** | ❌ 未做 | V3 的 AttnRes（跨层状态列表）+ SiTU-GLU MoE 需要扩 `TgArch`，不是"加字段"能解决 | 这是描述符抽象的真压力点，工作量 vs 剩余轮次不划算 |
| **B5 DeepSeek MLA** | ❌ 未开始 | MLA 的隐空间压缩注意力 + 共享专家路由 | 目标本身写"最后做，单独立子目标" |
| **CPU 版引擎的退出堆损坏** | ⚠️ 已定性未修 | `_native/libtgai_engine.dll` 在**进程退出/卸载**时损坏堆（`0xC0000374`/`0xC0000005`）；**工作本身正确**（load/encode/generate 都成功） | 需要 ASan/调试器定位；产线用 GPU 版，不受影响 |

---

## 5. 三条贯穿全局的教训（写给未来的自己）

1. **判据本身必须先被验证。** 本期四次被自己的判据骗：
   `退出码==0`（引擎退出阶段堆损坏，成功也非 0）、`nvidia-smi` 采样（同码读出 2.90/6.17 GiB）、
   `stdout 找 GEN`（崩溃早于刷新，把成功记成失败）、`--c-forward` 写死 `presets.llama`（qwen2 假失败 4.14e-01）。
2. **"源码树里能跑"不等于"装到用户机器能跑"。** 本期四例：`packages` 漏 `tgai_go.train`、
   `pipeline` 按路径读 `training/bpe_train.py`、全局 `*.exclude` 把 `*.c` 剔掉、`__version__` 与 pyproject 不一致。
   **每个交付都必须在干净 venv 里过一遍。**
3. **能重编才能修。** 之前"引擎有 bug 修不了"是因为以为没有编译器 —— 实际本机 MSVC 2022 + nvcc 12.4 都在
   （`cl.exe` 只在 Developer Command Prompt 的 PATH 里，`where cl` 查不到不代表没装）。
   打通构建后，零改动重编 → V100 验证 → 才敢谈修改。

---

## 6. 「能不能替代 PyTorch」——过拟合对照（新增，硬判据）

**方法**：同一份训练循环（`overfit_parity.py::train_once`）分别用 `tgai_go.train`(CUDA) 与真 `torch` 跑；
权重用 numpy 固定种子初始化后**灌进两边**（必须逐位相同），固定同一批 32 个样本做**过拟合**。

| 项 | TGAI CUDA | PyTorch |
|---|---|---|
| loss 首 → 末 | 4.161366 → **0.008179** | 4.161367 → 0.008179 |
| 过拟合（<0.05） | ✅ | ✅ |
| **400 步逐 step 最大相对差** | **5.684e-05** | — |

短程 20 步时该差值为 **2.322e-07**（fp32 舍入量级）。

**结论**：在本库覆盖的算子范围内，TGAI GO 的 CUDA 后端**可以替代 PyTorch 做训练**，
且同一份代码只需改导入。

### 迁移/自定义训练的实测（用户视角）

| 示例 | 有效代码 | 结果 |
|---|---|---|
| `examples/torch_style_overfit.py`（PyTorch 风格循环，只改导入） | 46 行 | CUDA：loss 4.8449 → **0.004463**（降 99.9%） |
| `examples/train_preset.py`（预设架构训练） | **15 行** | llama 5.1666→0.0500 / qwen2 4.9137→0.0502 / gpt2 4.8231→0.0443，三者重载逐位一致 |

### 两个"改导入就迁移"的真实障碍（已修）
1. `nn.Linear` 默认 `bias=False`（PyTorch 为 `True`）→ 已改为 `True`；
2. **CUDA 下 `p.data = ...` 赋初值无效**（值在 `_gpu`），必须用 `load_state_dict()`。

---

## 7. B5（DeepSeek MLA）——**第一步达成**，非整项

| 后端 | MLA logits 相对误差 |
|---|---|
| CPU | **1.944e-07** |
| CUDA | **3.092e-07** |

范围：隐空间压缩（`c_kv`）、`k_nope/k_rope/v` 上投影、**`k_rope` 全头共享**、Q 可选低秩、输出投影。
**未实现**：RoPE 本身、完整 DeepSeek MoE 路由、吸收矩阵优化。
顺带补上 CUDA `T.cat`（用户自写模型也会用到）。

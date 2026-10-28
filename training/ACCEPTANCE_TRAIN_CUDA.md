# tgai_go.train · CUDA 后端验收档

> 目标：给 `tgai_go.train` 补 GPU 后端，**不依赖 PyTorch**（方案 A：自研显存/张量层）。
> 判分纪律：判分台先自证（CPU vs CPU 必须恰好为 0），才允许用于考核。
> 环境：V100-32GB / Windows / 驱动 580.88；产线 11450 共驻，未改动。

---

## M1 · 显存底座 + matmul + cross_entropy ✅

**判据**：与 numpy 后端同 seed 跑 Linear + cross_entropy 一个 step，loss / W / dW 相对误差 < 1e-6。

**判分台**：`training/harness/train_backend_parity.py`（`--selftest` 自证，`--cuda` 考核）

**实测（2026-10-28，我在 V100 上独立复跑，用仓库版本而非 Grok 工作区）**：

| 项 | 结果 |
|---|---|
| `--selftest`：CPU vs CPU | loss **0.000e+00** / W **0.000e+00** / dW **0.000e+00** → ok=True |
| `--cuda`：CPU vs CUDA | loss CPU=1.65572155 CUDA=1.65572155 **rel=0.000e+00**；W **rel=0.000e+00**；dW **rel=6.181e-08** → **ok=True** |
| 是否 import torch | **否**（实测 `sys.modules` 里没有 torch） |
| `test_train_api.py` A–H | **全 PASS**，无回归（autograd 6.263e-05 / AdamW 0 / Muon 630→1.85 / MLP 1.1719→0.0054 / 注意力梯度 1.37e-04、1.99e-04 / 因果性 0.00 / 小 LM 100%） |

**实现要点**（Grok）：`set_device("cuda")` 后新建张量走设备指针；2D matmul 用 `cublasSgemm`（pedantic FP32，避免 TF32）；
`cross_entropy` 用内嵌 `tgai_ce.cu` 经 NVRTC 编译。

**只有真卡才暴露的坑（值得记下来）**：
Windows `nvcuda.dll` 同时导出**旧 32 位 ABI** 的无后缀符号 `cuMemAlloc` / `cuMemFree` / `cuMemcpyHtoD` /
`cuMemcpyDtoH` / `cuMemcpyDtoD`。在 primary context 上调用它们会返回 **rc=201 invalid device context** ——
而 context 当时其实是 current 的（`cuCtxGetCurrent` 与 `cuDevicePrimaryCtxRetain` 返回同一句柄）。
`cuda.h` 里的 `#define cuMemAlloc cuMemAlloc_v2` 对 ctypes **不生效**。
修法：这五个内存入口显式绑定 `_v2`，缺 `_v2` 直接报错，**不要用 try/except 吞掉 201**。

**仍在 CPU 的**：逐元素运算、softmax/silu、Embedding、RMSNorm、attention、3D matmul、AdamW/Muon，
以及带 `ignore_index` 的 cross_entropy（logits D2H、dlogits H2D）—— 这些属于 M2/M3/M4。

---

## M2 · 逐算子 GPU 化（待做）

判据：rmsnorm / silu_mul / moe_add / emb_gather / rope / cross_entropy 每个算子对 CPU 后端 < 1e-5。

---

## M3 · attention 子层接 CUDA ✅

**判据**：三层 attention 堆叠的小模型，loss 与全部 17 个参数梯度对 CPU 后端 < 1e-4。

**判分台**：`training/harness/train_layer_parity.py`（**已自证**：CPU vs CPU 17 项全 0）

模型：`x[B=2,S=16,D=32] → Linear(32,32) → CausalSelfAttention(32,nh=4)` ×3 `→ Linear(32,V=8) → CE`

**实测（2026-10-28，我在 4090 上用仓库版本独立复跑）**：

| 项 | 结果 |
|---|---|
| loss 相对误差 | **0.000e+00** |
| 最差参数项 | `attn2.q_proj.weight` **1.047e-06** |
| 判定 | **ok=True**（判据 1e-4） |
| 三个 `--selftest` | 仍全 0 |
| M1 / M2 `--cuda` | 无回归（仍 ok=True） |
| `test_train_api.py` A–H | 全 PASS |
| import torch | **否** |

### 一个重要的布局发现（影响 M5 的选择）

CPU 后端的 head 拆分是 `reshape(B,S,H,Dh).reshape(B*H,S,Dh)`，即 **C 顺序重解释**，
而仓库里的 attention kernel（flash / tiled / nop）要求的是**显式 `[seq, nh, dh]` 布局**，且 dh=64。
两者**不可直接互换**（Dh=8 时 kernel 还会直接 return）。

因此 M3 的 attention **没有调用**我们的 flash kernel，而是用与 CPU 同构的分解：
已验收的 `cublasSgemm_v2` 组出 bmm（batch = B*H = 8）+ `softmax_lastdim`；
反向用同一套 bmm 加 `y*(dy - sum(dy*y))`。

**这不算退步**（数值对上、且全部在 GPU），但要记住：**想用 flash 前向，必须先做布局适配**
（把 C 顺序的 head 拆分换成显式 permute）。这条留给 M5——真实模型 dh=64，那时 flash 才有意义。

---

## M5 · PyTorch 迁移一致性（CPU 11/11，CUDA 11/11 ✅）

**用户硬指标**：把 `import torch` 换成 `import tgai_go.train as torch`，**其余代码一字不改**必须跑通。

**判分台**：`training/harness/pytorch_conformance.py` —— 11 段**标准 PyTorch 写法**的用例：
`tensor_basic / linear_nd / sequential / functional_names / cross_entropy / no_grad /
optimizer_step / state_dict / reshape_permute / seed_repeat / rmsnorm_attn`

| 后端 | 结果 |
|---|---|
| **CPU**（本机 Windows 与 V100 各跑一次） | **11/11 PASS** |
| **CUDA**（V100 独立复跑） | **11/11 PASS** ✅ |

### CUDA 侧未通过的 6 项与真实原因（这是下一步的工作清单）

| 用例 | 失败原因 |
|---|---|
| `tensor_basic` | CUDA 标量广播坏了：`ewise_add 形状不同: (2,2) vs (1,)` |
| `linear_nd` | CUDA matmul 只支持 2D/3D，收到 `3D x 2D`（CPU 侧已修，CUDA 未修） |
| `functional_names` | `CudaRuntime` 缺 `silu_mul_fwd` |
| `optimizer_step` | `float * NoneType`（某参数梯度为 None） |
| `reshape_permute` | CUDA 后端**还没有 `permute`** |
| `rmsnorm_attn` | `CudaRuntime` 缺 `rmsnorm_fwd` |

### 已入库的 API 表面（CPU 后端）

补上了 32 个缺失的顶层名字（38 个里原有 6 个）：
`tensor / zeros / ones / full / randn / rand / arange / cat / stack / manual_seed / from_numpy /
float32 / int64 / save / load / topk / where / max / min / exp / log / sqrt / abs / clamp /
matmul / bmm / einsum / permute` 等，并把 `nn.Linear` 从"只吃 2D"改成支持任意前导维。

**Grok 主动标注的"近似而非等价"（诚实项，必须保留）**：
- 新算子是 CPU / float32 路径，**CUDA 张量上会 `NotImplementedError`**（正是 CUDA 5/11 的来源）
- `manual_seed` 只保证**我们自己的**两次 `randn` 一致，**不保证与 PyTorch 逐位相同**
- `save` / `load` **不是** `torch.save` 的 zip 格式
- 比较运算 `>` `<` **未实现**


### M5-CUDA 六项修复（2026-10-28 晚，V100 实测 11/11）

| 用例 | 根因 | 修法 |
|---|---|---|
| `tensor_basic` | CUDA 侧没有标量/广播加法 | GPU 上做广播加，反向按轴求和（也在设备上） |
| `linear_nd` | CUDA matmul 只吃 2D | 最后两维 matmul、前导维当 batch，仍走 `cublasSgemm`，反向 unbroadcast |
| `functional_names` | 缺 `silu_mul_fwd`、缺最后一维 `log_softmax` | 接上已验收的 silu_mul 内核，补 log_softmax |
| `optimizer_step` | **不是缺梯度**：`nn.Linear` 默认无 bias，`weight.grad` 正常；崩在 weight decay 读 `p.data`，而 CUDA 张量的值在 `_gpu`、`data` 是 `None` | AdamW 的更新改在 GPU 上做；`grad is None` 仍按 PyTorch 语义跳过 |
| `reshape_permute` | CUDA 后端没有 permute | permute 前向 + 逆置换反向都在 GPU；3D transpose 走 permute |
| `rmsnorm_attn` | 缺 `rmsnorm_fwd` | 接上已验收的 rmsnorm 内核，**eps=1e-6** |

**回归**（同一轮 V100 复跑）：CPU 一致性 11/11；四个 `--selftest` 全 0；A–H 全 PASS；
M1 `--cuda` ok=True；M2 四项最差 1.67e-07；M3 最差 7.997e-07。

> **口径说明（必须保留）**：11/11 只证明**这 11 种 PyTorch 写法**可以零改动迁移，
> **不等于**"任意 PyTorch 代码都能跑"。反例仍然存在（比较运算、`torch.save` zip 格式、
> dtype 只支持 float32/int64、`manual_seed` 不保证与 PyTorch 逐位一致）。
> 对外表述应为"常用写法可直接迁移，边界见文档"，而不是"完全兼容"。

### 我在判分台里发现自己一个错

`tensor_basic` 原本断言 `sum == 22`，但 `x*2+1` 四个元素是 `3,5,7,9`，和应为 **24** ——
**是我的判分台写错了**，Grok 指出并改正。判分台先自证这条纪律，反过来也保护了实现方不被冤枉。

---

## M4 · 优化器上 GPU（AdamW ✅ / Muon 未达标）

判分台 `train_opt_parity.py`（自证：CPU vs CPU 6 项全 0）。`Linear(8,8)→silu→Linear(8,4)→CE`，
每个优化器 3 步，判据「更新后参数相对误差 < 1e-7」。

**实测（48GB 4090 单卡实例，仓库版本）**：

| 优化器 | worst | 明细 | 判定 |
|---|---|---|---|
| **AdamW** | **9.055e-08** | loss 9.055e-08 / lin1 6.419e-08 / lin2 8.511e-08 | ✅ **PASS** |
| **Muon** | **6.995e-07** | loss **0** / lin1 7.307e-08 / lin2 6.995e-07 | ❌ 超判据 7 倍 |

### Muon 为什么卡在 7e-07（我的判断，附证据）

Muon 的核心是 `_newton_schulz5`：**5 轮** float32 多项式迭代，每轮都做 `X@X.T`、`(X@X.T)²`、`Y@X`。
即使 GPU 侧刻意绕开 cuBLAS、改用与 numpy 同序的 FMA kernel（Grok 已这么做，注释里写明理由），
numpy/OpenBLAS 的阻塞与 SIMD 累加顺序仍无法逐位复现 —— 单次 matmul 约 1e-8 的差，5 轮迭代后被放大到 ~7e-07。

**结论：这是「跨 BLAS 逐位一致」的固有困难，不是实现遗漏。**
AdamW（纯逐元素、无矩阵乘）能到 9e-08，正好印证这个解释。

### 待决策

- **方案 A（建议）**：Muon 的判据放宽到 **< 1e-6**，并在文档注明原因；
  AdamW 仍保持 < 1e-7。训练默认用 AdamW，Muon 属可选优化器。
- **方案 B**：继续追 < 1e-7（需要让 numpy 侧也用 GPU 的累加顺序，或两边都改 float64 —— 等于改算法）。

> 这条我**没有**擅自放宽：判据是目标里写的，改成 1e-6 需要你确认。

---

## MoE 模块（nn.MoE）✅ —— 为 V2.7 架构铺路

**动机**：V2.7 是 E=8 top-2 的 MoE，而 `tgai_go.train` 原先只有 Dense FFN，无法表达真 V2.7 结构。

**实现**：`nn.MoE(d_model, d_ff, n_experts, top_k, gate)`，参数命名与 .TG 对齐
（`router.weight [E,d]`、`experts.{e}.{gate,up,down}.weight`）。
`gate="softmax"` 对全部 E 个 logits 做 softmax 再取 top-k；`gate="sigmoid"` 为 `sigmoid(logit)`，
`top_k=1` 时与已验收的 `training/kernels/tgai_moeblk.cu` 语义一致（argmax，平局取小索引）。

**实测（48GB 4090，我独立复跑 `training/harness/moe_selfcheck.py`）**：

| 用例 | CPU vs CUDA | 说明 |
|---|---|---|
| `sigmoid-k1-tie-empty` | **1.370e-07** | 含空专家 `[1,2]`；未选中专家 router 梯度 **0.000e+00** |
| `softmax-k2-3d-empty` | **1.331e-07** | 3D 输入 |
| `sigmoid-k2-random` | **1.393e-07** | 未选中专家 router 梯度 **0.000e+00** |
| `softmax-k1-single` | **1.716e-07** | 3 个空专家 `[1,2,3]` |
| `tie->smaller` | **6.193e-08** | 精确平局取小索引（与解析参考比） |
| 有限差分 vs 解析 | PASS | dW / dx 均吻合到 1e-6 |

**VERDICT ok=True**（判据 < 1e-5）。

> **判分台自身修了两处**（记录在案，避免再犯）：
> 1. `grad` 在 CPU 后端是 ndarray，脚本却无条件 `.to_numpy()` → 崩。
> 2. 原 `tie` 测试用「把 router 某行设成 -20」来"强制"专家胜出 —— 那是**权重行**不是 bias，
>    logit = x·(-20) 对随机 x 接近 0，根本压不住，导致假失败。
>    改成「两行权重完全相同 + 第三行取负 + x 首维固定」后 PASS。
>    **判分台先自证这条纪律，又一次保护了实现方。**

**回归**：一致性 CPU/CUDA 均 11/11；四 selftest 全 0；单步性能未退化（0.15 s，13728 tokens/s）。

**仍未覆盖**：MoE 在 V2.7 规模（E=8/ff=4096/seq=8192）的端到端训练与吞吐 —— 下一轮做。

---

## M6/M7 · V2.7 形状 MoE 端到端对比（进行中，已暴露一个硬边界）

**配置**：d=1024 / nh=16 / ff=4096 / **E=8 top-2** / V=32779 / 1 层；同一份数据、同超参、同 init 尺度。
判分台：`training/harness/v27_moe_bench.py`（两边结构一致，参数量必须相等才算对齐）。

### 实测（48GB 4090 单卡）

| 项 | PyTorch | tgai_go.train | 比值 |
|---|---|---|---|
| 参数量 | **138.4 M** | **138.4 M** | 1.00（结构已对齐） |
| 单步（seq=2048） | 0.123 s | 0.218 s | **1.75×** |
| 吞吐（seq=2048） | 16672 tokens/s | 9553 tokens/s | 0.57× |
| **峰值显存（seq=2048）** | **2.97 GB** | **10.11 GiB** | **3.4×** |
| loss 曲线（seq=2048） | 9.93 → 7.71 → 8.32 | 10.51 → 6.75 → 5.48 | 都正常下降 |

### ★ 长上下文硬边界（seq=8192，MoE，1 层）

| | PyTorch | tgai_go.train |
|---|---|---|
| 结果 | ✅ 0.306 s/step，**峰值 6.06 GB** | ❌ **OOM**（采样峰值 46.17 GiB 后 `cuMemAlloc rc=2`） |

**根因（两条，按影响排序）**：
1. **attention 前向物化 P**：seq=8192、16 头时 P 矩阵 = 16 × 8192² × 4B ≈ **4.3 GB**，
   而且 scores / mask / softmax 中间量各有一份，反向还要留着 → 单层十几 GB。
   PyTorch 的 SDPA 走 flash 式实现，**完全不物化 P**（所以只要 6 GB）。
2. **算子级无内存复用**：每个算子新分配、autograd 保留全部中间量（seq=2048 时的 3.4× 就是这么来的）。

**现成的解法（已验证过，只是还没接进来）**：
`training/kernels/tgai_attn_nop.cu` 的 `attn_fwd_nop`（只写 O）+ `attn_bwd_recompute`（内部重算 P）。
之前实测：前向 **0.86×**、反向 **2.00×**，但**不物化 P** → 内存从 4.3 GB 降到接近 0。
**长上下文场景下这个交换是划算的**（用时间换显存），应当做成"长上下文模式"。

### 结论（当前）
- **短上下文（seq≤2048）**：MoE 可训，速度 1.75× 慢，显存 3.4× —— 可用但不划算。
- **长上下文（seq=8192）**：**当前不可用**，必须先接 no-P attention 路径。

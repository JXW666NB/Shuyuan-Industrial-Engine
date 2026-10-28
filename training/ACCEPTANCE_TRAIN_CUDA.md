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

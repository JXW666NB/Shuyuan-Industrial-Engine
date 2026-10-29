# 整层融合（训练图搬进 C）设计规格

> 目标：10M 对话模型（B=78, S=64, D=256, NH=8, FF=1280）4090 ≤ 20 ms/step、V100 ≤ 45 ms/step，
> 即相对 PyTorch（31.55 / 67.06 ms）至少 **1.5×** 快。
> 当前：4090 33.0 ms、V100 68.3 ms（已追平 PyTorch）。

## 1. 现状剖析（4090，33.0 ms/step）

| 段 | 耗时 | 备注 |
|---|---|---|
| forward | ~13 ms | 8 层，每层 2×RMSNorm + QKV/OUT 4 GEMM + RoPE + mask-softmax + GLU + 3 GEMM |
| backward | ~15 ms | elementwise 反向占大头（dRMSNorm 5 算子、dGLU 4 算子、dSoftmax） |
| CE | ~2.1 ms | 已是 GPU 选择矩阵 |
| optimizer | ~2.1 ms | AdamW 74 个参数张量 |

**结论**：GEMM 已经不是瓶颈（4.9 ms），剩下的是**每层 ~30 个 elementwise kernel + 它的反向**。
融合的目标就是把这 30 个 kernel 压成 fwd 3 个、bwd 4 个。

## 2. 要融合的算子（每层）

### 前向
| # | 现在 | 融合后 |
|---|---|---|
| 1 | `rms(x, ln1)` = mul, mean, add, pow, mul, mul（6 kernel） | `fused_rmsnorm_fwd(x, w, eps) -> (y, rstd)`（已有内核，返回 rstd 备用） |
| 2 | q/k/v = 3 GEMM + 6 个 permute/reshape | 3 次 cuBLAS（保持），布局直接写 [NH,S,DH]（省 permute） |
| 3 | `rope(q)`、`rope(k)` = 4 mul + 2 bmm + 2 add | 1 个 `fused_rope_fwd(z, cos, sin, S, NH, DH)` 原地 |
| 4 | `sdpa` = mask mul + sub + exp + sum + div + bmm（6 kernel） | `fused_attn_fwd(q, k, v) -> (o, lse)`（causal，在线 softmax） |
| 5 | `o @ Wo` + residual add | 1 GEMM + `fused_add_inplace` |
| 6 | `rms(ln2)` | 同 1 |
| 7 | `glu(gate, up)` | 已有 `silu_mul_fwd` |
| 8 | `@ Wd` + residual add | 1 GEMM + `fused_add_inplace` |

前向 kernel 数：**6+2 +2 +6 +2 +6 +1 +2 ≈ 27 → 8**（含 6 次 GEMM）。

### 反向（按层从后往前）
| # | 融合后 |
|---|---|
| 1 | `fused_add_bwd`（residual 分叉） |
| 2 | `fused_glu_bwd(g, gate, up) -> (dgate, dup)`（已有 `silu_mul_bwd`） |
| 3 | `fused_rmsnorm_bwd(dy, x, w, rstd) -> (dx, dw)`（已有内核） |
| 4 | `fused_attn_bwd(g, q, k, v, o, lse) -> (dq, dk, dv)`（在线 softmax 反向，1 个 kernel） |
| 5 | `fused_rope_bwd(g, cos, sin)`（1 个 kernel，原地） |
| 6 | GEMM 反向（cuBLAS） |

反向 kernel 数：**~30 → 9**。

## 3. 显存预算（B=78,S=64,D=256,FF=1280,NH=8,L=8）

| 需要保存的量 | 大小 |
|---|---|
| 每层 x_in、ln1 输出 y1、rstd1、q/k/v、attn 输出 o、lse、ln2 输出 y2、rstd2、gate、up | ≈ 78×64×(256×6 + 1280×2) × 4B ≈ **2.0 GB** |

⚠️ 这是最大的风险点：全量保存 8 层的中间激活要 ~2 GB（V100 16 GB 可承受，但要和 PyTorch 对照组同机跑就紧张）。
**对策**：先做「保存 + 融合」版本（简单、快），如果显存不够再上**分块重算**（recompute RMSNorm/GLU 的输出，只存 rstd 与输入）。

## 4. 代码落点

```
cuda/tgai_cuda.cu          # 新增 fused 内核（fwd/bwd）
python/tgai_go/train/cuda.py   # 新增 launch 封装
python/tgai_go/train/__init__.py # 新增 fused_block_fwd/bwd + autograd 接线
examples/exam2_chat10m_tgai.py   # 增加 --fused 开关（默认开，可回退纯 Python 路径做对拍）
```

**关键设计**：`examples` 里保留**纯 Python 参考路径**，用 `--fused 0/1` 切换，
这样对拍（数值）和回退（出问题）都是一行参数。

## 5. 验收流程（每一步都跑）

1. **单测对拍**：同一份随机权重，`--fused 0` 与 `--fused 1` 跑 1 步，
   比较 loss 与**每个参数的梯度**，相对误差 ≤ 1e-5。
2. **整步计时**：`profile2.py`（带 `rt.sync()`），记录 fwd/bwd/ce/opt 四段。
3. **训练曲线**：600 步 loss 曲线与旧路径对比（±2%）。
4. **回归**：`run_all.sh` 全绿。
5. **考试交付**：`chat10m.TG` 引擎三问 + `pelican.TG` 不回归。

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 显存不够（2 GB 激活） | 先测；不够就只存 rstd + 输入，重算 GLU/norm 输出 |
| 反向公式写错 | 先只融合**前向**（反向仍走 Python），单独验证前向数值；再融合反向 |
| causal mask 边界错 | 用 S=1..8 的小例子逐元素对拍 |
| kernel 写错导致 NaN | 每次改动加 `assert not nan` 的冒烟测试 |
| 4090 明天到期 | **优先在 4090 上把 fwd 融合跑通并测速**，反向可先用 Python |

## 7. 分阶段计划（每阶段都可独立交付）

| 阶段 | 内容 | 预期 |
|---|---|---|
| **P1** | 融合前向（RoPE + causal attention + add），反向仍 Python | 33.0 → ~26 ms（1.2× 快于 PyTorch） |
| **P2** | 融合反向（dRMSNorm + dGLU + dAttn） | ~20 ms（1.5×） |
| **P3** | 整层一个 C 函数（Python 每层 1 次调用） | ~17 ms（1.85×） |
| **P4** | 定位报告《为什么用 TGAI GO》 | 交付 |

---

## ★ 实现路径已确认（第 1 轮，2026-10-30）

`cuda.py` 里的 CUDA 内核是**源码字符串 + NVRTC 运行时编译**：

- 文件头注释写明：只通过 ctypes 调 **驱动 / cuBLAS / NVRTC**，不 import torch；
- 内核源码直接嵌在 `cuda.py` 里（例如 `extern "C" __global__ void sgemm_mode(...)` 在 L875、
  `silu_mul_fwd` 在 L445），由 `_fn(name)` 按需从 NVRTC 模块取函数指针；
- **没有单独的 .cu 构建步骤** —— 加一个融合核 = 加一段源码字符串 + 一个 `launch()` 封装。

⇒ P1 不需要改构建系统，可以直接在 `cuda.py` 里加：

```python
# 伪代码：加在现有内核字符串旁边
extern "C" __global__ void fused_rope_fwd(float* z, const float* cos, const float* sin,
                                          int rows, int seq, int nh, int dh) { ... }
```
再写一个 `def fused_rope_fwd(self, z, cos, sin, seq, nh, dh)` 用 `self.launch(...)`，
Python 侧 `Tensor._backward` 照 RMSNorm 的模式接（见 `__init__.py` 的 `RMSNorm._forward_cuda`）。

**因此下一轮的第一步**：
1. 在 `cuda.py` 加 `fused_rope_fwd` 源码 + 封装；
2. 在 `__init__.py` 加 `T.fused_rope(x, cos, sin)`，反向暂时复用现有 `rope` 的 Python 反向；
3. 4090 上对拍（与现有 `rope()` 比 fwd 输出，应 ≤1e-6）+ 测单层耗时；
4. 通过后再做 `fused_attn_fwd`（在线 softmax + 返回 lse），这是 P1 的大头。

---

## ★ 第 2 阶段实测（2026-10-30，V100）：先撞到物理上限

### 又省了一点：嵌入层改融合 gather

示例原来用 `np.eye(V)[ids] @ W`：每步造一个 [B*S, V] 稠密矩阵（10M 模型 7 MB）再做稠密 GEMM。
改用后端融合 `emb_fwd` gather 后：

| | V100 单步 |
|---|---|
| 改前 | 68.3 ms |
| 改后 | **66.3 ms** |

数值：前向 **0.000e+00**、权重梯度 **0.000e+00**（与 one-hot 版逐位相同）。

> 顺带修了一个真 BUG：早前给 `Embedding.forward` 加的"支持 Tensor 索引"补丁，
> 用 `return self.weight[np.asarray(...)]` **短路了 CUDA 的融合 gather 路径**，
> 导致 CUDA 上抛「切片暂不支持索引类型 ndarray」。已恢复成「CUDA 走 `_forward_cuda`、CPU 走 numpy」。

### 物理上限（重要，决定目标是否可达）

10M 模型一步的计算量（6·P·T，P=10.06M，T=B·S=4992）：

```
6 × 10.06e6 × 4992 ≈ 301 GFLOP
```

| 机器 | fp32 峰值（SIMT，非张量核） | 我们的耗时 | 有效算力 | 占峰值 |
|---|---|---|---|---|
| V100-32GB | **7.8 TFLOP/s** | 66.3 ms | 4.54 T | **58%** |
| PyTorch（V100） | 同上 | 67.06 ms | 4.49 T | **58%** |
| 4090 | 82 TFLOP/s | 33.0 ms | 9.1 T | 11% |

**结论**：在 V100 上，我们和 PyTorch 都已经跑到 fp32 峰值的 **58%**，两者基本重合。
要用 45 ms 完成 301 GFLOP，需要 **6.7 TFLOP/s = 峰值 86%** —— 对这批小矩阵（M=4992, K/N=256/1280）
不现实。

⇒ **V100 上"比 PyTorch 快 1.5×"这个判据，靠融合/消除开销是完不成的**；
   要更快只有两条路：
   1. **fp16/bf16 混合精度**（V100 张量核 120 TFLOP/s，理论 15×；实际 2–4×），需要 loss scaling；
   2. 换更大的 batch/序列让 GEMM 更饱满（当前 M=4992 已经不小，收益有限）。

4090 上还有很大空间（11% 占用），但 4090 已到期，无法继续验证。

### 建议

- **V100 目标改为可达值**：≤ 55 ms（约 1.22× 快于 PyTorch），或直接不设"反超"要求；
- 若要真正反超，**下一个工作流应该是混合精度**（fp16 + loss scaling + 数值对拍），而不是继续融合。

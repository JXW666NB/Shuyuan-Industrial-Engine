# 速度优化简报（给 Grok）—— TGAI GO 训练比 PyTorch 慢 19 倍

> 结论先行：**不是 cuBLAS 慢，也不数学模式问题**，是「逐算子 Python 路径」的实现效率问题。
> 目标：10M 对话模型 4090 上 609 ms/step → **< 60 ms/step**（PyTorch 31.55 ms）。

## 1. 实测事实（4090，10.06M 模型，B=78，S=64，V=360）

| 指标 | 数值 |
|---|---|
| 整步 | **605 ms** |
| 其中 forward | 237 ms |
| backward | 357 ms |
| optimizer | 2 ms |
| kernel 启动次数 | **1014 次/步** |
| 启动开销合计 | 17.5 ms（约 17 µs/次，**不是瓶颈**） |
| `__matmul__` 累计（fwd+bwd 中被包到的部分） | 218 ms / 90 次 = **2.42 ms/次** |
| 单次 GEMM 微基准（异步，不可信） | 0.03 ms |
| cuBLAS 数学模式 PEDANTIC ↔ DEFAULT | 606 ↔ 615 ms（**无差别**） |
| 后端 `cudaDeviceSynchronize` 出现次数 | **0** |

**推算**：一步约 150 GFLOP（fwd 50 + bwd 100），605 ms → **0.26 TFLOP/s**；
4090 fp32 峰值约 82 TFLOP/s，PyTorch 实测 31.55 ms → 4.8 TFLOP/s。**我们差 19 倍。**

## 2. 已排除

- ❌ cuBLAS 慢 / 数学模式：切换后无变化
- ❌ kernel 启动开销：1014 次只占 17.5 ms
- ❌ one-hot 嵌入：只多 1.4× 计算量
- ❌ 逐算子显存同步：后端没有 sync 调用

## 3. 最可能的原因（按可能性排序）

1. **逐算子内核未向量化 / 占用率低**
   1014 次启动里约 600 次是 elementwise（`ewise_mul` 298、`ewise_add` 131、
   `broadcast_to8` 117、`sum_axis8` 68）。它们逐个读写 1.28M–6.4M 个 float。
   若每个只跑到 50 GB/s（未用 float4、block 太小、非合并访问），
   单次 25 MB 读写就是 0.5 ms，累计正好是几百 ms 量级。

2. **算子间无融合**（每次都要写回显存再读出来）
   一个 RMSNorm 现在是 `mul → mean → rsqrt → mul → mul` 5 个 kernel；
   SiLU-GLU 是 `exp → add → pow/div → mul → mul` 5 个 kernel。
   L=8 层 → 每层至少 20 个 elementwise kernel。

3. **`broadcast_to8` 117 次**：说明形状广播在反复物化，可能在复制整块张量。

## 4. 建议的优化顺序（每步都要有前后对比数字）

### P0：把 elementwise 内核做成向量化 + grid-stride（预计 2–4×）
```cuda
// 现在大概是 1 元素/线程；改成：
for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n4; i += gridDim.x*blockDim.x)
    ((float4*)dst)[i] = op(((float4*)src)[i]);   // 4 元素/线程
```
验收：单 kernel 吞吐 > 400 GB/s（用 1.28M float 测量）。

### P1：融合三个热点（预计 1.5–2×）
| 融合后 kernel | 替代 | 省下的显存往返 |
|---|---|---|
| `rmsnorm_fwd`（含 mean/rstd/scale） | 5 个 kernel | 4 趟 |
| `silu_glu_fwd`（`silu(a)*b`） | 5 个 kernel | 4 趟 |
| `residual_add`（原地 `x += y`） | 2 个 kernel | 1 趟 |

### P2：消除 `broadcast_to8` 的物化（预计 10–20%）
把广播改成内核里的下标计算，不要生成中间张量。

### P3（可选，收益最大但要动 autograd）
把每层的前向写成一个 **fused layer kernel**（q/k/v 三次 GEMM 仍走 cuBLAS，
但 LN、RoPE、softmax 的 elementwise 部分合成一个 kernel）。

## 5. 验收标准（必须同侧先证明）

1. 建一个**带同步**的 GEMM 基准：`launch N 次 GEMM → cudaStreamSynchronize → 计时`，
   确认单次 GEMM 真实耗时（现在的 0.03 ms 是异步假象）。
2. 每改一个 kernel，跑同一份 `kprof.py`（按 kernel 名统计），记录：
   启动次数、启动耗时、**整步耗时**。
3. 回归：`run_all.sh` 必须全绿（conformance 13/13、c_arch、V3、pipeline）。

## 6. 现成的测量脚本（本仓库外，D:\kernel_lab\exam 或 4090 的 /home/featurize/tgai/e3）

| 脚本 | 用途 |
|---|---|
| `profile2.py` | forward / backward / opt 分段计时 |
| `kprof.py` | 按 kernel 名统计次数与耗时 |
| `gemmbench.py` | GEMM 微基准（**注意：需加同步才可信**） |
| `mathmode.py` | cuBLAS 数学模式 A/B |

## 7. 给用户一个用 TGAI GO 的理由（现状，不是吹）

**部署侧已经赢**（有实测）：
- `.TG` 单文件，C 引擎，**零 Python 依赖**（PyTorch 要 2 GB 环境）；
- 长上下文推理快：seq=8192 吞吐 56.7k tok/s（0.144 s/step）；
- 显存更省：MoE 基准 3.418 GiB vs PyTorch 2.966 GiB（同精度下峰值更低）；
- 训练出来的模型**同一份权重**能直接进产线引擎，不需要转换脚本。

**训练侧现在是短板**（这份简报就是要修的）：
手写模型走 Python 逐算子路径，比 PyTorch 慢 8.7–19×。
修完 P0–P2 预计能进到 **1.5–2× 以内**；要反超得靠 P3 + 把图搬进 C。

> 一句话：**TGAI GO 现在的价值是"训练完直接能上产线、部署极轻"，
> 不是"训练更快"。要把训练也变成卖点，就必须把 P0–P3 做完。**

---

## ★ 已修复（2026-10-30，第 37 轮）

**根因找到了：`bmm` 在 Python 里按 batch 循环调 cuBLAS。**

```python
# 旧（cuda.py bmm）
for i in range(batch):
    self._sgemm(...)          # attention 里 batch = B*NH = 624 ⇒ 624 次 ctypes 调用
```

每层注意力的 `rope`（2 次 bmm）与 `sdpa`（1 次 bmm）合计 3 次 bmm × 624 = 1872 次调用/层，
每次约 20 µs ⇒ 单层 28.4 ms，8 层 227 ms，正好是 forward 的 96%。

**修法**：新增 `sgemm_strided_batched()`，用 `cublasSgemmStridedBatched` 一次算完整个 batch
（行优先→列优先的转换与 `_sgemm` 完全一致）。

### 效果（实测）

| 指标 | 修前 | 修后 | 提升 |
|---|---|---|---|
| 注意力块 | 28.43 ms/层 | **1.59 ms/层** | **17.9×** |
| forward | 237 ms | **21.1 ms** | 11.2× |
| backward | 357 ms | **33.0 ms** | 10.8× |
| 整步（10M, B=78 S=64） | 605 ms | **64.9 ms** | **9.3×** |
| 4090 实跑 600 步 | 609.5 ms/step | **66.5 ms/step** | 9.2× |
| V100 实跑 600 步 | 588.7 ms/step | **118.7 ms/step** | 5.0× |
| 对 PyTorch（4090） | 19.3× 慢 | **2.11× 慢** | — |
| 对 PyTorch（V100） | 8.8× 慢 | **1.77× 慢** | — |

### 数值正确性（回归全绿，且与修前逐位一致）

- bmm 单测：与 numpy `einsum` 最大误差 **2.861e-06**
- llama CUDA 8.664e-06 / qwen2 4.403e-06 / gpt2 1.857e-07（与修前完全相同）
- c_arch 三架构 PASS、V3 前向 4.793e-05、pipeline 5.3091→0.1755、性能/显存不变
- 考试二模型重训后 loss 5.9257 → 0.0885，引擎三问回答**逐字不变**

## 接下来还能做什么（按收益排序）

| 项 | 现状 | 预计 |
|---|---|---|
| 反向的 elementwise 融合（RMSNorm/GLU 各 5→1 kernel） | 1014 kernel/步 | 1.3–1.6× |
| `cross_entropy` 8.6 ms（占整步 13%） | 物化 log_softmax | 1.1× |
| `broadcast_to8` 物化（117 次/步） | 生成中间张量 | 5–10% |
| 每层 fused kernel（P3） | — | 再 1.3× |

## ★ 第二轮修复（2026-10-30，同一轮后续）

**`cross_entropy(ignore_index=...)` 把整块 logits 拷到 CPU 算**：

```python
# 旧（__init__.py _cross_entropy_cuda_ignore）
host = Tensor(logits.to_numpy(), requires_grad=True)   # 4992x360 = 7 MB D2H
loss_cpu = _Functional._cross_entropy_cpu(host, tgt, ignore_index)
... rt.h2d(host.grad)                                  # 又 7 MB H2D
```

**修法**：用**选择矩阵**在 GPU 上算有效行。

```python
S = 单位阵取有效行   # [nvalid, n]，缓存在 GPU 上（mask 每步相同，只建一次）
lc = S @ logits                       # 只保留有效行
loss = ce_fwd(lc, tgt_valid)          # 融合 kernel
# backward: dlogits = S.T @ ce_bwd(lc, tgt_valid)，无效行天然为 0
```

**踩过的坑（重要）**：第一版用「无效行填 safe 目标 0」的偷懒法，
**梯度是对的（9.5e-08）但 loss 偏大 72%**（8.09 vs 4.69），因为那些行的 CE 被算进了均值。
必须真的只对有效行求均值。

### 效果

| 指标 | 修前 | 修后 |
|---|---|---|
| CE + backward（4992×360，20% 有效） | 8.8 ms | **0.59 ms（15×）** |
| 整步（4090） | 66.4 ms | **50.7 ms** |
| 整步（4090 实跑 600 步） | 66.5 ms | **53.3 ms** |
| 整步（V100 实跑 600 步） | 118.7 ms | **85.6 ms** |

### 累计战果

| | 最初 | bmm 修后 | CE 修后 | 对 PyTorch |
|---|---|---|---|---|
| 4090 | 609.5 ms | 66.5 ms | **53.3 ms** | 19.3× → **1.69×** |
| V100 | 588.7 ms | 118.7 ms | **85.6 ms** | 8.8× → **1.28×** |

### 剩余项（下一步）

| 项 | 现状 | 预计 |
|---|---|---|
| 反向 elementwise 融合（RMSNorm/GLU 各 5→1 kernel） | 1014 kernel/步 | 1.3–1.5× |
| `broadcast_to8` 物化（117 次/步） | 生中间张量 | 5–10% |
| 每层 fused kernel | — | 再 1.3× |

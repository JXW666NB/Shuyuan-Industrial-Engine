# TGAI GO · 训练侧反向算子（Training Kernels）

把 TGAI GO 从**纯推理引擎**扩展为**可训练引擎**的第一阶段成果。
推理侧（`cuda/tgai_cuda.cu`、`tgai_engine.c`、生产 DLL）**零改动**。

## 这是什么

7 组 device-only CUDA kernel，覆盖单层 Transformer 训练所需的全部前向/反向算子，
每一条都与 PyTorch 逐元素数值对拍通过。

| 文件 | 覆盖算子 | 行数 |
|---|---|---|
| `kernels/tgai_bwd.cu` | `rmsnorm_bwd`、`silu_mul_bwd`、`moe_add_bwd`、`emb_gather_bwd`、`rope_bwd`、`residual_bwd`、`moe_route_bwd` | 196 |
| `kernels/tgai_opt.cu` | `muon_step`（Newton-Schulz 正交化）、`adamw_step` | 132 |
| `kernels/tgai_attn.cu` | causal multi-head attention 前向 + 反向 | 122 |
| `kernels/tgai_mm.cu` | matmul 前向 + 反向（`dX = dY·Wᵀ`、`dW = Xᵀ·dY`） | 62 |
| `kernels/tgai_moeblk.cu` | MoE FFN 块（rmsnorm → top-1 路由 → 专家 SwiGLU → 残差） | 151 |
| `kernels/tgai_layer.cu` | **整层**（attention + MoE，11 个梯度） | 329 |
| `kernels/tgai_ce.cu` | softmax 交叉熵（loss + dlogits） | 57 |

## 怎么跑（不需要 nvcc / MSVC）

整套流程用 **NVRTC 运行期编译 + CUDA Driver API**，只要能加载 `nvrtc64_*.dll` 和 `nvcuda.dll` 即可：

```python
from tgai_nvrtc import Cuda
cu = Cuda()
mod = cu.compile(open("kernels/tgai_layer.cu").read(), "sm_70")
cu.launch(mod, "layer_fwd", (1,1,1), (1,1,1), args, shared=48*1024)
```

这条路径是**被逼出来的**：目标机（V100 / Windows）上 CUDA 12.4 的 `nvcc` 不接受 MSVC 19.44
（报 `Host compiler targets unsupported OS`），NVRTC 绕开了整条宿主编译链。

## 验证结果（实测，可复现）

| 判分台 | 内容 | 结果 |
|---|---|---|
| `harness/op_suite.py` | 7 个反向算子 | `ok=True` |
| `harness/opt_suite.py` | Muon + AdamW（对拍 `torch.optim.AdamW`） | `ok=True`（2.4e-07） |
| `harness/attn_suite.py` | attention（seq 16/24/32，nh 2/3/4，dh 8/16/32） | `ok=True`（2.4e-07） |
| `harness/mm_suite.py` | matmul（3 种形状） | `ok=True`（1.1e-05） |
| `harness/moe_block_suite.py` | MoE 块（2 种形状） | `ok=True`（6.9e-07） |
| `harness/layer_suite.py` | 整层（2 种形状） | `ok=True`（9.0e-07） |
| `harness/ce_suite.py` | 交叉熵（3 种 (N,V)） | `ok=True`（2.4e-07） |

### 端到端训练 + 过拟合测试

`harness/l2_train.py`：同一批 8 个样本重复 400 步（可记忆任务），
对同一模型做**双跑**——参考路径为 PyTorch autograd + 手写 Muon/AdamW，引擎路径为上述 kernel。

```
目标类别数 = 8 / 8 个样本        末步准确率 = 100.0%
loss[0]   ref=4.220753   eng=4.220753
loss[50]  ref=1.816050   eng=1.816050
loss[99]  ref=0.000000   eng=0.000000
末步 loss 精确值: ref=0.000000000 eng=0.000000000
曲线最大绝对偏差 = 6.980e-05
VERDICT ok=True
```

**结论：训练链路真的在学** —— loss 从 4.22 降到 0，样本全背对，且与 PyTorch 曲线一致。

> 说明：过拟合后**权重**相对偏差约 1.4e-02，这是 Muon 的特性——
> 它按 Frobenius 范数把梯度归一化到单位长度，`loss≈0` 时梯度只是数值噪声，
> 仍会被放大成满幅更新。因此收敛后**不应**用逐元素权重对齐做判据，只看 loss 曲线与准确率。

## 已知限制（不要当成 bug）

1. **单线程串行实现**：所有 kernel 用 `grid=(1,1,1) block=(1,1,1)`，正确性优先，**没有做过性能优化**。
   要拿速度必须并行化 + 接 cuBLAS 做 GEMM。
2. **形状有上限**：受静态 shared/局部数组限制（如 `d ≤ 64`、`rows*cols ≤ 2048`），生产用需改成动态。
3. **未接入生产**：`11450` 服务的 DLL 与模型文件未改动；这批 kernel 是独立的一套。

## 后续方向

见 [`docs/hetero_orchestrator.md`](../docs/hetero_orchestrator.md)：显存账、Muon+offload、
异构多卡（含 AMD/国产卡 Vulkan 路线）、1-bit 梯度压缩、MCU 集群算例与风险清单。

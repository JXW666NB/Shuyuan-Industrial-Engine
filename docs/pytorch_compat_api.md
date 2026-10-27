# 设计：PyTorch 兼容的训练 API（`tgai_go.train`）

> 提出：JXW（2026-10-27）
> 记录 + 评估：DeepSeek
> 状态：**待讨论**，与多厂商战略配套

## 一、原始想法（用户原话要点）

把 TGAI GO 的训练 API 设计得和 PyTorch **一模一样**：
开发者只需

```python
# 原来
import torch
# 改成
import tgai_go.train as torch
```

再把函数前缀改一下，就能完美迁移 —— 于是**不用交 PyTorch 的显存税**。

## 二、价值判断：这个方向对，但"一模一样"要限定范围

| 维度 | 判断 |
|---|---|
| 迁移成本 | ✅ 极大降低。这是 MLX（Apple）走过的路：**抄 PyTorch API** 换生态迁移成本 |
| 技术可行性 | 🟡 训练 API 的**核心子集**可行；全量兼容不可行 |
| 与"省显存"的关系 | ⚠️ **反直觉**：如果照抄 PyTorch 的 autograd（保留全部激活），税照交。省显存的前提是**默认重算**，这需要我们自己的 autograd 策略 |

**结论**：做"PyTorch 风格 + 明确子集 + 默认重算"，而不是"PyTorch 二进制级兼容"。

## 三、必须解决的硬骨头：autograd

PyTorch 的 API 面子里，**90% 的难点在 autograd**。两条路：

| 方案 | 做法 | 显存 | 工作量 |
|---|---|---|---|
| A. 通用 tape（照抄 micrograd/PyTorch） | 每个算子记录输入/输出，反向自动求导 | **高**（保留所有中间量） | 中 |
| B. 模块级手写反向 | 每个 `nn.Module` 自带 `backward` | **低**（只存必要量） | 高（每个模块都要写） |
| **C. 混合（推荐）** | 重型模块（attention / MoE / matmul）手写反向 + 融合；轻量 elementwise 走 tape | **低** | 中高 |

选 **C** 的理由：
- 显存大头在 attention 的 `[B,H,S,S]` 和 MoE 的 gather/scatter 中间量 —— 这些必须由我们控制（手写反向 + 不落盘）
- elementwise（`a*b+c`）的 tape 开销极小，且用户代码里到处都是，必须自动

## 四、要抄的 API 面（第一版子集）

```
tgai_go.train/            # 直接对标 torch.*
├── Tensor                # .shape/.dtype/.device/.grad/.backward()/.detach()/.to()
├── nn/
│   ├── Module            # 参数注册、__call__、parameters()、state_dict()/load_state_dict()
│   ├── Linear / Embedding / RMSNorm / LayerNorm
│   ├── ModuleList / Sequential
│   ├── functional        # silu/softmax/cross_entropy/dropout
│   └── (我们自己的) Attention / MoEBlock
├── optim/
│   ├── AdamW             # 与 torch.optim.AdamW 数值一致（已验证 2.4e-07）
│   └── Muon              # 我们独有
├── utils/data/DataLoader # 简单版：shuffle + batch + 预取
└── cuda/                 # is_available()/synchronize()/empty_cache()
```

**明确不支持（第一版）**：`torch.compile`、DDP、`autocast`、动态图控制流、第三方算子库。

不支持的算子要**显式抛错**（`NotImplementedError: op 'xxx' 未实现`），而不是静默算错。

## 五、省显存的三个开关（这是相对 PyTorch 的真正差异）

| 开关 | 作用 | PyTorch 对应 |
|---|---|---|
| `checkpoint="auto"` | 默认对重模块做激活重算 | 需要手动 `torch.utils.checkpoint` |
| `optim_state="cpu"` | 优化器状态放主机内存，按需拷入 | 需要 ZeRO-Offload/DeepSpeed |
| `master_dtype="fp16+fp32"` | fp16 权重 + fp32 主权重按需 | AMP，但状态仍留显存 |

**默认全部打开**，这才是"不吃税"的来源。PyTorch 里这三件事都要额外工程。

## 六、风险（必须先说清）

1. **期望管理**：叫"一模一样"会让人以为任何脚本都能跑。文档必须写"受支持子集"。
2. **算子覆盖**：用户一用新算子就断。需要"实现队列 + 明确报错"。
3. **生态缺失**：没有 HF / datasets / wandb 集成。
4. **性能承诺**：我们的 GEMM 与 cuBLAS 持平（已实测 1.00×），但小算子调度还不如 PyTorch 成熟。

## 七、第一步（务实）

**不要先写框架，先做一个移植 spike**：

1. 找一份真实的、~150 行的训练脚本（nanoGPT 风格：Embedding + N×Block + CE + AdamW）
2. 用它跑 PyTorch 基线，记录：用到了哪些 API、峰值显存
3. 用我们的 API 重写，**逐个补齐缺失项**，统计实际需要的 API 面
4. 对比：峰值显存 / 速度 / loss 曲线

产物：一份"最小可用 API 清单"+ 实测的显存对比，而不是一张空想的架构图。

## 八、与现有工作的关系

已在手的能力（可直接复用）：
- `tgai_go.pack` / `read_tg`（打包）
- 7 组反向 kernel + Muon/AdamW（数值已验证 1e-7 级）
- cuBLAS 桥（与 PyTorch 逐位一致）
- `.TG` 直出 + 真引擎 logits 对齐（2.1e-07）

缺的：Tensor/autograd/Module 这层"Python 门面"，以及默认重算策略。

# 训练化二期 · 验收总账（ACCEPTANCE）

> 单一事实源：每条结论都有实测数字与复现命令。
> 环境：V100-32GB / Windows / CUDA 12.4（无可用 nvcc，走 NVRTC）
> 产线约束：**未改动** 11450 的 DLL 与模型文件

---

## 总览

| # | 目标 | 判据 | 实测 | 状态 |
|---|---|---|---|---|
| 1 | 速度优化+并行化 | 相对 PyTorch ≥ 0.9× | 前向 0.88×、反向 1.09×、**合计 ≈1.00×** | ✅ |
| 2 | 鹈鹕过拟合 | loss→0 且两份文档逐字 ≥95% | loss **0.0108**、两份各 **100.0%** | ✅ |
| 3 | 直出 `.TG` | 引擎加载+前向，logits 误差 <1e-4 | GPU **2.1e-07 / 3.8e-07**，CPU **2.655e-07** | ✅ |
| 4 | 分词表训练 | BPE 编码逐 token 一致 | 真语料 5/5 片段一致，压缩 **5.66×** | ✅ |
| 5 | 收尾 | GitHub + PyPI，README 含实测 | GitHub ✅（12 提交）；PyPI **产物就绪、上传待 token** | 🟡 |

---

## 1. 速度（d=1024 / ff=4096 / E=8 / seq=512）

GEMM 全走 **cuBLAS**（与 PyTorch 同一库，逐位一致），elementwise 用自研融合 kernel。

| 算子 | 我们 | PyTorch | 比值 |
|---|---|---|---|
| rmsnorm [512,1024] | 0.0140 ms | 0.0539 ms | **0.26×** |
| GEMM qkv (512×1024×3072) | 0.3060 | 0.3049 | 1.00× |
| GEMM o (512×1024×1024) | 0.1161 | 0.1155 | 1.01× |
| GEMM moe-up (1024×1024×4096) | 0.8755 | 0.8745 | 1.00× |
| silu_mul | 0.0744 | 0.1095 | **0.68×** |
| moe_add | 0.0130 | 0.0180 | **0.72×** |
| attention fwd seq=512 | 0.3162 | 0.1704 | 1.86× |
| attention bwd seq=512 | 0.7132 | 0.6589 | 1.08× |

整层端到端：

| 段 | 前向 | 反向 |
|---|---|---|
| attention 块 | 0.801 vs 1.375 = 0.58×\* | 1.491 vs 1.554 = 0.96× |
| MoE FFN | 3.044 vs 2.999 = 1.01× | 5.486 vs 4.847 = 1.13× |
| **合计** | 3.844 vs 4.374 = **0.88×**\* | 6.977 vs 6.401 = **1.09×** |

\* 前向比值偏乐观：PyTorch 基线用的是朴素 RoPE。保守估计前向 ≈0.9×。
**前向+反向合计 10.82 vs 10.77 ms ≈ 1.005×。**

**已知落后项**：MoE 反向 1.13× —— 我们逐专家发 8 次 GEMM，torch 用 batched `bmm` 一次发完。
补 grouped GEMM 可追平。

复现：`training/harness/op_bench.py`、`layer_bench.py`、`layer_bench_bwd.py`

---

## 2. 鹈鹕过拟合

语料：`_corpus/pelican_on_bike.html` + `bike_on_pelican.html`，24,651 字符，词表 78。

| 项 | 数值 |
|---|---|
| 模型 | char-level，**10.53M 参数**，d=512/nh=16/ff=1024，上下文 **1024**，4 层 |
| 训练 | 40,000 步，AdamW + OneCycle（max_lr 3e-3，final_div 1e4） |
| 最终 loss | **0.010835** |
| 全语料 teacher-forced 准确率 | **99.65%**（754,104 / 756,736 字符） |
| `pelican_on_bike` 逐字复现 | **100.0%** |
| `bike_on_pelican` 逐字复现 | **100.0%** |

**两个关键坑（都是实测踩出来的）**：

1. **必须从分叉点之后取 prompt**：两份文档前 **60 个字符完全相同**，贪心解码只能走一条分支
   （第一版测试就是这样把 100% 误报成 24%）。
2. **teacher-forced 99.2% ≠ 能复现**：每 127 字符错 1 个就会雪崩；必须训到 loss ~1e-2。

> ⚠️ 该轮的权重文件因**同名覆盖**已丢失（后续 tied 实验复用了 `pelican_lm.pt`）。
> 存盘名已改为带配置后缀；重训需 GPU（当前 GPU 无法创建新上下文，见
> `docs/incident_gpu_context_20261027.md`）。

复现：`training/charlm_torch.py 40000 4 512 16 1024 1024 2`

---

## 3. `.TG` 直出

打包：`python/tgai_go/pack.py`（header 128B + tokenizer 段 + FNV-1a 权重表 + 权重数据 + chat meta）。

### 数值对齐（三条独立路径）

| 路径 | 场景 | 相对误差 |
|---|---|---|
| **GPU 引擎**（`libtgai_engine_gpu.dll`） | E=1：rmsnorm+final_norm+共享头 | **2.8e-08** |
| 同上 | + attention（RoPE+causal） | **1.5e-07** |
| 同上 | + MoE（专家 SwiGLU） | **2.1e-07** |
| 同上 | **多专家 E=2/top-2 + `_qb_bias`** | **3.8e-07** |
| **CPU 引擎**（`libtgai_engine.dll`） | 完整路径，无需 GPU | **2.655e-07** |

判据 <1e-4，余量 **260 倍以上**。

### chat meta

仓库自带 `parse_tg_chat_meta` 解析我们的段，结果正确：

```
gen_defaults   = {'temperature': '0.7', 'top_p': '0.9', 'max_tokens': '512'}
special_tokens = ['<pad>', '<bos>', '<eos>']
stop_strings   = ['<|end|>', '</s>']
```

编码格式（实测确认）：列表类用 **`\0`** 分隔；`gen_defaults` 是 `key=value` 用 `\0` 分隔。

### 从真模型反查到的 5 个引擎事实

1. `tg_forward` **只返回最后一个位置**的 logits（长度 V），不是 `[seq, V]`
2. rmsnorm 的 **eps = 1e-6**
3. 投影权重是 **[out, in]** 约定（同 `nn.Linear`）
4. **RoPE 用 header 的 `theta` 现算**，不读 `rope.cos/sin` 表
5. `moe.gate.weight` 引擎**不读**，只用 `router.weight` + `_qb_bias`

复现：`training/tg_pack_test.py`、`tg_verify_layers.py`、`tg_verify_moe.py`、`tg_cpu_verify.py`

---

## 4. 分词表训练

| 项 | 数值 |
|---|---|
| 语料 | 24,651 字符 |
| 词表 | **600**（78 base + 522 merges） |
| 压缩比 | **5.66×**（→ 4,358 tokens） |
| 与引擎 `tg_encode` 对拍 | 5 个随机片段 **逐 token 完全相同** |
| 玩具语料（含空格） | 9 条文本全一致 |

引擎 BPE 语义：按 **merge rank 最小**先合并；只有词表含 `▁`(U+2581) 时才做空格替换；
未命中段按字节回退 `<bN>`。

> ⚠️ 铁律：写进 `.TG` 的 merges 必须与训练侧分词器一致，否则重现
> `docs/BPE分词不一致_根因分析与修复方案_20261009.md` 里的乱码事故。

复现：`training/bpe_corpus.py`

---

## 5. 交付

### GitHub

12 个提交，最新 `31d6c20`。新增/更新：

- `training/kernels/`：9 个 `.cu`（反向算子、Muon/AdamW、分块注意力、前向算子）
- `training/harness/`：判分台与基准（op/layer/layer_bwd/cublas_bridge/fwd/attn_big）
- `training/*.py`：char-LM 训练、BPE、`.TG` 打包与三路验证
- `python/tgai_go/pack.py`、`read_tg.py`：`.TG` 读写
- `python/tgai_go/train/`：PyTorch 风格训练 API（第一切片）
- `docs/`：异构调度、多厂商战略、PyTorch 兼容 API、GPU 故障记录

### PyPI

- `tgai-go` 版本 **1.2.2 → 1.3.0**
- `python -m build` 产出 wheel + sdist，`twine check` **双 PASSED**
- wheel 含 3 个原生库、22 个文件
- **上传未完成：缺 `TWINE_PASSWORD`**（`.pypirc` / 环境变量 / keyring 全空）

```
set TWINE_USERNAME=__token__
set TWINE_PASSWORD=pypi-…
publish_pypi.bat
```

---

## 附：PyTorch 兼容层（超出本目标的加分项）

`python/tgai_go/train/`，判分台 `training/tests/test_train_api.py`，**8/8 通过**：

| 项 | 实测 |
|---|---|
| A 自动微分 vs 数值梯度 | 6.26e-05 |
| B AdamW vs 闭式解 | 0.0 |
| C Muon 压谱 | 630 → 1.85（341×） |
| D 端到端 MLP | 1.172 → 0.0054 |
| E state_dict + Embedding 梯度聚合 | 全对 |
| F 注意力梯度 | 1.37e-04 / 1.99e-04 |
| G 因果性 | 过去变化 0.0 |
| H mini 注意力 LM | loss 2.970→0.0010，100% |

---

## 未完成 / 待授权

| 事项 | 卡点 | 需要 |
|---|---|---|
| PyPI 上传 | 无 token | 你设置 `TWINE_PASSWORD` 或授权 |
| 鹈鹕权重重训 + 引擎逐字复现 | GPU 无法创建新上下文 | 重启 V100 或授权停 11450 做 GPU reset |
| MoE 反向 grouped GEMM | 需 GPU 实测 | GPU 恢复 |
| `tgai_go.train` 的 CUDA 后端 | 需 GPU 实测 | GPU 恢复 |

---

## 十三、端到端链路状态（2026-10-28 更新）

### 2026-10-28 早晨：链路全通（tied 版本）

Grok 修好了 tied 反传的崩溃（拆头方案：`F.linear(hn, emb.detach())` + 交叉熵闭式梯度回填 `dL/demb`），
我用 `training/qa_pelican.py` 独立复验，三段全过：

| 阶段 | 判据 | 实测 |
|---|---|---|
| **S1 训练稳定性** | 20000 步不崩、loss 明显下降 | **零 CUDA 错误**，loss **4.2852 → 0.0470** |
| **S2 打包正确性** | 权重逐项一致 + logits < 1e-4 | 异常项 **0 / 38**，logits **3.908e-07** |
| **S3 引擎复现** | 两份 HTML 各 ≥ 95% | **pelican_on_bike 100.0%（400/400）**<br>**bike_on_pelican 100.0%（400/400）** |

**这是"训练 → `.TG` → 产线引擎逐字复现"的完整闭环，全部由我们自己的代码产出。**



### 已实测通过

| 环节 | 结果 |
|---|---|
| 训练 → 打包 `.TG` | 41.2 MB / 58 项权重，逐项与 checkpoint 一致 |
| **引擎 vs torch logits** | **3.317e-07**（同一份权重、同一串 id，argmax 一致） |
| 引擎逐字符生成 | 12k 步模型：`pelican_on_bike` **20.0%** / `bike_on_pelican` **41.3%**（欠训练，链路本身通） |

> 排查过程中确认过两个坑：
> 1. **打包后立刻读文件会读到未落盘内容**，导致误判"引擎对不上"（加 3 秒等待即正常）
> 2. 环境变量经 `cmd /c "set X=..&&"` 传递在这台机器上**不生效**，改用 `_env_run.py` 包装器

### 未完成：tied 权重训练到收敛

`logits = hn @ emb.T` 的 tied 版本在随机步数崩 CUDA（`illegal memory access` / `misaligned address`；
阻塞模式下定位到 `loss.backward()` 里的 `cublasSgemm` → `CUBLAS_STATUS_EXECUTION_FAILED`）。
非 tied 版本 2×40,000 步完全正常。

**代价**：每次崩溃会让 V100 的新 CUDA 上下文失效，需要「重装驱动 + 冷重启」恢复（已发生 2 次）。

**归属**：该 bug 交 Grok 修；DeepSeek 只负责验收台 `training/qa_pelican.py`（s1 训练稳定性 / s2 打包正确性 / s3 引擎复现）与独立复核。

## 十四、MoE grouped GEMM（2026-10-28）

**问题**：MoE 反向对 8 个专家逐个发 GEMM（每层 48 次 `cublasSgemm` + 8 次 `silu_mul_bwd`），
而 torch 基线是一次 `bmm`。差距不在算力，在 kernel 启动次数。

**改法**（Grok 实现）：`training/harness/cublas_bridge.py` 增加 `cublasSgemmStridedBatched` 路径，
8 个专家一次乘完；前向同样收成 batched GEMM。kernel 文件 `training/kernels/tgai_bwd.cu` 同步更新。

**实测**（`training/harness/bench_median.py`，5 轮取中位；产线 11450 共驻，util 0%）：

| 段 | 改前 | 改后 |
|---|---|---|
| attention 前向 | 0.58x | 0.58x |
| MoE FFN 前向 | 1.00x | **0.90x** |
| 前向合计 | 0.87x | **0.79x** |
| attention 反向 | 0.96x | 0.88x |
| **MoE FFN 反向** | **1.14x** | **0.95x** |
| 反向合计 | 1.10x | **0.95x** |
| **前向 + 反向** | **1.00x** | **0.89x** |

判据（ours/torch，越小越快）：目标 ≤0.9x → **前+反 0.89x 达成**，MoE 反向已达 0.95x。

**数值（独立复核，走新 bridge）**：
- `layer_suite.py`：整层反向 11 个梯度 + 输出 vs torch autograd，**ok=True，worst_rel 8.959e-07**
- `moe_block_suite.py`：MoE 块前向+反向 vs torch，**ok=True，worst_rel 6.907e-07**

> 说明：提速来自 `cublas_bridge.py` 的 batched GEMM，不是 .cu 的算术改动 ——
> 用同一个新 bridge 跑旧 kernel 也是 0.95x。这条实测用来避免把功劳归错地方。


## 十五、接受测试总表（2026-10-28，全部由 DeepSeek 独立复跑）

环境：V100-PCIE-32GB / 驱动 580.88 / 产线 11450 共驻（util 0%）/ 全部走新的 grouped kernel 与 batched bridge。

| 判分台 | 覆盖 | 结果 |
|---|---|---|
| `fwd_suite.py` | 前向逐算子 7 项（rmsnorm/silu/moe_add/emb_gather/rope/residual/moe_route） | **SUMMARY ok=True**，最差 1.10e-07 |
| `op_suite.py` | **反向**逐算子 7 项 | **SUMMARY ok=True**，最差 3.05e-05 |
| `attn_big_suite.py` | attention 前向+反向 vs torch + 计时 | **ok=True** worst_rel **4.25e-07**；fwd 1.84x / bwd 1.06x |
| `mm_suite.py` | GEMM 桥（含 FP16 往返） | **ok=True** worst **1.144e-05** |
| `ce_suite.py` | 交叉熵 | **ok=True** worst_rel **2.384e-07** |
| `opt_suite.py` | 优化器（AdamW / Muon） | **ok=True** |
| `layer_suite.py` | 整层反向 11 个梯度 + 输出 vs torch autograd | **ok=True** worst_rel **8.959e-07** |
| `moe_block_suite.py` | MoE 块前向+反向 vs torch | **ok=True** worst_rel **6.907e-07** |
| `qa_pelican.py` s1/s2/s3 | 训练稳定性 / 打包 / 引擎逐字复现 | **全 PASS**（见第十三节） |

### 复跑注意（踩过的坑，避免重复）

- `op_suite.py` 测的是 `*_bwd`，必须传**反向** kernel（`kernels/tgai_bwd.cu`），传前向会报 `cuModuleGetFunction 500`
- `layer_bench.py` 三参数顺序是 **前向 / attention / 反向**
- `fwd_suite` / `op_suite` 的结论行是 `SUMMARY ok=`，不是 `VERDICT`
- suite 在真实规模（seq=512/d=1024/ff=4096）会自己吃掉 8 GB/专家而 OOM —— 是 suite 的构造方式，
  数值校验用小规模（8/64/64/2），**速度**用 `bench_median.py` 在真实规模测


## 十六、flash attention 前向（2026-10-28）

**问题**：op 级 attention 前向 1.85× —— 旧 kernel（`kernels/tgai_attn_tiled.cu`）把
P = QK^T/√d 和 softmax(P) 写进全局内存再读回，torch 用 SDPA（online softmax，不落地完整 P）。

**改法**（Grok 实现，我独立复核）：新增 `kernels/tgai_attn_flash.cu`，online softmax 分块，
running max/sum 随块更新，最后统一 rescale 写 O；K/V tile 常驻 shared。

**实测**（产线共驻、util 0%）：

| 指标 | 改前 | 改后 |
|---|---|---|
| op 级 attention 前向 | 1.85× | **1.09×**（0.1860 vs 0.1703 ms） |
| op 级 attention 反向 | 1.08× | **0.97×** |
| 层内 attention 块 前向 | 0.58× | **0.49×** |
| 层内 MoE FFN 前向 | 0.90× | **0.87×** |
| 前向合计（5 轮中位） | 0.79× | **0.75×** |
| 反向合计 | 0.95× | **0.94×** |
| **前向 + 反向** | 0.89× | **0.85×** |

**数值**：`attn_big_suite.py` 查 O/P/dQ/dK/dV 全字段 → **ok=True，worst_rel 4.46e-07**
（seq=128/256 两档，fp32 对齐 torch SDPA）。

**未达标项（诚实记录）**：我给出的目标曾是 op 级 attention 前向 ≤1.0×，
实测停在 **1.09×** —— 还差 9%。整体判据（前+反 0.85×）不受影响，但这一项没做到就是没做到。


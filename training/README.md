# TGAI GO · 训练侧（Training）

把 TGAI GO 从**纯推理引擎**扩展为**可训练引擎**。
推理侧（`cuda/tgai_cuda.cu`、`src/tgai_engine.c`、生产 DLL）**零改动**。

> 所有数字均为 V100-32GB / Windows / CUDA 12.4 上的实测，判分台源码同仓库可复现。

---

## 一、交付物

### 1. 反向算子（`training/kernels/`）

| 文件 | 覆盖 | 行数 | 对拍结果 |
|---|---|---|---|
| `tgai_bwd.cu` | rmsnorm / silu_mul / moe_add / emb_gather / rope / residual / moe_route 的**反向** | 196 | 7/7 PASS |
| `tgai_opt.cu` | **Muon**（Newton-Schulz 正交化）+ **AdamW** | 132 | 2.4e-07 |
| `tgai_attn.cu` | causal MHA 前向 + 反向（小 seq 版） | 122 | 2.4e-07 |
| `tgai_attn_tiled.cu` | tiled MHA 前向 + 反向（seq≥512 的早期版） | 479 | 4.3e-07 |
| `tgai_attn_flash.cu` | **flash 前向**（online softmax，不落地完整 P） | 559 | 4.5e-07 |
| `tgai_mm.cu` | matmul 前向 + 反向 | 62 | 1.1e-05 |
| `tgai_moeblk.cu` | MoE FFN 块（rmsnorm → top-1 路由 → 专家 SwiGLU → 残差） | 151 | 6.9e-07 |
| `tgai_layer.cu` | **整层**（attention + MoE，11 个梯度） | 329 | 9.0e-07 |
| `tgai_ce.cu` | softmax 交叉熵（loss + dlogits） | 57 | 2.4e-07 |
| `tgai_fwd.cu` | 6 个前向算子（并行版） | 148 | 全 PASS |

### 2. 判分台（`training/harness/`）

`op_suite` / `opt_suite` / `attn_suite` / `attn_big_suite` / `mm_suite` / `moe_block_suite` /
`layer_suite` / `ce_suite` / `fwd_suite` / `l2_train` / `op_bench` / `layer_bench` / `layer_bench_bwd`
以及 `bench_median`（多轮取中位，消除产线共驻带来的抖动）

每个判分台都**先自证**（用参考实现跑通）再用于考核；参考实现放在 `.*_private.cu` 里不随仓库分发。

### 3. `.TG` 直出（`python/tgai_go/pack.py`、`read_tg.py`）

训练完直接把权重 + 分词表 + 特殊 token + 默认推理参数写成一个 `.TG`，**能被 TGAI GO 直接加载**。

### 4. 分词表训练（`training/bpe_train.py`、`bpe_corpus.py`）

纯 Python 实现 BPE，产出 `vocab + merges`，写进 `.TG` 的 tokenizer 段。

---

## 二、速度：与 PyTorch 对比（d=1024 / ff=4096 / E=8 / seq=512）

GEMM 全部走 **cuBLAS**（与 PyTorch 同一个库、逐位一致），elementwise 用我们自己的融合 kernel。

| 算子 | 我们 | PyTorch | 比值 |
|---|---|---|---|
| rmsnorm [512,1024] | 0.0131 ms | 0.0534 ms | **0.25×** |
| GEMM qkv (512×1024×3072) | 0.3057 | 0.3059 | 1.00× |
| GEMM o (512×1024×1024) | 0.1154 | 0.1156 | 1.00× |
| GEMM moe-up (1024×1024×4096) | 0.8739 | 0.8736 | 1.00× |
| silu_mul | 0.0742 | 0.1096 | **0.68×** |
| moe_add | 0.0127 | 0.0178 | **0.71×** |
| attention fwd seq=512 | 0.1741 | 0.1706 | 1.02× |
| attention bwd seq=512 | 0.7114 | 0.7350 | **0.97×** |

> attention 前向的 1.02× 是在**额外产出 P** 的前提下取得的：我们的 `attn_bwd` 直接消费 P、不重算，
> 所以前向必须写这份 25 MB 的概率矩阵；torch 的 SDPA 基线只产出 O。详见 `ACCEPTANCE.md` §16。

整层端到端（attention 块 + MoE FFN），五轮取中位：

| 指标 | grouped GEMM 后 | **再加 flash attention** |
|---|---|---|
| 层内 attention 块 前向 | 0.58× | **0.49×** |
| 层内 MoE FFN 前向 | 0.90× | **0.87×** |
| 前向合计 | 0.79× | **0.75×**（中位 ms 3.069 / 4.350 = **0.71×**） |
| 反向合计 | 0.95× | **0.94×** |
| **前向 + 反向** | 0.89× | **0.85×** |
| op 级 attention 前向 | 1.85× | **1.02×**（且我们额外产出 P，见 ACCEPTANCE §16） |
| op 级 attention 反向 | 1.08× | **0.97×** |

\* 前向的 attention 0.49× 偏乐观：PyTorch 基线的 RoPE 是朴素实现。保守估计层内前向约 0.85×。
**前向+反向合计 = 0.85×（比 PyTorch 快约 15%）。**

> 以上为 `harness/bench_median.py` 五轮取中位（产线 11450 共驻、util 0%）。
> **MoE 反向曾落后 1.14×**：我们逐专家发 8 次 GEMM，torch 用 batched `bmm` 一次发完。
> 已由 grouped GEMM 修掉 —— `cublas_bridge.py` 增加 `cublasSgemmStridedBatched`，
> 8 个专家一次乘完，MoE 反向降到 **0.95×**。
> **attention 前向曾落后 1.85×**：旧 kernel 把 P 和 softmax(P) 写进全局内存再读回。
> 已换成 online-softmax 的 flash 前向（`kernels/tgai_attn_flash.cu`，不落地完整 P），
> 单算子降到 **1.02×**、层内降到 **0.48×**。
> 诚实说明：grouped GEMM 的提速来自 bridge 的 batched GEMM，不是 .cu 的算术改动
> （同一个新 bridge 跑旧 kernel 也是 0.95×）。

---

## 三、过拟合实测：两份 SVG HTML 逐字复刻

语料：`_corpus/pelican_on_bike.html` + `bike_on_pelican.html`（24,651 字符，词表 78）

| 项 | **tied 版（当前验收）** | 早期非 tied 版 |
|---|---|---|
| 结构 | char-level，**4 层**，`d=512 / nh=16 / ff=1024`，词表 **78** | 同左 |
| 参数量（读 checkpoint 实测） | **10.530 M**（`emb[78,512]`、`Wqkv[512,1536]`、`Wg/Wu[512,1024]`、`Wd[1024,512]`） | 同左 |
| 上下文 | **512** | 1024 |
| 训练步数 | **20,000** | 40,000 |
| 最终 loss | **0.0470** | 0.0108 |
| `pelican_on_bike` 逐字复现 | **100.0%（400/400）** | 100.0% |
| `bike_on_pelican` 逐字复现 | **100.0%（400/400）** | 100.0% |

> 语料实测 **24,651 字符 / 词表 78**；任务书里写的是 24,657 / 85，以实测为准。

> 关键坑：两份文档前 60 个字符完全相同，**必须从分叉点之后取 prompt**，否则贪心解码只会走一条分支
> （第一版测试就是这样误报 24%）。另外 teacher-forced 99.2% 并不等于能复现 —— 每 127 字符错 1 个就雪崩。

---

## 四、`.TG` 直出：真引擎加载并前向，logits 逐位对齐

用 `libtgai_engine_gpu.dll`（产线同款）加载我们写的 `.TG`：

| 场景 | 相对误差 |
|---|---|
| 单专家：rmsnorm + final_norm + 共享头 | **2.8e-08** |
| + attention（RoPE + causal） | **1.5e-07** |
| + MoE（专家 SwiGLU） | **2.1e-07** |
| **多专家 E=2/top-2 + `_qb_bias`** | **3.8e-07** |

过程中查实的 5 个引擎事实（都写在 `training/tg_verify_*.py` 注释里）：

1. `tg_forward` **只返回最后一个位置**的 logits（长度 V），不是 `[seq, V]`
2. rmsnorm 的 **eps = 1e-6**
3. 投影权重是 **[out, in]** 约定（同 `nn.Linear`）
4. **RoPE 用 header 的 `theta` 现算**，不读 `rope.cos/sin` 表（表是给别的 backend 的）
5. `moe.gate.weight` 引擎**不读**，只用 `router.weight` + `_qb_bias`

---

## 五、分词表训练：与引擎逐 token 一致

| 项 | 数值 |
|---|---|
| 语料 | 24,651 字符 |
| 词表 | **600**（78 base + 522 merges） |
| 压缩比 | **5.66×**（→ 4,358 tokens） |
| 与引擎 `tg_encode` 对拍 | 5 个随机片段 **逐 token 完全相同** |

引擎 BPE 语义（源码核对 + 实测）：按 **merge rank 最小**优先合并；只有词表含 `▁`(U+2581) 时
才做空格替换；未命中的连续段按字节回退 `<bN>`。

---

## 六、怎么跑（不需要 nvcc / MSVC）

整套流程用 **NVRTC 运行期编译 + CUDA Driver API**：

```python
from tgai_nvrtc import Cuda
cu = Cuda()
mod = cu.compile(open("kernels/tgai_layer.cu").read(), "sm_70")
cu.launch(mod, "layer_fwd", (1,1,1), (1,1,1), args, shared=48*1024)
```

这条路径是被逼出来的：目标机 CUDA 12.4 的 `nvcc` 不接受 MSVC 19.44
（`Host compiler targets unsupported OS`），NVRTC 绕开整条宿主编译链。

---

## 七、已知限制

1. **不是所有 kernel 都并行**：`tgai_layer.cu` / `tgai_moeblk.cu` / `tgai_mm.cu` 是单线程串行
   （正确性优先、规模小），生产形态是"cuBLAS + 并行算子 + 主机编排"（见 `layer_bench.py`）。
2. **形状有上限**：受静态 shared / 局部数组限制（如串行层 `d ≤ 64`）。
3. **MoE 已用 grouped GEMM**（`cublasSgemmStridedBatched`），反向 0.95×；要求各专家行数相同，
   行数不齐时该路径不适用，需回退逐专家 GEMM。
4. **只验了 CUDA**：HIP（海光 DCU / AMD）移植未开始，见 `docs/multi_vendor_strategy.md`。
5. **PyPI 包未含训练流程**：`tgai_go.pack` / `read_tg` 只是打包工具，训练脚本在仓库 `training/`。

---

## 八、CPU 侧独立复核（不需要 GPU）

GPU 被占用/异常时，`.TG` 仍可用**纯 CPU 引擎**验证：

```
[1] 打包 (含 chat meta)                     56,103 B
[2] chat meta 解析（仓库自带 parse_tg_chat_meta）
    gen_defaults  = {'temperature': '0.7', 'top_p': '0.9', 'max_tokens': '512'}
    special_tokens= ['<pad>', '<bos>', '<eos>']
    stop_strings  = ['<|end|>', '</s>']
[3] CPU 引擎加载: L=1 d=32 nh=4 ff=64 V=40 seq=128 E=1 topk=1
[4] CPU 引擎 logits vs NumPy 参考: 最大相对误差 = 2.655e-07
VERDICT ok=True  chat_meta=True  cpu_forward=True
```

脚本：`training/tg_cpu_verify.py`；引擎：`cpu/build/libtgai_engine.dll`。

### chat meta 的编码格式（实测确认）

| type | 名称 | 编码 |
|---|---|---|
| 1/6/7/8 | chat_template / system_prompt / render_mode / template_turn | 纯 UTF-8 字符串 |
| 2/3/4 | stop_strings / turn_markers / special_tokens | 多个字符串用 **`\0`** 分隔 |
| 5 | gen_defaults | `key=value`，多条用 **`\0`** 分隔 |

`tgai_go.pack.TGWriter.set_chat_meta()` 会按上表自动编码（传 dict / list / str 均可）。

---

## 九、⚠️ 写 merges 前必读

仓库 `docs/BPE分词不一致_根因分析与修复方案_20261009.md` 记录过一次事故：
`.TG` 里写了 merges，但训练侧分词器走的是字符级回退 ⇒ 引擎做 BPE 合并、训练没做 ⇒ **输出乱码**。

**规则：写进 `.TG` 的 `merges` 必须与训练时用的分词器完全一致。**
我们的对拍（`training/bpe_corpus.py`）就是为此设计的：训练侧编码结果与引擎 `tg_encode`
**逐 token 比对**，不一致立刻暴露。

---

## 十、端到端演示（训练 → `.TG` → 引擎生成）

`training/pelican_to_tg.py` 把训练好的 char-LM 直接打成 `.TG`：

| 环节 | 结果 |
|---|---|
| 训练 | 20000 步零 CUDA 错误，loss **4.2852 → 0.0470** |
| 打包 10.5M 鹈鹕模型 | 58 项权重，逐项与 checkpoint 一致（异常 0/38） |
| `libtgai_engine_gpu.dll` 加载 | ✅ `L=4 d=512 nh=16 ff=1024 V=78 seq=512 E=1 topk=1` |
| 引擎 vs torch logits | **3.908e-07**（判据 <1e-4） |
| **引擎逐字符生成** | **两份 HTML 各 100.0%（400/400）** |

> 引擎的 LM head 与 `token_embedding` **共享权重**，因此训练侧必须用 tied embedding 才能对齐；
> 非 tied 模型打成 `.TG` 后，引擎会用共享头算 logits，结果与训练侧不一致。

---

## 十一、相关设计文档

| 文档 | 内容 |
|---|---|
| `docs/hetero_orchestrator.md` | 异构多卡资源调度（显存账 / 1-bit 压缩 / MCU 算例） |
| `docs/multi_vendor_strategy.md` | 多厂商原生兼容（海光 DCU / AMD / 摩尔线程）与"显存税"拆解 |
| `docs/pytorch_compat_api.md` | PyTorch 兼容训练 API 的设计与边界 |
| `docs/incident_gpu_context_20261027.md` | V100 无法创建新 CUDA 上下文的排查记录 |

---

## 十二、`tgai_go.train`：PyTorch 风格训练 API（第一切片）

目标是让迁移只改两处：

```python
# 原来                          # 现在
import torch                    import tgai_go.train as torch
model = torch.nn.Linear(...)    model = torch.nn.Linear(...)   # 代码几乎不变
```

当前切片（**CPU/numpy 后端，已自证**）：自动微分（含广播还原、批量 matmul、切片索引）、
`nn.Module/Linear/Embedding/RMSNorm/CausalSelfAttention/Sequential/ModuleList`、
`optim.AdamW/Muon`、`functional.cross_entropy/softmax/silu/scaled_dot_product_attention`、
`state_dict/load_state_dict`。

判分台 `training/tests/test_train_api.py`（无需 GPU）：

| 项 | 判据 | 实测 |
|---|---|---|
| A 自动微分 | vs 中心差分，相对误差 < 1e-4 | **6.26e-05** |
| B AdamW | vs 闭式参考，< 1e-6 | **0.0** |
| C Muon | 病态输入条件数压平 <= 2.0 且改善 >= 50x | **630 -> 1.85（341x）** |
| D 端到端 | 小 MLP 拟合非线性函数，loss 降 >= 20x | **1.172 -> 0.0054（218x）** |
| E 参数管理 | state_dict 往返 + Embedding 梯度聚合 | 全对 |
| F 注意力梯度 | dx / dWq 对数值梯度 < 1e-3 | **1.37e-04 / 1.99e-04** |
| G 因果性 | 改未来 token 不影响过去位置 | 过去变化 **0.0**，末位 6.55 |
| H 注意力 LM | 周期序列 next-token，准确率 > 95% | loss 2.970→0.0010，**100%** |

```
汇总: A..H = 8/8 PASS   VERDICT ok=True
```

> Muon 的 NS5 系数（3.4445 / -4.7750 / 2.0315）不动点是 **0.868**，不是 1 ——
> 它的作用是**压平奇异值谱**，常数倍由学习率吸收。判据必须按"条件数"写，不能按"等于单位阵"写。

**尚未做**：CUDA 后端（接我们自己的 kernel + cuBLAS）、attention/MoE 模块、默认激活重算（`checkpoint="auto"`）。设计见 `docs/pytorch_compat_api.md`。

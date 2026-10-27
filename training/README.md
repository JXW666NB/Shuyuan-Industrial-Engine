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
| `tgai_attn_tiled.cu` | **tiled** MHA 前向 + 反向（seq≥512） | 479 | 4.3e-07 |
| `tgai_mm.cu` | matmul 前向 + 反向 | 62 | 1.1e-05 |
| `tgai_moeblk.cu` | MoE FFN 块（rmsnorm → top-1 路由 → 专家 SwiGLU → 残差） | 151 | 6.9e-07 |
| `tgai_layer.cu` | **整层**（attention + MoE，11 个梯度） | 329 | 9.0e-07 |
| `tgai_ce.cu` | softmax 交叉熵（loss + dlogits） | 57 | 2.4e-07 |
| `tgai_fwd.cu` | 6 个前向算子（并行版） | 148 | 全 PASS |

### 2. 判分台（`training/harness/`）

`op_suite` / `opt_suite` / `attn_suite` / `attn_big_suite` / `mm_suite` / `moe_block_suite` /
`layer_suite` / `ce_suite` / `fwd_suite` / `l2_train` / `op_bench` / `layer_bench` / `layer_bench_bwd`

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
| rmsnorm [512,1024] | 0.0140 ms | 0.0539 ms | **0.26×** |
| GEMM qkv (512×1024×3072) | 0.3060 | 0.3049 | 1.00× |
| GEMM o (512×1024×1024) | 0.1161 | 0.1155 | 1.01× |
| GEMM moe-up (1024×1024×4096) | 0.8755 | 0.8745 | 1.00× |
| silu_mul | 0.0744 | 0.1095 | **0.68×** |
| moe_add | 0.0130 | 0.0180 | **0.72×** |
| attention fwd seq=512 | 0.3162 | 0.1704 | 1.86× |
| attention bwd seq=512 | 0.7132 | 0.6589 | 1.08× |

整层端到端（attention 块 + MoE FFN）：

| 段 | 前向 | 反向 |
|---|---|---|
| attention 块 | 0.801 vs 1.375 = 0.58×\* | 1.491 vs 1.554 = 0.96× |
| MoE FFN | 3.044 vs 2.999 = 1.01× | 5.486 vs 4.847 = 1.13× |
| **合计** | 3.844 vs 4.374 = **0.88×**\* | 6.977 vs 6.401 = **1.09×** |

\* 前向的 0.58× 偏乐观：PyTorch 基线的 RoPE 是朴素实现。保守估计前向约 0.9×。
**前向+反向合计 ≈ 1.00×（与 PyTorch 持平）。**

> 唯一明显落后的是 **MoE 反向 1.13×**：我们逐专家发 8 次 GEMM，torch 用 batched `bmm` 一次发完。
> 补一个 grouped GEMM 即可追平。

---

## 三、过拟合实测：两份 SVG HTML 逐字复刻

语料：`_corpus/pelican_on_bike.html` + `bike_on_pelican.html`（24,651 字符，词表 78）

| 项 | 数值 |
|---|---|
| 模型 | char-level，2 层… **10.57 M 参数**，`d=512 / nh=16 / ff=1024`，上下文 **1024** |
| 训练 | 40,000 步，AdamW + OneCycle（max_lr 3e-3 → 1e-7） |
| 最终 loss | **0.0108** |
| 全语料 teacher-forced 准确率 | **99.65%** |
| `pelican_on_bike` 逐字复现 | **100.0%** |
| `bike_on_pelican` 逐字复现 | **100.0%** |

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
3. **MoE 反向未做 grouped GEMM**：比 torch 慢 13%。
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
| 打包 10.5M 鹈鹕模型 | **41.2 MB / 58 项权重** |
| `libtgai_engine_gpu.dll` 加载 | ✅ `L=4 d=512 nh=16 ff=1024 V=78 seq=1024 E=1 topk=1` |
| 引擎逐字符生成 | ✅ 跑通（文本乱码，因 tied 权重训练中断 —— 见 `docs/incident_gpu_context_20261027.md`） |

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

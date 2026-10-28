# TGAI GO 训练化二期 · 发布说明（v1.3.0）

> 全部数字为 V100-32GB / Windows / CUDA 12.4 实测；判分台源码同仓库，可复现。
> 推理侧（生产 DLL、11450）**零改动**。

## 五项判据

### (1) 速度：前向+反向 **0.85×**（比 PyTorch 快约 15%）

规模 d=1024 / ff=4096 / E=8 / seq=512，五轮取中位（`harness/bench_median.py`）：

| 段 | ours ms | torch ms | 比值 |
|---|---|---|---|
| attention 块 前向 | 0.614 | 1.251 | 0.49× |
| MoE FFN 前向 | 2.512 | 2.779 | 0.90× |
| **前向合计** | **3.069** | **4.350** | **0.71×** |
| attention 块 反向 | 1.51 | 1.57 | 0.96× |
| MoE FFN 反向 | 4.70 | 4.94 | 0.95× |
| **反向合计** | **6.093** | **6.473** | **0.94×** |
| **前向 + 反向** | **9.162** | **10.823** | **0.85×** |

单算子（`harness/op_bench.py`）：rmsnorm 0.25× / GEMM 1.00× / silu_mul 0.68× / moe_add 0.71× /
attention 前向 **1.02×**（我们额外产出反向所需的 P，torch SDPA 不物化）/ attention 反向 **0.97×**。

两条关键优化：
- **MoE grouped GEMM**：`cublasSgemmStridedBatched` 把 8 个专家的 GEMM 合成一次（1.14× → 0.95×）
- **flash attention 前向**：online softmax，不再把 P 与 softmax(P) 写进全局再读回（1.85× → 1.02×）

### (2) 鹈鹕过拟合：**100.0% / 100.0%**

tied char-LM，4 层 / d=512 / nh=16 / ff=1024 / **10.530 M 参数**（读 checkpoint 实测），语料 24,651 字符、词表 78。
20,000 步零 CUDA 错误，loss **4.2852 → 0.0470**；引擎逐字复现两份 HTML 各 **400/400**。

### (3) 直出 `.TG`：logits 相对误差 **3.908e-07**（判据 <1e-4）

真引擎 `libtgai_engine_gpu.dll` 加载并前向；58 项权重逐项与 checkpoint 一致（异常 0/38）。

### (4) BPE：引擎 `tg_encode` **逐 token 一致（5/5 片段）**

词表 **600**（78 base + 522 merges），压缩比 **5.66×**。

### (5) 收尾

- GitHub：仓库 **22 个提交** + 里程碑 tag **`v1.3.0-training`**；README 全部数字与实测对齐（并修掉旧版里 4 类错值）
- **PyPI：待上传** —— 产物已就绪（`python/dist/tgai_go-1.3.0-py3-none-any.whl` / `.tar.gz`，
  `twine check` 通过、元数据与已发布的 1.2.2 对等、隔离安装与打包往返冒烟测试通过），
  只差 `PYPI_TOKEN`：见 `python/tools/pypi_upload.py`

## 完整性

仓库 `training/kernels/` 里的 `.cu` 与我实测（判分台跑分）的版本**字节一致**：

| 文件 | sha256（前 16） |
|---|---|
| `tgai_bwd.cu` | `90ed05af167afd82` |
| `tgai_attn_flash.cu` | `b8239b27f26f3126` |
| `tgai_attn_nop.cu` | `ee15e0293fbd7cd1` |

## 一个被数据否掉的方案（记下来避免重复踩）

「前向不写 P + 反向重算」：前向确实降到 **0.86×**，但反向变成 **2.00×**
（无工作区、不能 grid sync，每个键块都要重算后续所有查询块）。
**结论：保留"前向写 P、反向读 P"，这是正确取舍。** 详见 `ACCEPTANCE.md` §17。

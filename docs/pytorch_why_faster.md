# PyTorch 为什么快：把它的底库拆开看

> 同一份 Python 代码（`exam2_chat10m_tgai.py` / `exam2_chat10m_torch.py`，**只差 SHIM 段的导入**），
> 在 V100 上：PyTorch **52.4 ms**，TGAI GO **62.6 ms**。
> 差距只能来自"同一条算子序列，两个后端各自做了什么"。本文全部是实测。

## 1. 先排除掉"大件"：GEMM 我们已经持平

同形状、都带 `sync` 的 A/B：

| 形状 (M,K,N) | TGAI (ms) | PyTorch (ms) | 比值 |
|---|---|---|---|
| 4992×256×256（qkv） | 0.0796 | 0.0763 | 1.04× |
| 4992×256×1280（FFN up） | 0.3017 | 0.2982 | 1.01× |
| 4992×1280×256（FFN down） | 0.3057 | 0.3050 | 1.00× |
| 1280×4992×256（dW） | 0.3193 | 0.2909 | 1.10× |
| 1992×4992×360（CE） | 0.7382 | 0.7283 | 1.01× |

⇒ **GEMM 不是问题**。我们和 PyTorch 都走 cuBLAS，差 0–10%。

## 2. PyTorch 那一步的 kernel 分布（torch.profiler 实测）

| kernel | 次数 | GPU 时间 |
|---|---|---|
| `aten::mm` | 203 | **28.62 ms** |
| `aten::mul` | 298 | 7.67 ms |
| `aten::copy_` | 92 | 2.66 ms |
| `aten::bmm` | 48 | 2.26 ms |
| `aten::reshape` | 330 | 1.86 ms |
| `aten::add_` | 82 | 1.60 ms |
| `aten::_local_scalar_dense` | 149 | 1.58 ms |
| `aten::pow` | 51 | 1.41 ms |
| `Optimizer.step#AdamW.step` | **1** | 1.24 ms |
| `aten::add` | 49 | 1.17 ms |
| `aten::transpose` | 235 | 1.03 ms |

**关键观察：PyTorch 的 kernel 数量（≈900 次）比我们（484 次）还多。**
所以它不是靠"少发 kernel"赢的，而是**每个 kernel 更便宜**，以及**有些操作根本不产生 kernel**。

## 3. 逐项对比：差距在哪

| 项 | PyTorch | TGAI | 差距来源 |
|---|---|---|---|
| GEMM | 28.6 ms（`mm` 203 + `bmm` 48） | ~31 ms（253 次 cuBLAS） | 基本持平（A/B 1.00–1.10×） |
| 逐元素吞吐 | `mul` 26 µs/次 | 我们 **1.05–1.48× 慢**（见下） | **内核未向量化** |
| `reshape` / `transpose` | 565 次，共 2.9 ms —— **视图，不拷贝** | `reshape` 是视图 ✓，但 `permute8` 64 次 + `swap_last2` 64 次 + `broadcast_to8` 32 次 **共 160 次真拷贝** | 布局策略 |
| 优化器 | `AdamW.step` **1 次** 1.24 ms（foreach 多张量） | `adamw_update` **74 次** 1.33 ms | foreach 封装 |
| 宿主同步 | `_local_scalar_dense` 149 次 1.58 ms | 我们只有 CE 反向 1 次 | 我们更少 ✓ |

### 逐元素 A/B（同形状、带 sync）

| 元素数 / 算子 | TGAI | PyTorch | 比值 |
|---|---|---|---|
| 1.28M / mul | 0.0328 ms | 0.0238 ms | **1.37×** |
| 1.28M / add | 0.0269 | 0.0224 | 1.20× |
| 1.28M / exp | 0.0237 | 0.0159 | **1.48×** |
| 6.39M / mul | 0.1042 | 0.0989 | 1.05× |
| 2.56M / mul | 0.0480 | 0.0413 | 1.16× |

小张量差距更大 ⇒ 主要是**每线程只处理 1 个 float**（PyTorch 的 TensorIterator 是向量化 + 循环展开）。

## 4. 结论：剩下的 10.2 ms 拆得开

| 来源 | 估计 |
|---|---|
| 逐元素内核未向量化（1.05–1.48×） | **4–6 ms** |
| `permute8`/`swap_last2`/`broadcast_to8` 的 160 次真拷贝 | **2–3 ms** |
| 逐算子派发（484 次 × ~17 µs，PyTorch 约 5–8 µs） | 2–3 ms |
| 优化器 74 次 vs 1 次 | ~1 ms |

**这就是"为什么 PyTorch 快"的全部答案**：不是黑魔法，是三件工程细节 ——
① 内核向量化 ② 视图语义（少拷贝） ③ 多张量融合算子（foreach）。

## 5. 对应的修法（按收益排序）

1. **给逐元素内核加 float4 路径**（`ewise_mul`/`add`/`scale`/`exp`/`bcast_add4`/`fused_rope`）：
   元素数 %4==0 且指针 16B 对齐时走向量化，否则回退标量。**逐元素运算逐位不变**（每个元素独立）。
   预期回收 4–6 ms → ~57 ms。
2. **消灭 `swap_last2` 的拷贝**：`q @ k.swap_last2()` 改用后端已有的 `gemm_nt`（cuBLAS op=T），
   省 64 次转置拷贝（0.85 ms + GPU 时间）。
3. **`permute8` 布局改造**：让 q/k/v 的 GEMM 直接产出 `[BH,S,DH]`（把权重按 head 重排，一次性离线做），
   省 64 次拷贝。
4. **优化器 foreach**：把 74 次 `adamw_update` 合成 1 次（需要多指针内核，收益 ~1 ms，优先级最低）。

做完 1–3，预计 **62.6 → ~55 ms**，差距 1.19× → **1.05×**。

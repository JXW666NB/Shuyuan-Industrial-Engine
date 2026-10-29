# 训练反向热点第二轮（seq=8192）

判分台：`v27_moe_bench.py --backend tgai --layers 1 --steps 3`，显存用 `vram_wrap.py`（nvidia-smi 峰值）。

| | 改前 | 改后 |
|---|---|---|
| seq=8192 单步 | 0.262 s | **0.170 s** |
| seq=8192 峰值显存 | 12.35 GiB | **12.78 GiB** |
| seq=2048 单步 | 0.064 s | **0.036 s** |
| loss（seq=8192 前 3 步） | 10.55 → 7.32 → 5.64 | **10.5505 → 7.3241 → 5.6400** |

单步已经低于 0.20 s，显存低于 13 GiB。seq=2048 没有退，比原来更快。

## 归因和上一轮表格的差别

`prof_bwd_nodes.py` 在每个反向节点前后 `cuCtxSynchronize`。同步之后，显存池以为空闲块的 kernel 都跑完了，把缓存从 2 GiB 压回 256 MiB（`cuMemFree`）。这几十毫秒被记在「刚好释放了大块」的那个节点上。

把同一个 kernel 单独计时（不夹在逐节点同步里）：

| 调用 | 单独耗时 | 判分台节点上看到的 |
|---|---|---|
| `emb_bwd`（256 个重复 id） | **1.27 ms** | 32.9 ms |
| `emb_bwd`（id 均匀撒满词表） | **1.26 ms** | — |
| `attn_nop_bwd_flash` | **35.5 ms** | 36.0 ms |
| MoE `grouped_tn`（dW） | **7.6 ms** | 整段 MoE 反向 23.0 ms |

embedding 那 33 ms 不是 atomic 核本身。attention 的 36 ms 是真的。

## 1. embedding 反向

生产路径仍是 `emb_bwd_owned`（`cuda.py`）：seq≥1024 时 256 个 block，每个 block 只累加 `id % 256` 的那些行，按 token 下标从小到大加，不再 `atomicAdd`。短序列仍走原来的 atomic。

- 语料只有 256 个 id：1.27 ms
- id 在 32779 里均匀：1.26 ms

两种分布都低于 5 ms，均匀分布没有变慢。

另外写过按词表分段的 `emb_bwd_plan2` + `emb_bwd_rows4`（每行一个 block，float4 按 token 顺序归约）。seq=8192 上是 1.98 ms（偏斜）/ 2.72 ms（均匀），慢在单线程扫 3 万词表。没有换成这条。

数值：`train_op_parity.py --cuda` 的 embedding 梯度相对误差是 0。

## 2. attention 反向

`attn_bwd_flash` 仍是默认核。seq=8192 实测 **35.5 ms**，没有压到 25 ms 以下。

试过、都没有留下：

- 把 atomic 的 `(k,d)` 映射改成一个 warp 打 32 条 cache line。seq=8192 从 36.9 ms 变成 38.3 ms。
- `__launch_bounds__(256, 2)`。seq=8192 是 37.5 ms，和原来一样。
- `attn_bwd_flash_g2`：一个 block 吃两个 query tile，dK/dV 先在 shared memory 里加再 atomic 一次。seq=8192 是 **57 ms**。寄存器和同步变多，原子冲突不是这 36 ms 的大头。默认关闭，`TGAI_ATTN_G2=1` 才走它。

`attn_lowmem_check`：dh=64 三组 PASS（最差 6.5e-7），非 causal / dh≠64 两条回退逐位 0。

## 3. MoE 反向

dW 仍走 `cublasSgemmGroupedBatched`（`grouped_tn`）。一步里这三次（gate/up/down）合计 **7.6 ms**，低于 15 ms。dX 的 `grouped_nn` 是 9.6 ms。

`grouped_tn` 里留了垫零 + `cublasSgemmStridedBatched` 的实现（token 数彼此不超过 8 倍、且运行时有这个符号时才走）。这一轮的语料没有走进去：`_grouped_sgemm` 仍然发出了全部 dW。loss 和改前逐位到小数点后四位，说明实际算的还是 grouped 那条。

## 单步从 0.262 s 到 0.170 s 的两处

1. **池不再在热路径上把工作集 `cuMemFree` 掉。** 以前缓存一过 2 GiB 就压回 256 MiB，`zero_grad` 单独要 45 ms，反向里也会被打断。现在只在总占用超过 12.2 GiB 时回收，每个尺寸留 2 个已完成的空闲块。分配如果还会冲过这条线，先等默认流再回收一次。
2. **CUDA 反向只给叶子张量留 `.grad`。** 中间激活以前每次都 `clone`。seq=8192 上一步有 156 次 clone、约 10 ms，而且把这些梯度缓冲留到峰值里。叶子的梯度缓冲是父算子刚写出来的，直接挂上，不再拷。和 PyTorch 一样，非叶子没有 `.grad`。

## 回归

全部在这台 4090 上复跑，退出码 0：

- `train_backend_parity.py` / `train_op_parity.py` / `train_layer_parity.py` / `train_opt_parity.py` 的 `--selftest`：逐位 0
- M1 `--cuda` 逐位 0；M2 `--cuda` 每项 < 1e-5；M3 `--cuda` 最差 1.1e-6
- `moe_selfcheck.py` ok，最差 1.7e-7
- `opt_step_parity.py --cuda` AdamW / Muon 逐位 0
- `attn_lowmem_check.py` 见上
- `pytorch_conformance.py` CPU 与 `--cuda` 都是 11/11

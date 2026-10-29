# A3：训练分配器峰值

口径是 `CudaRuntime.pool_stats()['peak_bytes']`，不是 nvidia-smi。
PyTorch 对照是同一脚本 `--backend torch` 的 `torch.cuda.max_memory_allocated()`，本轮复测与任务书一致。

改动只在 `python/tgai_go/train/cuda.py` 与 `python/tgai_go/train/__init__.py`。
改前备份：`docs/_backup_a3mem_20261029_124353/`。

## 结果

| seq | 改前 | 改后 | PyTorch | 改后比值 | 目标 |
|---|---|---|---|---|---|
| 2048 | 4.177 GiB | **3.386 GiB** | 2.966 GiB | 1.14× | ≤ 3.56 GiB（1.2×） |
| 8192 | 9.811 GiB | **6.650 GiB** | 7.101 GiB | 0.94× | ≤ 8.5 GiB |

live 高水位（`peak_live_bytes`）：2048 为 2.916 GiB，8192 为 5.613 GiB。
分配器峰值高于 live，差的是池里尚未 `cuMemFree` 的空闲块。

速度（`v27_moe_bench.py --layers 1 --steps 3`，编译不计入）：

| seq | 改前 | 改后 | loss |
|---|---|---|---|
| 2048 | 0.036 s/step | **0.035 s/step** | 10.5066 → 6.7539 → 5.4846 |
| 8192 | 约 0.158 s/step | **0.143 s/step** | 10.5505 → 7.3241 → 5.6400 |

两条 loss 与任务书逐位相同。没有走已否定的分块 CE。

## 改了什么

### 1. CE 反向原地写进 logits（省掉第二份 `[N,V]`）

`ce_bwd` 先把一整行的 max / sum(exp) 归约完，写的时候每个线程只读自己要覆盖的那个元素，所以输出可以和 logits 同址，数值与另开缓冲逐位相同。

只在同时满足时原地：

- 缓冲是池子的主人（不是 reshape 视图，否则会改掉共享的上游激活）
- `N*V ≥ 2^20`（约 4 MiB 以上）

小词表保持原样。`llama_parity` 在 `backward()` 之后还要读 logits，它的 `V=64, seq=16` 不走原地，前向值还在。训练步里的 loss 标量在覆盖之前已经写进独立的 loss 缓冲，所以打印的 loss 仍是前向值。

根梯度不是 1 时，用 `scale_inplace` 写回同一块，不再 `scale()` 出第三份。

这一项去掉的是任务书里 `:2275` 那份约 1024 MiB 的 dlogits。改后的归因名单里没有它。

### 2. logits GEMM 的反向

`:809` 的 `sgemm`（dHidden）和 `:811` 的 `gemm_tn`（dW）写出来的就是要留下的梯度，不是可以再删的临时量。真正多出来的那一份是 dlogits，已由上一条原地化吃掉。dW 在 `.grad is None` 时直接挂上叶子，没有再 clone。

改后 seq=8192 上这两行合计约 192 + 144 MiB（含注意力那几次同形状 GEMM，不只是词表那一次）。

### 3. reshape 零拷贝

本运行时的缓冲都是打包的 C 连续存储。`CudaRuntime.reshape` 以前每次 `cuMemcpyDtoD`。现在只新建一个 `GpuBuf` 视图：同一 `ptr`，`_token` 为空所以析构不归还池，`keepalive` 拉住主人。元素个数必须相等。

`permute` 仍会拷贝成新的连续缓冲；对它再 reshape 才是视图。seq=8192 上任务书记的 `:1059` / `:1064` 各约 512 MiB 拷贝已从 Top 名单消失。

### 4. MoE 中间量

- `silu(x)` 与分组路径里的 `silu(gate)` 不再分配一份全 1 再做 `silu_mul`。新 kernel `silu_fwd_inplace` / `silu_bwd_da` 与 `b==1` 的旧公式在有限数上逐位相同，并且不再分配用不到的 `db`。
- 分组反向用完 gate / up / hidden / rows 后立刻丢掉引用。默认流上后续 kernel 排在读它们的 kernel 之后，池可以复用，不必挂到整步结束。
- `grouped_tn` 的 strided 垫片（`Ap/Bp/Cp`）不再挂在梯度的 `keepalive` 上活过整段反传。主机侧的 alpha/beta 仍留着。

没有把 `grouped_*` 改成调用方预分配。那条路要改每一处形状，而上面三项已经把峰值打到目标以下。分块 logits+CE 仍默认关闭。

## 改后 Top（`prof_alloc_site.py 8192`）

该脚本按调用行累加本次 step 的分配字节，不扣减已归还的块，所以名次是「这一行一共申请了多少」，不是某一瞬间的同时占用。它报的合计 5.66 GiB，与 live 高水位 5.613 GiB 同量级。

| 调用点 | MiB | 是什么 |
|---|---|---|
| `__init__.py:801` `gemm_nt` | 1152.6 | 词表 logits 加上 Q/K/V/O 的前向 GEMM |
| `__init__.py:1303` `_acc` 里的 `add` | 288.0 | 梯度累加 |
| `__init__.py:2752` `grouped_nt` gate | 256.0 | MoE 前向 |
| `__init__.py:2753` `grouped_nt` up | 256.0 | MoE 前向 |
| `__init__.py:2756` `clone` + silu | 256.0 | 不再附带全 1 |
| `__init__.py:2759` `mul` hidden | 256.0 | silu * up |
| `__init__.py:2788` `grouped_nn` | 256.0 | dHidden |
| `__init__.py:2792` / `:2793` `mul` | 各 256.0 | dSilu、dUp |
| `__init__.py:2797` `silu_bwd_da` | 256.0 | 只有 da，没有 db |
| `__init__.py:809` `sgemm` | 192.0 | dHidden（含非词表 GEMM） |
| `__init__.py:811` `gemm_tn` | 144.1 | dW |

## 回归

全部在远端当前代码上复跑。

- `pytorch_conformance` CPU 与 `--cuda`：各 **13/13**
- `--selftest` 相对差均为 **0**：`train_op_parity`、`train_backend_parity`、`train_layer_parity`、`train_opt_parity`、`llama_parity`、`opt_step_parity`
- `moe_selfcheck`：`VERDICT ok=True`（CUDA 最差 1.72e-7）。CPU 有限差分的权重来自未播种的全局 RNG，偶发会贴着 1e-4，单独再跑一次为 PASS，与本次 CUDA 改动无关
- `opt_step_parity --cuda`：AdamW / Muon 相对差 **0**
- `attn_lowmem_check`：五条全 PASS（非 causal / dh≠64 为逐位 0）
- `llama_parity.py --cuda`：logits 8.7e-6，梯度最差 9.2e-6
- `llama_parity.py --c-forward`：logits 8.7e-6

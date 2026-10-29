# 任务：把训练显存降到 PyTorch 的 1.2 倍以内（A3）

> 这份文件是任务书全文。prompt 里只给入口，细节看这里。

## 0. 纪律
- 远端：单卡 RTX 4090，通过 `feat.py` 访问（本地工作目录 `D:\DSH-Workspace\TGAI-对话档案`）：
  ```
  python feat.py sh "命令"
  python feat.py put <本地> <远端绝对路径>
  python feat.py get <远端> <本地>
  ```
- **不改** SSH key / authorized_keys / 密码 / sudo（别人公司的机器）。
- 单次 GPU 任务 ≤10 分钟。
- 包：`/home/featurize/tgai/python/tgai_go/`；kernel：`/home/featurize/tgai/training/kernels/`；
  判分台：`/home/featurize/tgai/training/harness/`；探针：`/home/featurize/tgai/*.py`。
- 本地仓库：`_repos/Shuyuan-Industrial-Engine/`（改动放这里，再 put 到远端验证）。

## 1. 判据口径（别用 nvidia-smi）
A3 用**分配器高水位**比：
- 我们：`CudaRuntime.pool_stats()['peak_bytes']`（脚本 `training/harness/a3_mem_parity.py`）
- PyTorch：`torch.cuda.max_memory_allocated()`

当前：

| seq | 我们 | PyTorch | 比值 |
|---|---|---|---|
| 2048 | 4.177 GiB | 2.966 GiB | 1.41× |
| 8192 | 9.811 GiB | 7.101 GiB | 1.38× |

**目标**：2048 ≤ **3.56 GiB**、8192 ≤ **8.5 GiB**。

速度**不许退化**：seq=2048 **0.036 s**、seq=8192 约 **0.158 s**。

## 2. 分配点归因（探针 `prof_alloc_site.py`，峰值 8.46 GiB 活张量）

| `python/tgai_go/train/__init__.py` 行 | 占用 | 是什么 |
|---|---|---|
| `:801` `buf = rt.gemm_nt(...)` | **1152 MiB** | 前向 logits `[8192,32779]`，被 CE 反向一直占着 |
| `:2275` `dl = rt.ce_bwd(...)` | **1024 MiB** | dlogits，又一份 1 GiB |
| `:1059` / `:1064` `rt.reshape` | 各 512 MiB | reshape 与其反向的拷贝 |
| `:2786` `rt.silu_mul_bwd` | 512 MiB | MoE 反向 |
| `:2745/2746/2749/2750/2754/2779` `grouped_*` | 各 256 MiB | MoE 分组 GEMM 中间量 |

## 3. 已验证的负优化（不要重复走）
"分块 logits+CE"（`F.linear_cross_entropy`，按 token 分块）**实测峰值反而升到约 11.7 GiB**
（切片、反向散播、更多尺寸桶）。bench 的 `--chunk` 默认已改回 0。该函数保留但默认不用。

## 4. 要做的事（按收益排序）

### ① CE 反向原地化（省约 1 GiB）
`rt.ce_bwd` 目前新分配 `[N,V]` 的 dlogits。loss 算完后 logits 没有其他读者，
**dlogits 可以直接写进 logits 的缓冲**。先确认无第二处读取，再原地化。

### ② logits matmul 的 backward 少一份（`:811`）
`dw = gemm_tn(...)` 与 `dh = sgemm(...)` 至少一份是可省的中间量；
考虑复用缓冲或直接累加进 `weight.grad`。

### ③ reshape 零拷贝（`:1059/:1064` 各 512 MiB）
内存连续时 `rt.reshape` 应只改 shape 元数据、不分配新缓冲；
仅在确实不连续时拷贝。反向同样不该分配。

### ④ MoE 分组 GEMM 中间量（约 1.5 GiB）
让 `grouped_nt/grouped_nn` 写进调用方给的缓冲，或用完立刻归还池。

## 5. 验收（自己跑，数字贴报告）
```
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/a3_mem_parity.py --backend tgai --seq 2048"
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/a3_mem_parity.py --backend tgai --seq 8192"
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/v27_moe_bench.py --backend tgai --layers 1 --steps 3 --seq 2048"
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/v27_moe_bench.py --backend tgai --layers 1 --steps 3 --seq 8192"
```

## 6. 不许回归（主控每次会自己复跑）
- `pytorch_conformance` CPU 与 CUDA 均 **13/13**
- 四 `--selftest` 全 0；`moe_selfcheck` ok；`opt_step_parity --cuda` 逐位 0；`attn_lowmem_check` 全 PASS
- `llama_parity.py --cuda` 与 `--c-forward` 仍绿
- loss 曲线：8192 前 3 步 10.5505 → 7.3241 → 5.6400；2048 10.5066 → 6.7539 → 5.4846

## 7. 交付
- 代码改在本地仓库
- 报告 `docs/train_cuda_memory_a3_sites.md`：每项改法、前后峰值、速度、最后再跑一次 `prof_alloc_site.py` 的 Top 名单
- 回话最后 6-8 行：两个 seq 峰值前后、速度前后、回归是否全过

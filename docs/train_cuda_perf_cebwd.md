# 单步 19.9s → 0.15s：串行 CE 不是「拷贝慢」

测量机：单卡 RTX 4090，`/home/featurize/tgai`。宽度 d=1024 / nh=16 / ff=4096 / V=32779 / seq=2048 / 1 层。改前改后用同一份 `_prof_v27.py`。

## 根因（自己跑出来的，不是照抄）

`ce_fwd` / `ce_bwd` 都是 `grid=(1,1,1)`、`block=(1,1,1)`。一个线程用 `exp32` 把 2048×32779 扫完。前向计时只有 0.004s，因为 kernel 只是入队，没有同步。

改前反向 cProfile（`backward` 墙钟 **19.894s**）：

| 调用 | 时间 | 实际在等什么 |
|---|---|---|
| `full` → `h2d` | 6.746s | 同步 H2D 把默认流排空。流里主要是还没人等过的 **`ce_fwd`** |
| `bw` → `read_scalar` → `d2h` | 13.051s | `ce_bwd` 刚入队，`cuCtxSynchronize` 把 **`ce_bwd`** 等完 |
| 其余 4 次 `h2d` | 合计约 0.03s | 队列已经空了，不是每次 1.34s |

5 次 `h2d` 的 tottime 6.774s 被平均成「每次 1.34s」是错的。真正慢的是两条单线程 CE。只改 `ce_bwd` 的话，`ce_fwd` 仍会在反向开头的第一次同步拷贝里露出约 6.7s，单步过不了 1s。

`read_scalar(g)` 读的是 `backward()` 用 `full((), 1.0)` 填好的根梯度，数值在主机上已经知道，却排在 `ce_bwd` 之后做全设备同步。

## 改法

都在 `python/tgai_go/train/cuda.py` 和 `train/__init__.py`。`exp32` / `log32` / `__fadd_rn` 没换。

1. **一行一个 block、256 线程。** 每线程扫一段连续列，块内再由 thread 0 按线程号左折叠。V≤256 时每段至多一个元素，和原来的单线程左折叠逐位相同（M1–M4 的 V 都 ≤8）。`ce_fwd` 拆成 `ce_fwd_rows` + `ce_fwd_mean`。
2. **标量 1 不再回读。** `full` / 单元素 `h2d` 记下 `known_scalar`。`read_scalar` 命中就直接返回。CE 的 `bw` 在发射 `ce_bwd` 之前取 scale，避免把 `ce_bwd` 自己等进去。真要回读时，`d2h` 改为 `cuMemcpyDtoHAsync` + `cuStreamSynchronize(NULL)`，不再先 `cuCtxSynchronize`。
3. **`h2d` 改异步**（`cuMemcpyHtoDAsync`，主机缓冲挂在 `GpuBuf.host_hold` 上）。CE 的 target 和 Embedding 下标按内容缓存，同一步里的重复小拷贝不再上传。
4. **`full` / `zeros` 用 `cuMemsetD32Async`。** 不再 `np.full` + 整块 H2D。Embedding 反向那张 V×D 的零梯度走这条。

## 分段计时

`_prof_v27.py` 的数字是 CPU 侧包住异步发射的时间。改前 GPU 时间几乎全部堆进反向里那两次同步。

| 段 | 改前 | 改后 |
|---|---:|---:|
| embedding | 0.000s | 0.000s |
| rmsnorm | 0.000s | 0.000s |
| qkv Linear | 0.002s | 0.002s |
| reshape+permute | 0.001s | 0.001s |
| SDPA | 0.015s | 0.015s |
| attn out | 0.001s | 0.001s |
| FFN | 0.008s | 0.002s |
| logits | 0.000s | 0.000s |
| cross_entropy 发射 | 0.004s | 0.000s |
| **backward** | **19.894s** | **0.081s** |
| AdamW.step | 0.052s | 0.002s |
| 分段合计 | 19.98s | 0.11s |

端到端以 `v27_bench.py --backend tgai --layers 1 --steps 3 --seq 2048` 为准（每步 `loss.to_numpy()` 会同步）：

- 改前：这一次剖面的单步是 **19.9s**（任务里记的是 19.7s，同一量级）
- 改后：**0.15 s/step**（13775 tokens/s）
- loss：**10.5874 → 6.6718 → 5.3792**（10.58 → 6.67 → 5.38 那一串）

只计 kernel、不算 268MB 的 `cuMemAlloc`（预热后 5 次，`cuStreamSynchronize`）：

| kernel | 5 次 (ms) | 最快 |
|---|---|---:|
| `ce_bwd` | 3.739, 3.806, 3.788, 3.782, 3.793 | **3.74ms** |
| `ce_fwd`（rows+mean） | 1.545, 1.480, 1.484, 1.480, 1.480 | **1.48ms** |

N=16、V=8 时 loss 和 dlogits 与 CPU 的相对误差都是 **0**（V≤256 的左折叠）。

## 回归

| 检查 | 结果 |
|---|---|
| `pytorch_conformance.py` | CPU 11/11，CUDA 11/11 |
| 四个 `--selftest` | backend / op / layer / opt 全部恰好为 0 |
| M1 `--cuda` | ok=True，loss/W/dW rel=0 |
| M2 `--cuda` | ok=True，最差 2.06e-6（判据 1e-5） |
| M3 `--cuda` | ok=True，最差 1.10e-6（判据 1e-4） |
| M4 `--cuda` | **改前改后都是 ok=False**，同一个数：Muon `lin2` rel=6.995e-7（判据 1e-7）。AdamW 仍 PASS（9.06e-8）。不是这次 CE 改动带出来的 |
| `test_train_api.py` | **A 改前改后都是 1.083e-4，卡在 <1e-4 外面**。A 只走 CPU 的 silu/matmul，这次 diff 没动那条路径。B–H 全 PASS |

改前对照用的是 `docs/_backup_cebwd_20261028/` 里改前的 `cuda.py` / `__init__.py`，传回同一台机器重跑 M4 和 `test_train_api.py`。

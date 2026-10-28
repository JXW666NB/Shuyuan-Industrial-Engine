# seq=8192 不再物化 P

测量机：单卡 RTX 4090，47.4 GB，`/home/featurize/tgai`。模型仍是 V2.7 那档：d=1024 / nh=16 / dh=64 / ff=4096 / E=8 top-2 / V=32779 / 1 层。

改前（本次任务给出的实测）：seq=2048 我们 0.218 s/step、峰值 10.11 GiB；seq=8192 在吃满 46.17 GiB 后 `cuMemAlloc rc=2`。PyTorch 同一档 seq=8192 是 0.306 s/step、6.06 GB。根因是 SDPA 把 `P` 写成 `[BH, S, S]`，16 头、seq=8192 一份就约 4.3 GB，scores / mask / softmax 再各留一份，反向还要把它们留到反传。

## 接口

默认路径没动。`functional.scaled_dot_product_attention(q, k, v, ...)` 的 `lowmem` 默认 `False`，仍是原来的 `bmm + 下三角 -1e30 + softmax`。

低显存入口有两个，语义相同：

- `functional.scaled_dot_product_attention(q, k, v, causal=True, lowmem=True)`
- `functional.sdpa_lowmem(q, k, v, causal=True, scale=None)`

`q,k,v` 仍是训练里的 `[B*H, S, Dh]`。`scale` 省略时是 `1/sqrt(Dh)`。

只有下面几条同时成立才进内核，否则回退到物化实现，调用方不用改形状：

- 设备是 CUDA
- `causal=True`（内核只有因果；非因果不能假装）
- `Dh == 64`
- 三维，且 q/k/v 同形状，`B*H > 0`、`S > 0`

`functional.sdpa_lowmem_ready()` 只负责把 `training/kernels/tgai_attn_nop.cu` 编进第二份 NVRTC 模块。计时前调用，避免把编译算进 s/step。默认 SDPA 不会加载这份模块，主模块的源码和编译参数也没加这颗内核。

`v27_moe_bench.py` 的 tgai 后端默认 `--attn lowmem`。`--attn legacy` 切回物化路径。验收命令没有新参数，因此走的就是低显存路径。

## 布局和启动

内核要的是 `[seq, nh, dh]`，seq 主序。训练张量是 `[BH, S, Dh]`，head 主序。B=1 时 `BH` 就是 `H`。

前向做一次显式 `permute(1, 0, 2)` 得到 `[S, BH, Dh]`，调用 `attn_fwd_nop`，再 `permute(1, 0, 2)` 回到 `[BH, S, Dh]`。这两次都是运行时的 permute 内核，不接进 autograd，避免和下面的自定义 VJP 各算一遍。

反向不读 P。上游梯度同样 permute 成 `[S, BH, Dh]`，调用 `attn_bwd_recompute(dO, Q, K, V, dQ, dK, dV, ...)`，三个梯度再 permute 回去。Q/K/V 用的是前向那份已经换成 seq 主序的副本。

B>1 时各头本来就不互相看，把 `BH` 当成内核的 `nh` 是同一件事。验收只用了 B=1。

启动和内核注释一致：

- block = 256
- dynamic shared memory = 32 KB
- grid = `(nh, ceil(S/64), 1)`
- S 能被 64 整除时第二维就是 `S//64`。seq=128 / 2048 / 8192 都整除。

`Dh != 64` 时内核自己会直接 return，所以 Python 侧在发射前就回退，不把未写完的 O 交出去。

## 数值

`/tmp/sdpa_lowmem_check.py`，dh=64、seq=128、nh=4、B=1，同一份 `RandomState(11)` 输入。相对误差是 `max|a-b| / (max|b| + 1e-12)`。

对现有 CUDA 物化实现：

```
vs materialized CUDA  seq=128 nh=4 dh=64
  O rel=1.010e-07 maxabs=1.788e-07
  dQ rel=3.227e-07 maxabs=1.490e-07
  dK rel=6.279e-07 maxabs=3.576e-07
  dV rel=3.408e-07 maxabs=1.431e-06
  worst=6.279e-07
```

对 CPU 物化实现：

```
vs CPU materialized
  O rel=8.421e-08 maxabs=1.490e-07
  dQ rel=2.904e-07 maxabs=1.341e-07
  dK rel=3.401e-07 maxabs=1.937e-07
  dV rel=2.840e-07 maxabs=1.192e-06
  worst=3.401e-07
```

两条都小于 1e-4。

回退是逐位相同的，不是“差不多”：

```
dh=8 lowmem fallback vs legacy
  O/dQ/dK/dV rel=0
non-causal lowmem fallback vs legacy
  O/dQ/dK/dV rel=0
```

## seq=8192

命令就是验收那条（没有额外开关）：

```
cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python timeout 900 python3 -u vram_wrap.py python3 -u training/harness/v27_moe_bench.py --backend tgai --layers 1 --steps 1 --seq 8192
```

第一次：

```
配置: ... seq=8192 ... attn=lowmem
参数量: 138.4 M
loss: 10.5505
吞吐: 8925.3 tokens/s（0.918 s/step）
采样峰值显存: 14.45 GiB（nvidia-smi 轮询，间隔 100 ms）
```

紧接着的复跑（同一条，UTF-8 日志）：

```
配置: d=1024 nh=16 ff=4096 E=8 top2 V=32779 seq=8192 layers=1 steps=1（MoE） attn=lowmem
参数量: 138.4 M
loss: 10.5505
吞吐: 8873.6 tokens/s（0.923 s/step）
采样峰值显存: 14.17 GiB（nvidia-smi 轮询，间隔 100 ms）
exit=0
```

没有 OOM。两次峰值都低于 20 GiB。loss 同为 10.5505。

## 默认路径和回归

seq=2048、`--attn legacy`、1 步，物化路径仍是库的默认：

```
配置: ... seq=2048 ... attn=legacy
loss: 10.5066
吞吐: 13272.9 tokens/s（0.154 s/step）
采样峰值显存: 7.72 GiB
exit=0
```

改前记录是 0.218 s/step。0.154 s 没有变慢，更没有超过 +20%（0.262 s 那条线）。

同一轮后面接着跑的回归：

| 检查 | 结果 |
|---|---|
| `pytorch_conformance.py` | CPU 11/11，`VERDICT ok=True` |
| `pytorch_conformance.py --cuda` | 11/11，`VERDICT ok=True` |
| 四个 `--selftest` | backend / op / layer / opt 全部恰好为 0，`ok=True` |
| M1 `--cuda` | `ok=True`，loss/W/dW rel=0 |
| M2 `--cuda` | `ok=True`，最差 2.06e-6（判据 1e-5） |
| M3 `--cuda` | `ok=True`，最差 1.099e-6（判据 1e-4） |
| M4 `--cuda` | AdamW PASS（9.055e-8）。Muon `lin2` rel=6.995e-07，`VERDICT ok=False`（判据 1e-7） |
| `moe_selfcheck.py` | `VERDICT ok=True`，worst_cuda=1.716e-07 |

M4 的 Muon 不是这次带出来的。`docs/train_cuda_perf_cebwd.md` 里改 CE 之前就是这一个数：`lin2` rel=6.995e-07，AdamW 9.06e-8。`train_opt_parity` 只走 Linear，不经过 SDPA，这次也没改 Muon 内核。

小规模最差相对误差 6.279e-07（对现有 CUDA；对 CPU 是 3.401e-07），低于 1e-4。
seq=8192 低显存路径跑通：峰值 14.45 GiB / 14.17 GiB，0.918 s/step / 0.923 s/step，loss=10.5505。
默认物化路径 seq=2048 单步 0.154 s，比改前的 0.218 s 更快，没有变慢。
回归：一致性 CPU/CUDA 均 11/11；四个 selftest 全 0；M1/M2/M3 `--cuda` ok=True；moe_selfcheck ok=True。
M4 `--cuda` 仍是改前那个 Muon lin2=6.995e-07（ok=False），AdamW 仍 PASS，不是这条 attention 改动。

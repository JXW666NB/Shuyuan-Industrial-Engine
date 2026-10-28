# 长上下文 attention 反向：从整行重算改成 flash 单次扫描

测量机：单卡 RTX 4090，47.4 GB，`/home/featurize/tgai`。模型仍是 V2.7 那档：d=1024 / nh=16 / dh=64 / ff=4096 / E=8 top-2 / V=32779 / 1 层。低显存路径仍是 `functional.scaled_dot_product_attention(..., causal=True, lowmem=True)`。dh≠64、非因果、CPU 不进这颗内核。

改前（本次任务给出的实测，lowmem）：

| 配置 | 我们 | PyTorch | 差距 |
|---|---|---|---|
| seq=2048，1 层 | 0.116 s | 0.104 s | 1.12× |
| seq=8192，1 层 | 1.037 s | 0.189 s | 5.5× |
| seq=8192 峰值显存 | 14.17 GiB | 7.10 GB | 2.0× |

## 旧反向为什么是三次方

`attn_bwd_recompute` 里每个 block 同时拥有一块 query（BQ=64）和同下标的 key。`bwd_dq` 对这块 query 把 K/V 扫两遍（一遍统计量，一遍 dQ）。`bwd_dkv_one` 再对**每一个更靠后的 query 块**调用 `sweep_stats`，把那一行从 k=0 到该行末尾的 softmax **整行重算**，只为了拿出本 block 那 64 个 key 上的 P。

query 块数 N = seq/64。每个 key 块 i 对后面每个 query 块都做 O(i) 次 K 块加载，总加载量是 Σ j²，也就是 O(N³)。seq=8192 时 N=128，相对「每个 (Q,K) 块对只碰一次」大约多出几十倍。seq 越大，多出来的就是这一截。P 本身没有落全局，浪费在重算。

## 新反向

低显存训练路径改走两个新入口，旧的 `attn_fwd_nop` / `attn_bwd_recompute` 还在同一个 `.cu` 里，直接按旧签名发射时行为不变。

### 前向顺手记下 m、l

`attn_fwd_stats` 的 QK、在线 softmax、写 O 与 `attn_fwd_nop` 是同一段 `fa_tile`（Bc=32，Q 常驻，下一块 K/V 预取）。每个有效行在结束时多写两个 float：

- `m[seq, nh]`：该行因果分数的最终 max
- `l[seq, nh]`：`sum exp(s - m)`

seq=8192、nh=16 时这两块一共 1 MiB。O 仍是前向输出，反向用它算 D，不再重算 O。重建概率用和前向同一个公式：

```
P = exp(scale * Q Kᵀ - m) / l    （因果掩码外、越界处为 0）
```

点积的 fma 顺序与 `fa_tile` 一致（d 从 0 到 63，float4 取 Q 的 8 行），所以 P 和前向用来归一化 O 的那份是同一套在线 softmax，不是另写一套。

### 反向：每个 Q 块只扫一遍 K/V

`attn_bwd_flash`，grid 仍是 `(nh, ceil(seq/64), 1)`，block 256。一个 block 拥有 64 行 query：

1. Q、dO 各加载一次，转置进 shared memory（行距 72，减轻后面 dK/dV 的 bank conflict）。
2. `D = rowsum(dO * O)`。每个 lane 拿 dim `lane` 和 `lane+32`，warp 归约。越界行 D=0。
3. 对因果范围内每个 K/V 块（BK=32）只做一次：
   - `S = scale * Q Kᵀ` 与 `dP = dO Vᵀ` 合在同一个 d 循环里（两套累加器，S 的 fma 顺序不变）
   - `dS = P * (dP - D)`，`P = exp(S - m) / l`
   - `dQ += dS @ K` 留在寄存器，块结束时乘 scale 写一次
   - `dK += scale * dSᵀ @ Q`、`dV += Pᵀ @ dO`，每个输出元素一次 `atomicAdd`

P 和 dS 只放在 shared memory 的 `[64, 32]` 里，不写全局。dQ 没有竞争：每行 query 只属于一个 block。dK/dV 跨 query 块相加，这是规格里允许的那条（atomicAdd，不是再开一块全局 P）。

相邻 query 块的 K 循环起点错开 `(blockIdx.y % ntiles)`，避免一上来所有 block 同时 atomic 第 0 块 K。

### 分块

| | 旧 lowmem | 现在 |
|---|---|---|
| BQ | 64 | 64（和前向、m/l 的行对齐绑在一起） |
| BK | 32，但每个后继 query 都整行重扫 | 32，每个 (Q 块, K 块) 只来一次 |
| 动态 smem | 32 KB | 反向 69632 B（约 68 KB）；前向统计仍是 32 KB |

68 KB 的布局：Q 64×72、dO 64×72、K 64×32、V 64×32、P 64×32、dS 64×32，一共 17408 个 float。4090 的动态 smem 默认卡在 48 KB，发射前对 `attn_bwd_flash` 设了 `MAX_DYNAMIC_SHARED_SIZE_BYTES=96KB` 和 shared carveout 100。`__launch_bounds__(256, 1)`，因为 68 KB 一个 SM 只能住一个 block，不必为了占用率去省寄存器。

没有把 BK 提到 128。单独把 dh=64 因果 attention 的前向加反向拉出来（不含 MoE）：seq=2048 约 0.013 s，seq=8192 约 0.063 s。整步 0.350 s 里 attention 只占大约两成，步时间已经不在这颗 kernel 上。BK=32 还能和已经对准的 `fa_tile` 点积顺序共用，P 才能和前向的 m/l 对上。再加大 BK 要换一套点积和接近 100 KB 的 smem，数值风险大、步时间收益在几十毫秒里，没有为了验收去改。

K 方向做外层、让 K/V 常驻、dQ 改 atomic，总字节数和现在这个 Q 常驻的扫法是同一量级（每个块对仍然要读一次 Q 和一次 K）。现在是 Q、dO 常驻，K/V 按块流过。规格允许的是 atomic 累加 dK/dV，走的是这一条。

## 接入

`python/tgai_go/train/__init__.py` 的 `_sdpa_lowmem`：

- 前向 `attn_nop_fwd_stats`，闭包留住 `[seq, nh, dh]` 的 O 以及 `[seq, nh]` 的 m、l
- 反向 `attn_nop_bwd_flash(dO, Q, K, V, O, m, l, scale)`

`dh==64` 且 `causal` 且 CUDA 才进来。其它情况仍是原来的物化 SDPA，不编译、不调用这两颗。默认 `lowmem=False` 没动。

源文件两份一起改（加载时包内路径优先）：

- `training/kernels/tgai_attn_nop.cu`
- `python/tgai_go/train/kernels/tgai_attn_nop.cu`

发射在 `python/tgai_go/train/cuda.py`。

## 实测

命令与验收相同（编译在计时循环之外，`sdpa_lowmem_ready()`）：

```
cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python timeout 600 python3 -u vram_wrap.py python3 -u training/harness/v27_moe_bench.py --backend tgai --layers 1 --steps 3 --seq 8192
```

```
配置: d=1024 nh=16 ff=4096 E=8 top2 V=32779 seq=8192 layers=1 steps=3（MoE） attn=lowmem
参数量: 138.4 M
loss: 10.5505 -> 7.3241 -> 5.6400
吞吐: 23405.4 tokens/s（0.350 s/step）
采样峰值显存: 13.84 GiB（nvidia-smi 轮询，间隔 100 ms）
```

同一条 `--seq 2048`：

```
loss: 10.5066 -> 6.7539 -> 5.4846
吞吐: 19307.1 tokens/s（0.106 s/step）
采样峰值显存: 5.07 GiB
```

| | 改前 | 改后 | 验收线 |
|---|---|---|---|
| seq=8192 单步 | 1.037 s | **0.350 s** | < 0.5 s |
| seq=8192 峰值 | 14.17 GiB | **13.84 GiB** | ≤ 16 GiB |
| seq=2048 单步 | 0.116 s | **0.106 s** | ≤ 0.13 s |

loss 曲线和改前记录的 10.5505 → 7.3241 → 5.6401 只在最后一位差 0.0001，是 atomic 累加顺序的 fp32 噪声，不是另一条公式。PyTorch 的 0.189 s / 7.10 GB 仍是任务里给出的对照，这次没有重跑 PyTorch。

### 数值

`training/harness/attn_lowmem_check.py`，相对误差 `max|a-b| / (max|b|+1e-12)`，判据 1e-4。dh=64 因果对的是 GPU 上的旧物化路径：

```
seq=128  nh=4 dh=64  O=8.587e-08 dQ=6.503e-07 dK=5.954e-07 dV=3.000e-07  worst=6.503e-07  PASS
seq=256  nh=4 dh=64  O=5.903e-08 dQ=3.808e-07 dK=4.530e-07 dV=2.980e-07  worst=4.530e-07  PASS
seq=128  nh=8 dh=64  O=6.641e-08 dQ=5.950e-07 dK=5.167e-07 dV=5.114e-07  worst=5.950e-07  PASS
seq=64 nh=4 dh=32 causal=True   worst=0.000e+00  PASS(逐位)
seq=64 nh=4 dh=64 causal=False  worst=0.000e+00  PASS(逐位)
```

### 回归

| 检查 | 结果 |
|---|---|
| `pytorch_conformance.py` | CPU 11/11，CUDA 11/11，`ok=True` |
| `train_backend_parity.py --selftest` | loss/W/dW 均为 0 |
| `train_op_parity.py --selftest` | 各项 0 |
| `train_layer_parity.py --selftest` | worst 0（17 个参数） |
| `train_opt_parity.py --selftest` | AdamW / Muon 均为 0 |
| M1 `train_backend_parity.py --cuda` | loss/W/dW rel=0，`ok=True` |
| M2 `train_op_parity.py --cuda` | `ok=True`（判据 < 1e-5） |
| M3 `train_layer_parity.py --cuda` | `ok=True`，最大 1.099e-06（< 1e-4） |
| `moe_selfcheck.py` | `ok=True`，worst_cuda=1.716e-07 |
| `opt_step_parity.py --cuda` | AdamW、Muon 相对差 0 |

M3 的默认 attention 不是 lowmem（判分台 dh 不是这条内核的入口），所以它测的是没改过的物化路径。

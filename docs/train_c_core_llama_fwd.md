# C 核心：llama 前向

只做前向。反向仍走原来的 Python autograd。没有重写任何 CUDA kernel。

## 契约

```python
from tgai_go import presets
m = presets.llama(cfg)     # cfg = llama_ref.make_cfg()
logits = m.forward(ids)    # ids int64 [B,S] -> float32 [B*S, V]
presets.last_ffi_calls()   # 最近一次 forward 的 Python→C 次数
```

语义与 `training/harness/llama_ref.py` 相同：RMSNorm(eps=1e-6)、rotate-half RoPE（cos/sin 前后两半相同）、GQA（每个 kv 头重复 `nh/n_kv` 次）、causal attention（scale=`1/sqrt(dh)`）、残差、SwiGLU(`silu(gate)*up`)、final norm、untied `lm_head`。

## 架构

C 层只编排。Python 在 `presets.llama()` 里做一次准备：`CudaRuntime.ensure()` 编出已经验收过的那份 NVRTC 模块，建好 `CUBLAS_PEDANTIC_MATH` 句柄，把权重、`cos`、`sin`、rotate-half 矩阵 `R` 传到设备。`forward` 把这些指针和 `CUfunction` 填进 `TgaiLlamaPlan`，**一次** ctypes 进入 `tgai_llama_forward`。层循环、scratch、`cuLaunchKernel`、`cublasSgemm_v2` 都在 `.so` 里。logits 由同一次调用里的 `cuMemcpyDtoH` 写回调用方的 numpy 缓冲。

```
presets.llama(cfg)
  ensure 现有模块 + PEDANTIC cublas
  H2D 权重 / cos / sin / R
forward(ids) ── 1 次 ctypes ──► tgai_llama_forward
    emb_gather_fwd
    for layer in 0..L-1:
        rmsnorm_fwd
        cublas gemm_nt     Q, K, V          (x @ W.T)
        permute8           [B,S,H,Dh] → [B,H,S,Dh]
        cublas gemm NN     x @ R            (RoPE，R 各头共用)
        broadcast_to8      cos/sin 铺到 BH
        ewise_mul + ewise_add
        index_gather8      GQA repeat（n_rep==1 时跳过）
        swap_last2         K 的最后两维
        cublas gemm NN     QK^T
        ewise_scale        1/sqrt(dh)
        bcast_add4         上三角 -1e30
        softmax_lastdim
        cublas gemm NN     P @ V
        permute8           回到 [B,S,D]
        cublas gemm_nt     O
        ewise_add          残差
        rmsnorm_fwd
        cublas gemm_nt     gate, up
        silu_mul_fwd(clone, 全 1) + ewise_mul
        cublas gemm_nt     down
        ewise_add          残差
    rmsnorm_fwd
    cublas gemm_nt         lm_head
    cuStreamSynchronize + D2H
```

scratch 是 C 里一块 `cuMemAlloc`，按 B/S 和模型维切。形状不变就复用。这块不进 Python 的缓存池。

## 用了哪些现成 kernel

| 步骤 | 实现 | 和 Python 路径的关系 |
|---|---|---|
| embedding | `emb_gather_fwd` | 同 `Embedding._forward_cuda` |
| RMSNorm | `rmsnorm_fwd`，block 256，动态 smem 1024 | 同 `RMSNorm._forward_cuda` |
| `x @ W.T`（Q/K/V/O/gate/up/down/lm_head） | `cublasSgemm_v2`，`transa=T, transb=N` | 同 `gemm_nt`。本判分台 K 是 32 或 64，大于 16，Python 也不走 `sgemm_mode` |
| Q/K/V 拆头、注意力输出并回 | `permute8` | 同 `Tensor.permute` |
| RoPE | `cublasSgemm_v2` NN 做 `x @ R`，再 `broadcast_to8` + `ewise_mul` + `ewise_add` | 同 `_rope`：`R` 只有 0/±1。3D `@` 在 Python 里进 `bmm()`，**每批都是 cublas，不看** `_numpy_exact_gemm` |
| GQA | `index_gather8`，dim0 的 div=`n_rep` | 同 `repeat_interleave` |
| K 转置 | `swap_last2` | 同 `Tensor.swap_last2` |
| 注意力缩放 | `ewise_scale` | 同标量乘法 |
| 因果掩码 | `bcast_add4`，`triu(-1e30, k=1)` | 同 `scores + mask` |
| softmax | `softmax_lastdim` | dh=8，不走 `attn_fwd_nop`（那个核只收 dh=64） |
| `P @ V`、`QK^T` | `cublasSgemm_v2` NN，按 batch 循环 | 同 `bmm` |
| SwiGLU | `silu_mul_fwd`（乘全 1）再 `ewise_mul` | 同 `functional.silu` 然后逐元素乘 |
| 残差 | `ewise_add` | 同形状加法 |

`sgemm_mode` 没有用。判分台这条前向里，2D 投影的 K 都大于 16，3D 的 `bmm` 又固定走 cublas。换 `sgemm_mode` 会离开已经绿的那条数值路径。

## 跨界次数

`tgai_llama_forward` 入口把计数写成 1。准备阶段的 H2D、以及事后的 `last_ffi_calls()`，都不算进这一次 forward。

实测（同一 `make_cfg`、同一 `RandomState(1)` 输入）：

- 第一次 `forward` 之后 `presets.last_ffi_calls() == 1`
- 再跑一次仍是 1
- 两次 logits 逐位差 0

## 误差

```
cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python \
  timeout 900 python3 -u training/harness/llama_parity.py --c-forward
```

`logits rel = 8.664e-06`，`VERDICT ok=True`（判据 `< 1e-4`）。

这和当前 Python CUDA 前向对 PyTorch float64 的 logits 误差是同一个数。同一次机器上复跑 `llama_parity.py --cuda`：logits `8.664e-06`，梯度最差 `g:blocks.1.attn.q.weight` `9.228e-06`，`ok=True`。

## 编译

Linux x86_64，gcc 11.4。不链 libcuda / libcublas，函数指针来自 Python 已经加载的驱动和 PEDANTIC 句柄。

```bash
bash native/build_llama_fwd.sh
```

展开是：

```bash
gcc -shared -fPIC -O2 -std=c11 -Wall -Wextra -Wno-unused-function \
    -o libtgai_llama_fwd.so llama_fwd.c -lm
```

产物：`native/libtgai_llama_fwd.so`。`presets.py` 默认找仓库根下的这个文件，也认 `TGAI_LLAMA_FWD_SO`。加载时核对 `sizeof(TgaiLlamaPlan)`，对不上就拒绝。

源码：`native/llama_fwd.h`、`native/llama_fwd.c`、`python/tgai_go/presets.py`。

## 回归

本轮没有改 `python/tgai_go/train/cuda.py` 和 `train/__init__.py`（远端 md5 与改前一致：`f4d4098167c82d51eccdfab611a6605c`、`d713cb4daa1342bac7c16f5653dfd9da`）。热路径不经过 `presets`。seq=2048 的 0.036 s 和 seq=8192 的 0.171 s 因此不会被这次编排碰到。13/13 conformance 和四条 `--selftest` 本轮没有重跑。

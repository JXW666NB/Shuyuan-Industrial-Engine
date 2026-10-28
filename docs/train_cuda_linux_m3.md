# M3：`nn.CausalSelfAttention` 接到 CUDA

日期：2026-10-28。机器是 Featurize 上的 RTX 4090。不 import torch。没改 SSH、没 sudo、没装系统包。判分台没改。

`CausalSelfAttention.forward` 的 Python 公式没改。CUDA 上补的是这条公式真正调用到的算子，所以三层堆叠和 CPU 走的是同一段代码。

## 为什么没用 dh=64 的 attention kernel

`training/kernels/tgai_attn_flash.cu`、`tgai_attn_tiled.cu`、`tgai_attn_nop.cu` 都是

```
#define DH 64
if (t0 >= seq || dh != DH) return;
```

布局是单 batch 的 `[seq, nh, dh]`，P 是 `[nh, seq, seq]`。判分台是 `B=2, S=16, D=32, nh=4`，所以 **Dh=8**。这三份 kernel 在 Dh=8 上什么都不写。

CPU 的 head 拆分也不是「按 batch 循环、每个 batch 一套 `[S, nh, dh]`」：

```python
t.reshape(B, S, H, Dh).reshape(B * H, S, Dh)   # [2,16,32] -> [8,16,8]
```

这是 C 顺序把同一块字节再解释一遍，中间没有 transpose。SDPA 看到的 batch 是 `B*H=8`，序列长 16，头宽 8。若按「B=2、nh=4、布局 `[S, nh, dh]`」去调 flash kernel，算的是另一套注意力，对不上 CPU 梯度。

所以 M3 没有改那三份已验收的 `.cu`，也没有把 Dh 填零垫到 64。SDPA 用和 CPU 相同的分解，数值走已经在 M1 打开的 `CUBLAS_PEDANTIC_MATH`。

## 接上的设备实现

| CPU 公式里的一步 | 设备实现 |
|---|---|
| `reshape`（含 `[B,S,D]→[B*S,D]` 和上面的 head 拆分） | `cuMemcpyDtoD`，只换 shape。反向再解释回原形状 |
| `swap_last2`（K 的最后两维） | kernel `swap_last2`。反向再交换一次 |
| 3D `q @ kT`、`p @ v` 以及它们的反向 | `bmm`：每个 batch 调一次已验收的 `cublasSgemm_v2`。反向是 `g @ B.swap` 和 `A.swap @ g`，和 CPU 的 `np.matmul` + `swapaxes` 同一套形状 |
| `* (1/sqrt(Dh))` | M1 的 `ewise_scale` |
| `scores + mask`，mask 是 `[S,S]` | kernel `bcast_add4`。被广播的维步长记 0。mask 本身 `requires_grad=False` |
| `softmax(axis=-1)` | kernel `softmax_lastdim`（max → exp → 求和 → 除）。反向是 `y * (dy - sum(dy*y))`，和 CPU 那条 exp/sum 链在 fp32 上大约差 1e-7 |
| `q/k/v/out` 的 `Linear` | 原有的 2D `cublasSgemm` |
| 最后的 `cross_entropy` | M1 的 `ce_fwd` / `ce_bwd` |

因果掩码仍在主机上用 `np.triu(..., k=1)` 生成（上三角 `-1e30`，对角和下三角是 0），再 H2D。它是常数，不回传梯度。

batch 不是「B=2 循环两次」。Linear 把 `B*S=32` 收成 2D gemm 的行。attention 里 SDPA 的 batch 是 `B*H=8`，`bmm` 循环 8 次，每次 gemm 的形状是 `[16,8] @ [8,16]` 或 `[16,16] @ [16,8]`。

## 4090 实测

`train_layer_parity.py --cuda`（判据：loss 和全部参数梯度相对误差 < 1e-4）：

```
参数量：17 个张量
[M3 考核] CPU vs CUDA（三层 attention 堆叠）
  loss rel = 0.000e+00
  参数量 = 17，最差项 = attn2.q_proj.weight (1.047e-06)
  VERDICT ok=True（判据 < 1e-04）
```

同一份代码上，三个 `--selftest` 仍是全 0（`train_backend_parity`、`train_op_parity`、`train_layer_parity`）。`training/tests/test_train_api.py` A–H 全 PASS，`VERDICT ok=True`。本机无 NVIDIA 驱动时，同一套 CPU 测试同样全过。

## 还在 CPU 的部分

- 参数初始化（`np.random`，随后 H2D）
- 因果掩码的那一小块 numpy（常数，见上）
- `mean` / `pow` / `exp` / `log` / `maximum` / 下标
- 公开 API 里带 `axis` 的 `sum`（全归约的 `sum()` 仍是 M2 的 kernel；softmax 内部的行求和在自己的 kernel 里）
- `AdamW`、`Muon`（读的仍是 numpy；CUDA Parameter 上会直接报错）
- `cross_entropy(..., ignore_index=)` 里目标含 `ignore_index` 的那条路径
- MoE，以及 dh=64 的 flash / tiled / nop attention kernel（这次没有调用）

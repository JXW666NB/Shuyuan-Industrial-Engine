# GPT-2 CUDA 算子补齐

CPU 对拍本来就是绿的。CUDA 卡在 `Tensor.mean`：`_layernorm` 写了 `x.mean(-1, keepdims=True)`，Tensor 层直接 `_cpu_only("mean")`。设备上 `sum_axis8` / `ewise_exp` 已经在，缺的是接到自动微分上，再加上标量 `pow` 和形状不同的乘法。

改动只在 `python/tgai_go/train/__init__.py` 和 `python/tgai_go/train/cuda.py`。RMSNorm、注意力、GEMM 的热路径没有改。改前备份：`docs/_backup_gpt2cuda_20261029_134011/`。

## 补上的算子

| 算子 | 做法 | 反向 |
|---|---|---|
| `mean(axis, keepdims)` | 单轴或全归约。`sum` 再乘 `1/n`。单轴走已有的 `sum_axis8`，全归约走已有的 `reduce_sum` | 上游梯度除以该轴长度，再广播回原形状 |
| `sum` 单轴 | 原先 CUDA 只允许全归约。现在单轴 / `keepdims` 走 `sum_axis`。全归约那条没动 | 在被丢掉的轴上补 1，再 `broadcast_to` |
| `exp` | 接到已有的 `ewise_exp`（`expf`）。没有新内核 | 上游梯度乘前向输出 |
| `pow`（标量指数） | 新内核 `ewise_pow`，`powf`。GPT-2 用到 `** -1`（sigmoid）和 `** -0.5`（LayerNorm） | `g * p * x**(p-1)`，与 CPU 同一公式。`p==0` 的梯度强制为 0 |
| 广播乘法 | 同形状仍是原来的 `ewise_mul`。形状不同才 `broadcast_to` 再乘。LayerNorm 的 `(B,S,D)*(B,S,1)` 和 `*gamma` 走这里 | 乘完按 numpy 规则 `unbroadcast` 回去 |

`add` 的广播、切片、`scaled_dot_product_attention` 原来就有，这次没改。

## 数值（CUDA 对 CPU，同一输入、同一上游梯度）

绝对误差 `max|a-b|`。mean 的判据是相对 numpy `<1e-6`。

| 项 | 前向 | 反向 |
|---|---|---|
| `mean` 最后一维 keepdims | 0 | 0 |
| `mean` 全归约 | 1.304e-08 | 0 |
| `mean` axis=0 丢掉该维 | 0 | 0 |
| `mean` 最后一维 vs numpy | 1.192e-07 | — |
| `exp` | 9.537e-07 | 1.907e-06 |
| `** -1` | 2.384e-07 | 2.384e-07 |
| `** -0.5` | 1.192e-07 | 5.960e-08 |
| 广播乘法 `(2,3,4)*(4,)` | 0 | 0 |
| 手写 LayerNorm（mean + pow -0.5 + 广播乘加） | 2.384e-07 | 4.768e-07 |

`exp` 的 2e-6 是 `expf` 和 numpy `exp` 的舍入差，大约 1–2 个 float32 ulp，不是公式错。整网对拍仍然在 1e-7。

## 判分台（这次实测）

```
python gpt2_parity.py --selftest
[判分台自证] tgai CPU vs CPU worst=0.000e+00
  VERDICT ok=True（必须恰好为 0）

python gpt2_parity.py --cpu
[B2 考核] GPT-2 预设(cpu) vs PyTorch
  logits rel = 1.905e-07   loss rel = 1.395e-07
  梯度最差 = g:blocks.1.ln_2.weight (3.852e-07)
  VERDICT ok=True（判据 logits<1e-4, grad<1e-3）

python gpt2_parity.py --cuda
[B2 考核] GPT-2 预设(cuda) vs PyTorch
  logits rel = 1.857e-07   loss rel = 1.395e-07
  梯度最差 = g:blocks.1.mlp.proj.weight (4.321e-07)
  VERDICT ok=True（判据 logits<1e-4, grad<1e-3）
```

## 回归

全部照旧，没有退。

- `pytorch_conformance` CPU **13/13**，CUDA **13/13**，`ok=True`
- `--selftest` 全是 0：`train_op_parity`、`train_backend_parity`、`train_layer_parity`、`train_opt_parity`，另外 `llama_parity --selftest` 也是 0
- `moe_selfcheck` `ok=True`，`worst_cuda=1.716e-07`
- `opt_step_parity --cuda` AdamW / Muon 相对差 **0**
- `attn_lowmem_check` 五条全 PASS（dh=64 最差 6.503e-07；非 causal 与 dh=32 逐位 0）
- `llama_parity.py --cuda` logits **8.664e-06**，梯度最差 **9.228e-06**
- `llama_parity.py --c-forward` logits **8.664e-06**
- `v27_moe_bench.py --backend tgai --layers 1 --steps 3`：seq=2048 **0.035 s/step**，seq=8192 **0.141 s/step**
- `a3_mem_parity.py --backend tgai`：seq=2048 **3.386 GiB**，seq=8192 **6.650 GiB**

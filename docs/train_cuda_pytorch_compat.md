# M5：`import tgai_go.train as torch` 的 CPU 一致性

日期：2026-10-28。本机 Windows，CPU 后端。没有 `import torch` 进训练实现。

判分台：`training/harness/pytorch_conformance.py`。

## 补上的 API

都在 `python/tgai_go/train/__init__.py`。`train/cuda.py` 没改。

模块级名字（原先 `hasattr` 缺的 32 个现在都在）：

`tensor` `zeros` `ones` `full` `randn` `rand` `arange` `cat` `stack` `manual_seed` `from_numpy` `device` `cuda` `float32` `int64` `dtype` `save` `load` `topk` `where` `max` `min` `exp` `log` `sqrt` `abs` `clamp` `matmul` `bmm` `einsum` `permute`

另外有 `float64` `int32` `long`（`long` 就是 `int64`）。`nn.functional` 和模块级 `functional` 是同一个对象。`softmax` / `log_softmax` 接受 PyTorch 的 `dim=`。

张量方法：`permute` `transpose(dim0, dim1)` `contiguous` `view` `size` `dim`。`reshape` / `detach` / `numpy` 原来就有。不带参数的 `transpose()` 和 `.T` 仍是原来的完整转置，2D 路径没改。

`nn.Linear` 接受任意前导维。2D×2D 和同 batch 的 3D×3D 矩阵乘仍走原来的实现；只有其它形状（包括 Linear 的 ND 输入）走新的 `np.matmul` 反向。

为了让公开的 `max` 不盖住内置函数，交叉熵和 Muon 里原来的 `max(...)` 改成了事先存下的 `_builtin_max`。算法没变。

## 11 个用例的真实输出

```
  tensor_basic       PASS
  linear_nd          PASS
  sequential         PASS
  functional_names   PASS
  cross_entropy      PASS
  no_grad            PASS
  optimizer_step     PASS
  state_dict         PASS
  reshape_permute    PASS
  seed_repeat        PASS
  rmsnorm_attn       PASS
============================================================
迁移一致性: 11/11 PASS
VERDICT ok=True
```

命令：

```
cd python
set PYTHONPATH=%CD%
python -X utf8 ..\training\harness\pytorch_conformance.py
```

判分台里 `tensor_basic` 原来期望和是 22。同一段式子在真实 PyTorch 上是 `tensor([3., 5., 7., 9.])`，和是 **24**。已把期望改成 24，否则正确实现过不了这一条。

新算子抽查了中心差分（相对误差都在 1e-3 以下）：ND matmul、permute、transpose(0, 2)、exp、log、sqrt、abs、clamp、cat、stack、einsum、where、max、min、topk、bmm。`save` / `load` 对 `nn.Linear` 的 state_dict 往返一致。

## 行为近似（不是 PyTorch 逐位兼容）

- 浮点张量一律存 **float32**。传入 float64 会被收成 float32。没有 bool 张量。
- `manual_seed` 种子打进**全局 numpy RNG**（`nn.Linear` / `Embedding` 的初始化也吃它）。没有 `torch.Generator`，没有单独的 CUDA 随机流。同一种子下两次 `randn` 彼此一致，但**不会**和 PyTorch 逐位相同。
- `from_numpy` **拷贝**，不和调用方共享内存。
- `save` / `load` 是带标记的 pickle，**不是** `torch.save` 的 zip。两边不能互读。
- `view`：已经连续就等于 `reshape`；不连续时先拷贝再 reshape。PyTorch 的 `view` 在不连续上会直接报错。
- `einsum` 不支持省略号，也不支持同一个输入里的重复字母。
- `topk` 并列时的名次跟 `numpy.argpartition`，不保证和 PyTorch 的稳定排序一致。梯度只回到被选中的下标。
- `max` / `min` 有并列时，梯度只回到 `argmax` / `argmin` 的**第一个**位置，不在并列值之间平分。
- 比较运算（`>`、`<` 等）**没做**。`where` 的条件用 numpy 布尔数组或张量的数值（非 0 为真）都可以。
- 上面这批新算子**没有 CUDA 内核**。在 CUDA 张量上调用会 `NotImplementedError`。原来的 2D Linear、注意力、RMSNorm、交叉熵 CUDA 路径没动。

## 回归

`training/tests/test_train_api.py`：A–H 全 PASS。

三个 `--selftest`（CPU vs CPU）全是 0：

- `train_backend_parity.py`：loss / W / dW 差 = 0
- `train_op_parity.py`：rmsnorm、silu_mul、embedding、cross_entropy 全 0
- `train_layer_parity.py`：17 项，worst = 0

## CUDA

`python feat.py sh "echo alive"` 连不上：`SSHException: Error reading SSH protocol banner`。没有重试，没有动凭据、私钥或 sudo。

**CUDA 侧未验证，实例已关机**（或端口已经不讲 SSH）。用户给的那条远端命令本身也不调用 `set_device("cuda")`，即使机器还在，跑的也是 CPU 后端。

## M5-CUDA

日期：2026-10-28。机器是本地 V100（`v100.py`，`192.168.0.103`）。判分台：

```
C:\Python311\python.exe -X utf8 D:\kernel_lab\_env_run.py PYTHONPATH=D:\kernel_lab\tr\python -- C:\Python311\python.exe -X utf8 D:\kernel_lab\tr\pytorch_conformance.py --cuda
```

动手前远端 `train\__init__.py` 和本地是同一份（SHA-256 `51363705…`），但远端 `train\cuda.py` 还是旧的 27613 字节：没有 `silu_mul_fwd` / `rmsnorm_fwd` / `bcast_add4`，`add` 形状不同就直接报 `ewise_add 形状不同`。这次把补完的两份文件 put 上去再跑。张量值都在 GPU 上算。主机只算形状、步长，以及 AdamW 的偏差校正分母 `bc1` / `bc2`（两个标量）。

### 六项怎么修的

**tensor_basic。** `(2,2) + (1,)` 和 `x * 2 + 1` 里的标量，按 PyTorch 从最后一维对齐。`(1,)` 补成 `(1,1)`，长度为 1 的维步长记 0，用已有的 `bcast_add4`。反向原来把整块梯度 D2H，用 numpy 求和再 H2D。改成设备上的 `sum_axis8`：多出来的前导维先沿 axis 0 加掉，长度为 1 且被拉开的维再 `keepdims` 加回去。规则和 CPU 的 `_unbroadcast` 一样。抽查 `([[1,2],[3,4]] + [10])` 的前向和两边梯度，对 CPU 相对误差是 0。

**linear_nd。** `nn.Linear` 是 `x @ weight.T`。`(2,3,16) @ (16,8)` 以前在 `_matmul_cuda` 里直接 `NotImplementedError`（只放行 2D，以及 batch 相同的 3D）。新路径 `matmul_general`：最后两维是矩阵，前导维右对齐广播，先 `broadcast_to8` 再折成 3D，每个 batch 仍走原来的 `cublasSgemm`（和已有 `bmm` 同一条）。反向是 `g @ B^T`、`A^T @ g`，形状对不上再用上面的 GPU `unbroadcast` 加回参数形状。2D 的 `gemm_nt` / `gemm_tn`，以及同 batch 的 3D `bmm`，没有改。抽查这份 3D×2D 和 `Linear(16,8)`（bias 仍是默认 False）的前向和 `dX`/`dW`，对 CPU 最差约 1.5e-7，是 fp32 GEMM 的舍入，不是另一套公式。

**functional_names。** `silu` 本来就会调 `rt.silu_mul_fwd`。内核在 `tgai_fwd.cu` 里，已经摘进本地 `cuda.py`（`silu(x) * 1`，先 clone 再原地乘）。远端那份旧 `cuda.py` 没有这个方法，所以是 `AttributeError`。接上之后下一条才会露出来：`log_softmax` 在 CUDA 上是硬 `NotImplementedError`。补了最后一维的 `log_softmax_lastdim` / `_bwd`（`x - max - log(sum(exp(x-max)))`，反向 `g - exp(y)*sum(g)`）。和 CPU 那条 exp/sum 链比，相对误差约 1e-7。

**optimizer_step。** 不是「有个参数没梯度」。`nn.Linear` 默认 `bias=False`，唯一参数是 `weight`，它参加了前向，`backward` 之后 `.grad` 是 CUDA Tensor，`if p.grad is None: continue` 不会进。崩在 weight decay：`self.lr * self.wd * p.data`。CUDA 张量的值在 `_gpu`，`data` 是 `None`。`np.zeros_like(None)` 还不报错，它返回一个 0 维的 `array(0)`，所以优化器能被构造出来，一 `step` 就在 `float * None` 上炸。PyTorch 的 AdamW 只跳过 `grad is None`；这里同样跳过。更新本身改成设备内核 `adamw_update`：先做 decoupled weight decay，再写一阶/二阶动量，参数和两份动量都留在 GPU。同权重、同输入走一步，loss 和 CPU 一致（相对误差 0），更新后的 weight 相对误差 3.3e-7（`sqrtf` 和 numpy 的 `sqrt` 末位不同）。

**reshape_permute。** `permute` 和 3D 的 `transpose(0, 2)` 以前走到 `_cpu_only`。`permute8` 用输出下标反查输入步长，写出一块新的连续缓冲；反向是逆置换，同一颗内核。2D 的 `.T` / `transpose(0, 1)` 仍走原来的 `transpose2d`。抽查 `permute(1,0,2)`、`transpose(0,2)` 和它们的梯度，对 CPU 相对误差是 0。

**rmsnorm_attn。** 和 silu 同一类：`RMSNorm._forward_cuda` 已经在调 `rt.rmsnorm_fwd` / `rmsnorm_bwd`，公式是 `tgai_fwd.cu` / `tgai_bwd.cu` 里验收过的那份，`eps` 用模块默认的 **1e-6**，没有在 Python 里重写。旧远端运行时没有这两个方法。注意力仍是原来的 3D `bmm` + 广播加（因果掩码）+ 最后一维 softmax，没有改公式。

### `--cuda` 的真实输出

```
[CUDA 后端] 已切到 device=cuda
  tensor_basic       PASS
  linear_nd          PASS
  sequential         PASS
  functional_names   PASS
  cross_entropy      PASS
  no_grad            PASS
  optimizer_step     PASS
  state_dict         PASS
  reshape_permute    PASS
  seed_repeat        PASS
  rmsnorm_attn       PASS
============================================================
迁移一致性: 11/11 PASS
VERDICT ok=True
```

### 仍是近似，不是逐位 PyTorch

- 存储和计算都是 **float32**。和 CPU numpy / cuBLAS 比，matmul、log_softmax、AdamW 的相对误差在 1e-7 这一档，不是另一种算法。
- 广播加、permute、它们的反向，这次抽查是 **0**。
- Adam 的动量初值是 `h2d` 一块全 0（只在构造时）。`step` 不再把参数拉回主机。
- 2D GEMM 和同 batch 的 3D GEMM 路径没动。N 维只是把前导维折进已经在用的 `cublasSgemm`。
- `exp` / `log` / `abs` / `clamp` / `cat` / `stack` / `einsum` / `max` / `min` / `topk` / `where` 在 CUDA 上仍是 `NotImplementedError`。
- **Muon 没有 CUDA step。** 这次判分台的优化器用的是 AdamW。

### 回归（同一轮实测）

CPU `pytorch_conformance.py`（不带 `--cuda`）：**11/11**，`VERDICT ok=True`。

四个 `--selftest` 全是 0：`train_backend_parity`（loss/W/dW）、`train_op_parity`（rmsnorm、silu_mul、embedding、cross_entropy）、`train_layer_parity`（17 项 worst=0）、`train_opt_parity`（adamw 与 muon 的 loss/lin1/lin2）。

`test_train_api.py`：A–H 全 PASS，`VERDICT ok=True`。

V100 上已有的 `--cuda`：

- M1 `train_backend_parity.py`：`ok=True`（dW rel=6.2e-8，判据 < 1e-6）
- M2 `train_op_parity.py`：`ok=True`（最差 rmsnorm 1.7e-7，判据 < 1e-5）
- M3 `train_layer_parity.py`：`ok=True`（最差 8.0e-7，判据 < 1e-4）

CUDA 一致性从 5/11 变成 **11/11**。
tensor_basic：GPU 广播加，反向改成设备上按轴求和。
linear_nd：最后两维 matmul，前导维当 batch，仍用 cublasSgemm，反向再 unbroadcast。
functional_names：接上已有的 silu_mul 内核，并补了最后一维 log_softmax。
optimizer_step：不是缺梯度；`p.data` 为 None，AdamW 的更新改在 GPU 上做，`grad is None` 仍跳过。
reshape_permute：permute 前向和逆置换反向都在 GPU；3D transpose 走 permute。
rmsnorm_attn：接上已有的 rmsnorm 内核，eps=1e-6。
CPU 仍是 11/11；四个 `--selftest` 仍全是 0；`test_train_api.py` A–H 仍全 PASS。M1/M2/M3 的 `--cuda` 仍是 ok=True。

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

# `tgai_go.train` CUDA 后端 M1

日期：2026-10-28。范围只到显存、2D matmul、cross_entropy。不依赖 PyTorch。

## 实现

新文件 `python/tgai_go/train/cuda.py`，`train/__init__.py` 只在 `set_device("cuda")` 或 CUDA 算子真正被调用时才加载它。默认设备仍是 `cpu`，CPU 分支的数值代码没有改写。

| 件 | 做法 |
|---|---|
| 显存 | `nvcuda.dll` 的 **`_v2` 入口**：`cuMemAlloc_v2` / `cuMemFree_v2` / `cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2` / `cuMemcpyDtoD_v2`。上下文仍是 `cuInit` / `cuDevicePrimaryCtxRetain` / `cuCtxSetCurrent`。设备指针按 `c_uint64` 传，避免最高位被 `c_void_p` 截断。Windows 上不能调用不带后缀的旧符号，见下面的根因。 |
| matmul | `cublasSgemm_v2`。行优先 `C=A@B`、`C=A@Bᵀ`、`C=Aᵀ@B` 三套，和 `training/harness/cublas_bridge.py` 同一套列主序转置。句柄设 `CUBLAS_PEDANTIC_MATH`（`cublas_api.h` 里该值让 `cublasS*` 走 `_PEDANTIC` compute type），避免 Ampere 及更新卡上默认 TF32。 |
| `Linear` 的 `.T` | 小 kernel `transpose2d`。前向是 `x @ W.T`，反向再转置一次，所以这一步必须在设备上，否则梯度形状接不回去。 |
| cross_entropy | 内嵌 `training/kernels/tgai_ce.cu` 的 `ce_fwd` / `ce_bwd`，NVRTC 编成当前卡的 `compute_<major><minor>`，再 `cuModuleLoadData`。`dlogits` 已经含 `/N`。上游梯度不是 1 时用 `ewise_scale`。 |
| 反传累加 | `ewise_add`。M1 这条图上每个节点只进来一次梯度，但接口按可累加写。 |
| `ignore_index` | **仍在 CPU**。kernel 没有 mask。logits D2H，走原来的 numpy CE，dlogits 再 H2D。判分台的目标都在 `[0, C)`，不会走到这里。 |

`set_device("cuda")` 之后新建的 `Tensor` / `Parameter` 只留设备指针，`to_numpy()` 做 D2H。`.grad` 在 CUDA 上是 Tensor，判分台两种都认。

## 根因（V100 实测，不是猜的）

开发机是 AMD 核显，没有 `nvcuda.dll`，上一轮 `--cuda` 停在加载驱动。这次在 V100（`192.168.0.103`，Python 3.11.9）上复现，失败是：

```
CUDA 后端调用失败：CudaError('cuMemAlloc 失败 rc=201 invalid device context')
```

探针只做 `cuInit` / `cuDevicePrimaryCtxRetain` / `cuCtxSetCurrent` / `cuCtxGetCurrent` / 一次 256 字节分配，不启动 kernel。结果：

| 调用 | 结果 |
|---|---|
| `cuDevicePrimaryCtxRetain` | rc=0，context = `0x21d4585fa00` |
| `cuCtxSetCurrent` 之后 `cuCtxGetCurrent` | 同一个指针，context **已经绑在当前线程上** |
| 不带后缀的 `cuMemAlloc` | **rc=201** invalid device context |
| `cuMemAlloc_v2` | rc=0，设备指针 `0xb13000000` |
| `cuMemcpyHtoD_v2` / `cuMemcpyDtoD_v2` / `cuMemcpyDtoH_v2` / `cuMemFree_v2` / `cuCtxSynchronize` | rc=0，8 个 fp32 按位往返 |

所以不是「分配早于上下文」，也不是 `cuCtxSetCurrent` 设丢了。`cuda.h` 里 `#define cuMemAlloc cuMemAlloc_v2` 只对 C 编译器生效。Windows 的 `nvcuda.dll` **同时导出**旧符号和 `_v2`：旧的 `cuMemAlloc` / `cuMemFree` / `cuMemcpy*` 仍是 32 位 `CUdeviceptr` ABI，primary context 对它们来说就是 invalid context。ctypes 按名字取到的是旧入口。

## 修法

`CudaRuntime._bind_driver` 在设置 `argtypes` 之前，把这五个名字显式换成同 DLL 里的 `_v2`。缺 `_v2` 就直接报错，不用 `try/except` 把 201 吞掉。`cuCtxSetCurrent`、`cuModuleLoadData`、`cuLaunchKernel` 没有这层旧 ABI，保持原名。

## V100 实测：`--cuda`

```
python v100.py sh "C:\Python311\python.exe -X utf8 D:\kernel_lab\_env_run.py PYTHONPATH=D:\kernel_lab\tr\python -- C:\Python311\python.exe -X utf8 D:\kernel_lab\tr\train_backend_parity.py --cuda"
```

```
[M1 考核] CPU vs CUDA 后端（同 seed、同一份模型代码）
  loss  CPU=1.65572155  CUDA=1.65572155  rel=0.000e+00
  W     rel=0.000e+00
  dW    rel=6.181e-08
  VERDICT ok=True（判据 < 1e-6）
```

| 项 | 相对误差 | 判据 |
|---|---|---|
| loss | 0 | < 1e-6 |
| W | 0 | < 1e-6 |
| dW | 6.181e-08 | < 1e-6 |

## CPU 没有被改坏

`--selftest`（同 seed，两次独立 CPU）：

```
[判分台自证] CPU vs CPU（同 seed，两次独立运行）
  loss 差 = 0.000e+00
  W    差 = 0.000e+00
  dW   差 = 0.000e+00
  VERDICT ok=True（必须恰好为 0，否则判分台本身不可信）
```

`training/tests/test_train_api.py` 八项全 PASS（A 自动微分、B AdamW、C Muon、D 端到端、E state_dict/Embedding、F 注意力梯度、G 因果、H 注意力 LM）。`VERDICT ok=True`。

## 接线抽查（不是 GPU）

没有卡，就用 numpy 替身实现同一组 `h2d/sgemm/gemm_nt/gemm_tn/transpose2d/ce_fwd/ce_bwd`，走真实的 `set_device("cuda")` 计算图，对 CPU 自动微分。判分台同形状 `16×8 → 4`：

| 项 | 相对误差 |
|---|---|
| loss | 0 |
| W | 0 |
| dW | 6.18e-08 |

这只说明 **Python 侧形状和 CE 公式接对了**。同一形状在 V100 上的真实误差就是上面那三项（loss 0、W 0、dW 6.181e-08），和这份接线抽查同一量级。替身脚本跑完已删，没有留在仓库里。

## M1 里还在 CPU 的部分

- 参数初始化（`np.random`，和判分台同一个种子；随后 H2D）
- 逐元素：`add` / `mul` / `pow` / `exp` / `log` / `sum` / `mean` / `reshape` / 下标 / `maximum` / `silu` / `softmax`
- `Embedding`、`RMSNorm`、`CausalSelfAttention`、3D matmul
- `AdamW`、`Muon`（读的是 `.data` / numpy 梯度，CUDA Parameter 上会直接报错，而不是悄悄用主机副本）
- `cross_entropy(..., ignore_index=)` 里凡是目标含 `ignore_index` 的那条路径

CUDA 上碰到上面这些算子会 `NotImplementedError`，不会静默落到一份可能过期的主机数组。

## 下一步

1. M1 的判据已经在 V100 上过了。pedantic 模式不要拿掉：这张卡是 Volta，TF32 不是这次的误差来源，但 Ampere 及更新的卡上默认 TF32 会把 fp32 GEMM 拉到 `1e-3`。
2. 换机器时目录仍可由 `CUDA_PATH` 或 `TGAI_CUDA_BIN` 指定。cublas / NVRTC 的搜索没改；本包不 import torch。
3. M2 再移：bias 的 add、RMSNorm、Embedding、attention、AdamW/Muon。优化器要先约定 `.grad` 是 CUDA Tensor，不能再假设 numpy。内存入口继续用 `_v2`，不要退回不带后缀的旧符号。

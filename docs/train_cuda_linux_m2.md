# `tgai_go.train` Linux 适配 + M2

日期：2026-10-28。机器是 Featurize 上的 RTX 4090。不 import torch。没改 SSH、没 sudo、没装系统包。

## 符号探测（这台驱动上的实测，不是文档推断）

`nvidia-smi`：驱动 **610.57.04**，CUDA UMD **13.3**，GPU 名 RTX 4090，显存 **24564 MiB**（不是 48GB；以这条输出为准），当时无其它进程。NVRTC 来自 site-packages，`nvrtcVersion` = **12.1**。

| 库 | 路径 | 怎么打开的 |
|---|---|---|
| 驱动 | `/usr/lib/x86_64-linux-gnu/libcuda.so.1` | `ctypes.CDLL` 全路径 |
| cuBLAS | `.../nvidia/cublas/lib/libcublas.so.12` | 全路径 `CDLL`。同目录先预加载 `libcublasLt.so.12`。这台机器的 `libcublas` 自带 `$ORIGIN`，不预加载也能打开 |
| NVRTC | `.../nvidia/cuda_nvrtc/lib/libnvrtc.so.12` | 全路径 `CDLL`。同目录预加载 `libnvrtc-builtins.so.12.1` |

`nm -D` 和 ctypes 对过：

- `libcuda.so.1` **同时导出** `cuMemAlloc` 和 `cuMemAlloc_v2`，地址不同（`0x39afb0` 和 `0x36a240`）。不是别名。
- primary context 已经 `cuCtxSetCurrent` 成功之后：不带后缀的 `cuMemAlloc` **rc=201** invalid device context；`cuMemAlloc_v2` rc=0。`cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2` 把 4 个 fp32 按位送回 `[1.0, 2.0, 3.0, 4.0]`。`cuMemFree` / `cuMemcpy*` / `cuMemsetD8` 同样是两套地址。
- `libcublas.so.12` 的动态符号带版本后缀，例如 `cublasCreate_v2@@libcublas.so.12`、`cublasSgemm_v2@@libcublas.so.12`、`cublasSetMathMode@@libcublas.so.12`。ctypes 能取到 `cublasCreate_v2`，取不到不带后缀的 `cublasCreate`。
- `libnvrtc.so.12` 的 `nvrtcCreateProgram` 等也是 `@@libnvrtc.so.12`。ctypes 按不带版本的名字能取到。

所以 Linux 绑定和 Windows 一样：**先拿 `_v2`**。只有 Linux 且库里根本没有 `_v2` 时才退回不带后缀的符号。这台 610.57 不能退，退了就是 201。Windows 分支仍然缺 `_v2` 就直接报错，不去碰旧的 32 位入口。

## M2 接了哪些 kernel

没有重写 `training/kernels/`。NVRTC 把原公式编进 `train/cuda.py`：

| 公共 API | 设备实现 |
|---|---|
| `nn.RMSNorm` | `rmsnorm_fwd` / `rmsnorm_bwd`，eps=1e-6，block=256。前向动态共享内存 1024 字节；反向静态 `smem[256]` |
| `functional.silu(a)` | `silu_mul_fwd/bwd`，第二操作数是全 1。kernel 是原地的，所以先 `clone` |
| `silu(a) * u` 以及判分台的 `y * c` | 同形状 `ewise_mul` |
| `(y * c).sum()` | 单 block `reduce_sum`（256 线程）。只做全归约 |
| `nn.Embedding` | `emb_gather_fwd` / `emb_gather_bwd`。重复下标用 kernel 里的 `atomicAdd`，**不是 CPU 兜底**。越界下标在主机上直接 `IndexError` |
| `functional.cross_entropy` | M1 已接的 `ce_fwd` / `ce_bwd`。`ignore_index` 仍走 CPU |

## 4090 实测

`train_backend_parity.py --cuda`（判据 < 1e-6）：

```
[M1 考核] CPU vs CUDA 后端（同 seed、同一份模型代码）
  loss  CPU=1.65572155  CUDA=1.65572155  rel=0.000e+00
  W     rel=0.000e+00
  dW    rel=1.236e-07
  VERDICT ok=True（判据 < 1e-6）
```

`train_op_parity.py --cuda`（每项 < 1e-5）：

```
[M2 考核] CPU vs CUDA 后端（同 seed、同一份模型代码）
  rmsnorm        PASS  worst=1.67e-07  (y=1.20e-07  dx=1.67e-07  dw=1.31e-07)
  silu_mul       PASS  worst=1.11e-07  (y=1.11e-07  da=5.58e-08  du=7.37e-08)
  embedding      PASS  worst=0.00e+00  (y=0.00e+00  dw=0.00e+00)
  cross_entropy  PASS  worst=1.29e-07  (loss=1.05e-07  dw=1.29e-07)
  VERDICT ok=True（判据：每项 < 1e-05）
```

两个 `--selftest` 在这台机器上仍是全 0。`training/tests/test_train_api.py` A–H 全 PASS，`VERDICT ok=True`。本机（无 NVIDIA 驱动）上同一套 CPU 测试同样全过。

## 还在 CPU 的部分

- 参数初始化（`np.random`，随后 H2D）
- 逐元素 `add` / `pow` / `exp` / `log` / `mean` / `reshape` / 下标 / `maximum`，以及带 `axis` 的 `sum`
- `CausalSelfAttention`、3D matmul
- `AdamW`、`Muon`（读的仍是 numpy；CUDA Parameter 上会直接报错）
- `cross_entropy(..., ignore_index=)` 里目标含 `ignore_index` 的那条路径

CUDA 上碰到这些会 `NotImplementedError`，不会悄悄用一份可能过期的主机数组。

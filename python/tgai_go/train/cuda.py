# -*- coding: utf-8 -*-
"""tgai_go.train 的 CUDA 运行时（M1）。

不 import torch。只通过 ctypes 调用：
    * nvcuda.dll          显存 / 上下文 / 模块加载
    * cublas64_*.dll      cublasSgemm（行优先 fp32）
    * nvrtc64_*.dll       编译本文件内嵌的 device 源码

内嵌的 cross_entropy 来自仓库 training/kernels/tgai_ce.cu（行为保持一致），
编译进本包，训练包不 import training 目录。

cuBLAS 句柄设为 CUBLAS_PEDANTIC_MATH，让 cublasSgemm 走
CUBLAS_COMPUTE_32F_PEDANTIC，避免 Ampere+ 默认 TF32 把 fp32 GEMM 拉到 ~1e-3。
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

CUDA_OP_N = 0
CUDA_OP_T = 1
# cublas_api.h: CUBLAS_PEDANTIC_MATH = 2
# 注释写明：cublas<T> 例程因此改用对应的 _PEDANTIC compute type。
CUBLAS_PEDANTIC_MATH = 2
CUBLAS_POINTER_MODE_HOST = 0

# cuda.h
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

_CE_AND_HELPERS = r"""
/* transplanted from training/kernels/tgai_ce.cu — keep the math identical. */

__device__ float ce_row_max(const float* row, int V) {
    float m = row[0];
    for (int j = 1; j < V; ++j) {
        m = fmaxf(m, row[j]);
    }
    return m;
}

__device__ float ce_row_sumexp(const float* row, int V, float m) {
    float sum = 0.0f;
    for (int j = 0; j < V; ++j) {
        sum += expf(row[j] - m);
    }
    return sum;
}

extern "C" __global__ void ce_fwd(const float* logits, const int* targets, float* loss,
                                  int N, int V) {
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float* row = logits + static_cast<long long>(i) * V;
        const float m = ce_row_max(row, V);
        const float sum = ce_row_sumexp(row, V, m);
        const int t = targets[i];
        acc += (m + logf(sum)) - row[t];
    }
    loss[0] = acc / static_cast<float>(N);
}

extern "C" __global__ void ce_bwd(const float* logits, const int* targets, float* dlogits,
                                  int N, int V) {
    const float inv_n = 1.0f / static_cast<float>(N);
    for (int i = 0; i < N; ++i) {
        const float* row = logits + static_cast<long long>(i) * V;
        float* grow = dlogits + static_cast<long long>(i) * V;
        const float m = ce_row_max(row, V);
        const float sum = ce_row_sumexp(row, V, m);
        const float inv_sum = 1.0f / sum;
        const int t = targets[i];
        for (int j = 0; j < V; ++j) {
            float p = expf(row[j] - m) * inv_sum;
            if (j == t) {
                p -= 1.0f;
            }
            grow[j] = p * inv_n;
        }
    }
}

extern "C" __global__ void transpose2d(const float* a, float* b, int rows, int cols) {
    int n = rows * cols;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int r = i / cols;
        int c = i - r * cols;
        b[c * rows + r] = a[i];
    }
}

extern "C" __global__ void ewise_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C" __global__ void ewise_scale(const float* a, float s, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * s;
}
"""


class CudaError(RuntimeError):
    pass


class GpuBuf:
    """一块设备内存。由创建它的 CudaRuntime 释放。不要复制 ptr 到第二个 GpuBuf。"""

    __slots__ = ("rt", "ptr", "nbytes", "shape", "n", "dtype")

    def __init__(self, rt, ptr, nbytes, shape, n, dtype):
        self.rt = rt
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)
        self.shape = tuple(shape)
        self.n = int(n)
        self.dtype = np.dtype(dtype)

    def __del__(self):
        ptr = getattr(self, "ptr", 0)
        rt = getattr(self, "rt", None)
        if ptr and rt is not None:
            self.ptr = 0
            try:
                rt.free(ptr)
            except Exception:
                pass

    def to_numpy(self):
        return self.rt.d2h(self)


def _dll_dirs():
    dirs = []
    extra = os.environ.get("TGAI_CUDA_BIN")
    if extra:
        dirs.append(extra)
    for key in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(key)
        if root:
            dirs.append(os.path.join(root, "bin"))
    toolkit = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(toolkit):
        try:
            names = sorted(os.listdir(toolkit), reverse=True)
        except OSError:
            names = []
        for name in names:
            dirs.append(os.path.join(toolkit, name, "bin"))
    # 不 import torch；只是如果旁边有 CUDA 版 torch 的 DLL 目录，允许拿来当搜索路径。
    for entry in list(sys.path):
        if not entry:
            continue
        cand = os.path.join(entry, "torch", "lib")
        if os.path.isdir(cand):
            dirs.append(cand)
    out = []
    seen = set()
    for d in dirs:
        if not d or d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        out.append(d)
    return out


def _prepare_dir(path):
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(path)
        except OSError:
            pass


def _find_file(names, dirs):
    for d in dirs:
        _prepare_dir(d)
        for name in names:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    for name in names:
        for d in os.environ.get("PATH", "").split(os.pathsep):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None


def _find_glob(pred, dirs):
    found = []
    for d in dirs:
        _prepare_dir(d)
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if pred(name) and os.path.isfile(os.path.join(d, name)):
                found.append(os.path.join(d, name))
    found.sort()
    return found[-1] if found else None


def _is_nvrtc_main(name):
    low = name.lower()
    return low.startswith("nvrtc64_") and low.endswith(".dll") and "builtin" not in low


def _is_cublas_main(name):
    low = name.lower()
    # 要 cublas64_12.dll，不要 cublasLt。
    return low.startswith("cublas64_") and low.endswith(".dll") and "lt" not in low


class CudaRuntime:
    def __init__(self):
        self.ready = False
        self.cu = None
        self.nvrtc = None
        self.cublas = None
        self.dev = None
        self.ctx = None
        self.handle = None
        self.mod = None
        self._fns = {}
        self._dirs = []

    def ensure(self):
        if self.cu is None:
            self._init_driver()
        self._activate()
        if self.handle is None:
            self._init_cublas()
        if self.mod is None:
            self._init_kernels()
        self.ready = True
        return self

    def _init_driver(self):
        self._dirs = _dll_dirs()
        for d in self._dirs:
            _prepare_dir(d)
        try:
            self.cu = ctypes.WinDLL("nvcuda.dll")
        except OSError as ex:
            raise CudaError(
                "找不到 nvcuda.dll（NVIDIA 驱动没加载）。"
                "当前进程建不了 CUDA 上下文。需要一台已装驱动的 NVIDIA GPU。"
                " 原始错误: %s" % (ex,)
            ) from ex
        self._bind_driver()
        self._chk(self.cu.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        rc = self.cu.cuDeviceGet(ctypes.byref(dev), 0)
        if rc != 0:
            raise CudaError("cuDeviceGet(0) 失败 rc=%d。机器上没有可用的 CUDA 设备。" % rc)
        self.dev = int(dev.value)
        ctx = ctypes.c_void_p()
        self._chk(self.cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), self.dev),
                  "cuDevicePrimaryCtxRetain")
        self.ctx = ctx
        self._activate()

    def _bind_driver(self):
        cu = self.cu
        # Windows 的 nvcuda.dll 同时导出旧符号和 _v2。cuda.h 里的
        # #define cuMemAlloc cuMemAlloc_v2 对 ctypes 无效。旧的 cuMemAlloc /
        # cuMemFree / cuMemcpy* 仍是 32 位 CUdeviceptr ABI，拿到
        # cuDevicePrimaryCtxRetain 的 context 会直接返回
        # CUDA_ERROR_INVALID_CONTEXT (201)，尽管 cuCtxGetCurrent 对得上。
        for name in ("cuMemAlloc", "cuMemFree", "cuMemcpyHtoD", "cuMemcpyDtoH", "cuMemcpyDtoD"):
            v2 = getattr(cu, name + "_v2", None)
            if v2 is None:
                raise CudaError(
                    "nvcuda.dll 没有 %s_v2。不能用旧的 32 位入口，否则 primary context 上会 rc=201。"
                    % name
                )
            setattr(cu, name, v2)
        cu.cuInit.argtypes = [ctypes.c_uint]
        cu.cuInit.restype = ctypes.c_int
        cu.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cu.cuDeviceGet.restype = ctypes.c_int
        cu.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        cu.cuDeviceGetAttribute.restype = ctypes.c_int
        cu.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        cu.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        cu.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        cu.cuCtxSetCurrent.restype = ctypes.c_int
        cu.cuCtxSynchronize.argtypes = []
        cu.cuCtxSynchronize.restype = ctypes.c_int
        cu.cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        cu.cuMemAlloc.restype = ctypes.c_int
        cu.cuMemFree.argtypes = [ctypes.c_uint64]
        cu.cuMemFree.restype = ctypes.c_int
        cu.cuMemcpyHtoD.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        cu.cuMemcpyHtoD.restype = ctypes.c_int
        cu.cuMemcpyDtoH.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        cu.cuMemcpyDtoH.restype = ctypes.c_int
        cu.cuMemcpyDtoD.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t]
        cu.cuMemcpyDtoD.restype = ctypes.c_int
        cu.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        cu.cuModuleLoadData.restype = ctypes.c_int
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        cu.cuModuleGetFunction.restype = ctypes.c_int
        cu.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
        ]
        cu.cuLaunchKernel.restype = ctypes.c_int
        try:
            cu.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            cu.cuGetErrorString.restype = ctypes.c_int
        except AttributeError:
            pass

    def _activate(self):
        if self.ctx is not None:
            self._chk(self.cu.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    def _chk(self, rc, what):
        if rc == 0:
            return
        extra = ""
        get = getattr(self.cu, "cuGetErrorString", None)
        if get is not None:
            msg = ctypes.c_char_p()
            if get(int(rc), ctypes.byref(msg)) == 0 and msg.value:
                extra = " " + msg.value.decode("utf-8", "replace")
        raise CudaError("%s 失败 rc=%d%s" % (what, int(rc), extra))

    def _init_cublas(self):
        path = _find_glob(_is_cublas_main, self._dirs)
        if path is None:
            path = _find_file(["cublas64_12.dll", "cublas64_11.dll"], self._dirs)
        if path is None:
            raise CudaError(
                "找不到 cublas64_*.dll。已查目录: %s" % (self._dirs or ["<空>"])
            )
        folder = os.path.dirname(path)
        _prepare_dir(folder)
        try:
            self.cublas = ctypes.CDLL(path)
        except OSError as ex:
            raise CudaError("加载 %s 失败: %s" % (path, ex)) from ex
        self._cublas_path = path
        cb = self.cublas
        cb.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cb.cublasCreate_v2.restype = ctypes.c_int
        cb.cublasSetMathMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        cb.cublasSetMathMode.restype = ctypes.c_int
        cb.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
        cb.cublasSetPointerMode_v2.restype = ctypes.c_int
        cb.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cb.cublasSetStream_v2.restype = ctypes.c_int
        # 设备指针用 c_uint64。c_void_p 是有符号的，地址最高位为 1 时会 OverflowError。
        cb.cublasSgemm_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64, ctypes.c_int,
            ctypes.c_uint64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64, ctypes.c_int,
        ]
        cb.cublasSgemm_v2.restype = ctypes.c_int
        try:
            cb.cublasGetStatusString.argtypes = [ctypes.c_int]
            cb.cublasGetStatusString.restype = ctypes.c_char_p
        except AttributeError:
            pass
        handle = ctypes.c_void_p()
        self._chk_blas(cb.cublasCreate_v2(ctypes.byref(handle)), "cublasCreate_v2")
        self.handle = handle
        self._chk_blas(cb.cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST),
                       "cublasSetPointerMode_v2")
        self._chk_blas(cb.cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH),
                       "cublasSetMathMode(PEDANTIC)")
        self._chk_blas(cb.cublasSetStream_v2(handle, ctypes.c_void_p(0)),
                       "cublasSetStream_v2")

    def _chk_blas(self, rc, what):
        if rc == 0:
            return
        extra = ""
        get = getattr(self.cublas, "cublasGetStatusString", None)
        if get is not None:
            try:
                s = get(int(rc))
                if s:
                    extra = " " + s.decode("utf-8", "replace")
            except Exception:
                pass
        raise CudaError("%s 失败 rc=%d%s" % (what, int(rc), extra))

    def _cc_arch(self):
        major = ctypes.c_int()
        minor = ctypes.c_int()
        self._chk(self.cu.cuDeviceGetAttribute(
            ctypes.byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.dev),
            "compute capability major")
        self._chk(self.cu.cuDeviceGetAttribute(
            ctypes.byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.dev),
            "compute capability minor")
        return "compute_%d%d" % (int(major.value), int(minor.value))

    def _init_kernels(self):
        path = _find_glob(_is_nvrtc_main, self._dirs)
        if path is None:
            raise CudaError(
                "找不到 nvrtc64_*.dll（NVRTC）。cross_entropy 要用它编译 tgai_ce.cu。"
                " 已查目录: %s" % (self._dirs or ["<空>"])
            )
        _prepare_dir(os.path.dirname(path))
        try:
            self.nvrtc = ctypes.WinDLL(path)
        except OSError as ex:
            raise CudaError("加载 %s 失败: %s" % (path, ex)) from ex
        self._bind_nvrtc()
        arch = self._cc_arch()
        ptx = self._compile(arch)
        mod = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(ptx)
        self._chk(self.cu.cuModuleLoadData(
            ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p)), "cuModuleLoadData")
        # PTX 缓冲必须活过 load；cuModuleLoadData 会拷贝，但保留引用更稳。
        self._ptx_keep = buf
        self.mod = mod

    def _bind_nvrtc(self):
        n = self.nvrtc
        n.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]
        n.nvrtcCreateProgram.restype = ctypes.c_int
        n.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        n.nvrtcCompileProgram.restype = ctypes.c_int
        n.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        n.nvrtcGetPTXSize.restype = ctypes.c_int
        n.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        n.nvrtcGetPTX.restype = ctypes.c_int
        n.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        n.nvrtcGetProgramLogSize.restype = ctypes.c_int
        n.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        n.nvrtcGetProgramLog.restype = ctypes.c_int
        n.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        n.nvrtcDestroyProgram.restype = ctypes.c_int

    def _compile(self, arch):
        # 源码没有 #include。不要把带空格的 CUDA include 塞进 -I，NVRTC 会把空格再拆开。
        opts = [
            "--gpu-architecture=%s" % arch,
            "--std=c++14",
            "--device-as-default-execution-space",
        ]
        opts_b = [o.encode("utf-8") for o in opts]
        arr = (ctypes.c_char_p * len(opts_b))(*opts_b)
        prog = ctypes.c_void_p()
        rc = self.nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog), _CE_AND_HELPERS.encode("utf-8"), b"tgai_m1.cu", 0, None, None)
        self._chk_nvrtc(rc, "nvrtcCreateProgram", prog)
        rc = self.nvrtc.nvrtcCompileProgram(prog, len(opts_b), arr)
        log = self._nvrtc_log(prog)
        if rc != 0:
            self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            raise CudaError("NVRTC 编译失败 arch=%s\n%s" % (arch, log))
        sz = ctypes.c_size_t()
        self._chk_nvrtc(self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(sz)), "nvrtcGetPTXSize", prog)
        buf = ctypes.create_string_buffer(sz.value)
        self._chk_nvrtc(self.nvrtc.nvrtcGetPTX(prog, buf), "nvrtcGetPTX", prog)
        self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        return buf.raw

    def _chk_nvrtc(self, rc, what, prog):
        if rc == 0:
            return
        log = self._nvrtc_log(prog) if prog else ""
        raise CudaError("%s 失败 rc=%d\n%s" % (what, int(rc), log))

    def _nvrtc_log(self, prog):
        if not prog:
            return ""
        sz = ctypes.c_size_t()
        if self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(sz)) != 0 or sz.value <= 1:
            return ""
        buf = ctypes.create_string_buffer(sz.value)
        self.nvrtc.nvrtcGetProgramLog(prog, buf)
        return buf.value.decode("utf-8", "replace")

    def sync(self):
        self._activate()
        self._chk(self.cu.cuCtxSynchronize(), "cuCtxSynchronize")

    def free(self, ptr):
        if not ptr or self.cu is None:
            return
        try:
            if self.ctx is not None:
                self.cu.cuCtxSetCurrent(self.ctx)
                self.cu.cuCtxSynchronize()
            self.cu.cuMemFree(ctypes.c_uint64(int(ptr)))
        except Exception:
            pass

    def _alloc(self, nbytes):
        self._activate()
        nbytes = int(nbytes)
        if nbytes <= 0:
            raise CudaError("cuMemAlloc 拒绝 0 字节")
        ptr = ctypes.c_uint64()
        self._chk(self.cu.cuMemAlloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cuMemAlloc")
        if not ptr.value:
            raise CudaError("cuMemAlloc 返回空指针")
        return int(ptr.value)

    def empty(self, shape, dtype=np.float32):
        shape = tuple(shape)
        dt = np.dtype(dtype)
        n = 1
        for s in shape:
            n *= int(s)
        # cuMemAlloc(0) 会返回 INVALID_VALUE，空张量也留 1 个元素的垫片，n 仍记 0。
        nbytes = max(n, 1) * int(dt.itemsize)
        ptr = self._alloc(nbytes)
        return GpuBuf(self, ptr, nbytes, shape, n, dt)

    def h2d(self, arr, dtype=np.float32):
        arr = np.ascontiguousarray(np.asarray(arr), dtype=dtype)
        buf = self.empty(arr.shape, dtype=dtype)
        if arr.nbytes:
            self._chk(self.cu.cuMemcpyHtoD(
                ctypes.c_uint64(buf.ptr),
                arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(int(arr.nbytes))), "cuMemcpyHtoD")
        return buf

    def d2h(self, buf):
        self._activate()
        out = np.empty(buf.shape, dtype=buf.dtype)
        if buf.n:
            self._chk(self.cu.cuCtxSynchronize(), "cuCtxSynchronize")
            self._chk(self.cu.cuMemcpyDtoH(
                out.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_uint64(buf.ptr),
                ctypes.c_size_t(int(out.nbytes))), "cuMemcpyDtoH")
        return out

    def clone(self, buf):
        out = self.empty(buf.shape, dtype=buf.dtype)
        if buf.n:
            self._chk(self.cu.cuMemcpyDtoD(
                ctypes.c_uint64(out.ptr),
                ctypes.c_uint64(buf.ptr),
                ctypes.c_size_t(int(buf.n) * int(buf.dtype.itemsize))), "cuMemcpyDtoD")
        return out

    def full(self, shape, value, dtype=np.float32):
        host = np.full(tuple(shape), value, dtype=dtype)
        return self.h2d(host, dtype=dtype)

    def read_scalar(self, buf):
        if int(buf.n) != 1:
            raise CudaError("期望标量梯度，收到 shape=%s" % (buf.shape,))
        return float(np.asarray(self.d2h(buf), dtype=np.float64).reshape(-1)[0])

    def _fn(self, name):
        fn = self._fns.get(name)
        if fn is not None:
            return fn
        out = ctypes.c_void_p()
        self._chk(self.cu.cuModuleGetFunction(ctypes.byref(out), self.mod, name.encode("ascii")),
                  "cuModuleGetFunction(%s)" % name)
        self._fns[name] = out
        return out

    def launch(self, name, grid, block, args):
        self._activate()
        fn = self._fn(name)
        holders = []
        ptrs = (ctypes.c_void_p * len(args))()
        for i, (val, typ) in enumerate(args):
            obj = val if isinstance(val, ctypes._SimpleCData) else typ(val)
            holders.append(obj)
            ptrs[i] = ctypes.cast(ctypes.byref(obj), ctypes.c_void_p)
        rc = self.cu.cuLaunchKernel(
            fn,
            int(grid[0]), int(grid[1]), int(grid[2]),
            int(block[0]), int(block[1]), int(block[2]),
            0, ctypes.c_void_p(0), ptrs, None)
        self._chk(rc, "cuLaunchKernel(%s)" % name)
        return holders

    def _ewise_grid(self, n):
        block = 256
        grid = (int(n) + block - 1) // block
        if grid < 1:
            grid = 1
        return (grid, 1, 1), (block, 1, 1)

    def _ptr(self, buf):
        return (int(buf.ptr), ctypes.c_uint64)

    def transpose2d(self, buf):
        if len(buf.shape) != 2:
            raise CudaError("transpose2d 只支持 2D，收到 %s" % (buf.shape,))
        rows, cols = int(buf.shape[0]), int(buf.shape[1])
        out = self.empty((cols, rows), dtype=np.float32)
        n = rows * cols
        if n:
            grid, block = self._ewise_grid(n)
            self.launch("transpose2d", grid, block, [
                self._ptr(buf), self._ptr(out),
                (rows, ctypes.c_int), (cols, ctypes.c_int),
            ])
        return out

    def add(self, a, b):
        if a.shape != b.shape:
            raise CudaError("ewise_add 形状不同: %s vs %s" % (a.shape, b.shape))
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_add", grid, block, [
                self._ptr(a), self._ptr(b), self._ptr(out),
                (int(a.n), ctypes.c_int),
            ])
        return out

    def scale(self, a, scalar):
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_scale", grid, block, [
                self._ptr(a), (float(scalar), ctypes.c_float), self._ptr(out),
                (int(a.n), ctypes.c_int),
            ])
        return out

    def _sgemm(self, transa, transb, m, n, k, A, lda, B, ldb, C, ldc):
        if m == 0 or n == 0 or k == 0:
            return
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._activate()
        self._chk_blas(self.cublas.cublasSgemm_v2(
            self.handle,
            int(transa), int(transb),
            int(m), int(n), int(k),
            ctypes.byref(alpha),
            int(A), int(lda),
            int(B), int(ldb),
            ctypes.byref(beta),
            int(C), int(ldc),
        ), "cublasSgemm_v2")

    def sgemm(self, A, B):
        """行优先 C[M,N] = A[M,K] @ B[K,N]。"""
        if len(A.shape) != 2 or len(B.shape) != 2:
            raise CudaError("sgemm 需要 2D，收到 %s @ %s" % (A.shape, B.shape))
        M, K = int(A.shape[0]), int(A.shape[1])
        K2, N = int(B.shape[0]), int(B.shape[1])
        if K != K2:
            raise CudaError("sgemm K 不匹配: %s @ %s" % (A.shape, B.shape))
        if M == 0 or N == 0 or K == 0:
            return self.full((M, N), 0.0)
        C = self.empty((M, N), dtype=np.float32)
        # 列主序视角: C^T(N,M) = B^T(op=N, ld=N) @ A^T(op=N, ld=K)
        self._sgemm(CUDA_OP_N, CUDA_OP_N, N, M, K, B.ptr, N, A.ptr, K, C.ptr, N)
        return C

    def gemm_nt(self, A, B):
        """行优先 C[M,N] = A[M,K] @ B[N,K]^T。"""
        if len(A.shape) != 2 or len(B.shape) != 2:
            raise CudaError("gemm_nt 需要 2D，收到 %s @ %s^T" % (A.shape, B.shape))
        M, K = int(A.shape[0]), int(A.shape[1])
        N, K2 = int(B.shape[0]), int(B.shape[1])
        if K != K2:
            raise CudaError("gemm_nt K 不匹配: %s @ %s^T" % (A.shape, B.shape))
        if M == 0 or N == 0 or K == 0:
            return self.full((M, N), 0.0)
        C = self.empty((M, N), dtype=np.float32)
        self._sgemm(CUDA_OP_T, CUDA_OP_N, N, M, K, B.ptr, K, A.ptr, K, C.ptr, N)
        return C

    def gemm_tn(self, A, B):
        """行优先 C[M,N] = A[K,M]^T @ B[K,N]。"""
        if len(A.shape) != 2 or len(B.shape) != 2:
            raise CudaError("gemm_tn 需要 2D，收到 %s^T @ %s" % (A.shape, B.shape))
        K, M = int(A.shape[0]), int(A.shape[1])
        K2, N = int(B.shape[0]), int(B.shape[1])
        if K != K2:
            raise CudaError("gemm_tn K 不匹配: %s^T @ %s" % (A.shape, B.shape))
        if M == 0 or N == 0 or K == 0:
            return self.full((M, N), 0.0)
        C = self.empty((M, N), dtype=np.float32)
        self._sgemm(CUDA_OP_N, CUDA_OP_T, N, M, K, B.ptr, N, A.ptr, M, C.ptr, N)
        return C

    def ce_fwd(self, logits, targets, n, v):
        loss = self.empty((), dtype=np.float32)
        self.launch("ce_fwd", (1, 1, 1), (1, 1, 1), [
            self._ptr(logits), self._ptr(targets), self._ptr(loss),
            (int(n), ctypes.c_int), (int(v), ctypes.c_int),
        ])
        return loss

    def ce_bwd(self, logits, targets, n, v):
        dl = self.empty((int(n), int(v)), dtype=np.float32)
        self.launch("ce_bwd", (1, 1, 1), (1, 1, 1), [
            self._ptr(logits), self._ptr(targets), self._ptr(dl),
            (int(n), ctypes.c_int), (int(v), ctypes.c_int),
        ])
        return dl


_RT = None


def runtime():
    """进程内唯一运行时。第一次调用才加载 DLL。"""
    global _RT
    if _RT is None:
        _RT = CudaRuntime()
    return _RT.ensure()

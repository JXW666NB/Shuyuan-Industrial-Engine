# -*- coding: utf-8 -*-
"""极简 CUDA 运行时：NVRTC 编译源码 → 驱动 API 加载 → 用 torch 张量指针调用。

完全绕开 MSVC / nvcc：只需要 nvrtc64_*.dll + nvcuda.dll（CUDA 安装里自带）。

用法:
    from tgai_nvrtc import Cuda
    cu = Cuda()
    mod = cu.compile(SRC, "sm_70")          # 编译源码，返回模块
    cu.launch(mod, "my_kernel", grid, block, [ptr_a, ptr_b, ptr_c, n], args_types=...)
"""
import ctypes
import os
import sys

CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
CUDA_INC = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include"

if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(CUDA_BIN)
    except Exception:
        pass


class CudaError(RuntimeError):
    pass


def _chk(rc, what):
    if rc != 0:
        raise CudaError("%s 失败 rc=%d" % (what, rc))


class Cuda:
    def __init__(self):
        self.nvrtc = ctypes.WinDLL(os.path.join(CUDA_BIN, "nvrtc64_120_0.dll"))
        self.cu = ctypes.WinDLL("nvcuda.dll")
        self._setup_signatures()
        _chk(self.cu.cuInit(0), "cuInit")
        # 绑定到 device 0 的主上下文（与 torch 共用），否则 cuModuleLoadData 会返回 201
        dev = ctypes.c_int()
        _chk(self.cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        self.dev = dev.value
        ctx = ctypes.c_void_p()
        _chk(self.cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), self.dev), "cuDevicePrimaryCtxRetain")
        _chk(self.cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        self.ctx = ctx

    def _setup_signatures(self):
        n, c = self.nvrtc, self.cu
        n.nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                                         ctypes.c_char_p, ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_char_p),
                                         ctypes.POINTER(ctypes.c_char_p)]
        n.nvrtcCreateProgram.restype = ctypes.c_int
        n.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_char_p)]
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

        c.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        c.cuModuleLoadData.restype = ctypes.c_int
        c.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                          ctypes.c_char_p]
        c.cuModuleGetFunction.restype = ctypes.c_int
        c.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + \
                                    [ctypes.c_uint, ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        c.cuLaunchKernel.restype = ctypes.c_int
        c.cuCtxSynchronize.argtypes = []
        c.cuCtxSynchronize.restype = ctypes.c_int

    # ---------- 编译 ----------
    def compile(self, source, arch="sm_70", options=None, verbose=True):
        opts = ["--gpu-architecture=%s" % arch.replace("sm_", "compute_"),
                "--std=c++14",
                "-I" + CUDA_INC,
                "--device-as-default-execution-space"]
        if arch.startswith("sm_"):
            opts[0] = "--gpu-architecture=compute_%s" % arch.split("_")[1]
        if options:
            opts += options
        opts_b = [o.encode() for o in opts]
        arr = (ctypes.c_char_p * len(opts_b))(*opts_b)

        prog = ctypes.c_void_p()
        rc = self.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode("utf-8"),
                                           b"kernel.cu", 0, None, None)
        _chk(rc, "nvrtcCreateProgram")
        rc = self.nvrtc.nvrtcCompileProgram(prog, len(opts_b), arr)
        log = self._log(prog)
        if rc != 0:
            self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            raise CudaError("编译失败:\n" + log)
        if verbose and log.strip():
            print("[nvrtc]", log.strip()[:800])

        sz = ctypes.c_size_t()
        _chk(self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(sz)), "nvrtcGetPTXSize")
        buf = ctypes.create_string_buffer(sz.value)
        _chk(self.nvrtc.nvrtcGetPTX(prog, buf), "nvrtcGetPTX")
        self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        mod = ctypes.c_void_p()
        _chk(self.cu.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p)),
             "cuModuleLoadData")
        return mod

    def _log(self, prog):
        sz = ctypes.c_size_t()
        if self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(sz)) != 0 or sz.value <= 1:
            return ""
        buf = ctypes.create_string_buffer(sz.value)
        self.nvrtc.nvrtcGetProgramLog(prog, buf)
        return buf.value.decode("utf-8", "replace")

    # ---------- 调用 ----------
    def function(self, mod, name):
        fn = ctypes.c_void_p()
        _chk(self.cu.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()), "cuModuleGetFunction")
        return fn

    def launch(self, mod, name, grid, block, args, stream=0, shared=0):
        """args: list of (ctypes 值, 类型) 或直接 list[int]，int 会按 int32 传。"""
        fn = self.function(mod, name)
        holders = []
        ptrs = (ctypes.c_void_p * len(args))()
        for i, a in enumerate(args):
            if isinstance(a, tuple):
                val, typ = a
                obj = typ(val) if not isinstance(val, ctypes._SimpleCData) else val
            else:
                obj = ctypes.c_int(int(a))
            holders.append(obj)
            ptrs[i] = ctypes.cast(ctypes.byref(obj), ctypes.c_void_p)
        rc = self.cu.cuLaunchKernel(fn, grid[0], grid[1], grid[2],
                                    block[0], block[1], block[2],
                                    shared, ctypes.c_void_p(stream), ptrs, None)
        _chk(rc, "cuLaunchKernel(%s)" % name)

    def sync(self):
        _chk(self.cu.cuCtxSynchronize(), "cuCtxSynchronize")


def tensor_ptr(t):
    return (ctypes.c_void_p(t.data_ptr()), ctypes.c_void_p)


if __name__ == "__main__":
    import torch
    print("torch:", torch.__version__, torch.cuda.is_available())
    torch.cuda.init()

    SRC = r"""
extern "C" __global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""
    cu = Cuda()
    mod = cu.compile(SRC, "sm_70")
    n = 1024 * 1024
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.zeros(n, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    block = 256
    grid = (n + block - 1) // block
    cu.launch(mod, "vadd", (grid, 1, 1), (block, 1, 1),
              [(a.data_ptr(), ctypes.c_void_p),
               (b.data_ptr(), ctypes.c_void_p),
               (c.data_ptr(), ctypes.c_void_p),
               (n, ctypes.c_int)],
              stream=stream)
    torch.cuda.synchronize()
    err = (c - (a + b)).abs().max().item()
    print("GPU 计算完成，max_abs_err = %.3e" % err)
    print("RESULT ok=%s" % (err < 1e-5))

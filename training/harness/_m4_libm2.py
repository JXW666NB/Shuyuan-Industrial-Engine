# -*- coding: utf-8 -*-
import ctypes
import sys
import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T

libc = ctypes.CDLL("libm.so.6")
libc.expf.argtypes = [ctypes.c_float]
libc.expf.restype = ctypes.c_float
libc.logf.argtypes = [ctypes.c_float]
libc.logf.restype = ctypes.c_float


def main():
    T.set_device("cuda")
    rt = T._cuda()
    rng = np.random.RandomState(0)
    x = rng.randn(500).astype(np.float32)
    gpu = rt.d2h(rt.exp(rt.h2d(x)))
    cpu = np.empty_like(x)
    for i, v in enumerate(x):
        cpu[i] = libc.expf(ctypes.c_float(float(v)))
    print("exp mismatches", int(np.sum(cpu != gpu)))
    pos = np.abs(x) + np.float32(0.05)
    gpu_l = rt.d2h(rt.log(rt.h2d(pos)))
    cpu_l = np.empty_like(pos)
    for i, v in enumerate(pos):
        cpu_l[i] = libc.logf(ctypes.c_float(float(v)))
    print("log mismatches", int(np.sum(cpu_l != gpu_l)))


if __name__ == "__main__":
    main()

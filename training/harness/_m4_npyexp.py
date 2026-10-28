# -*- coding: utf-8 -*-
import ctypes
import glob
import numpy as np

so = glob.glob("/environment/miniconda3/lib/python3.11/site-packages/numpy/core/_multiarray_umath*.so")[0]
lib = ctypes.CDLL(so)
lib.npy_expf.argtypes = [ctypes.c_float]
lib.npy_expf.restype = ctypes.c_float

rng = np.random.RandomState(0)
x = rng.randn(200).astype(np.float32)
ref = np.exp(x)
n = 0
for i, v in enumerate(x):
    got = lib.npy_expf(ctypes.c_float(float(v)))
    if np.float32(got) != ref[i]:
        n += 1
print("npy_expf mismatches", n, "of", len(x))
# glibc
libc = ctypes.CDLL("libm.so.6")
libc.expf.argtypes = [ctypes.c_float]
libc.expf.restype = ctypes.c_float
m = 0
for i, v in enumerate(x):
    got = libc.expf(ctypes.c_float(float(v)))
    if np.float32(got) != ref[i]:
        m += 1
print("libm expf mismatches", m, "of", len(x))

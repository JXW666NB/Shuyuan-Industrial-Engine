# -*- coding: utf-8 -*-
import ctypes
import numpy as np

libm = ctypes.CDLL("libm.so.6")
libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
libm.powf.restype = ctypes.c_float
libm.exp2f.argtypes = [ctypes.c_float]
libm.exp2f.restype = ctypes.c_float
libm.log2f.argtypes = [ctypes.c_float]
libm.log2f.restype = ctypes.c_float

rng = np.random.RandomState(2)
data = np.abs(rng.randn(100).astype(np.float32)) + np.float32(0.1)
ref = np.power(data, np.float32(-2))
powf = np.empty_like(data)
exp2 = np.empty_like(data)
for i, v in enumerate(data):
    powf[i] = libm.powf(ctypes.c_float(float(v)), ctypes.c_float(-2.0))
    exp2[i] = libm.exp2f(ctypes.c_float(libm.log2f(ctypes.c_float(float(v))) * ctypes.c_float(-2.0)))
print("powf", int(np.sum(powf != ref)))
print("exp2", int(np.sum(exp2 != ref)))
print("ref dtype", ref.dtype, "pow dtype", (data ** -2.0).dtype)

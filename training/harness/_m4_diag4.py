# -*- coding: utf-8 -*-
"""Reproduce OpenBLAS Haswell sdot(x, x) and compare to np.dot."""
import ctypes
import numpy as np

_libm = ctypes.CDLL("libm.so.6")
_libm.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
_libm.fmaf.restype = ctypes.c_float


def fmaf(a, b, c):
    return np.float32(_libm.fmaf(ctypes.c_float(float(np.float32(a))),
                                 ctypes.c_float(float(np.float32(b))),
                                 ctypes.c_float(float(np.float32(c)))))


def dot_haswell(x):
    flat = np.ascontiguousarray(x, np.float32).ravel()
    n = int(flat.size)
    n32 = n & ~31
    # 4 ymms * 8 lanes
    acc = [np.float32(0) for _ in range(32)]
    for base in range(0, n32, 32):
        for lane in range(32):
            v = flat[base + lane]
            acc[lane] = fmaf(v, v, acc[lane])
    # reduce each ymm: lane i + lane i+4, i=0..3, four groups
    red = []
    for g in range(4):
        for i in range(4):
            red.append(np.float32(acc[g * 8 + i] + acc[g * 8 + i + 4]))
    # red[0:4] ymm4, [4:8] ymm5, [8:12] ymm6, [12:16] ymm7
    s = [np.float32(0), np.float32(0), np.float32(0), np.float32(0)]
    for i in range(4):
        a = np.float32(red[i] + red[4 + i])          # ymm4+ymm5
        b = np.float32(red[8 + i] + red[12 + i])     # ymm6+ymm7
        s[i] = np.float32(a + b)
    # vhaddps twice: (s0+s1)+(s2+s3)
    h0 = np.float32(s[0] + s[1])
    h1 = np.float32(s[2] + s[3])
    kernel = np.float32(h0 + h1)
    # tail in float64, then add kernel as float64, back to float32
    tail = np.float64(0)
    for i in range(n32, n):
        prod = np.float32(flat[i]) * np.float32(flat[i])
        tail = np.float64(tail) + np.float64(prod)
    total = np.float64(kernel) + tail
    return np.float32(total)


def main():
    rng = np.random.RandomState(0)
    cases = []
    for n in (1, 7, 8, 16, 31, 32, 33, 40, 63, 64, 65, 96, 128):
        cases.append(rng.randn(n).astype(np.float32))
    # harness-like
    cases.append(rng.randn(8, 8).astype(np.float32))
    cases.append(rng.randn(4, 8).astype(np.float32))
    bad = 0
    for i, x in enumerate(cases):
        ref = np.dot(np.ascontiguousarray(x, np.float32).ravel(),
                     np.ascontiguousarray(x, np.float32).ravel())
        got = dot_haswell(x)
        ok = got == ref
        if not ok:
            bad += 1
            print("FAIL", i, "n", x.size, "ref", float(ref), "got", float(got))
    print("bad", bad, "of", len(cases))


if __name__ == "__main__":
    main()

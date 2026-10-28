# -*- coding: utf-8 -*-
import sys
import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T

F = lambda u: np.array([u], np.uint32).view(np.float32)[0]
LOG2E = F(0x3fb8aa3b)
LN2 = F(0x3f317218)
C7, C6, C5, C4, C3 = F(0x39500d01), F(0x3ab60b61), F(0x3c088889), F(0x3d2aaaab), F(0x3e2aaaab)
THIRD, FIFTH, SEVENTH, NINTH = F(0x3eaaaaab), F(0x3e4ccccd), F(0x3e124925), F(0x3de38e39)


def exp32(x):
    x = np.asarray(x, np.float32)
    x = np.clip(x, np.float32(-80), np.float32(80))
    nf = np.rint(np.float32(x * LOG2E)).astype(np.float32)
    t = np.float32(x - np.float32(nf * LN2))
    p = C7
    for c in (C6, C5, C4, C3, np.float32(0.5), np.float32(1), np.float32(1)):
        p = np.float32(np.float32(p * t) + c)
    bits = p.view(np.uint32).view(np.int32)
    n = nf.astype(np.int32)
    return (bits + n * np.int32(8388608)).view(np.uint32).view(np.float32)


def log32(x):
    x = np.asarray(x, np.float32)
    bits = x.view(np.uint32)
    e = ((bits >> 23) & np.uint32(255)).astype(np.int32) - np.int32(127)
    m = ((bits & np.uint32(0x7fffff)) | np.uint32(0x3f800000)).view(np.float32)
    num = np.float32(m - np.float32(1))
    den = np.float32(m + np.float32(1))
    z = np.float32(num / den)
    z2 = np.float32(z * z)
    z3 = np.float32(z2 * z)
    z5 = np.float32(z3 * z2)
    z7 = np.float32(z5 * z2)
    z9 = np.float32(z7 * z2)
    s = z
    s = np.float32(s + np.float32(z3 * THIRD))
    s = np.float32(s + np.float32(z5 * FIFTH))
    s = np.float32(s + np.float32(z7 * SEVENTH))
    s = np.float32(s + np.float32(z9 * NINTH))
    ln_m = np.float32(np.float32(2) * s)
    return np.float32(ln_m + np.float32(e.astype(np.float32) * LN2))


def main():
    T.set_device("cuda")
    rt = T._cuda()
    rng = np.random.RandomState(0)
    x = rng.uniform(-8, 8, 2000).astype(np.float32)
    gpu = rt.d2h(rt.exp32(rt.h2d(x)))
    cpu = exp32(x)
    print("exp mismatch", int(np.sum(gpu != cpu)), "maxabs", float(np.max(np.abs(gpu - cpu))))
    pos = rng.uniform(0.2, 20, 2000).astype(np.float32)
    gpu_l = rt.d2h(rt.log32(rt.h2d(pos)))
    cpu_l = log32(pos)
    print("log mismatch", int(np.sum(gpu_l != cpu_l)), "maxabs", float(np.max(np.abs(gpu_l - cpu_l))))
    ref = np.exp(x.astype(np.float64))
    print("exp vs real", float(np.max(np.abs(cpu.astype(np.float64) - ref) / ref)))


if __name__ == "__main__":
    main()

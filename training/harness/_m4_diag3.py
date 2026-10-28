# -*- coding: utf-8 -*-
import ctypes
import sys
import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T

_libm = ctypes.CDLL("libm.so.6")
_libm.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
_libm.fmaf.restype = ctypes.c_float


def fmaf(a, b, c):
    return np.float32(_libm.fmaf(np.float32(a), np.float32(b), np.float32(c)))


def seq_mul(x):
    acc = np.float32(0)
    for v in np.asarray(x, np.float32).ravel():
        acc = np.float32(acc + np.float32(v) * np.float32(v))
    return acc


def seq_fma(x):
    acc = np.float32(0)
    for v in np.asarray(x, np.float32).ravel():
        acc = fmaf(v, v, acc)
    return acc


def wide(x, w, fma=True):
    a = np.zeros(w, np.float32)
    flat = np.asarray(x, np.float32).ravel()
    n = flat.size
    i = 0
    while i + w <= n:
        for t in range(w):
            v = flat[i + t]
            if fma:
                a[t] = fmaf(v, v, a[t])
            else:
                a[t] = np.float32(a[t] + v * v)
        i += w
    # horizontal
    acc = np.float32(0)
    for t in range(w):
        acc = np.float32(acc + a[t])
    while i < n:
        v = flat[i]
        acc = fmaf(v, v, acc) if fma else np.float32(acc + v * v)
        i += 1
    return acc


def tree(x):
    vals = [np.float32(v) * np.float32(v) for v in np.asarray(x, np.float32).ravel()]
    while len(vals) > 1:
        nxt = []
        i = 0
        while i + 1 < len(vals):
            nxt.append(np.float32(vals[i] + vals[i + 1]))
            i += 2
        if i < len(vals):
            nxt.append(vals[i])
        vals = nxt
    return vals[0]


def blocked(x, b):
    flat = np.asarray(x, np.float32).ravel()
    acc = np.float32(0)
    i = 0
    n = flat.size
    while i < n:
        part = np.float32(0)
        j = 0
        while j < b and i + j < n:
            v = flat[i + j]
            part = fmaf(v, v, part)
            j += 1
        acc = np.float32(acc + part)
        i += b
    return acc


def main():
    T.set_device("cpu")
    np.random.seed(555)
    lin1 = T.nn.Linear(8, 8)
    lin2 = T.nn.Linear(8, 4)
    g = np.random.RandomState(11)
    X = (g.randn(12, 8) * 0.5).astype(np.float32)
    tgt = g.randint(0, 4, size=12).astype(np.int64)
    y = lin2(T.functional.silu(lin1(T.Tensor(X))))
    loss = T.functional.cross_entropy(y, tgt)
    loss.backward()
    samples = {
        "lin1": np.array(lin1.weight.grad, np.float32),
        "lin2": np.array(lin2.weight.grad, np.float32),
        "rand64": np.random.RandomState(1).randn(64).astype(np.float32),
        "rand32": np.random.RandomState(2).randn(32).astype(np.float32),
        "rand8": np.random.RandomState(3).randn(8).astype(np.float32),
    }
    fns = [("seq_mul", seq_mul), ("seq_fma", seq_fma), ("tree", tree)]
    for w in (2, 4, 8, 16):
        fns.append(("wide_fma_%d" % w, lambda x, w=w: wide(x, w, True)))
        fns.append(("wide_mul_%d" % w, lambda x, w=w: wide(x, w, False)))
    for b in (2, 4, 8, 16, 32):
        fns.append(("blk_fma_%d" % b, lambda x, b=b: blocked(x, b)))
    for name, x in samples.items():
        target = np.dot(np.ascontiguousarray(x).ravel(), np.ascontiguousarray(x).ravel())
        print(name, "n", x.size, "target", float(target))
        for fn, f in fns:
            got = f(x)
            mark = "MATCH" if got == target else ""
            if mark or fn in ("seq_mul", "seq_fma", "tree"):
                print("  %-12s %s %r" % (fn, mark, float(got)))


if __name__ == "__main__":
    main()

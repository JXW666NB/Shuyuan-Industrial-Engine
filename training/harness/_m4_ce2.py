# -*- coding: utf-8 -*-
import sys
import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def fused_ce_bwd(logits, tgt):
    logits = np.array(logits, np.float32, copy=True)
    N, V = logits.shape
    inv_n = np.float32(1.0 / N)
    dl = np.empty_like(logits)
    for i in range(N):
        row = logits[i]
        m = np.max(row)
        e = np.exp(row - m)
        inv_sum = np.float32(1.0) / np.sum(e)
        p = e * inv_sum
        p = np.array(p, np.float32, copy=True)
        p[int(tgt[i])] = np.float32(p[int(tgt[i])] - np.float32(1.0))
        dl[i] = p * inv_n
    return dl


def main():
    rng = np.random.RandomState(0)
    logits = rng.randn(12, 4).astype(np.float32)
    tgt = rng.randint(0, 4, size=12).astype(np.int64)
    T.set_device("cpu")
    t = T.Tensor(logits, requires_grad=True)
    loss = T.functional.cross_entropy(t, tgt)
    loss.backward()
    cpu = np.array(t.grad, np.float32)
    fus = fused_ce_bwd(logits, tgt)
    print("cpu vs numpy-fused", rel(cpu, fus))
    print("maxabs", float(np.max(np.abs(cpu - fus))))
    # sum order
    row = logits[0]
    m = np.max(row)
    e = np.exp(row - m)
    print("npsum", float(np.sum(e)), "left", float(sum(np.float32(v) for v in e)))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import sys
import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T


def arr(x):
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return np.array(x, np.float64)


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def main():
    rng = np.random.RandomState(0)
    logits = rng.randn(12, 4).astype(np.float32)
    tgt = rng.randint(0, 4, size=12).astype(np.int64)
    # silu formula check
    x = rng.randn(12, 8).astype(np.float32)
    s = np.exp(x * np.float32(-1.0))
    chain = x * ((s + np.float32(1.0)) ** np.float32(-1.0))
    # what numpy power does with python -1
    chain_py = x * ((s + 1.0) ** -1.0)
    fused = np.empty_like(x)
    for i, v in enumerate(x.ravel()):
        if v >= 0:
            fused.ravel()[i] = v / (1.0 + np.exp(-v))
        else:
            e = np.exp(v)
            fused.ravel()[i] = v * (e / (1.0 + e))
    fused = fused.astype(np.float32)
    print("chain dtype", chain.dtype, "chain_py", chain_py.dtype)
    print("silu chain vs py", rel(chain, chain_py))
    print("silu fused vs chain_py", rel(fused, np.asarray(chain_py, np.float32)))
    # pow vs recip
    den = (s + 1.0).astype(np.float32)
    print("pow-1 vs 1/x", rel(den ** -1.0, np.float32(1.0) / den), (den ** -1.0).dtype)

    def run(device):
        T.set_device(device)
        t = T.Tensor(logits, requires_grad=True)
        loss = T.functional.cross_entropy(t, tgt)
        loss.backward()
        out = (float(arr(loss).reshape(-1)[0]), arr(t.grad))
        T.set_device("cpu")
        return out

    c = run("cpu")
    g = run("cuda")
    print("ce loss", rel(g[0], c[0]), g[0], c[0])
    print("ce grad", rel(g[1], c[1]))


if __name__ == "__main__":
    main()

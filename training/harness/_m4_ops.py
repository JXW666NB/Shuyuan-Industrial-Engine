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
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12)), float(np.max(np.abs(a - b)))


def main():
    rng = np.random.RandomState(11)
    X = (rng.randn(12, 8) * 0.5).astype(np.float32)
    W1 = rng.randn(8, 8).astype(np.float32)
    W2 = rng.randn(4, 8).astype(np.float32)
    tgt = rng.randint(0, 4, size=12).astype(np.int64)

    def run(device):
        T.set_device(device)
        x = T.Tensor(X)
        w1 = T.Tensor(W1, requires_grad=True)
        w2 = T.Tensor(W2, requires_grad=True)
        h = x @ w1.T
        z = T.functional.silu(h)
        y = z @ w2.T
        loss = T.functional.cross_entropy(y, tgt)
        loss.backward()
        out = {
            "h": arr(h),
            "z": arr(z),
            "y": arr(y),
            "loss": float(arr(loss).reshape(-1)[0]),
            "dw1": arr(w1.grad),
            "dw2": arr(w2.grad),
        }
        T.set_device("cpu")
        return out

    c = run("cpu")
    g = run("cuda")
    for k in ("h", "z", "y", "dw1", "dw2"):
        print(k, rel(g[k], c[k]))
    print("loss", abs(g["loss"] - c["loss"]) / (abs(c["loss"]) + 1e-12), g["loss"], c["loss"])


if __name__ == "__main__":
    main()

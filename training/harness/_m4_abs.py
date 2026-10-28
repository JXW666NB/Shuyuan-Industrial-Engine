# -*- coding: utf-8 -*-
import sys
import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T


def arr(x):
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return np.array(x, np.float64)


def snap(device):
    T.set_device(device)
    np.random.seed(555)
    lin1 = T.nn.Linear(8, 8)
    lin2 = T.nn.Linear(8, 4)
    opt = T.optim.Muon([lin1.weight, lin2.weight], lr=0.02)
    g = np.random.RandomState(11)
    X = (g.randn(12, 8) * 0.5).astype(np.float32)
    tgt = g.randint(0, 4, size=12).astype(np.int64)
    y = lin2(T.functional.silu(lin1(T.Tensor(X))))
    loss = T.functional.cross_entropy(y, tgt)
    lin1.zero_grad(); lin2.zero_grad()
    loss.backward()
    out = {
        "g2": arr(lin2.weight.grad),
        "w2_before": arr(lin2.weight),
    }
    opt.step()
    out["w2"] = arr(lin2.weight)
    T.set_device("cpu")
    return out


def main():
    c = snap("cpu")
    g = snap("cuda")
    dw = np.abs(g["w2"] - c["w2"])
    dg = np.abs(g["g2"] - c["g2"])
    print("w2 max", float(np.max(np.abs(c["w2"]))), "dw max", float(dw.max()), "at", np.unravel_index(dw.argmax(), dw.shape))
    print("g2 max", float(np.max(np.abs(c["g2"]))), "dg max", float(dg.max()), "at", np.unravel_index(dg.argmax(), dg.shape))
    print("dw/g scale", float(dw.max()) / (float(dg.max()) + 1e-30))
    print("rel", float(dw.max()) / (float(np.max(np.abs(c["w2"]))) + 1e-12))


if __name__ == "__main__":
    main()

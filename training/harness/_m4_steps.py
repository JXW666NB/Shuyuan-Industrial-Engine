# -*- coding: utf-8 -*-
import sys
import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T

D, C, N = 8, 4, 12
SEED = 555


def rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def one(device):
    T.set_device(device)
    np.random.seed(SEED)
    lin1 = T.nn.Linear(D, D)
    lin2 = T.nn.Linear(D, C)
    opt = T.optim.Muon([lin1.weight, lin2.weight], lr=0.02)
    g = np.random.RandomState(11)
    X = (g.randn(N, D) * 0.5).astype(np.float32)
    tgt = g.randint(0, C, size=N).astype(np.int64)
    snaps = []
    for step in range(3):
        y = lin2(T.functional.silu(lin1(T.Tensor(X))))
        loss = T.functional.cross_entropy(y, tgt)
        lin1.zero_grad()
        lin2.zero_grad()
        loss.backward()
        g1 = np.array(lin1.weight.grad.to_numpy() if hasattr(lin1.weight.grad, "to_numpy") else lin1.weight.grad, np.float64)
        g2 = np.array(lin2.weight.grad.to_numpy() if hasattr(lin2.weight.grad, "to_numpy") else lin2.weight.grad, np.float64)
        opt.step()
        w1 = np.array(lin1.weight.to_numpy(), np.float64)
        w2 = np.array(lin2.weight.to_numpy(), np.float64)
        snaps.append((float(np.array(loss.to_numpy()).reshape(-1)[0]), g1, g2, w1, w2))
    T.set_device("cpu")
    return snaps


def main():
    cpu = one("cpu")
    cuda = one("cuda")
    for i, (a, b) in enumerate(zip(cpu, cuda)):
        print("step", i,
              "loss", rel(b[0], a[0]),
              "g1", rel(b[1], a[1]),
              "g2", rel(b[2], a[2]),
              "w1", rel(b[3], a[3]),
              "w2", rel(b[4], a[4]))


if __name__ == "__main__":
    main()

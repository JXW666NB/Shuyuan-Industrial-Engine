# -*- coding: utf-8 -*-
import sys
import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T


def main():
    rng = np.random.RandomState(1)
    x = rng.randn(12, 8).astype(np.float32)
    logits = rng.randn(12, 4).astype(np.float32)
    tgt = rng.randint(0, 4, size=12).astype(np.int64)

    def silu(device):
        T.set_device(device)
        t = T.Tensor(x, requires_grad=True)
        y = T.functional.silu(t)
        (y * y).sum().backward()
        out = (np.array(y.to_numpy(), np.float32), np.array(t.grad if device == "cpu" else t.grad.to_numpy(), np.float32))
        T.set_device("cpu")
        return out

    def ce(device):
        T.set_device(device)
        t = T.Tensor(logits, requires_grad=True)
        loss = T.functional.cross_entropy(t, tgt)
        loss.backward()
        g = t.grad if device == "cpu" else t.grad.to_numpy()
        out = (float(np.array(loss.to_numpy()).reshape(-1)[0]), np.array(g, np.float32))
        T.set_device("cpu")
        return out

    sc, sg = silu("cpu"), silu("cuda")
    print("silu y", int(np.sum(sc[0] != sg[0])), "dy", int(np.sum(sc[1] != sg[1])))
    cc, cg = ce("cpu"), ce("cuda")
    print("ce loss", cc[0], cg[0], "eq", cc[0] == cg[0])
    print("ce g", int(np.sum(cc[1] != cg[1])), "maxabs", float(np.max(np.abs(cc[1] - cg[1]))))


if __name__ == "__main__":
    main()

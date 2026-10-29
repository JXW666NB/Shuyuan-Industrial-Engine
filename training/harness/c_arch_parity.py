# -*- coding: utf-8 -*-
"""C 核心的架构描述符判分台：同一份 C 前向，用描述符区分 llama / qwen2。

链路：C 前向  vs  已验证的 Python 参考  vs  PyTorch
  - Python 参考对 PyTorch 的误差已实测 ~2e-7（llama_parity / gpt2_parity）
  - 所以判据取 **C vs Python 参考 < 1e-4** 即可

判据：每个架构 < 1e-4；自证：C 连跑两次必须逐位一致（差 0）。

用法：python c_arch_parity.py --selftest | --compare
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, HERE)

import tgai_go.train as T            # noqa: E402
import llama_ref                     # noqa: E402
import gpt2_ref                      # noqa: E402

TOL = 1e-4
ARCHS = ("llama", "qwen2", "gpt2")


def _np(x):
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy(), np.float64)
    return np.asarray(x, np.float64)


def make_cfg(arch):
    """各架构的小配置（权重固定 seed，判分台可复现）。"""
    if arch == "gpt2":
        return gpt2_ref.make_cfg(d=32, L=2, nh=4, ff=64, V=64, seq=16, tie=False, seed=0)
    kw = dict(d=32, L=2, nh=4, n_kv=2, ff=64, V=64, seq=16, theta=10000.0, seed=0)
    if arch == "qwen2":
        kw["qkv_bias"] = True
    return llama_ref.make_cfg(**kw)


def ref_logits(arch, cfg, ids):
    """已验证的 Python 参考。"""
    if arch == "gpt2":
        m = gpt2_ref.GPT2(cfg, cfg["weights"])
    else:
        m = llama_ref.Llama(cfg, cfg["weights"])
    return _np(m.forward(ids))


def c_logits(arch, cfg, ids):
    """C 核心（presets）。返回 None 表示该架构在 C 侧还没实现。"""
    from tgai_go import presets
    fn = getattr(presets, arch, None)
    if fn is None:
        return None
    m = fn(cfg)
    return _np(m.forward(ids))


def make_inputs(cfg):
    g = np.random.RandomState(1)
    ids = g.randint(0, cfg["V"], size=(2, cfg["seq"])).astype(np.int64)
    tgt = g.randint(0, cfg["V"], size=2 * cfg["seq"]).astype(np.int64)
    return ids, tgt


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    T.set_device("cuda" if a.cuda else "cpu")

    if a.selftest:
        print("[判分台自证] C 前向连跑两次必须逐位一致")
        worst = 0.0
        for arch in ARCHS:
            cfg = make_cfg(arch)
            ids, _ = make_inputs(cfg)
            r1 = c_logits(arch, cfg, ids)
            if r1 is None:
                print("  %-8s C 侧未实现（跳过自证）" % arch)
                continue
            r2 = c_logits(arch, cfg, ids)
            d = float(np.max(np.abs(r1 - r2)))
            worst = max(worst, d)
            print("  %-8s 两次差 = %.3e" % (arch, d))
        ok = worst == 0.0
        print("  VERDICT ok=%s（必须恰好为 0）" % ok)
        return 0 if ok else 1

    if a.compare:
        print("[C 架构描述符考核] C 前向 vs Python 参考")
        allok = True
        for arch in ARCHS:
            cfg = make_cfg(arch)
            ids, _ = make_inputs(cfg)
            ref = ref_logits(arch, cfg, ids)
            got = c_logits(arch, cfg, ids)
            if got is None:
                print("  %-8s C 侧未实现  FAIL" % arch)
                allok = False
                continue
            r = rel(got, ref)
            okk = r < TOL
            allok = allok and okk
            print("  %-8s 相对误差 = %.3e  %s" % (arch, r, "PASS" if okk else "FAIL"))
        print("  VERDICT ok=%s（判据 <%.0e）" % (allok, TOL))
        return 0 if allok else 1

    print("请指定 --selftest 或 --compare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

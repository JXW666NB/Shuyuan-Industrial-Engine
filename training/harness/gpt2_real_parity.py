# -*- coding: utf-8 -*-
"""B2 的最后一环：加载**真实 GPT-2 权重**，与 PyTorch 参考实现对齐 logits。

权重：`gpt2`（124M）官方 `pytorch_model.bin`（522.7 MB），本地路径见 `--bin`。
真实 GPT-2 的关键事实（与我们的默认配置不同，必须显式对齐）：
- `wte` 与 `lm_head` **共享**（tie）⇒ 用 tie=True
- 所有 Linear 在 HF 里是 **Conv1D**（权重形状 `[in, out]`）⇒ 要**转置**成我们的 `[out, in]`
- `layer_norm_epsilon = 1e-5`、激活 `gelu_new`（tanh 近似）、位置编码是学习式 `wpe`

输入用 GPT-2 分词器的已知序列（"Hello, my dog is cute"），避免依赖分词器实现：
  [15496, 11, 616, 3290, 318, 13779]

用法：
  python gpt2_real_parity.py --bin <pytorch_model.bin>
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
import gpt2_ref                      # noqa: E402

TOL = 1e-4
IDS = [15496, 11, 616, 3290, 318, 13779]      # GPT-2 tokens for "Hello, my dog is cute"


def map_weights(sd):
    """HF GPT-2 state_dict -> 我们 gpt2_ref 的命名（并做 Conv1D -> Linear 转置）。

    兼容两种前缀：`transformer.h.0...`（新版 HF）与 `h.0...`（本文件实测下载的这个）。
    跳过 `attn.bias` / `attn.masked_bias`（因果掩码 buffer，不参与计算）。
    """
    def _keep(k):
        # ★ 只跳过"因果掩码 buffer"：它的最后两段恰好是 attn.bias / attn.masked_bias。
        #   注意不能用 k.endswith("attn.bias") —— attn.c_attn.bias（真参数）也会被误删。
        parts = k.split(".")
        if len(parts) >= 2 and parts[-2:] == ["attn", "bias"]:
            return False
        if len(parts) >= 2 and parts[-2:] == ["attn", "masked_bias"]:
            return False
        return True

    sd = {(k[len("transformer."):] if k.startswith("transformer.") else k): v
          for k, v in sd.items() if _keep(k)}
    w = {}
    w["wte.weight"] = sd["wte.weight"].float().numpy()
    w["wpe.weight"] = sd["wpe.weight"].float().numpy()
    w["ln_f.weight"] = sd["ln_f.weight"].float().numpy()
    w["ln_f.bias"] = sd["ln_f.bias"].float().numpy()
    nL = 0
    while ("h.%d.ln_1.weight" % nL) in sd:
        nL += 1
    for i in range(nL):
        h = "h.%d." % i
        b = "blocks.%d." % i
        w[b + "ln_1.weight"] = sd[h + "ln_1.weight"].float().numpy()
        w[b + "ln_1.bias"] = sd[h + "ln_1.bias"].float().numpy()
        # Conv1D: [in, out] -> 我们的 [out, in]
        w[b + "attn.qkv.weight"] = sd[h + "attn.c_attn.weight"].float().numpy().T.copy()
        w[b + "attn.qkv.bias"] = sd[h + "attn.c_attn.bias"].float().numpy()
        w[b + "attn.proj.weight"] = sd[h + "attn.c_proj.weight"].float().numpy().T.copy()
        w[b + "attn.proj.bias"] = sd[h + "attn.c_proj.bias"].float().numpy()
        w[b + "ln_2.weight"] = sd[h + "ln_2.weight"].float().numpy()
        w[b + "ln_2.bias"] = sd[h + "ln_2.bias"].float().numpy()
        w[b + "mlp.fc.weight"] = sd[h + "mlp.c_fc.weight"].float().numpy().T.copy()
        w[b + "mlp.fc.bias"] = sd[h + "mlp.c_fc.bias"].float().numpy()
        w[b + "mlp.proj.weight"] = sd[h + "mlp.c_proj.weight"].float().numpy().T.copy()
        w[b + "mlp.proj.bias"] = sd[h + "mlp.c_proj.bias"].float().numpy()
    return nL, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    T.set_device("cpu")

    import torch
    sd = torch.load(a.bin, map_location="cpu", weights_only=True)
    nL, W = map_weights(sd)
    D = W["wte.weight"].shape[1]
    V = W["wte.weight"].shape[0]
    print("真实权重: L=%d d=%d V=%d" % (nL, D, V))

    cfg = dict(d=D, L=nL, nh=D // 64, ff=W["blocks.0.mlp.fc.weight"].shape[0],
               V=V, seq=len(IDS), tie=True, seed=0, weights=W)
    ids = np.asarray([IDS], dtype=np.int64)
    tgt = np.asarray([IDS[1:] + [IDS[0]]], dtype=np.int64).reshape(-1)

    # 我们的实现
    m = gpt2_ref.GPT2(cfg, W)
    ours = np.asarray(m.forward(ids).to_numpy(), np.float64)
    print("我们的 logits", ours.shape)

    # PyTorch 参考（复用 gpt2_parity 里已验证过的那份实现）
    import gpt2_parity
    ref = gpt2_parity.run_torch(cfg, ids, tgt)["logits"]
    print("PyTorch logits", ref.shape)

    r = gpt2_parity.rel(ours, ref)
    print("真实 GPT-2 权重下 logits 相对误差 = %.3e" % r)
    a_ours = np.argmax(ours.reshape(-1, ours.shape[-1]), axis=-1)
    print("前 3 个位置的 argmax（我们）:", a_ours[:3].tolist())
    a_ref = np.argmax(ref.reshape(-1, ref.shape[-1]), axis=-1)
    print("前 3 个位置的 argmax（torch）:", a_ref[:3].tolist())
    print("argmax 完全一致:", bool(np.array_equal(a_ours, a_ref)))
    ok = r < TOL
    print("  VERDICT ok=%s（判据 <%.0e）" % (ok, TOL))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

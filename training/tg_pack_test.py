# -*- coding: utf-8 -*-
"""`.TG` 打包回环自证：写 → 读 → 逐元素比对（含 tokenizer / chat meta / fp16 精度）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tgai_pack import TGWriter          # noqa: E402
from tg_read_tg import read_tg, load_tensor   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tgtest")


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "demo.tg")
    rng = np.random.default_rng(0)

    # 一个小模型（形状贴近真实命名规范）
    d, ff, V, E, L = 32, 64, 40, 2, 2
    w = {
        "token_embedding.weight": rng.standard_normal((V, d)).astype(np.float32) * 0.1,
        "final_norm.weight": np.ones(d, np.float32),
    }
    for i in range(L):
        w[f"blocks.{i}.ln1.weight"] = np.ones(d, np.float32)
        w[f"blocks.{i}.ln2.weight"] = np.ones(d, np.float32)
        w[f"blocks.{i}.attn.q_proj.weight"] = rng.standard_normal((d, d)).astype(np.float32) * 0.1
        w[f"blocks.{i}.attn.k_proj.weight"] = rng.standard_normal((d, d)).astype(np.float32) * 0.1
        w[f"blocks.{i}.attn.v_proj.weight"] = rng.standard_normal((d, d)).astype(np.float32) * 0.1
        w[f"blocks.{i}.attn.out_proj.weight"] = rng.standard_normal((d, d)).astype(np.float32) * 0.1
        w[f"blocks.{i}.moe.router.weight"] = rng.standard_normal((E, d)).astype(np.float32) * 0.1
        w[f"blocks.{i}.moe._qb_bias"] = rng.standard_normal((E,)).astype(np.float32) * 0.01
        w[f"blocks.{i}.moe.gate.weight"] = rng.standard_normal((E, d)).astype(np.float32) * 0.1
        for e in range(E):
            w[f"blocks.{i}.moe.experts.{e}.w1.weight"] = rng.standard_normal((ff, d)).astype(np.float32) * 0.1
            w[f"blocks.{i}.moe.experts.{e}.w2.weight"] = rng.standard_normal((ff, d)).astype(np.float32) * 0.1
            w[f"blocks.{i}.moe.experts.{e}.w3.weight"] = rng.standard_normal((d, ff)).astype(np.float32) * 0.1

    tokens = [chr(ord("a") + i) for i in range(V - 3)] + ["<pad>", "<bos>", "<eos>"]
    merges = [(0, 1), (2, 3)]

    wr = TGWriter(n_layers=L, d_model=d, n_heads=4, d_ff=ff, vocab=V, max_seq=128,
                  n_experts=E, n_activated=2, rope_theta=10000,
                  pad=V - 3, bos=V - 2, eos=V - 1, unk=0, weight_dtype=1)
    wr.add_weights(w)
    wr.set_tokenizer(tokens, merges=merges, special_ids=[V - 3, V - 2, V - 1, 0])
    wr.set_chat_meta({5: b"temp=0.7,top_p=0.9", 4: b"<pad>,<bos>,<eos>"})
    info = wr.write(path)
    print("写出: %(path)s  %(size)d B  权重 %(n_weights)d 项" % info)

    rd = read_tg(path)
    h = rd["header"]
    print("\n=== header 回读 ===")
    print("  magic=0x%08X version=%d n_layers=%d d_model=%d n_heads=%d d_ff=%d"
          % (h["magic"], h["version"], h["n_layers"], h["d_model"], h["n_heads"], h["d_ff"]))
    print("  vocab=%d max_seq=%d n_experts=%d topk=%d rope_theta=%d arch=%d"
          % (h["vocab"], h["max_seq"], h["n_experts"], h["n_activated"], h["rope_theta"], h["arch"]))
    print("  pad/bos/eos/unk = %d/%d/%d/%d"
          % (h["pad"], h["bos"], h["eos"], h["unk"]))
    ok_hdr = (h["magic"] == 0x49414754 and h["n_layers"] == L and h["d_model"] == d
              and h["vocab"] == V and h["n_experts"] == E and h["n_activated"] == 2
              and h["pad"] == V - 3 and h["bos"] == V - 2 and h["eos"] == V - 1)

    print("\n=== tokenizer 回读 ===")
    ok_tok = (rd["tokens"] == tokens and rd["merges"] == merges
              and rd["specials"][:4] == [V - 3, V - 2, V - 1, 0])
    print("  tokens 一致=%s merges 一致=%s specials[:4]=%s"
          % (rd["tokens"] == tokens, rd["merges"] == merges, rd["specials"][:4]))

    print("\n=== 权重回读（fp16 量化误差）===")
    worst = 0.0
    worst_name = ""
    for name, arr in w.items():
        got = load_tensor(rd, name)
        if got is None:
            print("  缺失:", name)
            worst = 1e9
            break
        if got.shape != arr.shape:
            print("  形状不符:", name, got.shape, arr.shape)
            worst = 1e9
            break
        e = np.abs(got - arr).max() / (np.abs(arr).max() + 1e-12)
        if e > worst:
            worst, worst_name = e, name
    print("  共 %d 项，最大相对误差 = %.3e（%s）" % (len(w), worst, worst_name))

    print("\n=== chat meta 回读 ===")
    ok_meta = rd["meta"].get(5) == b"temp=0.7,top_p=0.9"
    print("  meta keys=%s  type5 一致=%s" % (list(rd["meta"].keys()), ok_meta))

    ok = ok_hdr and ok_tok and worst < 1e-3 and ok_meta
    print("\nVERDICT ok=%s header=%s tokenizer=%s weights=%.2e meta=%s"
          % (ok, ok_hdr, ok_tok, worst, ok_meta))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""反查真模型权重表里到底有哪些参数名（表里只存 FNV-1a 哈希，用候选名去撞）。

用法: python tg_probe_names.py <big.tg> [n_layers]
"""
import os
import struct
import sys

def fnv1a64(name: str) -> int:
    h = 0xCBF29CE484222325
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main(path, L=None):
    fsz = os.path.getsize(path)
    with open(path, "rb") as f:
        h = struct.unpack("<16I6Q4I", f.read(128))
        (magic, version, wdt, n_layers, d_model, n_heads, d_ff, vocab, max_seq,
         n_experts, n_act, rope_theta, pad, bos, eos, unk,
         tok_off, tok_len, wt_off, wt_cnt, wd_off, wd_len, arch, nkv, modflag, ext_off) = h
        print(f"文件 {fsz/2**30:.1f} GB  magic=0x{magic:08X} ver={version} L={n_layers} "
              f"d={d_model} nh={n_heads} ff={d_ff} V={vocab} seq={max_seq} E={n_experts} "
              f"topk={n_act} theta={rope_theta} arch={arch} dtype={wdt}")
        print("权重表: %d 项 @ %d" % (wt_cnt, wt_off))
        f.seek(wt_off)
        table = []
        for _ in range(wt_cnt):
            rec = f.read(52)
            nh, off, ln, dt, nd = struct.unpack_from("<QQQII", rec, 0)
            shape = struct.unpack_from("<IIII", rec, 32)
            table.append((nh, off, ln, dt, nd, shape))
    hashes = {t[0]: t for t in table}

    LL = L if L else n_layers
    cands = ["token_embedding.weight", "lm_head.weight", "final_norm.weight",
             "output_norm.weight", "norm.weight"]
    for i in range(min(LL, 2)):
        for base in ("blocks.%d.ln1.weight" % i, "blocks.%d.ln2.weight" % i,
                     "blocks.%d.attn_norm.weight" % i, "blocks.%d.ffn_norm.weight" % i,
                     "blocks.%d.attn.q_proj.weight" % i, "blocks.%d.attn.k_proj.weight" % i,
                     "blocks.%d.attn.v_proj.weight" % i, "blocks.%d.attn.out_proj.weight" % i,
                     "blocks.%d.attn.rope.cos" % i, "blocks.%d.attn.rope.sin" % i,
                     "blocks.%d.attn.rope_cos" % i, "blocks.%d.attn.rope_sin" % i,
                     "blocks.%d.moe.router.weight" % i, "blocks.%d.moe.gate.weight" % i,
                     "blocks.%d.moe._qb_bias" % i, "blocks.%d.moe.qb_bias" % i,
                     "blocks.%d.moe.experts.0.w1.weight" % i,
                     "blocks.%d.moe.experts.0.w2.weight" % i,
                     "blocks.%d.moe.experts.0.w3.weight" % i,
                     "blocks.%d.moe.experts.0.gate_proj.weight" % i,
                     "blocks.%d.moe.experts.0.up_proj.weight" % i,
                     "blocks.%d.moe.experts.0.down_proj.weight" % i):
            cands.append(base)

    print("\n%-42s %-10s %s" % ("候选名", "存在", "shape/len"))
    print("-" * 78)
    found = {}
    for c in cands:
        hh = fnv1a64(c)
        if hh in hashes:
            nh, off, ln, dt, nd, shape = hashes[hh]
            found[c] = shape[:nd]
            print("%-42s %-10s shape=%s len=%d dtype=%d" % (c, "YES", shape[:nd], ln, dt))
    print("\n命中 %d / %d 个候选名" % (len(found), len(cands)))

    # 统计表里总项数，判断是否还有我们没猜到的
    print("表内总项数 = %d；已知命名可解释 %d 项（按 L=%d、E=%d 估）"
          % (wt_cnt, len(found), n_layers, n_experts))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None))

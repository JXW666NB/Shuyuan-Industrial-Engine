# -*- coding: utf-8 -*-
"""`.TG` 回读校验器：解析我们写出的文件，与源张量逐元素比对。

用法:
    python tg_read_tg.py <file.tg> [--dump]
"""
import struct
import sys

import numpy as np

TG_MAGIC = 0x49414754
DT = {0: ("fp32", 4, np.float32), 1: ("fp16", 2, np.float16),
      2: ("int8", 1, np.int8), 3: ("int32", 4, np.int32)}
CHAT_MAGIC = 0x46434754


def fnv1a64(name: str) -> int:
    h = 0xCBF29CE484222325
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def read_tg(path):
    raw = open(path, "rb").read()
    h = struct.unpack("<16I6Q4I", raw[:128])
    (magic, version, wdt, n_layers, d_model, n_heads, d_ff, vocab, max_seq,
     n_experts, n_act, rope_theta, pad, bos, eos, unk,
     tok_off, tok_len, wt_off, wt_cnt, wd_off, wd_len, arch, nkv, modflags, ext_off) = h
    assert magic == TG_MAGIC, hex(magic)

    # tokenizer
    o = tok_off
    v_size, n_merges = struct.unpack_from("<II", raw, o)
    o += 8
    merges = []
    for _ in range(n_merges):
        a, b = struct.unpack_from("<HH", raw, o)
        o += 4
        merges.append((a, b))
    tokens = []
    for _ in range(v_size):
        ln = raw[o]
        o += 1
        tokens.append(raw[o:o + ln].decode("utf-8"))
        o += ln
    specials = list(raw[o:o + 16])
    o += 16

    # weights
    table = []
    for i in range(wt_cnt):
        p = wt_off + i * 52
        nh, off, ln, dt, nd = struct.unpack_from("<QQQII", raw, p)
        shape = list(struct.unpack_from("<IIII", raw, p + 32))
        (sc,) = struct.unpack_from("<I", raw, p + 48)
        table.append(dict(hash=nh, off=off, len=ln, dtype=dt, ndim=nd,
                          shape=shape[:nd] if nd else shape, scale=sc))

    # chat meta
    meta = {}
    if ext_off:
        oo = ext_off
        m, ver, cnt = struct.unpack_from("<III", raw, oo)
        assert m == CHAT_MAGIC, "chat meta magic 不对: %x" % m
        oo += 12
        for _ in range(cnt):
            t, ln = struct.unpack_from("<II", raw, oo)
            oo += 8
            meta[t] = raw[oo:oo + ln]
            oo += ln

    return dict(header=dict(magic=magic, version=version, weight_dtype=wdt,
                            n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                            d_ff=d_ff, vocab=vocab, max_seq=max_seq,
                            n_experts=n_experts, n_activated=n_act,
                            rope_theta=rope_theta, pad=pad, bos=bos, eos=eos, unk=unk,
                            arch=arch, num_kv_heads=nkv, modality_flags=modflags,
                            extension_off=ext_off),
            tokens=tokens, merges=merges, specials=specials,
            table=table, data_off=wd_off, raw=raw, meta=meta)


def load_tensor(info, name):
    """按参数名取出张量（用 FNV-1a 匹配）"""
    want = fnv1a64(name)
    for e in info["table"]:
        if e["hash"] == want:
            _, esize, np_t = DT[e["dtype"]]
            start = info["data_off"] + e["off"]
            buf = info["raw"][start:start + e["len"]]
            return np.frombuffer(buf, dtype=np_t).reshape(e["shape"]).astype(np.float32)
    return None


def main(path):
    info = read_tg(path)
    h = info["header"]
    print("=== header ===")
    for k, v in h.items():
        print("  %-16s %s" % (k, v))
    print("=== tokenizer ===")
    print("  vocab=%d merges=%d specials[:8]=%s" % (len(info["tokens"]), len(info["merges"]),
                                                    info["specials"][:8]))
    print("  前 10 个 token: %s" % info["tokens"][:10])
    print("=== weights (%d 项) ===" % len(info["table"]))
    for i, e in enumerate(info["table"][:12]):
        print("  #%-2d off=%-8d len=%-8d dtype=%-2d shape=%s"
              % (i, e["off"], e["len"], e["dtype"], e["shape"]))
    if info["meta"]:
        print("=== chat meta ===")
        for t, d in info["meta"].items():
            print("  type=%-2d len=%-4d %r" % (t, len(d), d[:40]))
    return info


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])

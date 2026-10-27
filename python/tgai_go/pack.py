# -*- coding: utf-8 -*-
"""`.TG` 打包器：按 tgai_format.h 写 header + tokenizer + 权重表 + 权重数据 + chat meta。

字段来源（已核对源码）：
  TGHeader        embedded/TGAI_GO/src/tgai_format.h:166  (128B, pack(1))
  TGWeightEntry   同文件 (52B)：name_hash/offset/length/dtype/ndim/shape[4]/scale
  tokenizer 段    同文件注释：u32 vocab_size, u32 n_merges, merges(u16,u16)*n,
                  tokens(u8 len + bytes)*vocab, u8[16] special ids
  chat meta 段    src/tgai_chat_meta.c：12B 头 [magic,version,count] + [u32 type,u32 len,data]*
                  由 header.extension_off 定位（u32；>=2^31 表示相对文件尾的负数偏移）
  name_hash       src/tgai_engine.c fnv1a_64（seed 0xcbf29ce484222325, prime 0x100000001b3）

用法:
    from tgai_pack import TGWriter
    w = TGWriter(n_layers=1, d_model=256, n_heads=8, d_ff=1024, vocab=78,
                 max_seq=256, n_experts=1, n_activated=1, rope_theta=10000,
                 pad=0, bos=1, eos=2, unk=3)
    w.add_weights({"token_embedding.weight": emb, ...})
    w.set_tokenizer(vocab_list, merges=[(a,b), ...])
    w.set_chat_meta({5: b"...", 4: b"..."})
    w.write("model.tg")
"""
import struct

import numpy as np

TG_MAGIC = 0x49414754
TG_VERSION = 1
TG_ARCH_TGAI_MOE = 0
DT_FP32, DT_FP16, DT_INT8, DT_INT32 = 0, 1, 2, 3
TG_CHAT_META_MAGIC = 0x46434754
TG_CHAT_META_VERSION = 1


def fnv1a64(name: str) -> int:
    h = 0xCBF29CE484222325
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


class TGWriter:
    def __init__(self, n_layers, d_model, n_heads, d_ff, vocab, max_seq,
                 n_experts=8, n_activated=2, rope_theta=10000,
                 pad=0, bos=1, eos=2, unk=3, n_kv_heads=0, arch=TG_ARCH_TGAI_MOE,
                 weight_dtype=DT_FP16):
        self.cfg = dict(n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                        vocab=vocab, max_seq=max_seq, n_experts=n_experts,
                        n_activated=n_activated, rope_theta=rope_theta,
                        pad=pad, bos=bos, eos=eos, unk=unk, n_kv_heads=n_kv_heads,
                        arch=arch, weight_dtype=weight_dtype)
        self.weights = []          # [(name, np.ndarray, dtype)]
        self.vocab_tokens = []
        self.merges = []
        self.specials = [0] * 16
        self.chat_meta = {}        # {type: bytes}

    # ---------- 内容 ----------
    def add_weight(self, name, arr):
        a = np.ascontiguousarray(arr)
        self.weights.append((name, a))

    def add_weights(self, mapping):
        for k, v in mapping.items():
            self.add_weight(k, v.detach().float().cpu().numpy() if hasattr(v, "detach") else v)

    def set_tokenizer(self, tokens, merges=None, special_ids=None):
        self.vocab_tokens = list(tokens)
        self.merges = list(merges or [])
        if special_ids:
            for i, v in enumerate(special_ids[:16]):
                self.specials[i] = int(v)

    def set_chat_meta(self, entries: dict):
        """{type_int: bytes}"""
        self.chat_meta = dict(entries)

    # ---------- 序列化 ----------
    def _tokenizer_bytes(self):
        out = bytearray()
        out += struct.pack("<II", len(self.vocab_tokens), len(self.merges))
        for a, b in self.merges:
            out += struct.pack("<HH", a, b)
        for tok in self.vocab_tokens:
            bs = tok.encode("utf-8")
            if len(bs) > 255:
                raise ValueError("token 过长: %r" % tok)
            out += struct.pack("<B", len(bs)) + bs
        sp = list(self.specials[:16])
        sp += [0] * (16 - len(sp))
        out += bytes(sp)
        return bytes(out)

    def _chat_meta_bytes(self):
        if not self.chat_meta:
            return b""
        out = bytearray()
        out += struct.pack("<III", TG_CHAT_META_MAGIC, TG_CHAT_META_VERSION, len(self.chat_meta))
        for t, data in self.chat_meta.items():
            out += struct.pack("<II", int(t), len(data)) + data
        return bytes(out)

    def write(self, path):
        tok = self._tokenizer_bytes()
        meta = self._chat_meta_bytes()

        # 权重：按目标 dtype 转换
        blob = bytearray()
        entries = []
        dtype = self.cfg["weight_dtype"]
        for name, arr in self.weights:
            flat = arr.reshape(-1)
            if dtype == DT_FP16:
                data = flat.astype(np.float16).tobytes()
                dt = DT_FP16
            else:
                data = flat.astype(np.float32).tobytes()
                dt = DT_FP32
            off = len(blob)
            blob += data
            shape = list(arr.shape)[:4] + [1] * max(0, 4 - len(arr.shape))
            entries.append((fnv1a64(name), off, len(data), dt, min(len(arr.shape), 4),
                            shape, 0))

        tok_off = 128
        wt_off = tok_off + len(tok)
        wd_off = wt_off + 52 * len(entries)
        meta_off = wd_off + len(blob)
        ext_off = meta_off if meta else 0
        if ext_off >= 0x80000000:
            raise ValueError("文件过大，extension_off 需要负数编码（本项目暂不支持 >2GB）")

        hdr = struct.pack(
            "<16I6Q4I",
            TG_MAGIC, TG_VERSION, dtype,
            self.cfg["n_layers"], self.cfg["d_model"], self.cfg["n_heads"],
            self.cfg["d_ff"], self.cfg["vocab"], self.cfg["max_seq"],
            self.cfg["n_experts"], self.cfg["n_activated"], self.cfg["rope_theta"],
            self.cfg["pad"], self.cfg["bos"], self.cfg["eos"], self.cfg["unk"],
            tok_off, len(tok),
            wt_off, len(entries),
            wd_off, len(blob),
            self.cfg["arch"], self.cfg["n_kv_heads"], 0, ext_off)
        assert len(hdr) == 128, "header 长度 %d != 128" % len(hdr)

        with open(path, "wb") as f:
            f.write(hdr)
            f.write(tok)
            for (nh, off, ln, dt, nd, shape, sc) in entries:
                f.write(struct.pack("<QQQII", nh, off, ln, dt, nd))
                f.write(struct.pack("<IIII", *shape))
                f.write(struct.pack("<I", sc))
            f.write(blob)
            if meta:
                f.write(meta)
        return dict(path=path, size=128 + len(tok) + 52 * len(entries) + len(blob) + len(meta),
                    n_weights=len(entries), tokenizer_len=len(tok), has_meta=bool(meta))

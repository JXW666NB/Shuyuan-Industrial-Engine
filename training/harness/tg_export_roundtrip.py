# -*- coding: utf-8 -*-
"""A4-③ 闭环判分台：训练侧参数 → 写 `.TG` → 自家引擎加载 → 生成。

判据（本阶段）：
1. 导出**字节确定**：同样输入写两次，文件逐字节一致（自证）
2. `tgai_go.load()` 能加载
3. `generate()` 能跑出非空文本而不报错

更高一级的判据（训练侧 logits 与引擎逐值一致）需要引擎暴露 logits，
当前 `Model` 只有 generate/文本 API，故留待原生调试接口。

用法：python tg_export_roundtrip.py --selftest | --check
"""
import argparse
import hashlib
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "training"))

V, D, NH, FF, L, E, MAXSEQ = 64, 32, 4, 64, 1, 1, 64


def make_params(seed=0):
    """造一份"训练侧命名"的参数（形状与引擎一致）。"""
    g = np.random.RandomState(seed)
    s = 0.02

    def rn(*shape, scale=s):
        return (g.randn(*shape) * scale).astype(np.float32)

    p = {"token_embedding.weight": rn(V, D),
         "final_norm.weight": np.ones(D, np.float32)}
    for i in range(L):
        b = "blocks.%d." % i
        p[b + "ln1.weight"] = np.ones(D, np.float32)
        p[b + "ln2.weight"] = np.ones(D, np.float32)
        for nm in ("q_proj", "k_proj", "v_proj", "out_proj"):
            p[b + "attn.%s.weight" % nm] = rn(D, D)
        p[b + "moe.router.weight"] = rn(E, D)
        for e in range(E):
            p[b + "experts.%d.gate.weight" % e] = rn(FF, D)
            p[b + "experts.%d.up.weight" % e] = rn(FF, D)
            p[b + "experts.%d.down.weight" % e] = rn(D, FF)
    return p


def vocab_tokens():
    toks = ["<pad>", "<unk>", "<bos>", "<eos>"]
    for i in range(V - len(toks)):
        toks.append(chr(0x4E00 + i))          # 一批汉字，保证 decode 非空
    return toks


def export_to(path, seed=0):
    from tgai_go.train.export import save_tg
    save_tg(path, make_params(seed),
            n_layers=L, d_model=D, n_heads=NH, d_ff=FF,
            vocab_tokens=vocab_tokens(), merges=[], max_seq=MAXSEQ,
            n_experts=E, n_activated=1, rope_theta=10000.0,
            special_ids=[0, 1, 2, 3], weight_dtype=0)
    return path


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = os.path.join(tempfile.gettempdir(), "tg_roundtrip")
    os.makedirs(d, exist_ok=True)

    if a.selftest:
        p1 = export_to(os.path.join(d, "a.TG"))
        p2 = export_to(os.path.join(d, "b.TG"))
        h1, h2 = digest(p1), digest(p2)
        same = h1 == h2
        print("[自证] 两次导出字节一致: %s" % same)
        print("  %s  %d 字节" % (h1[:16], os.path.getsize(p1)))
        print("  VERDICT ok=%s（必须逐字节一致）" % same)
        return 0 if same else 1

    if a.check:
        ok_all = True
        p = export_to(os.path.join(d, "m.TG"))
        sz = os.path.getsize(p)
        print("[A4-③] 导出 %d 字节" % sz)
        try:
            import tgai_go
            m = tgai_go.load(p)
            print("  load OK: %s" % type(m).__name__)
            txt = m.generate("ab", max_tokens=6, temperature=0.0, top_k=1, top_p=1.0)
            print("  generate -> %r" % (txt,))
            ok_all = ok_all and bool(txt)
        except Exception as ex:
            print("  FAIL: %s: %s" % (type(ex).__name__, str(ex)[:200]))
            ok_all = False
        print("  VERDICT ok=%s" % ok_all)
        return 0 if ok_all else 1

    print("请指定 --selftest 或 --check")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

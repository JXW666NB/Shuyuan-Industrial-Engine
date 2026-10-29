# -*- coding: utf-8 -*-
"""从 TGAI 版生成 PyTorch 孪生版：**只替换 SHIM 段**，其余逐字节相同。

用法：
    python make_twins.py            # 生成 exam1/exam2 的 *_torch.py
    python make_twins.py --check    # 只校验现有孪生是否与生成结果一致

保证：`diff examN_*_tgai.py examN_*_torch.py` 的差异只出现在 SHIM 段。
"""
import argparse
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

IMPORT_TGAI = "# === SHIM-IMPORT-BEGIN ===\nimport tgai_go.train as torch\n# === SHIM-END ==="
IMPORT_TORCH = "# === SHIM-IMPORT-BEGIN ===\nimport torch\n# === SHIM-END ==="

SAVE_TGAI = '''def save_artifacts(path, params, vocab, meta):
    from tgai_go.train.export import save_tg
    save_tg(path, params, n_layers=meta["L"], d_model=meta["D"], n_heads=meta["NH"],
            d_ff=meta["FF"], vocab_tokens=vocab, merges=[], max_seq=64,
            n_experts=1, n_activated=1, rope_theta=10000.0,
            special_ids=[0, 1, 2, 3], weight_dtype=0)
    with open(os.path.splitext(path)[0] + ".tokens.txt", "w", encoding="utf-8") as f:
        for i, t in enumerate(vocab):
            f.write("%d\\t%s\\n" % (i, t))'''

SAVE_TORCH = '''def save_artifacts(path, params, vocab, meta):
    torch.save({"params": {k: torch.tensor(v) for k, v in params.items()},
                "vocab": vocab, "meta": meta}, path)
    with open(os.path.splitext(path)[0] + ".tokens.txt", "w", encoding="utf-8") as f:
        for i, t in enumerate(vocab):
            f.write("%d\\t%s\\n" % (i, t))'''

PAIRS = [("exam1_pelican_tgai.py", "exam1_pelican_torch.py"),
         ("exam2_chat10m_tgai.py", "exam2_chat10m_torch.py")]


def build(src_text):
    s = src_text
    assert IMPORT_TGAI in s, "找不到 SHIM① 标记"
    s = s.replace(IMPORT_TGAI, IMPORT_TORCH, 1)
    assert SAVE_TGAI in s, "找不到 SHIM② 函数体"
    s = s.replace(SAVE_TGAI, SAVE_TORCH, 1)
    # 输出默认名：.TG -> .pt（仅这一处与格式有关）
    s = s.replace('ap.add_argument("--out", default="pelican")',
                  'ap.add_argument("--out", default="pelican.pt")')
    s = s.replace('ap.add_argument("--out", default="chat10m")',
                  'ap.add_argument("--out", default="chat10m.pt")')
    s = s.replace('out_path = a.out + (".TG" if a.out.endswith("pelican") else "")', 'out_path = a.out')
    s = s.replace('out_path = a.out + (".TG" if a.out.endswith("chat10m") else "")', 'out_path = a.out')
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bad = 0
    for tg, to in PAIRS:
        p_tg = os.path.join(HERE, tg)
        p_to = os.path.join(HERE, to)
        if not os.path.exists(p_tg):
            print("%-28s 跳过（源不存在）" % tg)
            continue
        src = io.open(p_tg, encoding="utf-8", newline=None).read().replace("\\r\\n", "\\n")
        want = build(src)
        if a.check:
            have = (io.open(p_to, encoding="utf-8", newline=None).read().replace("\\r\\n", "\\n")
                    if os.path.exists(p_to) else "")
            ok = have == want
            bad += 0 if ok else 1
            print("%-28s -> %-28s %s" % (tg, to, "一致 ✅" if ok else "不一致 ❌"))
        else:
            io.open(p_to, "w", encoding="utf-8", newline="\n").write(want)
            print("%-28s -> %-28s 已生成" % (tg, to))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

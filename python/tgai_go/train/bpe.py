# -*- coding: utf-8 -*-
"""纯 Python BPE 训练/编码（随包分发版）。

来源：`training/bpe_train.py` 里的 `train_bpe` / `encode_ref`，逻辑逐行保留。
搬进包的原因：`training/` 不会随 wheel 分发，而一键流水线（B6）需要自训分词表 ——
如果按文件路径去读仓库里的 `bpe_train.py`，`pip install` 之后就会失败。

编码规则与引擎 `tgai_engine.c::tg_encode` 对齐：
  1) 特殊标记做贪心最长匹配，命中直接出 id
  2) 普通文本按 UTF-8 字符切成 base 序列
  3) 反复合并 **rank 最小** 的相邻对（rank = merges 里的顺序）
  4) 未命中的连续段按字节回退
"""
from collections import Counter


def train_bpe(text, target_vocab):
    """最朴素的 BPE：每次合并频次最高的相邻对，直到词表达到 target_vocab。"""
    chars = sorted(set(text))
    vocab = list(chars)
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = [stoi[c] for c in text]
    merges = []
    while len(vocab) < target_vocab:
        pairs = Counter(zip(ids, ids[1:]))
        if not pairs:
            break
        (a, b), cnt = pairs.most_common(1)[0]
        if cnt < 2:
            break
        new_tok = vocab[a] + vocab[b]
        vocab.append(new_tok)
        new_id = len(vocab) - 1
        merges.append((a, b))
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return vocab, merges


def encode_ref(text, vocab, merges):
    """按 merges 的 rank 顺序做合并（与训练时的规则一致）。"""
    stoi = {c: i for i, c in enumerate(vocab)}
    rank = {p: r for r, p in enumerate(merges)}
    ab = {}
    for (a, b) in merges:
        s = vocab[a] + vocab[b]
        ab[(a, b)] = stoi[s]
    ids = [stoi[ch] for ch in text]
    while True:
        best = None
        for i in range(len(ids) - 1):
            r = rank.get((ids[i], ids[i + 1]))
            if r is not None and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        i = best[1]
        ids[i:i + 2] = [ab[(ids[i], ids[i + 1])]]
    return ids

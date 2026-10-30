# -*- coding: utf-8 -*-
"""流式数据管线：语料按行读、按批建样本，内存只与 batch 有关。

为什么需要它：`build_examples()` 会把**整个语料**一次性物化成 (N, S) 的 int64 数组，
1 GB 语料约产生 9 GB 的 X/Y + 数百 MB 文本（实测 host 峰值 15–20 GB）。
10 GB 级语料直接不可用。

这里的做法：
  1. `iter_records()` 逐行解析，永不把全文留在内存；
  2. `build_vocab_stream()` 流式统计字符频次（一个 dict，几十 KB）；
  3. `batch_iter()` 每次只把 batch 条对话编码成 (B, S) 数组。

S 固定为 `seq`（不足补 eos、超出截断），这样 CUDA Graph / 形状缓存都能用。
本模块不依赖 tgai_go，也不依赖 torch —— 两边的训练脚本可以共用同一份。
"""
import json
import os

import numpy as np

BOS = "\x02"
EOS = "<eos>"
PAD = -100


def iter_records(paths, max_chars=None):
    """流式产出 (q, a)。规则与原实现一致：text 里要有 <TGAI>，回答 ≤40 字符。"""
    used = 0
    for fp in paths:
        if not fp or not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line).get("text", "")
                except Exception:
                    continue
                used += len(t)
                if "<TGAI>" not in t:
                    if max_chars and used >= max_chars:
                        return
                    continue
                qpart, _, apart = t.partition("<TGAI>")
                q = qpart.replace("<用户>", "", 1).strip()
                a = apart.replace("<回答>", "").strip()
                if not q or not a or len(a) > 40:
                    if max_chars and used >= max_chars:
                        return
                    continue
                yield q, a
                if max_chars and used >= max_chars:
                    return


def _row(q, a, stoi, seq):
    """把一条对话编成定长序列（与引擎 build_prompt 逐字符一致）。"""
    head = "%s\n%s<TGAI><回答>\n%s" % (BOS, q, a)
    ids = [stoi.get(ch, stoi.get("<unk>", 0)) for ch in head]
    ids.append(stoi.get(EOS, 0))
    if len(ids) > seq:
        ids = ids[:seq]
        ids[-1] = stoi.get(EOS, 0)
    n = len(ids)
    x = np.zeros((seq,), np.int64)
    y = np.full((seq,), PAD, np.int64)
    x[:n] = ids
    y[:n] = ids
    return x, y


def build_vocab_stream(paths, limit=4000, max_chars=None, min_count=1):
    """流式建词表：先统计字符频次，再按频次取前 limit 个。"""
    freq = {}
    total = 0
    for q, a in iter_records(paths, max_chars):
        for ch in "%s\n%s<TGAI><回答>\n%s" % (BOS, q, a):
            freq[ch] = freq.get(ch, 0) + 1
            total += 1
    items = [(c, n) for c, n in freq.items() if n >= min_count]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    toks = [c for c, _ in items[:limit]]
    for special in (BOS, EOS, "<unk>"):
        if special not in toks:
            toks.append(special)
    stoi = {c: i for i, c in enumerate(toks)}
    return toks, stoi


def count_records(paths, max_chars=None):
    """数一遍有多少条可用对话（流式，不占内存）。"""
    n = 0
    for _ in iter_records(paths, max_chars):
        n += 1
    return n


def batch_iter(paths, stoi, seq, batch, max_chars=None, shuffle=True, seed=1234,
               epochs=None):
    """按批产出 (X, Y)，形状 (B, seq)。内存 O(batch)。

    shuffle=True 时用「缓冲池」洗牌：只在视野内打乱，不需要把全部样本读进内存。
    epochs=None 表示无限循环（训练按步数停）。
    """
    rng = np.random.RandomState(seed)
    buf = []
    ep = 0
    while True:
        produced = 0
        for q, a in iter_records(paths, max_chars):
            buf.append(_row(q, a, stoi, seq))
            produced += 1
            if len(buf) >= batch * 8:            # 攒够就洗一次，取出 batch 条
                if shuffle:
                    idx = rng.permutation(len(buf))
                    buf[:] = [buf[i] for i in idx]
                while len(buf) >= batch:
                    chunk = buf[:batch]
                    del buf[:batch]
                    yield (np.stack([c[0] for c in chunk]),
                           np.stack([c[1] for c in chunk]))
        if buf:                                   # 收尾：把剩下的发完（可能不足 B）
            if shuffle:
                idx = rng.permutation(len(buf))
                buf[:] = [buf[i] for i in idx]
            while buf:
                chunk = buf[:batch]
                del buf[:batch]
                yield (np.stack([c[0] for c in chunk]),
                       np.stack([c[1] for c in chunk]))
        ep += 1
        if epochs is not None and ep >= epochs:
            return
        if max_chars:                             # 小语料重复读意义不大，避免死循环
            if produced == 0:
                return

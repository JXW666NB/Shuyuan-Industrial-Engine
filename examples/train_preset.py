# -*- coding: utf-8 -*-
"""示例 ②：训练一个预设架构，**几行代码就看到 loss**。

    python examples/train_preset.py 语料目录/

全部工作都在这 4 行里：
    from tgai_go.train.pipeline import run
    r = run(corpus_dir=..., out_dir=..., arch="gpt2", steps=200)
    print(r["first_loss"], "->", r["final_loss"])
分词表由流水线**自训**（BPE），用户不需要准备。
"""
import sys
import tgai_go.train as T
from tgai_go.train.pipeline import run

corpus = sys.argv[1] if len(sys.argv) > 1 else "corpus"
T.set_device("cpu")          # 小配置 CPU 足够；要大模型把这里换成 "cuda"
r = run(corpus_dir=corpus, out_dir="run_demo", arch="gpt2",
        vocab_size=192, seq=32, steps=60, d_model=64, n_layers=1, n_heads=4, d_ff=128)
print("架构 gpt2 | 自训词表 %d | loss %.4f -> %.4f"
      % (r["vocab_size"], r["first_loss"], r["final_loss"]))

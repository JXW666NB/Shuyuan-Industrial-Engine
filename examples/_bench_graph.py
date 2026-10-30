# -*- coding: utf-8 -*-
"""只量：整图重放 vs 每步改 74 个 adamw 节点。"""
import importlib
import time

import numpy as np

exam = importlib.import_module("exam2_chat10m_tgai")
exam.T.set_default_device("cuda")
rt = exam.T._cuda()
assert rt.graph_arm(), rt.graph_stats()
text, _src = exam.load_corpus()
_toks, stoi = exam.build_vocab(text)
x, y = exam.build_examples(text, stoi, exam.CFG["MAXLEN"])
cfg = dict(exam.CFG, V=len(_toks), S=int(x.shape[1]))
params = exam.make_model(cfg, np.random.RandomState(0))
opt = exam.T.optim.AdamW(list(params.values()), lr=2e-3)

def step(numpy_target):
    logits = exam.forward(params, x, cfg)
    tgt = y.reshape(-1)
    flat = logits.reshape(cfg["S"] * x.shape[0], cfg["V"])
    if numpy_target:
        loss = exam.F.cross_entropy(flat, tgt, ignore_index=-100)
    else:
        loss = exam.F.cross_entropy(flat, exam.T.tensor(tgt), ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss

for _ in range(3):
    step(False)
assert rt.graph_begin()
loss = step(True)
assert rt.graph_end(), rt.graph_stats()
assert rt.graph_launch()
rt.sync()
print("stats", rt.graph_stats(), flush=True)

def bench(label, fn, n=30):
    for _ in range(5):
        fn()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    rt.sync()
    ms = (time.perf_counter() - t0) / n * 1000.0
    print("%s %.3f ms" % (label, ms), flush=True)

bench("launch_only", lambda: rt.graph_launch())
bench("launch_patch", lambda: rt.graph_launch(0.9, 0.6))
# 拆开 SetParams 和 launch
def patch_only():
    b1 = float(np.float32(0.8))
    b2 = float(np.float32(0.5))
    for node, params, holders, _ptrs in rt._adamw_nodes:
        holders[12].value = b1
        holders[13].value = b2
        rt._chk(rt.cu.cuGraphExecKernelNodeSetParams(
            rt._graph_exec, node, __import__("ctypes").byref(params)),
            "set")

bench("patch_only", patch_only)
bench("d2h_loss", lambda: loss.to_numpy())

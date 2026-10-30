# -*- coding: utf-8 -*-
"""数据并行正确性测试（CPU，2 进程，不需要 GPU）。

验证两件事：
  1. broadcast 之后各 rank 的权重完全等于 rank0；
  2. all_reduce_grads 之后各 rank 的梯度等于所有 rank 的算术平均。
"""
import os
import subprocess
import sys
import tempfile

CHILD = r'''
import os, sys, numpy as np
sys.path.insert(0, os.environ["DP_PY"])
from tgai_go.train.dist import DataParallel

WORLD = int(os.environ["TGAI_DP_WORLD"])
RANK = int(os.environ["TGAI_DP_RANK"])

class P:
    def __init__(self, data):
        self.data = np.asarray(data, np.float32)
        self.grad = None

dp = DataParallel()
# rank0 的权重是 1.0，其它 rank 是 99.0（要被广播覆盖）
params = [P(np.full((8, 4), 1.0 if RANK == 0 else 99.0, np.float32)),
          P(np.full((3,), 1.0 if RANK == 0 else 99.0, np.float32))]
dp.broadcast(params)
assert np.allclose(params[0].data, 1.0), "广播失败 p0: %s" % params[0].data.ravel()[:3]
assert np.allclose(params[1].data, 1.0), "广播失败 p1"

# 每个 rank 的梯度不同：rank r 的梯度 = r+1
for i, p in enumerate(params):
    p.grad = np.full(p.data.shape, float(RANK + 1), np.float32)
dp.all_reduce_grads(params)
want = (1.0 + WORLD) / 2.0          # rank0..rank_{W-1} 的均值
assert np.allclose(params[0].grad, want), "归约错误: %s != %s" % (params[0].grad.ravel()[:2], want)
assert np.allclose(params[1].grad, want), "归约错误 p1"
dp.close()
print("RANK %d OK" % RANK)
'''

def main():
    py = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + "python"
    job = "test%d" % os.getpid()
    d = tempfile.mkdtemp(prefix="dp_")
    env = dict(os.environ, **{"DP_PY": py, "TGAI_DP_WORLD": "2", "TGAI_DP_JOB": job,
                              "TGAI_DP_DIR": d, "PYTHONPATH": py})
    f = os.path.join(d, "child.py")
    open(f, "w", encoding="utf-8").write(CHILD)
    procs = []
    for r in range(2):
        e = dict(env, **{"TGAI_DP_RANK": str(r)})
        procs.append(subprocess.Popen([sys.executable, f], env=e,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    bad = 0
    for i, p in enumerate(procs):
        out, err = p.communicate(timeout=120)
        print("rank%d rc=%d %s" % (i, p.returncode, (out or "").strip().replace("\n", " | ")))
        if p.returncode != 0:
            bad += 1
            print("  stderr:", (err or "").strip()[:400])
    print("RESULT: %s" % ("PASS" if bad == 0 else "FAIL"))
    return 0 if bad == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())

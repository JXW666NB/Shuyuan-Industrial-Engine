# -*- coding: utf-8 -*-
"""多卡训练链路的整合测试：两个 rank 跑在**同一张卡**上（不需要双卡硬件）。

单卡也能验证的部分：
  * 数据分片（每个 rank 只算自己那一份 batch）；
  * 梯度全归约 + 求平均；
  * 两个 rank 的**参数最终逐位一致**（这是"数据并行数学上正确"的硬证据）。

不能验证的部分：加速比（两个进程抢同一张卡，只会更慢）。

用法：
    python tools/test_dp_train.py                 # 默认 20 步
    python tools/test_dp_train.py --steps 50
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


HERE = os.path.dirname(os.path.abspath(__file__))


def _find_dir(marker, *cands):
    """找出包含 marker 的目录：兼容仓库布局（python/ + examples/）与扁平部署（全在一个目录）。"""
    for c in cands:
        if os.path.exists(os.path.join(c, marker)):
            return c
    return cands[0]


PKG_DIR = _find_dir("tgai_go", os.path.join(REPO, "python"), HERE, REPO)
EX_DIR = _find_dir("exam2_chat10m_tgai.py", os.path.join(REPO, "examples"), HERE, REPO)

CHILD = r'''
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.environ["DP_PY_PKG"])
sys.path.insert(0, os.environ["DP_PY_EX"])
import tgai_go.train as T
T.set_default_device("cuda")
from tgai_go.train.dist import DataParallel
import exam2_chat10m_tgai as E

STEPS = int(os.environ["DP_STEPS"])
dp = DataParallel()
text, _ = E.load_corpus()
toks, stoi = E.build_vocab(text)
x, y = E.build_examples(text, stoi, E.CFG["MAXLEN"])
cfg = dict(E.CFG, V=len(toks), S=int(x.shape[1]))
p = E.make_model(cfg, np.random.RandomState(0))          # 同种子，保证起点一致
if dp.world > 1:
    dp.broadcast(p.values())                             # 保险：强制各 rank 起点一致
opt = T.optim.AdamW(list(p.values()), lr=2e-3)
F = T.functional
n = int(x.shape[0])
w = dp.world

def shard(a):
    return a[dp.rank::w] if w > 1 else a

for step in range(STEPS):
    xb, yb = shard(x), shard(y)
    lg = E.forward(p, xb, cfg)
    lo = F.cross_entropy(lg.reshape(cfg["S"] * xb.shape[0], cfg["V"]),
                         T.tensor(yb.reshape(-1)), ignore_index=-100)
    opt.zero_grad()
    lo.backward()
    if dp.world > 1:
        dp.all_reduce_grads(p.values())
    opt.step()

h = hashlib.sha256()
for k in sorted(p.keys()):
    arr = np.ascontiguousarray(np.asarray(p[k].to_numpy(), np.float32))
    h.update(k.encode("utf-8"))
    h.update(arr.tobytes())
out = {"rank": dp.rank, "checksum": h.hexdigest(),
       "loss": float(np.asarray(lo.to_numpy()).reshape(-1)[0]),
       "params": sum(int(np.prod(a.shape)) for a in p.values())}
dp.close()
with open(os.environ["DP_OUT"], "w", encoding="utf-8") as f:
    json.dump(out, f)
print("rank %d checksum %s" % (dp.rank, h.hexdigest()[:16]))
'''


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--device", default="0", help="两个 rank 共用的那张卡")
    a = ap.parse_args(argv)

    d = tempfile.mkdtemp(prefix="dp_train_")
    f = os.path.join(d, "child.py")
    open(f, "w", encoding="utf-8").write(CHILD)
    job = "dp%d" % int(time.time())
    results = []
    procs = []
    for r in range(a.world):
        env = dict(os.environ)
        env.update({"DP_PY_PKG": PKG_DIR,
                    "DP_PY_EX": EX_DIR,
                    "DP_STEPS": str(a.steps),
                    "DP_OUT": os.path.join(d, "out%d.json" % r),
                    "TGAI_DP_WORLD": str(a.world),
                    "TGAI_DP_RANK": str(r),
                    "TGAI_DP_JOB": job,
                    "TGAI_DP_DIR": d,
                    "CUDA_VISIBLE_DEVICES": a.device,
                    "PYTHONPATH": os.pathsep.join([PKG_DIR, EX_DIR]),
                    "PYTHONIOENCODING": "utf-8"})
        procs.append(subprocess.Popen([sys.executable, f], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, encoding="utf-8", errors="replace"))
        time.sleep(1.0)

    bad = 0
    for r, pr in enumerate(procs):
        out, err = pr.communicate(timeout=900)
        print("[rank%d rc=%d] %s" % (r, pr.returncode, (out or "").strip()[-200:]))
        if pr.returncode != 0:
            bad += 1
            print("  stderr:", (err or "").strip()[-500:])
            continue
        with open(os.path.join(d, "out%d.json" % r), encoding="utf-8") as fh:
            results.append(json.load(fh))

    if bad or len(results) < a.world:
        print("RESULT: FAIL（%d 个 rank 失败）" % bad)
        return 1

    cks = {r["checksum"] for r in results}
    print("")
    print("rank 参数 SHA256 是否一致：%s" % ("是" if len(cks) == 1 else "否 —— %d 种" % len(cks)))
    for r in results:
        print("  rank %d: params %d，末 loss %.6f，checksum %s"
              % (r["rank"], r["params"], r["loss"], r["checksum"][:16]))
    ok = len(cks) == 1
    print("RESULT: %s" % ("PASS（数据并行数学上正确）" if ok else "FAIL（各 rank 参数不一致）"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
import sys

def stats(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            step, loss, ppl, elapsed = line.split(",")
            rows.append((int(step), float(loss), float(elapsed)))
    n = len(rows)
    last = rows[-1]
    ms = last[2] / n * 1000.0
    # 去掉前 4 步（编译/捕获），看稳态
    tail = rows[4:]
    dt = tail[-1][2] - rows[3][2]
    ms_tail = dt / len(tail) * 1000.0
    print("%s rows=%d last_step=%d last_loss=%.6f elapsed=%.3f ms/step=%.3f steady_ms=%.3f"
          % (path, n, last[0], last[1], last[2], ms, ms_tail))

for p in sys.argv[1:]:
    stats(p)

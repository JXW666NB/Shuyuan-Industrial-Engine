#!/bin/bash
set -e
cd /mnt/workspace/tgai
export PYTHONPATH=/mnt/workspace/tgai/python
unset TG_BF16 TG_BF16_MAX_K TG_BF16_MIN_OUTPUT TG_FP16 TG_FP16_SPLIT TG_CUDA_GRAPH TG_CORPUS

echo "===== OFF ====="
python3 examples/exam2_chat10m_tgai.py --device cuda --steps 600 --out /tmp/tg_bf16_off --log-csv /tmp/tg_bf16_off.csv

echo "===== BF16_K1280 ====="
TG_BF16=1 python3 examples/exam2_chat10m_tgai.py --device cuda --steps 600 --out /tmp/tg_bf16_k1280 --log-csv /tmp/tg_bf16_k1280.csv

echo "===== BF16_K1e9 ====="
TG_BF16=1 TG_BF16_MAX_K=1000000000 python3 examples/exam2_chat10m_tgai.py --device cuda --steps 600 --out /tmp/tg_bf16_k1e9 --log-csv /tmp/tg_bf16_k1e9.csv

python3 - << 'PY'
import csv, math
def load(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["step"]), float(row["loss"]), float(row["elapsed_s"])))
    return rows
def summarize(path):
    rows = load(path)
    nan = sum(1 for _, loss, _ in rows if not math.isfinite(loss))
    ms_all = rows[-1][2] / len(rows) * 1000.0
    ms_steady = (rows[-1][2] - rows[5][2]) / (len(rows) - 6) * 1000.0
    print("FILE %s steps=%d first=%.6f last=%.6f nan=%d ms_all=%.3f ms_steady_from_step5=%.3f" % (
        path, len(rows), rows[0][1], rows[-1][1], nan, ms_all, ms_steady))
    return rows
off = summarize("/tmp/tg_bf16_off.csv")
k1280 = summarize("/tmp/tg_bf16_k1280.csv")
kbig = summarize("/tmp/tg_bf16_k1e9.csv")
base = off[-1][1]
for name, rows in (("k1280", k1280), ("k1e9", kbig)):
    last = rows[-1][1]
    dev = abs(last - base) / max(abs(base), 1e-12)
    print("DEVIATION %s last=%.6f base=%.6f rel=%.6f pct=%.4f%%" % (
        name, last, base, dev, dev * 100.0))
print("BASELINE_VS_00885 off_last=%.6f abs_diff=%.6f" % (base, abs(base - 0.0885)))
PY

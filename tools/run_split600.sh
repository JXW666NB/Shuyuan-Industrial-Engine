#!/bin/bash
set -u
cd /home/featurize/tgai/e3
export TG_CORPUS=/home/featurize/tgai/e3/SFT/persona.jsonl
export PYTHONPATH=/home/featurize/tgai/e3
unset TG_FP16
unset TG_FP16_SPLIT
echo "===== FP32 ====="
timeout 560 python3 -u exam2_chat10m_tgai.py --device cuda --steps 600 --out x_fp32.TG --log-csv x_fp32.csv
echo FP32_EXIT:$?
export TG_FP16_SPLIT=1
echo "===== SPLIT ====="
timeout 560 python3 -u exam2_chat10m_tgai.py --device cuda --steps 600 --out x_split.TG --log-csv x_split.csv
echo SPLIT_EXIT:$?
python3 - << 'PY'
import csv
for name in ("x_fp32.csv", "x_split.csv"):
    rows = list(csv.DictReader(open(name, encoding="utf-8")))
    last = rows[-1]
    n = len(rows)
    ms = float(last["elapsed_s"]) / n * 1000.0
    bad = [r for r in rows if r["loss"] in ("nan", "inf", "-inf")]
    print("%s nrows %d first %s last %s elapsed %s ms/step %.3f bad %d"
          % (name, n, rows[0]["loss"], last["loss"], last["elapsed_s"], ms, len(bad)))
PY

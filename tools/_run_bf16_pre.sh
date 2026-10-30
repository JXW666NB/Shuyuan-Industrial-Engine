#!/bin/bash
set +e
LIVE=/mnt/workspace/tgai/python/tgai_go/train/cuda.py
cp "$LIVE" /tmp/cuda.py.bf16
cp /tmp/cuda.py.pre "$LIVE"
cd /mnt/workspace/tgai
unset TG_BF16 TG_BF16_MAX_K TG_FP16 TG_FP16_SPLIT TG_CUDA_GRAPH TG_CORPUS
PYTHONPATH=/mnt/workspace/tgai/python python3 examples/exam2_chat10m_tgai.py --device cuda --steps 600 --out /tmp/tg_pre --log-csv /tmp/tg_pre.csv
rc=$?
cp /tmp/cuda.py.bf16 "$LIVE"
echo "RESTORE_MD5 $(md5sum "$LIVE")"
python3 - << 'PY'
import csv
rows=list(csv.DictReader(open("/tmp/tg_pre.csv", newline="")))
print("PRE first=%s last=%s steps=%d elapsed=%s" % (rows[0]["loss"], rows[-1]["loss"], len(rows), rows[-1]["elapsed_s"]))
PY
exit $rc

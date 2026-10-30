#!/bin/bash
set -e
ts=$(date +%Y%m%d_%H%M%S)
echo "TS=$ts"
cd /home/featurize/tgai/e3
ls -la tgai_go/train/cuda.py exam2_chat10m_tgai.py
md5sum tgai_go/train/cuda.py tgai_go/train/__init__.py exam2_chat10m_tgai.py
cp -a tgai_go/train/cuda.py "tgai_go/train/cuda.py.bak_graph_$ts"
cp -a tgai_go/train/__init__.py "tgai_go/train/__init__.py.bak_graph_$ts"
cp -a exam2_chat10m_tgai.py "exam2_chat10m_tgai.py.bak_graph_$ts"
echo BACKUP_OK
python3 /home/featurize/tgai/e3/_probe_graph_syms.py

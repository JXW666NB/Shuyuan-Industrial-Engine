# -*- coding: utf-8 -*-
"""跑一个命令，同时采样 GPU 显存峰值（用 nvidia-smi 轮询，独立进程）。"""
import subprocess
import sys
import time

cmd = sys.argv[1:]
interval = 0.1


def mem_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return 0


p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
peak = 0
while p.poll() is None:
    peak = max(peak, mem_used())
    time.sleep(interval)
out = p.stdout.read()
peak = max(peak, mem_used())
print(out.rstrip())
print("采样峰值显存: %.2f GiB（nvidia-smi 轮询，间隔 %.0f ms）" % (peak / 1024.0, interval * 1000))

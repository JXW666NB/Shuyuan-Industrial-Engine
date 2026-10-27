# -*- coding: utf-8 -*-
"""多轮基准取中位：消除共驻产线带来的抖动。

用法（V100 上）：
    python _bench_median.py <bwd.cu> <attn.cu> [轮数]
输出每段 ours/torch 的 中位/最小/最大。
"""
import re
import statistics
import subprocess
import sys

PY = r"C:\Python311\python.exe"
LAB = r"D:\kernel_lab"


def run(script, args):
    r = subprocess.run([PY, "-X", "utf8", script] + args,
                       capture_output=True, text=True, errors="replace", timeout=900)
    return (r.stdout or "") + (r.stderr or "")


def parse(out):
    """抓形如 '名字   1.234   5.678   0.95x' 的行。"""
    rows = {}
    for line in out.splitlines():
        m = re.match(r"^\s*(\S.*?)\s{2,}([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)x\s*$", line.strip())
        if m:
            name = m.group(1).strip()
            rows[name] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return rows


def agg(samples, key):
    vals = [s[key][2] for s in samples if key in s]
    return (statistics.median(vals), min(vals), max(vals)) if vals else None


def main():
    bwd = sys.argv[1]
    attn = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    fwd = sys.argv[4] if len(sys.argv) > 4 else bwd

    print("=== 反向 %d 轮 ===" % n)
    bwd_samples = []
    for i in range(n):
        out = run(LAB + r"\layer_bench_bwd.py", [bwd, attn])
        rows = parse(out)
        bwd_samples.append(rows)
        moe = rows.get("MoE FFN 反向")
        print("  轮%d  MoE=%s  合计=%s" % (
            i + 1,
            ("%.2fx" % moe[2]) if moe else "?",
            ("%.2fx" % rows["合计 反向"][2]) if "合计 反向" in rows else "?"))

    for key in ("attention 块 反向", "MoE FFN 反向", "合计 反向"):
        a = agg(bwd_samples, key)
        if a:
            print("  %-14s 中位 %.2fx  (%.2f-%.2f)" % (key, a[0], a[1], a[2]))

    print("=== 前向 %d 轮 ===" % n)
    fwd_samples = []
    for i in range(n):
        out = run(LAB + r"\layer_bench.py", [fwd, attn, bwd])
        rows = parse(out)
        fwd_samples.append(rows)
        moe = rows.get("MoE FFN 前向")
        print("  轮%d  MoE=%s  合计=%s" % (
            i + 1,
            ("%.2fx" % moe[2]) if moe else "?",
            ("%.2fx" % rows["合计 前向"][2]) if "合计 前向" in rows else "?"))

    for key in ("attention 块 前向", "MoE FFN 前向", "合计 前向"):
        a = agg(fwd_samples, key)
        if a:
            print("  %-14s 中位 %.2fx  (%.2f-%.2f)" % (key, a[0], a[1], a[2]))

    # 合计（前向+反向，用中位数的 ours/torch 之和）
    fa = agg(fwd_samples, "合计 前向")
    ba = agg(bwd_samples, "合计 反向")
    if fa and ba:
        ours = sum(fwd_samples[0]["合计 前向"][:2]) if False else None
        # 直接用中位比值近似不够准，这里用单轮原始 ms 的中位
        o1 = statistics.median([s["合计 前向"][0] for s in fwd_samples if "合计 前向" in s])
        t1 = statistics.median([s["合计 前向"][1] for s in fwd_samples if "合计 前向" in s])
        o2 = statistics.median([s["合计 反向"][0] for s in bwd_samples if "合计 反向" in s])
        t2 = statistics.median([s["合计 反向"][1] for s in bwd_samples if "合计 反向" in s])
        print("=== 合计（中位 ms）===")
        print("  前向 %.3f / %.3f = %.2fx" % (o1, t1, o1 / t1))
        print("  反向 %.3f / %.3f = %.2fx" % (o2, t2, o2 / t2))
        print("  前+反 %.3f / %.3f = %.2fx" % (o1 + o2, t1 + t2, (o1 + o2) / (t1 + t2)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

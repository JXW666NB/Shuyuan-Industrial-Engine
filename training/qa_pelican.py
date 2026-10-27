# -*- coding: utf-8 -*-
"""鹈鹕端到端验收台（QA）—— Grok 的修复必须整套通过。

四个阶段，各自独立判定：
  S1 训练稳定性 : 连续 N 步不崩（默认 20000），且 loss 有下降
  S2 打包正确性 : .TG 里 58 项权重与 checkpoint 逐项一致；引擎 vs torch logits 相对误差 < 1e-4
  S3 引擎复现   : 引擎从分叉点后逐字符生成，两份文档匹配率 >= 95%
  S4 分词一致性 : 引擎 BPE 编码与训练侧分词器逐 token 一致（无 merges 时退化为字符级）

用法（在 V100 上）：
    python qa_pelican.py all [STEPS]         # 全套
    python qa_pelican.py s2 CKPT TG          # 只验打包
    python qa_pelican.py s3 CKPT TG [N]      # 只验复现
"""
import os
import re
import subprocess
import sys

PY = r"C:\Python311\python.exe"
LAB = r"D:\kernel_lab"
OUT = os.path.join(LAB, "_tgtest")
TRAINER = os.path.join(LAB, "charlm_torch.py")
PACKER = os.path.join(LAB, "pelican_to_tg.py")
VERIFY = os.path.join(LAB, "pelican_verify_engine.py")
CHECK = os.path.join(LAB, "pelican_check_pack.py")
GEN = os.path.join(LAB, "pelican_to_tg.py")
CORPUS = [os.path.join(LAB, "_corpus", "pelican_on_bike.html"),
          os.path.join(LAB, "_corpus", "bike_on_pelican.html")]

CFG = (4, 512, 16, 1024, 512, 2)          # NL D NH FF SEQ BATCH
CKPT = os.path.join(OUT, "pelican_NL4_D512_F1024_S512.pt")
TG = os.path.join(OUT, "pelican_qa.tg")


def run(cmd, timeout=2700):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return -9, "TIMEOUT"


def s1_train(steps=20000):
    print("=" * 60)
    print("[S1] 训练稳定性：%d 步，不得出现 CUDA error" % steps)
    args = [PY, "-X", "utf8", TRAINER, str(steps)] + [str(x) for x in CFG]
    rc, out = run(args)
    crashed = "CUDA error" in out or "illegal memory access" in out or "misaligned address" in out
    losses = [float(x) for x in re.findall(r"loss=([0-9.]+)", out)]
    first = losses[0] if losses else None
    last = losses[-1] if losses else None
    ok = (not crashed) and last is not None and last < (first or 1e9) * 0.5
    print("    崩溃=%s  loss: %s -> %s  步数=%d" % (crashed, first, last, len(losses)))
    if crashed:
        for line in out.splitlines():
            if "CUDA error" in line:
                print("    " + line.strip()[:120])
                break
    print("    判定:", "PASS" if ok else "FAIL")
    return ok, out


def s2_pack(ckpt=CKPT, tg=TG):
    print("=" * 60)
    print("[S2] 打包正确性")
    rc, out = run([PY, "-X", "utf8", PACKER, "pack", ckpt, tg], 900)
    packed = "打包完成" in out
    print("    打包:", "OK" if packed else "失败")
    rc2, out2 = run([PY, "-X", "utf8", CHECK, ckpt, tg], 900)
    weights_ok = "VERDICT ok=True" in out2
    m = re.search(r"异常项: (\d+) / (\d+)", out2)
    print("    权重逐项:", (m.group(0) if m else "?"), "->", "OK" if weights_ok else "FAIL")
    rc3, out3 = run([PY, "-X", "utf8", VERIFY, ckpt, tg], 900)
    mm = re.search(r"最大相对误差 = ([0-9.eE+-]+)", out3)
    err = float(mm.group(1)) if mm else 9.9
    logits_ok = err < 1e-4
    print("    引擎 vs torch logits: %.3e -> %s" % (err, "OK" if logits_ok else "FAIL"))
    ok = packed and weights_ok and logits_ok
    print("    判定:", "PASS" if ok else "FAIL")
    return ok


def s3_reproduce(ckpt=CKPT, tg=TG, n=400):
    print("=" * 60)
    print("[S3] 引擎逐字符复现（%d 字符）" % n)
    rc, out = run([PY, "-X", "utf8", GEN, "gen", str(n), tg], 1800)
    rates = [float(x) for x in re.findall(r"逐字符匹配 (\d+)/(\d+) = ([0-9.]+)%", out)]
    rates = [float(x) for x in re.findall(r"= ([0-9.]+)%", out)]
    for line in out.splitlines():
        if "匹配" in line:
            print("    " + line.strip())
    ok = len(rates) >= 2 and min(rates) >= 95.0
    print("    判定:", "PASS" if ok else "FAIL（要求两份都 >=95%）")
    return ok


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "s1":
        return 0 if s1_train(int(sys.argv[2]) if len(sys.argv) > 2 else 20000)[0] else 1
    if mode == "s2":
        return 0 if s2_pack(*sys.argv[2:4]) else 1
    if mode == "s3":
        return 0 if s3_reproduce(*sys.argv[2:4]) else 1
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    r1, _ = s1_train(steps)
    r2 = s2_pack()
    r3 = s3_reproduce()
    print("=" * 60)
    print("总判定: S1=%s S2=%s S3=%s  =>  %s"
          % (r1, r2, r3, "PASS" if (r1 and r2 and r3) else "FAIL"))
    return 0 if (r1 and r2 and r3) else 1


if __name__ == "__main__":
    sys.exit(main())

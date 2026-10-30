# -*- coding: utf-8 -*-
"""多卡启动器：为每个 rank 起一个进程，各绑一张卡。

用法：
    python tools/launch_dp.py --world 2 -- python examples/exam2_chat10m_tgai.py --device cuda --steps 600

它会为每个 rank 设置：
    CUDA_VISIBLE_DEVICES=<rank>     （让每个进程只看到自己那张卡）
    TGAI_DP_WORLD / TGAI_DP_RANK    （数据并行身份）
    TGAI_DP_JOB                     （本次运行的 rendezvous 命名空间，避免互相干扰）

注意：这是「单机多卡」。跨机目前不支持（rendezvous 用的是本机共享内存）。
"""
import argparse
import os
import subprocess
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--devices", default="", help="逗号分隔，如 0,1；默认 0..world-1")
    ap.add_argument("--job", default="")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- 之后的完整命令")
    a = ap.parse_args(argv)

    cmd = a.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("缺少要执行的命令（放在 -- 之后）")

    devs = [d.strip() for d in a.devices.split(",") if d.strip()] or \
           [str(i) for i in range(a.world)]
    if len(devs) < a.world:
        ap.error("--devices 只有 %d 个，但 --world=%d" % (len(devs), a.world))

    job = a.job or ("%d" % int(time.time()))
    print("数据并行启动：world=%d devices=%s job=%s" % (a.world, devs[:a.world], job))
    print("命令：%s" % " ".join(cmd))

    procs = []
    for r in range(a.world):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = devs[r]
        env["TGAI_DP_WORLD"] = str(a.world)
        env["TGAI_DP_RANK"] = str(r)
        env["TGAI_DP_JOB"] = job
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)
        time.sleep(1.0)                 # 错开启动，避免同时抢共享内存建名字

    bad = 0
    for r, p in enumerate(procs):
        rc = p.wait()
        print("rank%d 退出码 %d" % (r, rc))
        if rc != 0:
            bad += 1
    print("RESULT: %s" % ("PASS" if bad == 0 else "FAIL（%d 个 rank 非零退出）" % bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

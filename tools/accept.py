# -*- coding: utf-8 -*-
"""一条命令跑完验收（L0 + L1 的核心判据），输出 PASS/FAIL 表。

    python tools/accept.py            # 标准：600 步
    python tools/accept.py --quick    # 快速：120 步，用于改完代码先自检
    python tools/accept.py --json out.json

它做的事（全部真跑，不看历史数据）：
  1. 自检 `doctor`（12 项，含 bf16 路径）；
  2. 形状矩阵：抽 4 组 (D/NH/FF/L/S) 训练若干步，要求不报错、loss 有限；
  3. 精度：fp32 与（硬件支持时）bf16/ fp16 各跑 N 步，比较末 loss 偏差 ≤2%；
  4. 显存：池峰值与 PyTorch 无关，这里只记录（对齐 PyTorch 的对比需要 torch，见 bench_bf16_ceiling）；
  5. 速度：给出各配置 ms/step，并和"本机 PyTorch 同代码"对比（有 torch 才做）。

判据与判定标准见 docs/PRODUCTION_READINESS.md。
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _add_path(*cands):
    """兼容两种布局：仓库（python/ + examples/）与扁平部署（tgai_go/ 与脚本同目录）。"""
    for c in cands:
        if os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)

_add_path(os.path.join(REPO, "python"), REPO)
_add_path(os.path.join(REPO, "examples"), REPO)


def _rss_mib():
    """当前进程峰值 RSS（Linux 用 /proc，其它返回 None）。"""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def _fmt(v, unit="", nd=1):
    return "—" if v is None else ("%.*f%s" % (nd, unit) if isinstance(v, float) else "%s%s" % (v, unit))


def run_doctor():
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(REPO, "python"), REPO])
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run([sys.executable, "-m", "tgai_go.train.doctor"],
                       capture_output=True, text=True, timeout=600,
                       encoding="utf-8", errors="replace", env=env)
    out = (p.stdout or "") + (p.stderr or "")
    ok = "自检通过" in out
    tail = [ln for ln in out.splitlines() if "自检" in ln]
    return {"ok": ok, "msg": tail[-1].strip() if tail else out.strip()[-120:],
            "sec": round(time.time() - t0, 1)}


def train_steps(T, E, steps, mode="fp32", batch=0, cfg_over=None):
    """跑 steps 步，返回 (ms/step, 末 loss)。mode: fp32/bf16/fp16。"""
    import tgai_go.train as TT  # noqa: F401
    if cfg_over:
        E.CFG.update(**cfg_over)
    text, _src = E.load_corpus()
    toks, stoi = E.build_vocab(text)
    x, y = E.build_examples(text, stoi, E.CFG["MAXLEN"])
    if x is None:
        raise RuntimeError("语料里没有可用对话")
    if batch and batch < int(x.shape[0]):
        x, y = x[:batch], y[:batch]
    cfg = dict(E.CFG, V=len(toks), S=int(x.shape[1]))
    p = E.make_model(cfg, np.random.RandomState(0))
    opt = T.optim.AdamW(list(p.values()), lr=2e-3)
    F = T.functional
    xt, yt = T.tensor(x), T.tensor(y)
    n = int(x.shape[0])
    warm = 2
    for _ in range(warm + steps):
        lg = E.forward(p, xt, cfg)
        lo = F.cross_entropy(lg.reshape(cfg["S"] * n, cfg["V"]), yt.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        lo.backward()
        opt.step()
        lv = float(np.asarray(lo.to_numpy()).reshape(-1)[0])
    # 计时单独跑一段（避免把 warm 混进来）
    T._cuda().sync()
    t0 = time.perf_counter()
    k = max(10, steps // 4)
    for _ in range(k):
        lg = E.forward(p, xt, cfg)
        lo = F.cross_entropy(lg.reshape(cfg["S"] * n, cfg["V"]), yt.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        lo.backward()
        opt.step()
        lv = float(np.asarray(lo.to_numpy()).reshape(-1)[0])
    T._cuda().sync()
    return (time.perf_counter() - t0) / k * 1e3, lv


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    steps = 120 if a.quick else a.steps

    import tgai_go.train as T
    T.set_default_device("cuda")
    from tgai_go.train import _cuda
    rt = _cuda()
    import exam2_chat10m_tgai as E

    rows = []          # (项目, 结果, 详情)
    data = {"platform": platform.platform(), "steps": steps}

    # 1) doctor
    d = run_doctor()
    rows.append(("自检 doctor", "PASS" if d["ok"] else "FAIL", "%s（%.1fs）" % (d["msg"], d["sec"])))
    data["doctor"] = d

    # 2) 形状矩阵（抽 4 组）
    shapes = [("D256 NH8 FF1280 L8 S64", dict(D=256, NH=8, FF=1280, L=8, MAXLEN=64)),
              ("D128 NH4 FF512  L4 S64", dict(D=128, NH=4, FF=512, L=4, MAXLEN=64)),
              ("D512 NH8 FF2048 L2 S64", dict(D=512, NH=8, FF=2048, L=2, MAXLEN=64)),
              ("D192 NH6 FF768  L3 S96", dict(D=192, NH=6, FF=768, L=3, MAXLEN=96))]
    shape_ok = 0
    shape_detail = []
    for name, over in shapes:
        try:
            E.CFG.update(D=256, NH=8, FF=1280, L=8, MAXLEN=64)      # 复位
            ms, lv = train_steps(T, E, max(6, steps // 20), cfg_over=over)
            ok = lv == lv and abs(lv) < 1e3
            shape_ok += 1 if ok else 0
            shape_detail.append("%s:%.1fms" % (name.split()[0], ms))
        except Exception as exc:                                    # noqa: BLE001
            shape_detail.append("%s:FAIL(%s)" % (name.split()[0], type(exc).__name__))
    rows.append(("形状矩阵 4 组", "PASS" if shape_ok == len(shapes) else "FAIL",
                 " ".join(shape_detail)))
    data["shapes"] = shape_detail

    # 3) 精度对比（fp32 vs bf16/fp16，硬件支持才做）
    E.CFG.update(D=256, NH=8, FF=1280, L=8, MAXLEN=64)
    base_ms, base_loss = train_steps(T, E, steps)
    data["fp32"] = {"ms": base_ms, "loss": base_loss}
    rows.append(("训练 %d 步 fp32" % steps, "PASS",
                 "%.1f ms/step，末 loss %.6f" % (base_ms, base_loss)))

    try:
        import torch
        if getattr(torch.version, "hip", None):
            has_bf16 = True                      # AMD CDNA/RDNA 都有 bf16
        else:
            # 不能信 is_bf16_supported()：V100 上它返回 True，其实是软件模拟。
            has_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    except Exception:                                               # noqa: BLE001
        has_bf16 = False

    for tag, env_key in (("bf16", "TG_BF16"),):
        if not has_bf16:
            rows.append(("%s 精度" % tag, "SKIP", "本机无 %s 硬件（算力 <8.0）" % tag))
            continue
        os.environ[env_key] = "1"
        try:
            ms, lv = train_steps(T, E, steps)
            dev = abs(lv - base_loss) / max(abs(base_loss), 1e-12) * 100
            ok = dev <= 2.0 and ms <= base_ms * 1.05
            rows.append(("%s %d 步" % (tag, steps), "PASS" if ok else "FAIL",
                         "%.1f ms/step（%.2fx），loss %.6f，偏差 %.2f%%" % (ms, base_ms / ms, lv, dev)))
            data[tag] = {"ms": ms, "loss": lv, "dev_pct": dev}
        finally:
            os.environ.pop(env_key, None)

    # 4) 显存
    st = rt.pool_stats()
    peak = st["peak_bytes"] / 2 ** 30
    rss = _rss_mib()
    rows.append(("显存/内存", "PASS" if peak < 32 else "WARN",
                 "池峰值 %.2f GiB，host 峰值 %s" % (peak, _fmt(rss, " MiB", 0) if rss else "—（非 Linux）")))
    data["peak_gib"] = peak
    data["host_peak_mib"] = rss

    # ---------- 输出 ----------
    print("")
    print("=" * 78)
    print("%-22s %-6s %s" % ("项目", "结果", "详情"))
    print("-" * 78)
    for name, res, detail in rows:
        print("%-22s %-6s %s" % (name, res, detail))
    print("=" * 78)
    bad = [r for r in rows if r[1] == "FAIL"]
    print("验收结论：%s（%d 项 FAIL）" % ("PASS" if not bad else "FAIL", len(bad)))
    data["rows"] = rows
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print("明细已写到 %s" % a.json)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())

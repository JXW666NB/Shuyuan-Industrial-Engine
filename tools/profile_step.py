# -*- coding: utf-8 -*-
"""分层计时 / 分阶段剖析（L3 可观测性）。

不改训练脚本，从外面量：

  1. **分阶段**：fwd / ce / bwd / opt（每段前后 `sync()`，看占比）；
  2. **按 kernel 名**：启动次数与 host 时间（找出该融谁）；
  3. **每层成本**：用「层数扫描 + 线性差分」求——分别跑 L=1,2,4,8，
     对层数做最小二乘拟合，斜率就是每层成本（比硬算单层更稳）。

用法：
    python tools/profile_step.py                 # 默认 10 步热身 3 步
    python tools/profile_step.py --steps 20 --layers 1,2,4,8
    python tools/profile_step.py --device cuda --batch 64
"""
import argparse
import collections
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "examples"))


def _fit(xs, ys):
    """最小二乘 y = a*x + b，返回 (a, b)。"""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else float("nan")), 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan"), my
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return a, my - a * mx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch", type=int, default=0, help="0=整语料")
    ap.add_argument("--layers", default="", help="如 1,2,4,8；给了就做每层成本拟合")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--gpu", action="store_true", help="按 kernel 统计 GPU 时间（CUDA event）")
    a = ap.parse_args(argv)

    import tgai_go.train as T
    T.set_default_device(a.device)
    from tgai_go.train import _cuda
    import exam2_chat10m_tgai as E

    rt = _cuda()
    text, src = E.load_corpus()
    toks, stoi = E.build_vocab(text)
    x, y = E.build_examples(text, stoi, E.CFG["MAXLEN"])
    if x is None:
        raise SystemExit("语料里没有可用对话")
    n = int(x.shape[0])
    bs = int(a.batch) if a.batch > 0 else n
    bs = max(1, min(bs, n))
    x, y = x[:bs], y[:bs]
    print("语料=%s 对话 %d 条（用前 %d）序列见下 device=%s" % (src, n, bs, a.device), flush=True)

    F = T.functional

    def build(n_layer):
        cfg = dict(E.CFG, V=len(toks), S=int(x.shape[1]), L=n_layer)
        p = E.make_model(cfg, np.random.RandomState(0))
        n_par = sum(int(np.prod(tuple(t.shape))) for t in p.values())
        opt = T.optim.AdamW(list(p.values()), lr=2e-3)
        return cfg, p, opt, n_par

    def stage_times(cfg, p, opt):
        """分阶段计时（每段前后 sync）。返回 dict。"""
        t = {}
        # 热身
        for _ in range(a.warmup):
            logits = E.forward(p, x, cfg)
            loss = F.cross_entropy(logits.reshape(cfg["S"] * x.shape[0], cfg["V"]),
                                   T.tensor(y.reshape(-1)), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
        rt.sync()
        fwd = ce = bwd = optm = 0.0
        for _ in range(a.steps):
            t0 = time.perf_counter()
            logits = E.forward(p, x, cfg)
            rt.sync(); t1 = time.perf_counter()
            loss = F.cross_entropy(logits.reshape(cfg["S"] * x.shape[0], cfg["V"]),
                                   T.tensor(y.reshape(-1)), ignore_index=-100)
            rt.sync(); t2 = time.perf_counter()
            opt.zero_grad(); loss.backward()
            rt.sync(); t3 = time.perf_counter()
            opt.step()
            rt.sync(); t4 = time.perf_counter()
            fwd += t1 - t0; ce += t2 - t1; bwd += t3 - t2; optm += t4 - t3
        k = float(a.steps)
        t.update(fwd=fwd / k * 1e3, ce=ce / k * 1e3, bwd=bwd / k * 1e3, opt=optm / k * 1e3)
        t["total"] = t["fwd"] + t["ce"] + t["bwd"] + t["opt"]
        return t

    # ---------- 主配置：分阶段 + kernel 计数 ----------
    cfg, p, opt, n_par = build(E.CFG["L"])
    print("参数量 %.3fM  序列 %d  层数 %d" % (n_par / 1e6, cfg["S"], cfg["L"]), flush=True)

    CNT = collections.Counter()
    TM = collections.Counter()
    _orig = rt.launch

    def launch(name, grid, block, args, shared=0, module=None):
        t0 = time.perf_counter()
        try:
            return _orig(name, grid, block, args, shared, module)
        finally:
            CNT[name] += 1
            TM[name] += time.perf_counter() - t0

    rt.launch = launch
    try:
        t = stage_times(cfg, p, opt)
    finally:
        rt.launch = _orig

    print("\n==== 分阶段（每段前后 sync）====")
    print("%-10s %9s %8s" % ("阶段", "ms", "占比"))
    for k in ("fwd", "ce", "bwd", "opt", "total"):
        print("%-10s %9.2f %7.1f%%" % (k, t[k], 100.0 * t[k] / max(t["total"], 1e-9)))

    st = rt.pool_stats()
    print("\n池：峰值 %.3f GiB / live %.3f / cached %.3f，命中率 %.1f%%，未命中 %d 次"
          % (st["peak_bytes"] / 2**30, st["live_bytes"] / 2**30, st["cached_bytes"] / 2**30,
             st["hit_rate"] * 100, st["misses"]))

    print("\n==== kernel 启动前 %d（host 时间，不代表 GPU 时间）====" % a.top)
    print("%-26s %7s %9s" % ("kernel", "次数", "host ms"))
    iters = max(1, a.warmup + a.steps)
    total_launch = sum(CNT.values())
    for name, c in CNT.most_common(a.top):
        print("%-26s %7.0f %9.2f" % (name[:26], c / iters, TM[name] * 1e3 / iters))
    print("合计 %.0f 次启动/步，host 侧 %.2f ms/步（%d 步合计 %.1f ms）"
          % (total_launch / iters, sum(TM.values()) * 1e3 / iters, iters, sum(TM.values()) * 1e3))

    # ---------- 按 kernel 的 GPU 时间（CUDA event，事后统一 sync）----------
    if a.gpu:
        import ctypes
        GEV = collections.defaultdict(float)
        GCNT = collections.Counter()
        evs = []
        _orig2 = rt.launch

        def _mk():
            p_ = ctypes.c_void_p()
            rc = rt.cu.cuEventCreate(ctypes.byref(p_), 0)   # 默认 flags：计时
            return p_

        def launch_g(name, grid, block, args, shared=0, module=None):
            e0, e1 = _mk(), _mk()
            rt.cu.cuEventRecord(e0, rt._stream)
            r = _orig2(name, grid, block, args, shared, module)
            rt.cu.cuEventRecord(e1, rt._stream)
            evs.append((name, e0, e1))
            return r

        # 先热身再开测，避免把首次编译算进来
        rt.launch = launch_g
        try:
            for _ in range(a.warmup):
                lg = E.forward(p, cfg and None or p, cfg) if False else None
            # 直接复用 stage_times 的循环体做一次计时
            for _ in range(max(1, a.steps)):
                logits = E.forward(p, x, cfg)
                loss = F.cross_entropy(logits.reshape(cfg["S"] * x.shape[0], cfg["V"]),
                                       T.tensor(y.reshape(-1)), ignore_index=-100)
                opt.zero_grad(); loss.backward(); opt.step()
        finally:
            rt.launch = _orig2
        rt.sync()
        for name, e0, e1 in evs:
            ms = ctypes.c_float()
            if rt.cu.cuEventElapsedTime(ctypes.byref(ms), e0, e1) == 0:
                GEV[name] += ms.value
                GCNT[name] += 1
            rt.cu.cuEventDestroy(e0)
            rt.cu.cuEventDestroy(e1)
        steps = float(max(1, a.steps))
        print("\n==== GPU 时间前 %d（每步，CUDA event 实测）====" % a.top)
        print("%-26s %8s %10s" % ("kernel", "次数/步", "GPU ms/步"))
        tot = 0.0
        for nm, t in sorted(GEV.items(), key=lambda kv: -kv[1])[:a.top]:
            print("%-26s %8.0f %10.2f" % (nm[:26], GCNT[nm] / steps, t / steps))
        for nm, t in GEV.items():
            tot += t
        print("%-26s %8.0f %10.2f" % ("【全部 kernel 合计】", sum(GCNT.values()) / steps, tot / steps))

    # ---------- 每层成本：层数扫描 + 拟合 ----------
    if a.layers:
        ls = [int(v) for v in a.layers.split(",") if v.strip()]
        xs, ys = [], []
        print("\n==== 每层成本（层数扫描）====")
        print("%-6s %9s %9s %9s" % ("L", "total ms", "参数 M", "每层估 ms"))
        for nl in ls:
            cfg2, p2, opt2, npar2 = build(nl)
            t2 = stage_times(cfg2, p2, opt2)
            xs.append(nl)
            ys.append(t2["total"])
            print("%-6d %9.2f %9.3f %9s" % (nl, t2["total"], npar2 / 1e6, "-"))
            del p2, opt2
            try:
                rt.empty_cache()
            except Exception:
                pass
        a_slope, b0 = _fit(xs, ys)
        print("拟合：每层 %.2f ms，固定开销 %.2f ms（R² 见下）" % (a_slope, b0))
        if len(xs) >= 2:
            my = sum(ys) / len(ys)
            ss_tot = sum((v - my) ** 2 for v in ys)
            ss_res = sum((v - (a_slope * xx + b0)) ** 2 for xx, v in zip(xs, ys))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print("R² = %.4f（越接近 1 说明线性假设越成立）" % r2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

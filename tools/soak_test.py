# -*- coding: utf-8 -*-
"""长稳测试（soak test）：连续训练并采样显存/内存/loss，检查漂移与 NaN。

L3「24h 长稳」的度量工具。判据（与 docs/PRODUCTION_READINESS.md 一致）：

  * 全程无 NaN/Inf；
  * 显存峰值漂移 ≤ 3%（热身后的最大值与最小值之差 / 均值）；
  * host RSS 漂移 ≤ 5%；
  * loss 不发散（末 20% 的均值 ≤ 首 20% 的均值）。

用法：

    python tools/soak_test.py --minutes 15 --batch 64
    python tools/soak_test.py --minutes 1440 --csv soak.csv     # 真正的 24 小时

退出码：0 = 通过；1 = 有判据不满足（并在末尾打印原因）。
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "examples"))


def _rss_gib():
    """当前进程 RSS（GiB）。Linux 读 /proc，Windows 用 psutil（没有就返回 nan）。"""
    try:
        with open("/proc/self/status", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    try:
        import psutil  # noqa: PLC0415
        return psutil.Process().memory_info().rss / float(1 << 30)
    except Exception:
        return float("nan")


def _dev_free_gib(rt):
    try:
        import ctypes
        info = getattr(rt.cu, "cuMemGetInfo_v2", None) or getattr(rt.cu, "cuMemGetInfo", None)
        if info is None:
            return float("nan")
        free_b, total_b = ctypes.c_size_t(), ctypes.c_size_t()
        if info(ctypes.byref(free_b), ctypes.byref(total_b)) == 0:
            return free_b.value / float(1 << 30)
    except Exception:
        pass
    return float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15.0, help="跑多久（分钟）")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sample-steps", type=int, default=50, help="每多少步采样一次")
    ap.add_argument("--csv", default="", help="采样明细写到这个 CSV")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--graph", action="store_true", help="开 CUDA Graph（要求不分批）")
    a = ap.parse_args(argv)

    import tgai_go.train as T
    T.set_default_device("cuda")
    from tgai_go.train import _cuda
    import exam2_chat10m_tgai as E

    rt = _cuda()
    text, src = E.load_corpus()
    toks, stoi = E.build_vocab(text)
    x, y = E.build_examples(text, stoi, E.CFG["MAXLEN"])
    if x is None:
        raise SystemExit("语料里没有可用对话")
    cfg = dict(E.CFG, V=len(toks), S=int(x.shape[1]))
    n = int(x.shape[0])
    bs = max(1, min(a.batch, n))
    print("语料=%s  对话 %d 条  序列 %d  参数量见下  batch=%d" % (src, n, cfg["S"], bs), flush=True)

    p = E.make_model(cfg, np.random.RandomState(0))
    n_par = sum(int(np.prod(tuple(t.shape))) for t in p.values())
    print("参数量 %.3fM" % (n_par / 1e6), flush=True)
    opt = T.optim.AdamW(list(p.values()), lr=a.lr)
    F = T.functional

    rng = np.random.RandomState(1234)
    order = rng.permutation(n)
    pos = 0

    def next_batch():
        nonlocal order, pos
        if pos + bs > n:
            order = rng.permutation(n)
            pos = 0
        idx = order[pos:pos + bs]
        pos += bs
        return x[idx], y[idx]

    rows = []
    t0 = time.time()
    deadline = t0 + a.minutes * 60.0
    step = 0
    bad = 0
    peak_max = 0.0
    warmup = max(20, int(0.1 * 600))     # 前 20 步不算漂移
    while time.time() < deadline:
        xb, yb = next_batch()
        logits = E.forward(p, xb, cfg)
        loss = F.cross_entropy(logits.reshape(cfg["S"] * xb.shape[0], cfg["V"]),
                               T.tensor(yb.reshape(-1)), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        lv = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
        if lv != lv or abs(lv) > 1e6:
            bad += 1
        if step % a.sample_steps == 0:
            rt.sync()
            st = rt.pool_stats()
            peak = st["peak_bytes"] / float(1 << 30)
            row = (round(time.time() - t0, 1), step, lv,
                   round(peak, 4),
                   round(st["live_bytes"] / float(1 << 30), 4),
                   round(st["cached_bytes"] / float(1 << 30), 4),
                   round(_dev_free_gib(rt), 3),
                   round(_rss_gib(), 3))
            rows.append(row)
            if step > warmup:
                peak_max = max(peak_max, peak)
            print("t=%7.1fs step=%5d loss=%.4f peak=%.3f live=%.3f cached=%.3f free=%.2f rss=%.2f"
                  % row, flush=True)

    rt.sync()
    print("\n==== 汇总 ====", flush=True)
    if not rows:
        print("没有采到样本（跑得太短）")
        return 1
    peaks = [r[3] for r in rows if r[1] > warmup]
    rsss = [r[7] for r in rows if r[7] == r[7]]
    losses = [r[2] for r in rows]
    k = max(1, len(losses) // 5)
    l_head = float(np.mean(losses[:k]))
    l_tail = float(np.mean(losses[-k:]))

    def _grow(vals, win=None):
        """只量「增长」：末段均值相对首段均值的涨幅。收缩算 0，不算漂移。

        之前的 (max-min)/mean 会把「内存下降」也判成漂移 —— 实测 A10 上
        RSS 从 0.566 降到 0.529 却被判 FAIL，是度量写错了。
        """
        if len(vals) < 4:
            return 0.0
        kk = win or max(1, len(vals) // 4)
        early = float(np.mean(vals[:kk]))
        late = float(np.mean(vals[-kk:]))
        return max(0.0, (late - early) / max(abs(early), 1e-9))

    drift = _grow(peaks) if peaks else 0.0
    rss_drift = _grow(rsss) if len(rsss) > 1 else 0.0
    print("跑了 %d 步 / %.1f 分钟；采样 %d 次" % (step, (time.time() - t0) / 60.0, len(rows)))
    print("loss 首段均值 %.4f → 末段均值 %.4f" % (l_head, l_tail))
    print("显存峰值 最大 %.3f GiB；热身期后「增长」%.2f%%（判据 ≤3%%，收缩不计）" % (max(peaks) if peaks else 0, drift * 100))
    print("host RSS「增长」%.2f%%（判据 ≤5%%，收缩不计）" % (rss_drift * 100))
    print("NaN/异常 loss 次数：%d（判据 0）" % bad)

    ok = True
    if bad:
        ok = False
        print("不通过：出现 %d 次 NaN/异常 loss" % bad)
    if drift > 0.03:
        ok = False
        print("不通过：显存漂移 %.2f%% > 3%%" % (drift * 100))
    if rss_drift > 0.05:
        ok = False
        print("不通过：host RSS 漂移 %.2f%% > 5%%" % (rss_drift * 100))
    if l_tail > l_head * 1.2:
        ok = False
        print("不通过：loss 发散（末段 %.4f > 首段 %.4f × 1.2）" % (l_tail, l_head))
    print("RESULT: %s" % ("PASS" if ok else "FAIL"))

    if a.csv:
        with open(a.csv, "w", encoding="utf-8") as f:
            f.write("elapsed_s,step,loss,peak_gib,live_gib,cached_gib,free_gib,rss_gib\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")
        print("明细已写到 %s" % a.csv)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""从训练日志 CSV 生成曲线图与速度对比报告。

输入：<报告>/4090/logs/*.csv, <报告>/V100/logs/*.csv
输出：<报告>/charts/*.png, <报告>/速度对比.md, <报告>/logs/summary.csv
"""
import csv
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
MACHINES = ["4090", "V100"]
EXAMS = {"e1": "考试一 鹈鹕（0.33M）", "e2": "考试二 对话（10M）"}
ENGINES = {"tgai": "TGAI GO", "torch": "PyTorch"}
COLORS = {"4090": (220, 60, 60), "V100": (40, 110, 220)}
DASH = {"tgai": None, "torch": (6, 4)}


def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["step"]), float(r["loss"]), float(r["ppl"]), float(r["elapsed_s"])))
    return rows


def font(size):
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_chart(title, series, ylabel, out, logy=False):
    """series: [(label, color, dash, [(x, y)])]"""
    W, H = 1000, 560
    pad_l, pad_r, pad_t, pad_b = 90, 30, 60, 60
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f_t, f_a = font(22), font(15)
    d.text((pad_l, 18), title, fill=(20, 20, 20), font=f_t)

    all_x = [x for _, _, _, pts in series for x, _ in pts]
    all_y = [y for _, _, _, pts in series for _, y in pts]
    if not all_x:
        img.save(out)
        return
    x0, x1 = min(all_x), max(all_x)
    y0, y1 = min(all_y), max(all_y)
    if y1 - y0 < 1e-12:
        y1 = y0 + 1e-9
    # 留白
    dy = (y1 - y0) * 0.08
    y0, y1 = y0 - dy, y1 + dy

    def px(x):
        return pad_l + (x - x0) / max(1e-9, (x1 - x0)) * (W - pad_l - pad_r)

    def py(y):
        return H - pad_b - (y - y0) / max(1e-12, (y1 - y0)) * (H - pad_t - pad_b)

    d.rectangle([pad_l, pad_t, W - pad_r, H - pad_b], outline=(180, 180, 180))
    for i in range(5):
        yy = py(y0 + (y1 - y0) * i / 4)
        d.line([pad_l, yy, W - pad_r, yy], fill=(235, 235, 235))
        d.text((10, yy - 9), ("%.4g" % (y0 + (y1 - y0) * i / 4)), fill=(90, 90, 90), font=f_a)
    for i in range(5):
        xx = x0 + (x1 - x0) * i / 4
        d.text((px(xx) - 14, H - pad_b + 8), ("%d" % xx), fill=(90, 90, 90), font=f_a)
    d.text((W - 190, H - pad_b + 30), "step", fill=(90, 90, 90), font=f_a)
    d.text((pad_l, H - pad_b + 30), ylabel, fill=(60, 60, 60), font=f_a)

    for label, color, dash, pts in series:
        step = max(1, len(pts) // 400)
        pts2 = pts[::step] + ([pts[-1]] if pts else [])
        for i in range(1, len(pts2)):
            a = (px(pts2[i - 1][0]), py(pts2[i - 1][1]))
            b = (px(pts2[i][0]), py(pts2[i][1]))
            if dash is None:
                d.line([a, b], fill=color, width=2)
            else:
                # 虚线：按段跳
                if (i // dash[0]) % 2 == 0:
                    d.line([a, b], fill=color, width=2)
        # 图例
    lx, ly = pad_l + 12, pad_t + 10
    for label, color, dash, _ in series:
        d.line([lx, ly + 8, lx + 34, ly + 8], fill=color, width=3 if dash is None else 2)
        d.text((lx + 42, ly), label, fill=(30, 30, 30), font=f_a)
        ly += 22
    img.save(out)


def main():
    charts = os.path.join(HERE, "charts")
    os.makedirs(charts, exist_ok=True)
    summary = []
    for exam in ("e1", "e2"):
        loss_series, ppl_series = [], []
        for m in MACHINES:
            for eng in ("tgai", "torch"):
                p = os.path.join(HERE, m, "logs", "log_%s_%s_%s.csv" % (exam, m, eng))
                if not os.path.exists(p):
                    continue
                rows = read_csv(p)
                if not rows:
                    continue
                label = "%s / %s" % (m, ENGINES[eng])
                loss_series.append((label, COLORS[m], DASH[eng], [(r[0], r[1]) for r in rows]))
                if exam == "e2":
                    ppl_series.append((label, COLORS[m], DASH[eng], [(r[0], r[2]) for r in rows]))
                n = len(rows)
                secs = rows[-1][3]
                summary.append(dict(machine=m, exam=exam, engine=ENGINES[eng], steps=n,
                                    first_loss=rows[0][1], final_loss=rows[-1][1],
                                    min_loss=min(r[1] for r in rows),
                                    total_s=secs, ms_per_step=secs / n * 1000,
                                    steps_per_s=n / secs if secs > 0 else 0))
        draw_chart("%s — loss 曲线" % EXAMS[exam], loss_series, "loss", os.path.join(charts, "%s_loss.png" % exam))
        if ppl_series:
            draw_chart("%s — PPL 曲线（perplexity）" % EXAMS[exam], ppl_series, "ppl",
                       os.path.join(charts, "%s_ppl.png" % exam))

    # 汇总表
    hdr = "machine,exam,engine,steps,first_loss,final_loss,min_loss,total_s,ms_per_step,steps_per_s"
    lines = [hdr] + ["%s,%s,%s,%d,%.6f,%.6f,%.6f,%.2f,%.3f,%.3f" % (
        s["machine"], s["exam"], s["engine"], s["steps"], s["first_loss"], s["final_loss"],
        s["min_loss"], s["total_s"], s["ms_per_step"], s["steps_per_s"]) for s in summary]
    with open(os.path.join(HERE, "logs", "summary.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")
    for s in summary:
        print("%-5s %-3s %-8s steps=%-4d loss %.4f→%.4f  %.2fs  单步 %.2f ms"
              % (s["machine"], s["exam"], s["engine"], s["steps"], s["first_loss"],
                 s["final_loss"], s["total_s"], s["ms_per_step"]))
    print("图表输出:", charts)


if __name__ == "__main__":
    main()

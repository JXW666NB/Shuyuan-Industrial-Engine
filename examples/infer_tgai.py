# -*- coding: utf-8 -*-
"""推理程序 ①：用 TGAI GO 引擎跑 `.TG` 模型。

默认走引擎官方接口 `model.generate()`（`.TG` 里已写入 v27 九标记对话模板）。
两个关键参数：
    temperature=1e-6, top_k=1        —— 贪心（`temperature=0` 会被当成"用默认"，实测会变采样）
    repeat_penalty=1.0               —— **必须显式传**。模板默认 1.05，对字符级词表太激进，
                                        会把正常的重复字（"鹈鹕""TGAI"）罚成乱码。

用法：
    python infer_tgai.py                       # 交互式（考试二：对话）
    python infer_tgai.py --task pelican        # 考试一：从空 prompt 续写
    python infer_tgai.py --prompt "你是谁研发的"  # 单次提问
    python infer_tgai.py --raw --prompt "..."   # 不走模板，直接原始 token 续写（对照用）
"""
import argparse
import ctypes
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GREEDY = dict(max_tokens=48, temperature=1e-6, top_k=1, top_p=1.0, repeat_penalty=1.0)


def pick(*cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["chat", "pelican"], default="chat")
    ap.add_argument("--tg", default=None)
    ap.add_argument("--dll", default=os.environ.get("TGAI_DLL", r"D:\TGAI\libtgai_engine_gpu.dll"))
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--raw", action="store_true", help="不走对话模板，直接续写")
    a = ap.parse_args()

    root = os.path.dirname(HERE)
    if a.task == "chat":
        tg = pick(a.tg, os.path.join(root, "V100", "models", "chat10m.TG"),
                  os.path.join(root, "4090", "models", "chat10m.TG"))
    else:
        tg = pick(a.tg, os.path.join(root, "V100", "models", "pelican.TG"),
                  os.path.join(root, "4090", "models", "pelican.TG"))
    if not tg:
        raise SystemExit("找不到 .TG 文件，请用 --tg 指定")

    import tgai_go
    m = tgai_go.load(tg, dll=a.dll) if os.path.exists(a.dll) else tgai_go.load(tg)
    print("模型: %s" % tg)

    if a.raw:
        e = m._e
        lib, model = e.lib, e.model
        lib.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
        V, eos = m.vocab_size, 2

        def gen_raw(text, n):
            ids = m.encode(text, add_bos=True)
            out = []
            for _ in range(n):
                cur = ids + out
                cache = lib.tg_kvcache_new(model, 128)
                arr = (ctypes.c_int32 * len(cur))(*cur)
                lp = lib.tg_forward(model, arr, len(cur), cache, 0)
                best = max(range(V), key=lambda i: float(lp[i]))
                lib.tg_kvcache_free(cache)
                if best == eos:
                    break
                out.append(best)
            return "".join(m.decode(int(i)) for i in out)

        text = "" if a.task == "pelican" else "<用户>\n%s\n<TGAI><回答>\n" % (a.prompt or "")
        print(gen_raw(text, a.max_tokens))
        return 0

    kw = dict(GREEDY)
    kw["max_tokens"] = a.max_tokens

    if a.task == "pelican":
        for _ in range(3):
            print(" -> %r" % m.generate("", **kw))
        return 0

    if a.prompt:
        print("Q: %s\nA: %s" % (a.prompt, m.generate(a.prompt, **kw)))
        return 0
    print("输入问题，回车发送（空行退出）")
    while True:
        try:
            q = input("你 > ").strip()
        except EOFError:
            break
        if not q:
            break
        print("小模型 > %s" % m.generate(q, **kw))
    return 0


if __name__ == "__main__":
    sys.exit(main())

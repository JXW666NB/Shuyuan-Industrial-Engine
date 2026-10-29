import os, sys, tempfile
sys.path.insert(0, r"D:\DSH-Workspace\TGAI-对话档案\_repos\Shuyuan-Industrial-Engine\training")
out = os.path.join(tempfile.gettempdir(), "tg_probe")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "dummy.TG")
import bpe_train
bpe_train.build_dummy_tg(path, ["<pad>", "<a>", "<b>", "ab"], merges=[(1, 2)])
print("写出:", path, os.path.getsize(path), "字节")
import tgai_go
print("tgai_go 版本:", getattr(tgai_go, "__version__", "?"))
try:
    m = tgai_go.load(path)
    print("load OK:", type(m).__name__)
    txt = m.generate("ab", max_tokens=4)
    print("forward/generate OK:", repr(txt)[:80])
except Exception as ex:
    print("load/generate 失败:", type(ex).__name__, str(ex)[:300])

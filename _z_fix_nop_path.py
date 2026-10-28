import io, py_compile, tempfile
p = r"D:\DSH-Workspace\TGAI-对话档案\_repos\Shuyuan-Industrial-Engine\python\tgai_go\train\cuda.py"
s = io.open(p, encoding="utf-8").read()
old = '''    env = os.environ.get("TGAI_ATTN_NOP_CU")
    here = os.path.dirname(os.path.abspath(__file__))
    cands = []
    if env:
        cands.append(env)
    cands.append(os.path.normpath(os.path.join(
        here, "..", "..", "..", "training", "kernels", "tgai_attn_nop.cu")))
    cands.append("/home/featurize/tgai/training/kernels/tgai_attn_nop.cu")'''
new = '''    env = os.environ.get("TGAI_ATTN_NOP_CU")
    here = os.path.dirname(os.path.abspath(__file__))
    cands = []
    if env:
        cands.append(env)
    # ① 包内自带（pip 安装后的正常位置）
    cands.append(os.path.join(here, "kernels", "tgai_attn_nop.cu"))
    # ② 源码树（开发时从仓库根运行）
    cands.append(os.path.normpath(os.path.join(
        here, "..", "..", "..", "training", "kernels", "tgai_attn_nop.cu")))'''
assert old in s, "锚点未匹配"
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)
py_compile.compile(p, cfile=tempfile.mktemp(), doraise=True)
print("attn_nop_source_path 已改为优先包内路径，并去掉写死的开发机路径")

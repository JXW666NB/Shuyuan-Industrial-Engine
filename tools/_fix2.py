import io
# 1) ms_agent：/gpu 支持 AMD（rocm-smi）
p = r"ms_agent.py"
s = io.open(p, encoding="utf-8").read()
old = '''            try:
                out = subprocess.run(["nvidia-smi",
                                      "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                                      "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=20).stdout.strip()
            except Exception as exc:  # noqa: BLE001
                out = "nvidia-smi 失败: %s" % exc'''
new = '''            out = ""
            try:
                out = subprocess.run(["nvidia-smi",
                                      "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                                      "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=20).stdout.strip()
            except Exception:
                pass
            if not out:                      # AMD / ROCm
                for cmd in (["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
                            ["rocm-smi"]):
                    try:
                        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                        if r.returncode == 0 and r.stdout.strip():
                            out = "[rocm-smi]\\n" + r.stdout.strip()[:3000]
                            break
                    except Exception:
                        continue
            if not out:
                out = "没有 nvidia-smi / rocm-smi"'''
assert old in s
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)

# 2) bench_bf16_ceiling：ROCm 感知
p2 = r"bench_bf16_ceiling.py"
s2 = io.open(p2, encoding="utf-8").read()
old2 = '''    cap = torch.cuda.get_device_capability(0)
    print("设备: %s" % torch.cuda.get_device_name(0))
    print("算力: %s   torch %s   cuda %s" % (cap, torch.__version__, torch.version.cuda))
    hw_bf16 = cap[0] >= 8
    print("BF16 硬件: %s（算力 %d.%d；sm_80+ 才有真 BF16）"
          % ("有" if hw_bf16 else "没有，PyTorch 会软件模拟", cap[0], cap[1]))'''
new2 = '''    is_rocm = bool(getattr(torch.version, "hip", None))
    print("设备: %s" % torch.cuda.get_device_name(0))
    if is_rocm:
        print("后端: ROCm %s（HIP）  torch %s" % (torch.version.hip, torch.__version__))
        hw_bf16 = True          # CDNA/RDNA 均有 BF16；MI300 还有 FP8
        print("BF16 硬件: 有（AMD ROCm，CDNA3 原生支持 bf16/fp8）")
    else:
        cap = torch.cuda.get_device_capability(0)
        print("后端: CUDA %s   算力 %s" % (torch.version.cuda, cap))
        hw_bf16 = cap[0] >= 8
        print("BF16 硬件: %s（算力 %d.%d；sm_80+ 才有真 BF16）"
              % ("有" if hw_bf16 else "没有，PyTorch 会软件模拟", cap[0], cap[1]))'''
assert old2 in s2
s2 = s2.replace(old2, new2, 1)
s2 = s2.replace('    with torch.autocast("cuda", dtype=dt, enabled=dt is not None):',
                '    with torch.autocast("cuda", dtype=dt, enabled=dt is not None):  # ROCm 也用 "cuda" 这个名字')
io.open(p2, "w", encoding="utf-8", newline="\n").write(s2)
import py_compile, tempfile
for f in (p, p2):
    py_compile.compile(f, cfile=tempfile.mktemp(), doraise=True)
print("已改：ms_agent 支持 rocm-smi；bench_bf16_ceiling 感知 ROCm（不再用算力判断 BF16）")

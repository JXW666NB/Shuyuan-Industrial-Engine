from pathlib import Path

p = Path(r"_repos/Shuyuan-Industrial-Engine/training/kernels/tgai_attn_nop.cu")
text = p.read_text(encoding="utf-8")
start = text.find("extern \"C\" __global__ void __launch_bounds__(256, 1)\nattn_bwd_flash")
end = text.find("// ---------------------------------------------------------------------------\n// Backward. Online softmax is recomputed")
if start < 0 or end < 0 or end <= start:
    raise SystemExit("markers %s %s" % (start, end))
new = Path(r"_repos/Shuyuan-Industrial-Engine/docs/_backup_hotspots_r1_20261029/attn_bwd_flash_new.cu").read_text(encoding="utf-8")
if not new.endswith("\n"):
    new += "\n"
text = text[:start] + new + "\n" + text[end:]
p.write_text(text, encoding="utf-8")
print("spliced", start, end)

import io, subprocess
n = subprocess.run(["git", "rev-list", "--count", "HEAD"], capture_output=True, text=True).stdout.strip()
p = r"D:\DSH-Workspace\TGAI-对话档案\_repos\Shuyuan-Industrial-Engine\training\RELEASE_NOTES_v1.3.0.md"
s = io.open(p, encoding="utf-8").read()
s = s.replace("GitHub：25 个提交 + 里程碑 tag **`v1.3.0-training`**",
              "GitHub：仓库 **%s 个提交** + 里程碑 tag **`v1.3.0-training`**" % n)
io.open(p, "w", encoding="utf-8").write(s)
print("提交总数 =", n, "| 已把发布说明里的提交数改成实测值")

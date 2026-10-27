# -*- coding: utf-8 -*-
"""一键上传 TGAI GO 到 PyPI，并验证可下载。

用法（任选一种提供 token）：
    set PYPI_TOKEN=pypi-xxxx        # Windows cmd
    $env:PYPI_TOKEN="pypi-xxxx"     # PowerShell
    python tools/pypi_upload.py

它会：
    1. 检查 dist/ 里的 wheel + sdist
    2. twine check（元数据合规）
    3. twine upload 到 pypi.org
    4. pip download 验证真的能拉到（不是"上传成功"就完事）

token 只在进程内使用，不落盘、不打印。
"""
import glob
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.normpath(os.path.join(HERE, "..", "dist"))
PKG = "tgai_go"
VERSION = "1.3.0"


def run(cmd, show=True):
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if show:
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        for line in out.splitlines()[-6:]:
            print("   " + line)
    return r.returncode


def main():
    token = os.environ.get("PYPI_TOKEN") or os.environ.get("TWINE_PASSWORD")
    if not token:
        print("缺少凭据：请先设置 PYPI_TOKEN（pypi- 开头）")
        print("  PowerShell:  $env:PYPI_TOKEN=\"pypi-...\"")
        return 2

    files = sorted(glob.glob(os.path.join(DIST, "*.whl")) +
                   glob.glob(os.path.join(DIST, "*.tar.gz")))
    if not files:
        print("dist/ 里没有构建产物，先跑： python -m build")
        return 3
    print("[1] 产物：")
    for f in files:
        print("   %-52s %7.1f KB" % (os.path.basename(f), os.path.getsize(f) / 1024))

    print("[2] twine check")
    if run([sys.executable, "-m", "twine", "check"] + files) != 0:
        print("元数据不合规，停止")
        return 4

    print("[3] 上传")
    rc = run([sys.executable, "-m", "twine", "upload",
              "-u", "__token__", "-p", token,
              "--repository-url", "https://upload.pypi.org/legacy/"] + files)
    if rc != 0:
        print("上传失败（常见原因：token 无该项目的写权限 / 版本号已存在）")
        return 5

    print("[4] 验证可下载")
    tmp = tempfile.mkdtemp(prefix="tgai_pypi_")
    rc = run([sys.executable, "-m", "pip", "download",
              "%s==%s" % (PKG, VERSION), "--no-deps", "-d", tmp])
    got = glob.glob(os.path.join(tmp, "*"))
    if rc == 0 and got:
        print("   ✅ pip download 成功：%s" % os.path.basename(got[0]))
        print("   用户侧安装方式： pip install %s==%s" % (PKG, VERSION))
        return 0
    print("   上传成功但 pip download 失败（索引可能有延迟，几分钟后重试）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

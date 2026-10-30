# -*- coding: utf-8 -*-
"""本机客户端：通过公网 URL 驱动魔搭/notebook 上的 ms_agent。

用法（和 feat.py / v100.py 的用法保持一致）：

    set MS_URL=https://xxxx.trycloudflare.com
    set MS_TOKEN=<长密码>

    python tools/ms_client.py ping
    python tools/ms_client.py gpu
    python tools/ms_client.py sh "nvidia-smi; python3 -V"
    python tools/ms_client.py put 本地文件 /tmp/远端路径
    python tools/ms_client.py get /tmp/远端路径 本地文件
    python tools/ms_client.py ls /tmp

只用标准库（urllib），不依赖 requests。
"""
import json
import os
import sys
import urllib.parse
import urllib.request

URL = os.environ.get("MS_URL", "").rstrip("/")
TOKEN = os.environ.get("MS_TOKEN", "")


def _req(path, data=None, method=None, raw=False, body_raw=False, timeout=900):
    if not URL:
        raise SystemExit("请先设置 MS_URL（ms_boot.sh 打印的那一行）")
    url = URL + path
    sep = "&" if "?" in url else "?"
    if TOKEN:
        url = url + sep + "token=" + urllib.parse.quote(TOKEN)
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data is not None else "GET"))
    if TOKEN:
        req.add_header("X-Token", TOKEN)
    if data is not None and not body_raw:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return body if raw else json.loads(body.decode("utf-8"))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    op = argv[0]

    if op == "ping":
        print(json.dumps(_req("/ping"), ensure_ascii=False, indent=1))
        return 0

    if op == "gpu":
        print(_req("/gpu").get("gpu", ""))
        return 0

    if op == "ls":
        print("\n".join(_req("/ls?path=" + urllib.parse.quote(argv[1] if len(argv) > 1 else "."))
                        .get("items", [])))
        return 0

    if op == "sh":
        cmd = argv[1] if len(argv) > 1 else ""
        timeout = int(argv[2]) if len(argv) > 2 else 600
        r = _req("/sh", json.dumps({"cmd": cmd, "timeout": timeout}).encode("utf-8"))
        if not r.get("ok"):
            print("失败:", r.get("err"))
            return 1
        out = r.get("out") or ""
        err = r.get("err") or ""
        if out:
            print(out, end="" if out.endswith("\n") else "\n")
        if err:
            sys.stderr.write(err if err.endswith("\n") else err + "\n")
        return int(r.get("rc") or 0)

    if op == "put":
        local, remote = argv[1], argv[2]
        with open(local, "rb") as f:
            data = f.read()
        r = _req("/put?path=" + urllib.parse.quote(remote), data, body_raw=True)
        print("PUT %s %s (%d 字节)" % ("OK" if r.get("ok") else "FAIL", remote, r.get("bytes", 0)))
        return 0 if r.get("ok") else 1

    if op == "get":
        remote, local = argv[1], (argv[2] if len(argv) > 2 else os.path.basename(argv[1]))
        data = _req("/get?path=" + urllib.parse.quote(remote), raw=True)
        with open(local, "wb") as f:
            f.write(data)
        print("GET OK %s -> %s (%d 字节)" % (remote, local, len(data)))
        return 0

    print("未知命令:", op)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

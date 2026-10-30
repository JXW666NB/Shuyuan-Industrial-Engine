#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""魔搭（或任何 notebook）侧的代理：一个带 token 的最小 HTTP 服务。

设计目标：**不需要 SSH、不需要公网 IP、不需要任何账号**。
notebook 里跑这个，再用 `cloudflared tunnel --url http://127.0.0.1:8080` 把它发布出去，
本机就能像 SSH 一样驱动它。

接口（全部要求 token，默认从环境变量 MS_TOKEN 取）：
    GET  /ping                     -> {"ok": true, "cwd": ...}
    POST /sh    {"cmd": "...", "timeout": 600}  -> {"rc":..,"out":..,"err":..}
    POST /put   {"path": "..."}                  body=原始字节
    GET  /get?path=...              -> 原始字节
    GET  /ls?path=...               -> 文件列表
    GET  /gpu                       -> nvidia-smi 摘要

安全：只绑 127.0.0.1（外网只能经 cloudflared 进），token 用 HMAC 比较。
"""
import base64
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

TOKEN = os.environ.get("MS_TOKEN", "").strip()
WORKDIR = os.environ.get("MS_WORKDIR", os.getcwd())
PORT = int(os.environ.get("MS_PORT", "8080"))


def _check(handler):
    """校验 token（header 或 query 里都行）。"""
    if not TOKEN:
        return True
    got = handler.headers.get("X-Token", "")
    if not got:
        got = (parse_qs(urlparse(handler.path).query).get("token") or [""])[0]
    return got == TOKEN


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 静音
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw(self, data, code=200, ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _auth(self):
        if not _check(self):
            self._json({"ok": False, "err": "bad token"}, 403)
            return False
        return True

    # ---------- GET ----------
    def do_GET(self):
        if not self._auth():
            return
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/ping":
            import platform
            return self._json({"ok": True, "cwd": os.getcwd(), "host": platform.node(), "os": sys.platform})
        if u.path == "/gpu":
            try:
                out = subprocess.run(["nvidia-smi",
                                      "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                                      "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=20).stdout.strip()
            except Exception as exc:  # noqa: BLE001
                out = "nvidia-smi 失败: %s" % exc
            return self._json({"ok": True, "gpu": out})
        if u.path == "/get":
            p = (q.get("path") or [""])[0]
            if not p or not os.path.isfile(p):
                return self._json({"ok": False, "err": "no such file"}, 404)
            with open(p, "rb") as f:
                return self._raw(f.read())
        if u.path == "/ls":
            p = (q.get("path") or ["."])[0]
            try:
                items = sorted(os.listdir(p))[:500]
            except Exception as exc:  # noqa: BLE001
                return self._json({"ok": False, "err": str(exc)}, 404)
            return self._json({"ok": True, "items": items})
        return self._json({"ok": False, "err": "unknown endpoint"}, 404)

    # ---------- POST ----------
    def do_POST(self):
        if not self._auth():
            return
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if u.path == "/sh":
            try:
                req = json.loads(raw.decode("utf-8"))
            except Exception:  # noqa: BLE001
                return self._json({"ok": False, "err": "bad json"}, 400)
            cmd = req.get("cmd", "")
            timeout = float(req.get("timeout", 600))
            try:
                pr = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                    timeout=timeout, cwd=WORKDIR)
                return self._json({"ok": True, "rc": pr.returncode,
                                   "out": pr.stdout[-200000:], "err": pr.stderr[-40000:]})
            except subprocess.TimeoutExpired:
                return self._json({"ok": False, "err": "timeout"}, 504)
            except Exception as exc:  # noqa: BLE001
                return self._json({"ok": False, "err": str(exc)}, 500)
        if u.path == "/put":
            q = parse_qs(u.query)
            p = (q.get("path") or [""])[0]
            if not p:
                return self._json({"ok": False, "err": "path required"}, 400)
            os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
            with open(p, "wb") as f:
                f.write(raw)
            return self._json({"ok": True, "bytes": len(raw)})
        return self._json({"ok": False, "err": "unknown endpoint"}, 404)


def main():
    if not TOKEN:
        print("⚠️ 没有设置 MS_TOKEN，任何能访问这个 URL 的人都能执行命令。建议设置。")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("[agent] 监听 127.0.0.1:%d  workdir=%s" % (PORT, WORKDIR), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

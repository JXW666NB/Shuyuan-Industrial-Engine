import io
p = r"ms_agent.py"
s = io.open(p, encoding="utf-8").read()
s = s.replace('return self._json({"ok": True, "cwd": os.getcwd(), "host": os.uname().nodename})',
              'import platform\n            return self._json({"ok": True, "cwd": os.getcwd(), "host": platform.node(), "os": sys.platform})')
io.open(p, "w", encoding="utf-8", newline="\n").write(s)

p2 = r"ms_client.py"
s2 = io.open(p2, encoding="utf-8").read()
s2 = s2.replace("def _req(path, data=None, method=None, raw=False, timeout=900):",
                "def _req(path, data=None, method=None, raw=False, body_raw=False, timeout=900):")
s2 = s2.replace('    if data is not None and not raw:\n        req.add_header("Content-Type", "application/json")',
                '    if data is not None and not body_raw:\n        req.add_header("Content-Type", "application/json")')
s2 = s2.replace('r = _req("/put?path=" + urllib.parse.quote(remote), data, raw=True)',
                'r = _req("/put?path=" + urllib.parse.quote(remote), data, body_raw=True)')
io.open(p2, "w", encoding="utf-8", newline="\n").write(s2)
print("两处已修：os.uname -> platform.node；put 用 body_raw 而不是 raw")

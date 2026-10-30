#!/usr/bin/env bash
# 魔搭（或任何 notebook）一键上线：起代理 + 反向隧道，打印公网 URL。
#
# 用法（在 notebook 的 cell 里）：
#   !bash ms_boot.sh <随便一个长密码>
# 或者先把两个文件传上去再跑：
#   MS_TOKEN=<长密码> bash ms_boot.sh
#
# 成功后它会打印一行：
#   PUBLIC_URL=https://xxxx.trycloudflare.com
# 把这一行发回来，本机就能像 SSH 一样驱动这台机器。
set -u

TOKEN="${1:-${MS_TOKEN:-}}"
PORT="${MS_PORT:-8080}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$TOKEN" ]; then
  echo "错误：必须给一个 token（防别人扫到 URL 就能执行命令）"
  echo "  用法: bash ms_boot.sh <长密码>"
  exit 1
fi

echo "[1/4] 检查 python / cloudflared"
python3 -V || { echo "没有 python3"; exit 1; }
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "  未找到 cloudflared，尝试 pip 安装（约 20 MB）"
  python3 -m pip install --quiet cloudflared >/dev/null 2>&1 || \
    python3 -m pip install --quiet --index-url https://pypi.tuna.tsinghua.edu.cn/simple cloudflared >/dev/null 2>&1 || true
fi
if command -v cloudflared >/dev/null 2>&1; then
  CF="cloudflared"
elif python3 -c "import cloudflared,os,sys; sys.stdout.write(os.path.join(os.path.dirname(cloudflared.__file__),'cloudflared'))" >/dev/null 2>&1; then
  CF="$(python3 -c "import cloudflared,os;print(os.path.join(os.path.dirname(cloudflared.__file__),'cloudflared'))")"
else
  echo "  仍然没有 cloudflared：尝试直接下载"
  URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
  curl -fsSL "$URL" -o /tmp/cloudflared 2>/dev/null || \
    curl -fsSL "https://ghproxy.net/$URL" -o /tmp/cloudflared 2>/dev/null || true
  chmod +x /tmp/cloudflared 2>/dev/null || true
  CF="/tmp/cloudflared"
fi
[ -x "$CF" ] || { echo "cloudflared 不可用（$CF），请手动装：pip install cloudflared"; exit 1; }
echo "  cloudflared = $CF"

echo "[2/4] 启代理（token 已设置）"
pkill -f ms_agent.py >/dev/null 2>&1 || true
MS_TOKEN="$TOKEN" MS_PORT="$PORT" nohup python3 "$HERE/ms_agent.py" > "$HERE/agent.log" 2>&1 &
sleep 2
curl -fsS "http://127.0.0.1:$PORT/ping?token=$TOKEN" >/dev/null || { echo "  代理没起来，看 $HERE/agent.log"; tail -5 "$HERE/agent.log"; exit 1; }
echo "  代理 OK"

echo "[3/4] 起反向隧道"
pkill -f "cloudflared tunnel" >/dev/null 2>&1 || true
rm -f "$HERE/cf.log"
nohup "$CF" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > "$HERE/cf.log" 2>&1 &

echo "[4/4] 等公网地址（最多 60 秒）"
URL=""
for _ in $(seq 1 60); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$HERE/cf.log" 2>/dev/null | head -1)"
  [ -n "$URL" ] && break
  sleep 1
done
if [ -z "$URL" ]; then
  echo "  没拿到地址，cf.log 尾部："
  tail -12 "$HERE/cf.log"
  exit 1
fi
echo
echo "================================================================"
echo "PUBLIC_URL=$URL"
echo "TOKEN=$TOKEN"
echo "验证: curl \"$URL/ping?token=$TOKEN\""
echo "================================================================"

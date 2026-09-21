#!/bin/sh
# Web アプリを起動して http://localhost:35607 で待ち受ける
cd "$(dirname "$0")"
[ -x .venv/bin/uvicorn ] || { echo "先に ./setup.sh を実行してください"; exit 1; }
PORT="${PORT:-35607}"
echo "→ http://localhost:${PORT} をブラウザで開いてください"
exec .venv/bin/uvicorn app:app --host 127.0.0.1 --port "$PORT"

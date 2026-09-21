#!/bin/sh
# 初回セットアップ: Python 3.12 の仮想環境を作って依存を入れる (uv があれば uv、無ければ python3 -m venv)
set -e
cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
else
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
echo "セットアップ完了。./run.sh でサーバーを起動できます。"

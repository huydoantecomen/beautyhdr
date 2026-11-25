#!/usr/bin/env bash
# Cai dat tren Linux / macOS
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
echo "Xong. Chay: . .venv/bin/activate && python -m net_anh"

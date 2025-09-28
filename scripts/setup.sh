#!/usr/bin/env bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Environment ready. Use: source .venv/bin/activate"

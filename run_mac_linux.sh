#!/bin/sh
# Base SKU Demand Planner - run with:  sh run_mac_linux.sh
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up. This takes a few minutes and needs internet access."
  python3 -m venv .venv || { echo "Python 3.10 or newer is required."; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || exit 1
fi
.venv/bin/python -m pip install -q -r requirements.txt || exit 1
echo "Starting. Open http://localhost:8501 in your browser. Press Ctrl+C here to stop."
( sleep 5; (open http://localhost:8501 || xdg-open http://localhost:8501) >/dev/null 2>&1 ) &
exec .venv/bin/python -m streamlit run app.py

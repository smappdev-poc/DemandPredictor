#!/bin/sh
# Demand Predictor - DEMO copy, on its own database and port 8502.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || { echo "Python 3.10 or newer is required."; exit 1; }
fi
.venv/bin/python -m pip install -q -r requirements.txt || exit 1
export PLANNER_DB="$(pwd)/data/demo.db" PLANNER_DEMO=1
echo "Starting the DEMO at http://localhost:8502. Press Ctrl+C to stop."
( sleep 5; (open http://localhost:8502 || xdg-open http://localhost:8502) >/dev/null 2>&1 ) &
exec .venv/bin/python -m streamlit run app.py --server.port 8502

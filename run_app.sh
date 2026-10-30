#!/usr/bin/env bash
# Starts the Streamlit app on THIS computer only (nothing is installed or published).
# Run from a terminal:  bash run_app.sh      Stop with Ctrl+C.

cd "$(dirname "$0")" || exit 1

if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "Error: no Python found. Create the environment first (see SETUP.md): uv venv .venv && uv pip install -r requirements.txt" >&2
  exit 1
fi

if ! "$PY" -c "import streamlit" >/dev/null 2>&1; then
  echo "Error: streamlit is not installed for $PY. Run: uv pip install -r requirements.txt" >&2
  exit 1
fi

if [ ! -f streamlit_app.py ] || [ ! -f m11.py ]; then
  echo "Error: streamlit_app.py or m11.py is missing from $(pwd)." >&2
  exit 1
fi

echo "Starting the app. Open http://localhost:8501 in your browser. Press Ctrl+C to stop."
exec "$PY" -m streamlit run streamlit_app.py --server.port 8501

#!/usr/bin/env bash
# Launch the face <-> sketch web UI on this machine with a checkpoint trained on Colab.
# Usage: ./run_local.sh [checkpoint]      (default: checkpoints/cyclegan_final.pth)
#        PORT=8000 ./run_local.sh         (default port: 5000)
set -euo pipefail
cd "$(dirname "$0")"

CKPT="${1:-checkpoints/cyclegan_final.pth}"
PORT="${PORT:-5000}"

if [[ ! -f "$CKPT" ]]; then
  echo "error: checkpoint not found: $CKPT" >&2
  echo "Download cyclegan_final.pth (the last step of colab_run.ipynb) into q2/checkpoints/." >&2
  exit 1
fi
if ! command -v python3 >/dev/null; then
  echo "error: python3 is not installed" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "First run: creating .venv and installing requirements (torch is large, this takes a few minutes)..."
  python3 -m venv .venv
fi
.venv/bin/pip install -q -r requirements.txt

echo "Open http://127.0.0.1:$PORT  (Ctrl+C to stop)"
exec .venv/bin/python app.py --checkpoint "$CKPT" --port "$PORT"

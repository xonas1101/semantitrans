#!/usr/bin/env bash
# Cross-platform (Linux/macOS) setup. Creates a venv on a supported Python
# (3.11–3.13; NOT 3.14, which lacks ML wheels) and installs requirements.
#
# Usage:  ./setup.sh            # CPU/default torch
#         ./setup.sh --optional # also install spaCy/peft/COMET extras
#
# Windows users: use setup_windows.bat (the Python entry points are identical).
set -euo pipefail

# Pick a supported interpreter: prefer 3.12, then 3.11/3.13.
PY=""
for cand in python3.12 python3.11 python3.13; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "No supported Python (3.11-3.13) found. On Fedora: sudo dnf install -y python3.12" >&2
  exit 1
fi
echo "Using interpreter: $($PY --version)"

$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ "${1:-}" = "--optional" ]; then
  python -m pip install -r requirements-optional.txt
  python -m spacy download en_core_web_sm || true
fi

echo
echo "Setup complete. Activate with:  source .venv/bin/activate"
python verify_setup.py || true

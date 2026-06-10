#!/usr/bin/env bash
# Cross-platform (Linux/macOS) setup. Creates a venv on a supported Python
# (3.11–3.13; NOT 3.14, which lacks ML wheels) and installs requirements.
#
# Usage:  ./setup.sh            # CPU/default torch
#         ./setup.sh --optional # also install spaCy/peft/COMET extras
#
# Windows users: use setup_windows.bat (the Python entry points are identical).
set -euo pipefail

# Preferred path: uv (no root, fetches a managed CPython, cross-platform).
# This is how the project was bootstrapped on Fedora where only Python 3.14
# (no ML wheels) was available system-wide.
if command -v uv >/dev/null 2>&1 || python3 -m pip install --user -q uv 2>/dev/null; then
  export PATH="$HOME/.local/bin:$PATH"
  echo "Using uv: $(uv --version)"
  uv python install 3.12
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
  if [ "${1:-}" = "--optional" ]; then
    uv pip install --python .venv/bin/python -r requirements-optional.txt
    .venv/bin/python -m spacy download en_core_web_sm || true
  fi
else
  # Fallback: a system Python 3.11-3.13 on PATH (NOT 3.14).
  PY=""
  for cand in python3.12 python3.11 python3.13; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
  if [ -z "$PY" ]; then
    echo "No uv and no supported Python (3.11-3.13). Install uv or: sudo dnf install -y python3.12" >&2
    exit 1
  fi
  echo "Using interpreter: $($PY --version)"
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  if [ "${1:-}" = "--optional" ]; then
    .venv/bin/python -m pip install -r requirements-optional.txt
    .venv/bin/python -m spacy download en_core_web_sm || true
  fi
fi

echo
echo "Setup complete. Activate with:  source .venv/bin/activate"
python verify_setup.py || true

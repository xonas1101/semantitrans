"""Sanity check: report Python, torch, CUDA, ffmpeg, deps, and the chosen device.

Run after installing requirements:  python verify_setup.py
Exits non-zero if a hard requirement (torch / ffmpeg) is missing.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK " if ok else "MISSING"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info >= (3, 14):
        print("  WARNING: Python 3.14+ has no ML wheels yet; use a 3.11–3.13 venv.")

    hard_ok = True

    # ffmpeg (Whisper dependency)
    ffmpeg = shutil.which("ffmpeg")
    hard_ok &= check("ffmpeg", ffmpeg is not None, ffmpeg or "install ffmpeg")

    # torch + device
    try:
        import torch

        cuda = torch.cuda.is_available()
        detail = torch.__version__ + (
            f", CUDA available ({torch.cuda.get_device_name(0)})" if cuda else ", CPU only"
        )
        check("torch", True, detail)
    except ImportError:
        hard_ok &= check("torch", False, "pip install -r requirements.txt")
        cuda = False

    # soft deps
    for mod, hint in [
        ("whisper", "openai-whisper"),
        ("transformers", "transformers"),
        ("sentencepiece", "sentencepiece"),
        ("datasets", "datasets"),
        ("pandas", "pandas"),
        ("sacrebleu", "sacrebleu"),
        ("gtts", "gTTS"),
    ]:
        check(mod, importlib.util.find_spec(mod) is not None, f"pip install {hint}")

    for mod, hint in [
        ("spacy", "optional: lemmatized idiom matching"),
        ("peft", "optional: LoRA adapter"),
        ("comet", "optional: COMET metric"),
    ]:
        present = importlib.util.find_spec(mod) is not None
        print(f"[{'OK ' if present else '— '}] {mod} ({hint})")

    # device decision (mirrors config.get_device)
    try:
        import config

        device = config.log_device()
        print(f"\nChosen device: {device}")
    except Exception as e:  # pragma: no cover
        print(f"\nCould not import config: {e}")

    print("\nResult:", "READY" if hard_ok else "NOT READY (fix MISSING items above)")
    return 0 if hard_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

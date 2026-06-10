"""Central configuration: model IDs, paths, and device detection.

Device selection is automatic: CUDA when a working GPU is present, otherwise
CPU. We never hardcode "cuda" so the exact same code runs on Linux, Windows and
macOS. On startup the chosen device is logged once (see `log_device`).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("semantitrans")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TESTSET_DIR = DATA_DIR / "testset"
AUDIO_DIR = TESTSET_DIR / "audio"
GLOSS_DIR = DATA_DIR / "glosses"
MODELS_CACHE = ROOT_DIR / ".model_cache"  # local HF cache (override with HF_HOME)

# The idiom gloss knowledge base consumed by the resolver.
GLOSS_KB_PATH = GLOSS_DIR / "idiom_glosses.json"
MANIFEST_PATH = TESTSET_DIR / "manifest.csv"

for _d in (DATA_DIR, TESTSET_DIR, AUDIO_DIR, GLOSS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model identifiers (all pretrained; cited in README)
# --------------------------------------------------------------------------- #
# Stage 1 — ASR. "small" is a good speed/quality tradeoff on a 4060; use
# WHISPER_MODEL=medium for higher accuracy. Override via env var.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

# Stage 3 — EN->HI translation (MarianMT, Apache-2.0).
TRANSLATOR_MODEL = os.environ.get("TRANSLATOR_MODEL", "Helsinki-NLP/opus-mt-en-hi")

# Optional LoRA adapter directory (applied on top of TRANSLATOR_MODEL if present
# and enabled). Built by train_lora.py.
LORA_ADAPTER_DIR = ROOT_DIR / "adapters" / "opus-mt-en-hi-idioms"

# Test-set source corpus (CC-BY-4.0).
MAGPIE_DATASET = "gsarti/magpie"


# --------------------------------------------------------------------------- #
# Device detection
# --------------------------------------------------------------------------- #
def get_device() -> str:
    """Return "cuda" if a usable GPU is present, else "cpu".

    Imported lazily so that config can be imported without torch installed
    (e.g. by verify_setup.py before dependencies exist).
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_dtype():
    """Half precision on GPU, full precision on CPU."""
    try:
        import torch
    except ImportError:
        return None
    return torch.float16 if get_device() == "cuda" else torch.float32


def log_device() -> str:
    """Log and return the chosen device. Call once at startup."""
    device = get_device()
    if device == "cuda":
        try:
            import torch

            name = torch.cuda.get_device_name(0)
            logger.info("Device: cuda (%s)", name)
        except Exception:  # pragma: no cover - defensive
            logger.info("Device: cuda")
    else:
        logger.info("Device: cpu (no CUDA GPU detected)")
    return device


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

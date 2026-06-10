"""Stage 1 — Whisper ASR wrapper (pretrained, frozen).

Cited as: OpenAI Whisper (Radford et al. 2022), MIT license.
We do not fine-tune or modify Whisper; it is used purely for inference.
"""

from __future__ import annotations

import logging

import config

logger = logging.getLogger("semantitrans.asr")


class WhisperASR:
    """Thin wrapper around openai-whisper that is device-aware.

    The model is loaded lazily on first use so importing this module is cheap.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or config.WHISPER_MODEL
        self.device = device or config.get_device()
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            import whisper

            logger.info("Loading Whisper '%s' on %s", self.model_name, self.device)
            # fp16 is only valid on CUDA; whisper handles dtype internally but
            # we pass the device explicitly so it never defaults to cuda.
            self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """Transcribe an audio file to English text."""
        model = self._ensure_loaded()
        use_fp16 = self.device == "cuda"
        result = model.transcribe(str(audio_path), language=language, fp16=use_fp16)
        text = (result.get("text") or "").strip()
        logger.debug("ASR: %s", text)
        return text

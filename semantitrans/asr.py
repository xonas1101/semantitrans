"""Stage 1 — Whisper ASR wrapper (transformers backend).

Base model: OpenAI Whisper (Radford et al. 2022), MIT. We additionally support
OUR fine-tuned checkpoint: if config.WHISPER_FT_DIR exists it is loaded in place
of the base model (built by train_whisper.py). Otherwise the pretrained base is
used as-is.

We use the `transformers` Whisper implementation (not the openai-whisper pip
package) so a local fine-tuned directory and a Hub id load through the same path.
"""

from __future__ import annotations

import logging

import config

logger = logging.getLogger("semantitrans.asr")


class WhisperASR:
    """Device-aware Whisper wrapper. Model loads lazily on first transcribe."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        # Prefer our fine-tuned checkpoint when present.
        if model_name is None and config.WHISPER_FT_DIR.exists():
            model_name = str(config.WHISPER_FT_DIR)
            logger.info("Using fine-tuned Whisper at %s", model_name)
        self.model_name = model_name or config.WHISPER_MODEL
        self.device = device or config.get_device()
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info("Loading Whisper '%s' on %s", self.model_name, self.device)
        self._processor = WhisperProcessor.from_pretrained(self.model_name)
        self._model = (
            WhisperForConditionalGeneration.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """Transcribe an audio file to text."""
        import librosa
        import torch

        self._ensure_loaded()
        audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
        features = self._processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)
        forced = self._processor.get_decoder_prompt_ids(language=language, task="transcribe")
        with torch.no_grad():
            ids = self._model.generate(features, forced_decoder_ids=forced)
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        logger.debug("ASR: %s", text)
        return text

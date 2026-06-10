"""Stage 3 — EN->HI translation wrapper (pretrained MarianMT).

Cited as: Helsinki-NLP/opus-mt-en-hi (Tiedemann & Thottingal 2020), Apache-2.0.

Supports an optional LoRA adapter (config.LORA_ADAPTER_DIR) layered on top of
the base model when `use_lora=True` and the adapter exists. The adapter is the
only optional *trained* component; the base model is used as-is otherwise.
"""

from __future__ import annotations

import logging

import config

logger = logging.getLogger("semantitrans.translator")


class Translator:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        use_lora: bool = False,
    ):
        self.model_name = model_name or config.TRANSLATOR_MODEL
        self.device = device or config.get_device()
        self.use_lora = use_lora
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading translator '%s' on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        if self.use_lora and config.LORA_ADAPTER_DIR.exists():
            try:
                from peft import PeftModel

                logger.info("Applying LoRA adapter from %s", config.LORA_ADAPTER_DIR)
                model = PeftModel.from_pretrained(model, str(config.LORA_ADAPTER_DIR))
            except ImportError:
                logger.warning("peft not installed; ignoring LoRA adapter")
        elif self.use_lora:
            logger.warning(
                "use_lora=True but no adapter at %s; using base model",
                config.LORA_ADAPTER_DIR,
            )

        self._model = model.to(self.device).eval()

    def translate(self, text: str, max_length: int = 512) -> str:
        """Translate a single English string to Hindi."""
        if not text or not text.strip():
            return ""
        self._ensure_loaded()
        import torch

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)
        with torch.no_grad():
            generated = self._model.generate(**inputs, max_length=max_length)
        hindi = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return hindi.strip()

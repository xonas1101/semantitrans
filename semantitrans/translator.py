"""Stage 3 — EN->HI translation wrapper.

Default: facebook/nllb-200-distilled-600M (Meta NLLB, CC-BY-NC-4.0) — much more
fluent for Hindi than opus-mt-en-hi. Works with any seq2seq HF model; when the
model name contains "nllb" we set the source language and force the Hindi
(hin_Deva) target token. Override with TRANSLATOR_MODEL (e.g. an opus model).

Supports an optional LoRA adapter (config.LORA_ADAPTER_DIR) when `use_lora=True`
and the adapter matches the base model; otherwise it is skipped with a warning.
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
        self._is_nllb = "nllb" in self.model_name.lower()
        self._forced_bos = None  # NLLB target-language token

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
            except Exception as e:  # not installed, or adapter trained on a different base
                logger.warning("Could not apply LoRA adapter (%s); using base model", e)
        elif self.use_lora:
            logger.warning(
                "use_lora=True but no adapter at %s; using base model",
                config.LORA_ADAPTER_DIR,
            )

        if self._is_nllb:
            self._tokenizer.src_lang = config.NLLB_SRC_LANG
            self._forced_bos = self._tokenizer.convert_tokens_to_ids(config.NLLB_TGT_LANG)

        self._model = model.to(self.device).eval()

    def translate(self, text: str, max_length: int = 512) -> str:
        """Translate a single English string to Hindi."""
        if not text or not text.strip():
            return ""
        self._ensure_loaded()
        import torch

        if self._is_nllb:
            self._tokenizer.src_lang = config.NLLB_SRC_LANG  # tokenizer is stateful
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)
        gen_kwargs = {"max_length": max_length}
        if self._forced_bos is not None:
            gen_kwargs["forced_bos_token_id"] = self._forced_bos
        with torch.no_grad():
            generated = self._model.generate(**inputs, **gen_kwargs)
        hindi = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return hindi.strip()

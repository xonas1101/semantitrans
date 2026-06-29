"""Stage 2 — Idiom-aware resolution module (the contribution).

This is an INFERENCE-TIME module (no training). It does three things:

  1. DETECT potentially idiomatic spans in the transcript by matching against an
     idiom inventory (the keys of the gloss KB, which is itself built from the
     MAGPIE inventory). Matching is done on a normalized token stream so that
     light inflectional variation is tolerated; lemmatization via spaCy is used
     when available and a pure-Python normalizer is the fallback.

  2. RESOLVE each detected idiom to a literal paraphrase via the gloss KB
     (IdiomKB-style lookup). This is the knowledge-injection step.

  3. INJECT the literal meaning into the text handed to the translator, either
     by SUBSTITUTING the idiom span with its literal paraphrase, or by APPENDING
     the gloss inline as a hint. Both modes are supported so the test set can
     decide which is better.

Honest framing: gloss-injection for idiom MT exists in text-only settings
(IdiomKB 2024; Baziotis et al. EACL 2023). The new angle here is applying it
inside an English-SPEECH -> Hindi-text cascade with a Hindi idiom test set.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import config

logger = logging.getLogger("semantitrans.idiom_resolver")

# Injection modes
MODE_SUBSTITUTE = "substitute"  # replace the idiom span with its literal meaning
MODE_APPEND = "append"          # keep the idiom, add "(literal)" inline as a hint
MODE_OFF = "off"                # passthrough (baseline cascade)

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Our trained figurative-vs-literal detector (built by train_idiom_detector.py).
DETECTOR_DIR = config.ROOT_DIR / "models" / "idiom-detector"


class _Detector:
    """Lazy wrapper around our fine-tuned figurative/literal classifier.

    Given (idiom, sentence) it returns True when the idiom is used figuratively.
    If the model dir is missing or transformers/torch aren't installed, every
    call returns True so the resolver degrades to its rule-based behavior.
    """

    def __init__(self):
        self._model = None
        self._tok = None
        self._ok = DETECTOR_DIR.exists()
        if not self._ok:
            logger.info("No trained detector at %s; resolver gates nothing", DETECTOR_DIR)

    def _ensure(self):
        if self._model is not None or not self._ok:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(str(DETECTOR_DIR))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(DETECTOR_DIR)).eval()
            logger.info("Loaded idiom detector from %s", DETECTOR_DIR)
        except Exception as e:  # missing torch, corrupt dir, etc.
            logger.warning("Could not load detector (%s); gating disabled", e)
            self._ok = False

    def is_figurative(self, idiom: str, context: str) -> bool:
        if not self._ok:
            return True
        self._ensure()
        if self._model is None:
            return True
        import torch

        inputs = self._tok(idiom, context, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        return int(logits.argmax(-1)) == 1  # 1 == figurative


@dataclass
class Detection:
    idiom: str           # canonical KB key
    surface: str         # exact substring from the original text
    literal: str         # literal paraphrase from the KB
    start: int           # char offset in original text
    end: int             # char offset (exclusive)


@dataclass
class ResolutionResult:
    original_text: str
    output_text: str
    mode: str
    detections: list[Detection] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.output_text != self.original_text


class _Normalizer:
    """Token normalizer: lowercase + optional lemmatization (spaCy if present)."""

    def __init__(self, use_lemmas: bool = True):
        self._nlp = None
        self.use_lemmas = use_lemmas
        if use_lemmas:
            try:
                import spacy

                try:
                    self._nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                except OSError:
                    # model not downloaded
                    self._nlp = spacy.blank("en")
                    self._nlp = None  # blank has no lemmatizer; fall back
                    logger.info(
                        "spaCy installed but 'en_core_web_sm' missing; "
                        "using surface normalization. "
                        "Run: python -m spacy download en_core_web_sm"
                    )
            except ImportError:
                logger.info("spaCy not installed; using surface normalization")

    def tokens(self, text: str) -> list[str]:
        """Return normalized tokens (no offsets). Used for KB keys."""
        if self._nlp is not None:
            return [t.lemma_.lower() for t in self._nlp(text) if t.is_alpha or t.is_digit]
        return [m.group(0).lower() for m in _WORD_RE.finditer(text)]

    def tokens_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        """Return (normalized_token, start, end) preserving original offsets."""
        if self._nlp is not None:
            out = []
            for t in self._nlp(text):
                if t.is_alpha or t.is_digit:
                    start = t.idx
                    out.append((t.lemma_.lower(), start, start + len(t.text)))
            return out
        return [
            (m.group(0).lower(), m.start(), m.end())
            for m in _WORD_RE.finditer(text)
        ]


class IdiomResolver:
    """Detect, resolve, and inject idiom literal meanings.

    Interface contract for the pipeline: `literalize(text) -> str`.
    """

    def __init__(
        self,
        gloss_kb_path: str | Path | None = None,
        mode: str = MODE_SUBSTITUTE,
        use_lemmas: bool = True,
        use_detector: bool = True,
    ):
        self.mode = mode
        self.normalizer = _Normalizer(use_lemmas=use_lemmas)
        self.detector = _Detector() if use_detector else None
        self.glosses: dict[str, dict] = {}
        self._index: dict[tuple[str, ...], dict] = {}
        self._max_len = 1
        self._load(gloss_kb_path or config.GLOSS_KB_PATH)

    # ------------------------------------------------------------------ #
    # KB loading / indexing
    # ------------------------------------------------------------------ #
    def _load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            logger.warning("Gloss KB not found at %s; resolver is a no-op", path)
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.glosses = data.get("glosses", {})
        for idiom, entry in self.glosses.items():
            key = tuple(self.normalizer.tokens(idiom))
            if not key:
                continue
            literal = entry.get("literal")
            if not literal:
                continue  # detect-only entries without a gloss can't be resolved
            # keep the longest idiom if two normalize to the same key
            if key not in self._index or len(idiom) > len(self._index[key]["idiom"]):
                self._index[key] = {"idiom": idiom, "literal": literal}
            self._max_len = max(self._max_len, len(key))
        logger.info(
            "Idiom resolver: %d glossed idioms indexed (max %d tokens), mode=%s",
            len(self._index),
            self._max_len,
            self.mode,
        )

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    def detect(self, text: str) -> list[Detection]:
        """Greedy longest-match, non-overlapping idiom detection."""
        if not self._index or not text:
            return []
        toks = self.normalizer.tokens_with_offsets(text)
        n = len(toks)
        detections: list[Detection] = []
        i = 0
        while i < n:
            matched = False
            # try longest windows first so "let the cat out of the bag" beats
            # any shorter sub-idiom
            for span in range(min(self._max_len, n - i), 0, -1):
                key = tuple(t[0] for t in toks[i : i + span])
                entry = self._index.get(key)
                if entry is None:
                    continue
                start = toks[i][1]
                end = toks[i + span - 1][2]
                # Skip injection when our detector says this usage is literal.
                if self.detector is not None and not self.detector.is_figurative(
                    entry["idiom"], text
                ):
                    i += span
                    matched = True
                    break
                detections.append(
                    Detection(
                        idiom=entry["idiom"],
                        surface=text[start:end],
                        literal=entry["literal"],
                        start=start,
                        end=end,
                    )
                )
                i += span
                matched = True
                break
            if not matched:
                i += 1
        return detections

    # ------------------------------------------------------------------ #
    # Injection
    # ------------------------------------------------------------------ #
    def _inject(self, text: str, detections: list[Detection], mode: str) -> str:
        if not detections or mode == MODE_OFF:
            return text
        # apply right-to-left so earlier offsets stay valid
        out = text
        for det in sorted(detections, key=lambda d: d.start, reverse=True):
            if mode == MODE_SUBSTITUTE:
                replacement = det.literal
            elif mode == MODE_APPEND:
                replacement = f"{det.surface} ({det.literal})"
            else:
                raise ValueError(f"unknown injection mode: {mode}")
            out = out[: det.start] + replacement + out[det.end :]
        return out

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def resolve(self, text: str, mode: str | None = None) -> ResolutionResult:
        mode = mode or self.mode
        if mode == MODE_OFF:
            return ResolutionResult(text, text, mode, [])
        detections = self.detect(text)
        output = self._inject(text, detections, mode)
        return ResolutionResult(text, output, mode, detections)

    def literalize(self, text: str, mode: str | None = None) -> str:
        """Pipeline-facing entry point: English in, literalized English out."""
        return self.resolve(text, mode=mode).output_text

"""Chains the three stages with per-stage timing.

The idiom-aware stage 2 is TOGGLEABLE: set idiom_mode="off" to run the plain
ASR -> translate cascade (the baseline), or "substitute"/"append" to enable the
idiom-aware resolution module. Models are loaded lazily and reused across calls.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import config
from semantitrans.asr import WhisperASR
from semantitrans.idiom_resolver import (
    MODE_OFF,
    MODE_SUBSTITUTE,
    Detection,
    IdiomResolver,
)
from semantitrans.translator import Translator

logger = logging.getLogger("semantitrans.pipeline")


@dataclass
class PipelineResult:
    english: str          # stage 1 output (raw transcript)
    intermediate: str     # stage 2 output (literalized English fed to translator)
    hindi: str            # stage 3 output
    detections: list[Detection] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    idiom_mode: str = MODE_OFF


class Pipeline:
    def __init__(
        self,
        idiom_mode: str = MODE_SUBSTITUTE,
        use_lora: bool = False,
        use_lemmas: bool = True,
        device: str | None = None,
    ):
        self.device = device or config.log_device()
        self.idiom_mode = idiom_mode
        self.asr = WhisperASR(device=self.device)
        self.resolver = IdiomResolver(mode=idiom_mode, use_lemmas=use_lemmas)
        self.translator = Translator(device=self.device, use_lora=use_lora)

    def run(self, audio_path: str) -> PipelineResult:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        english = self.asr.transcribe(audio_path)
        timings["asr"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        resolution = self.resolver.resolve(english, mode=self.idiom_mode)
        intermediate = resolution.output_text
        timings["resolve"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        hindi = self.translator.translate(intermediate)
        timings["translate"] = time.perf_counter() - t0

        timings["total"] = sum(timings.values())
        return PipelineResult(
            english=english,
            intermediate=intermediate,
            hindi=hindi,
            detections=resolution.detections,
            timings=timings,
            idiom_mode=self.idiom_mode,
        )

    def translate_text(self, english: str) -> PipelineResult:
        """Run stages 2-3 only on already-transcribed English (for eval reuse)."""
        resolution = self.resolver.resolve(english, mode=self.idiom_mode)
        hindi = self.translator.translate(resolution.output_text)
        return PipelineResult(
            english=english,
            intermediate=resolution.output_text,
            hindi=hindi,
            detections=resolution.detections,
            idiom_mode=self.idiom_mode,
        )

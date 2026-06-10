"""semantitrans — idiom-aware English-speech to Hindi-text pipeline.

Stage 1  Whisper ASR (pretrained, frozen)         -> English transcript
Stage 2  Idiom-aware resolution module (the new part) -> literalized English
Stage 3  opus-mt-en-hi translation (pretrained)    -> Hindi text
"""

__version__ = "0.1.0"

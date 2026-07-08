# Results — semantitrans (idiom-aware EN speech → HI text over a noisy channel)

All plots and raw numbers in one place. Each result has a matching
`*_summary.txt` with the takeaway in plain words.

| Result | Plot | Data | Summary |
|---|---|---|---|
| Idiom detector accuracy | `detector_cm.png` | — | `detector_summary.txt` |
| Test-set composition | `testset.png` | — | (250 clips: 200 figurative + 50 literal) |
| ASR robustness (WER vs SNR) | `wer_snr.png` | `wer_snr.csv` | `wer_snr_summary.txt` |
| Semantic communication (3 schemes vs SNR) | `semcom_snr.png` | `semcom_snr.csv` | `semcom_summary.txt` |

## Pipeline

English speech → Whisper ASR (fine-tuned by us) → idiom detection + resolution
(DistilBERT detector trained by us on MAGPIE) → NLLB-200 translation → Hindi text.

## Components trained by us

- `models/whisper-ft` — Whisper fine-tuned on LibriSpeech (train_whisper.py)
- `models/idiom-detector` — DistilBERT figurative/literal classifier, MAGPIE (train_idiom_detector.py)
- `models/semcodec` — semantic channel codec trained FROM SCRATCH (train_semcodec.py)
- `adapters/opus-mt-en-hi-idioms` — LoRA adapter (train_lora.py)

Pretrained backbones (cited, standard practice): OpenAI Whisper, Meta NLLB-200,
DistilBERT. Semantic-codec architecture follows the DeepSC paradigm
(Xie et al., 2021); implementation and weights are ours.

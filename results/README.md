# Results — semantitrans (idiom-aware EN speech → HI text over a noisy channel)

All plots and raw numbers in one place. Each result has a matching
`*_summary.txt` with the takeaway in plain words.

**Start with `KEY_RESULTS.md` — every important number with its meaning, in one file.**
`PRESENTATION.md` is the 5-minute speaker script.

| Result | Plot | Data | Summary |
|---|---|---|---|
| Idiom detector accuracy | `detector_cm.png` | — | `detector_summary.txt` |
| Test-set composition | `testset.png` | — | (250 clips: 200 figurative + 50 literal) |
| ASR robustness (WER vs SNR) | `wer_snr.png` | `wer_snr.csv` | `wer_snr_summary.txt` |
| Semantic communication (robustness) | `semcom_snr.png` | `semcom_snr.csv` | `semcom_summary.txt` |
| Semantic communication (vs gold refs) | `semcom_snr_gold.png` | `semcom_snr_gold.csv` | `KEY_RESULTS.md` §6 |
| Idiom-aware vs baseline + LTE (gold) | — | manifest.csv | `KEY_RESULTS.md` §1–2 |

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

# semantitrans — idiom-aware English-speech → Hindi-text

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xonas1101/semantitrans/blob/main/colab_demo.ipynb)

Translate **English audio → Hindi text**, with special handling for
**idiomatic / figurative** English (e.g. *"raining cats and dogs"*) that standard
speech-translation cascades render literally and wrong.

**Fastest way to try it: open the Colab notebook above** (free GPU, nothing to
install). Local install instructions are further down.

## Pipeline

```
audio
 ─▶ [Stage 1] Whisper ASR (pretrained, frozen)            → English transcript
 ─▶ [Stage 2] Idiom-aware resolution module (the new part) → literalized English
 ─▶ [Stage 3] opus-mt-en-hi translation (pretrained)       → Hindi text
```

Stages 1 and 3 are off-the-shelf pretrained models, used as-is. The contribution
is **Stage 2** plus a **purpose-built Hindi idiom test set** and evaluation.

### What Stage 2 does (inference-time, no training)

1. **Detect** idioms in the transcript by matching an idiom inventory (the gloss
   KB, built from the MAGPIE inventory) over a normalized token stream
   (lemmatized via spaCy when available, surface-normalized otherwise).
2. **Resolve** each idiom to a literal paraphrase via a gloss knowledge base
   (IdiomKB-style lookup): `idiom → literal meaning`.
3. **Inject** that meaning into the text sent to the translator, either by
   **substituting** the idiom span or **appending** the gloss as a hint. Both
   modes ship; the test set decides which is better.

> **Honest framing.** Gloss-injection for idiom MT already exists in text-only
> settings (IdiomKB 2024; Baziotis et al. EACL 2023). The new angle here is
> applying it inside an English-**speech** → Hindi-**text** cascade, evaluated
> with a Hindi idiom test set. We do not claim to have invented idiom resolution.

## Example (real output)

Input audio says: *"It was raining cats and dogs last night, so we called it a day."*

| Mode | Hindi output | Notes |
|------|--------------|-------|
| `off` (baseline) | `यह रात हम इसे एक दिन कहा जाता था` | both idioms lost / garbled |
| `substitute` (idiom-aware) | `रात बहुत ही भारी वर्षा हो रही थी, इसलिए हम दिन के लिए काम करना बंद कर देते थे` | *"it rained very heavily ... so we stopped working for the day"* — correct |

The resolver detected `raining cats and dogs → raining very heavily` and
`called it a day → stop working for the day` (lemma-matched from "called").

## Run in Google Colab (recommended — free GPU)

The local machine's GPU may be unavailable (e.g. missing driver); Colab gives a
free CUDA GPU with nothing to install. Click the **Open in Colab** badge above,
or open `colab_demo.ipynb`. The notebook clones this repo, installs deps (reusing
Colab's CUDA torch), and runs the demo + lets you upload your own audio.

## Install (local)

Requires **Python 3.11–3.13** (not 3.14 — no ML wheels yet) and **ffmpeg**.

```bash
# Linux / macOS — setup.sh prefers uv (no root; fetches managed CPython 3.12)
./setup.sh                 # core deps
./setup.sh --optional      # + spaCy / peft extras

# Windows (optional convenience; Python entry points are identical)
setup_windows.bat
```

If you don't have a supported Python, the easiest no-root route is
[`uv`](https://docs.astral.sh/uv/) (what `setup.sh` uses):

```bash
python3 -m pip install --user uv     # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

(Alternatively on Fedora: `sudo dnf install -y python3.12`.)
For **GPU**, install the matching CUDA torch build from <https://pytorch.org>
first; the code auto-detects CUDA and falls back to CPU. Verify:

```bash
python verify_setup.py     # reports torch, CUDA, ffmpeg, and the chosen device
python download_models.py  # optional: pre-cache Whisper + opus-mt-en-hi
```

## Usage

```bash
python run.py audio.wav                 # idiom-aware (span substitution, default)
python run.py audio.wav --mode append   # idiom-aware (gloss-hint injection)
python run.py audio.wav --mode off      # plain cascade (baseline)
python run.py audio.wav --lora          # use the optional LoRA adapter
python run.py audio.wav --json          # machine-readable output
```

## Evaluation workflow

```bash
python build_testset.py --n 60          # MAGPIE figurative → TTS audio → manifest
# → then fill the 'hindi_reference' (and optional 'literal_trap_hi') columns
#   in data/testset/manifest.csv with correct Hindi (the gold standard).

python run_testset.py --mode off        # baseline predictions
python run_testset.py --mode substitute # idiom-aware predictions
python run_testset.py --mode append     # idiom-aware (hint) predictions
python evaluate.py                      # chrF++ + literal-translation-error
python evaluate.py --comet              # also COMET (~2GB, GPU recommended)
```

**Metrics:** chrF++ (sacrebleu), a literal-translation-error rate (LTE, inspired
by Baziotis et al. EACL 2023; needs the optional `literal_trap_hi` column), and
optional COMET.

## Gloss knowledge base

`data/glosses/idiom_glosses.json` ships a small, project-authored seed of common
idiom paraphrases (source tag `seed`). Expand or audit coverage against MAGPIE:

```bash
python build_glosses.py --audit                       # coverage report
python build_glosses.py --emit-template todo.json     # unglossed MAGPIE idioms
python build_glosses.py --merge ext.json --merge-license "<license>"
```

**Verify the license of any external gloss source before redistributing** the
merged KB, and keep each entry's `source` tag honest.

## Optional LoRA adapter

`train_lora.py` fits a small LoRA adapter on opus-mt-en-hi over idiom-containing
EN-HI pairs (tens of minutes on a 4060 — not a from-scratch/full fine-tune).
Report it as exactly that.

## Layout

| Path | Role |
|------|------|
| `config.py` | model IDs, paths, device detection (`get_device`, `log_device`) |
| `semantitrans/asr.py` | Stage 1 Whisper wrapper |
| `semantitrans/idiom_resolver.py` | **Stage 2 — the contribution** |
| `semantitrans/translator.py` | Stage 3 opus-mt-en-hi wrapper (+ optional LoRA) |
| `semantitrans/pipeline.py` | chains stages, timing, idiom toggle |
| `run.py` | CLI (audio → Hindi) |
| `build_glosses.py` | gloss KB audit / expand / merge |
| `build_testset.py` · `run_testset.py` · `evaluate.py` | test set + eval |
| `train_lora.py` | optional LoRA adapter |

## Credits / licenses

- **Whisper** — Radford et al. 2022 (MIT).
- **opus-mt-en-hi** — Tiedemann & Thottingal 2020, Helsinki-NLP (Apache-2.0).
- **MAGPIE** — Haagsma et al. 2020; `gsarti/magpie` (CC-BY-4.0).
- Method context: **IdiomKB** (2024); **Baziotis et al.** EACL 2023
  (literal-translation-error metric); **Zaitova et al.** ACL 2025 (TTS-based
  idiom speech-translation test set); Dankers et al. 2022.

Non-commercial academic project. Everything derived from pretrained models or
external datasets is cited as such.

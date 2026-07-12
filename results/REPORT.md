# semantitrans — Project Report

**Idiom-aware English-speech → Hindi-text translation as a semantic communication system**

BTech project · github.com/xonas1101/semantitrans · July 2026

---

## Abstract

Standard speech-translation cascades destroy idiom meaning: "it's raining cats
and dogs" becomes a literal Hindi sentence about animals. This project builds a
three-stage pipeline (Whisper ASR → a gated idiom-resolution module trained by
us → NLLB-200 translation) and then treats the whole system as a **semantic
communication** problem: instead of transmitting the speech signal over a noisy
wireless channel, the sender extracts the *meaning* and transmits only that.
On a 250-sentence gold-annotated test set, idiom resolution cuts literal
translation errors from **48.0% to 13.5%** with provably zero harm on literal
usages. Over a simulated AWGN channel, transmitting meaning as text needs
**2,418× fewer bits** than the waveform, and our from-scratch DeepSC-style
channel codec (8-bit quantized symbols) preserves meaning gracefully at SNRs
where digital text transmission collapses.

---

## 1. The idea

Two failures of a naive "record → transmit → transcribe → translate" system,
and one fix for both:

1. **Linguistic failure.** Neural MT translates idioms word-by-word. The words
   survive; the meaning dies.
2. **Communication failure.** The raw waveform is huge (~1.7 Mbit per
   utterance) and fragile — at -5 dB SNR its transcription WER exceeds 100%.

Both are the same disease: the system is faithful to the **signal** instead of
the **meaning**. The fix is to extract the meaning at the sender (ASR + idiom
resolution) and make *that* the transmitted message. This is exactly the
semantic-communication paradigm (Shannon & Weaver's "level B"; revived by
DeepSC, Xie et al. 2021): success is measured by whether the meaning survives,
not whether the bits do.

## 2. System architecture

```
                        SENDER                                 RECEIVER
audio ─▶ Whisper ASR ─▶ idiom detection ─▶ gloss     channel   NLLB-200 ─▶ Hindi
         (fine-tuned)    (our DistilBERT    substitution ───▶  EN→HI       text
                          classifier)       ("meaning text")
```

- **Stage 1 — ASR.** `openai/whisper-tiny`, demonstratively fine-tuned by us on
  a LibriSpeech slice (`train_whisper.py`).
- **Stage 2 — idiom resolution (the core contribution).** An idiom inventory
  (built from MAGPIE) is matched over the lemmatized transcript. A
  DistilBERT classifier **we trained on MAGPIE** (96.5% accuracy, F1 0.976
  held-out) decides figurative vs literal for each match; only confidently
  figurative usages get their literal meaning substituted from a gloss
  knowledge base. The gate suppresses substitution when p(literal) ≥ 0.70, so
  idioms used literally ("the dog was raining water off its fur"?) pass through
  untouched.
- **Stage 3 — MT.** `facebook/nllb-200-distilled-600M` EN→HI (optional LoRA
  adapter on opus-mt-en-hi also trained, `train_lora.py`).
- **Semantic communication layer.** `semcom_eval.py` simulates an AWGN wireless
  channel and compares four ways to deliver one utterance (§5). One of them is
  **our own channel codec** (`semantitrans/semcodec.py`): a transformer
  encoder → 16 channel symbols per token (8-bit quantized, power-normalized) →
  AWGN → transformer decoder, **trained from scratch** on 60k MAGPIE sentences
  with the noisy channel *inside* the training loop at random SNRs (-4..12 dB).

## 3. Models trained in this project

| Component | Base | Trained on | Result |
|---|---|---|---|
| Idiom detector | DistilBERT | MAGPIE figurative/literal | 96.5% acc, F1 0.976 |
| Whisper fine-tune | whisper-tiny | LibriSpeech slice | demonstrative |
| LoRA adapter | opus-mt-en-hi | IIT-B idiom pairs (299) | thin, optional |
| **Semantic channel codec** | **none — from scratch** | 60k MAGPIE sentences, 90 epochs (best-val checkpoint, cosine LR decay) | graceful degradation (§5) |

All backbones are cited pretrained models (standard research practice); the
codec is entirely our code and weights, following the DeepSC architecture
paradigm.

## 4. Test set and metrics

- **Test set:** 250 TTS-generated clips from MAGPIE (200 figurative + 50
  literal idiom usages), with **gold Hindi references annotated by hand** and a
  `literal_trap_hi` column holding the wrong-literal trap word per idiom
  (e.g. बिल्ली for "raining cats and dogs"). TTS-based idiom speech-translation
  test sets are an accepted methodology (Zaitova et al., ACL 2025).
- **chrF++** (sacrebleu): character n-gram F-score, standard for Hindi MT.
- **LTE — Literal Translation Error rate** (inspired by Baziotis et al.,
  EACL 2023): % of figurative sentences whose output contains the wrong
  literal rendering of the idiom.
- **chrF vs SNR**: meaning preservation across the noisy channel, both against
  each scheme's own clean output (robustness) and against gold references.

## 5. Results

### 5.1 Idiom resolution (gold references, 250 sentences)

| mode | LTE % ↓ | chrF++ figurative ↑ | chrF++ literal |
|---|---|---|---|
| baseline (off) | 48.0 | 37.31 | 43.05 |
| gloss appended | 46.0 | 38.40 | 43.05 |
| **gloss substituted** | **13.5** | 38.00 | 43.05 (byte-identical) |

Three findings: (a) substitution cuts literal idiom errors **3.6×**; (b) merely
*hinting* the meaning (append) barely helps — the idiom must be *replaced*;
(c) all modes are byte-identical on literal-usage sentences, i.e. the detector
gate causes **zero over-correction**.

### 5.2 Bits per message (the bandwidth result)

| scheme | what crosses the channel | bits/message | vs waveform |
|---|---|---|---|
| traditional | 16 kHz × 16-bit waveform | 1,692,058 | 1× |
| semantic (text) | idiom-resolved English, UTF-8 | 700 | **2,418× fewer** |
| semantic (text, rep-3 coded) | same + rate-1/3 repetition code | 2,100 | 806× fewer |
| semantic (our codec) | learned symbols, 8-bit quantized | 2,309 | **733× fewer** |

The waveform's megabits carry voice, accent, and room noise the receiver never
needed. Intelligence at the endpoints replaces bandwidth in the channel.

### 5.3 Meaning vs channel SNR

(plots: `semcom_snr.png` robustness, `semcom_snr_gold.png` absolute)

Meaning preservation (chrF vs each scheme's clean-channel output, n=25):

| SNR (dB) | traditional | text uncoded | text rep-3 coded | our codec |
|---|---|---|---|---|
| 10 | 0.53 | **1.00** | **1.00** | 0.63 |
| 5 | 0.33 | 0.65 | **0.98** | 0.35 |
| 2 | 0.28 | 0.12 | **0.69** | 0.25 |
| 0 | 0.20 | 0.04 | **0.37** | 0.17 |
| -2 | 0.17 | 0.02 | **0.16** | 0.12 |
| -5 | 0.14 | 0.01 | 0.03 | **0.09** |

- Good channel (10 dB): text bits deliver the meaning **perfectly** at 1/2418
  of the bandwidth.
- Below ~3 dB: uncoded text falls off the digital cliff (chrF 0.04 at 0 dB —
  bit errors shred UTF-8). The rep-3 repetition code pushes the cliff ~3-4 dB
  left but still collapses (0.03 at -5 dB); coding delays the cliff, it does
  not remove it.
- **Our codec has no cliff**: trained with the channel in the loop, it degrades
  gracefully, is the only semantic scheme still working at -5 dB (3× the
  coded text), and roughly matches the full 1.7-Mbit waveform at a small
  fraction of the bits. Graceful degradation is the
  signature result of learned semantic communication.

(Full numbers: `KEY_RESULTS.md` and the CSVs in this folder.)

### 5.4 Rayleigh fading (the realistic wireless channel)

(plots: `semcom_snr_rayleigh.png`, `semcom_snr_rayleigh_gold.png`)

Same experiment over quasi-static flat Rayleigh fading with perfect CSI (one
fade drawn per message, shared by all schemes):

| SNR (dB) | traditional | text uncoded | text rep-3 coded | our codec |
|---|---|---|---|---|
| 10 | 0.39 | 0.78 | **0.87** | 0.59 |
| 5 | 0.36 | 0.60 | **0.81** | 0.38 |
| 2 | 0.28 | 0.28 | **0.55** | 0.27 |
| 0 | 0.18 | 0.08 | **0.30** | 0.18 |
| -2 | 0.14 | 0.05 | **0.17** | 0.14 |
| -5 | 0.11 | 0.02 | 0.08 | **0.12** |

Deep fades break digital text long before AWGN does (uncoded text is no longer
perfect even at 10 dB), and the codec's advantage *grows*: at -5 dB it beats
every scheme including the full 1.7-Mbit waveform. Random-SNR training is in
effect training on a fading channel, so this is expected — and matches
DeepSC's Rayleigh findings.

### 5.5 WER of the received Hindi

(plots: `semcom_wer.png`, `semcom_wer_rayleigh.png`, `_gold` variants)

All sweeps also report word error rate vs the same references. WER exceeds
1.0 when corrupted text decodes to garbage longer than the reference —
uncoded text hits WER **7.3** at -2 dB AWGN while the codec stays at 0.91.
The digital cliff is even more dramatic in WER than in chrF.

### 5.6 Semantic noise (meaning corruption at the sender)

(plot: `semcom_semnoise.png` · data: `semcom_semnoise.csv`)

Each sender-side word is replaced with probability p on a *clean* channel,
isolating semantic noise from channel noise. chrF at p = 0/0.1/0.3: plain
text 1.00/0.76/0.43; codec 1.00/0.56/0.26. **Honest negative result:** the
codec amplifies semantic noise (corrupted words are out-of-distribution
inputs) — it defends against channel noise only. Meaning corruption must be
fixed at the source; nothing downstream can undo it.

---

## 6. Comparative study

How this project relates to the published work it builds on, and where it
differs. Links in §8.

| Work | What it does | What we share | What we do differently |
|---|---|---|---|
| **DeepSC** (Xie et al., IEEE TSP 2021) | Transformer joint semantic-channel coding for *text*; beats Huffman+RS/Turbo baselines at low SNR (~8× BLEU gain at 9 dB vs conventional) | Same architecture paradigm and training recipe (channel in the loop, random SNR); same graceful-degradation finding | Ours is a small from-scratch CPU-trained codec inside a *cross-lingual speech* pipeline; DeepSC transmits English to reproduce English — we transmit meaning that a receiver *translates* |
| **DeepSC-S** (Weng & Qin, 2021) | Semantic communication for *speech signals* (SE-network); reconstructs the waveform | Speech over a learned channel | DeepSC-S still reconstructs the *signal*; we discard the signal entirely and send stage-2 text — orders of magnitude fewer bits, because our receiver needs meaning, not audio |
| **DeepSC-SR** (Weng et al., 2021) | Transmits text-related semantic features of speech; receiver outputs a transcript | Closest paradigm relative: speech in, text semantics over the channel | We add the *idiom-resolution* stage before transmission (the transmitted meaning is already disambiguated) and a translation receiver (EN speech → HI text) |
| **Farsad, Rao & Goldsmith (2018)** | First deep joint source-channel coding of text (LSTM) | Text JSCC lineage our codec descends from | Transformer, per-token symbols, quantized; embedded in an application pipeline rather than studied in isolation |
| **Baziotis et al.** (EACL 2023) | Automatic evaluation of idioms in NMT; shows NMT translates idioms literally; introduces targeted literal-error evaluation | Our LTE metric is directly inspired by their targeted evaluation | They diagnose text-to-text MT; we *intervene* (gated gloss substitution) and measure the fix in a speech pipeline, for EN→HI |
| **IdiomKB** (Li et al., 2024) | Multilingual idiom→meaning knowledge base; feeds figurative meanings to LLM translators | Same core intervention: give the translator the meaning, not the idiom | IdiomKB targets LLM prompting for text; we use a compact gloss KB + a *trained figurative/literal gate* in a real-time speech cascade, and show substitution ≫ hinting (13.5% vs 46.0% LTE) |
| **Zaitova et al.** (ACL 2025) | Shows idiom translation is a major weakness of speech-to-text systems (Whisper, SeamlessM4T) on DE/RU→EN | Confirms our problem statement in SLT; validates TTS-built idiom test sets | They benchmark the failure; we build and evaluate a *fix*, on a language pair (EN→HI) they don't cover |
| **Dankers et al.** (2022) | Analyzes how transformers (mis)handle idioms | Motivation for the detector | Analysis paper; ours is a system |

**Positioning in one sentence:** existing work either builds semantic channels
for generic text/speech (DeepSC family) *or* diagnoses idiom failure in
translation (Baziotis, Zaitova) *or* injects idiom meaning into text LLMs
(IdiomKB) — this project is, to our knowledge, the first student-scale system
that chains all three: a gated idiom-meaning extractor inside an English-speech
→ Hindi-text cascade, transmitted over a learned noisy-channel codec, with a
purpose-built gold-annotated Hindi idiom test set.

## 7. Honest limitations & future work

- The Whisper fine-tune is demonstrative; the LoRA adapter is thin (299 idiom
  pairs). The substantive trained models are the **detector** and the **codec**.
- The fading model is quasi-static flat Rayleigh with perfect CSI; no
  Doppler or frequency selectivity.
- chrF under-credits the codec's paraphrases; an embedding-based semantic
  metric (LaBSE cosine / COMET) would score meaning directly.
- The codec still paraphrases (~28% reconstruction WER on a clean channel);
  a larger model (d_model 256+) trained on GPU would raise its ceiling —
  DeepSC-scale codecs reconstruct near-losslessly.
- Gloss KB coverage is a seed set; scaling it (or borrowing IdiomKB with
  license attribution) widens idiom coverage.
- Research-grade systems (DeepSC) use BLEU/sentence-similarity on 100k+
  sentence corpora; our study is deliberately small-scale (250 gold
  sentences, 25-clip channel sweeps) but methodologically parallel — including
  AWGN + Rayleigh channels and a semantic-noise ablation.

## 8. References

- Xie, Qin, Li, Juang. *Deep Learning Enabled Semantic Communication Systems.*
  IEEE Trans. Signal Processing, 2021. <https://www.semanticscholar.org/paper/f9314fd99be5f2b1b3efcfab87197d578160d553>
- Weng, Qin. *Semantic Communication Systems for Speech Transmission.* 2021.
  <https://arxiv.org/abs/2102.12605>
- Weng, Qin, Li. *Semantic Communications for Speech Recognition.* 2021.
  <https://arxiv.org/abs/2107.11190>
- Farsad, Rao, Goldsmith. *Deep Learning for Joint Source-Channel Coding of
  Text.* ICASSP 2018. <https://arxiv.org/abs/1802.06832>
- Baziotis, Mathur, Hasler. *Automatic Evaluation and Analysis of Idioms in
  Neural Machine Translation.* EACL 2023. <https://aclanthology.org/2023.eacl-main.267/>
- Li et al. *Translate Meanings, Not Just Words: IdiomKB's Role in Optimizing
  Idiomatic Translation with Language Models.* AAAI 2024. <https://arxiv.org/abs/2308.13961>
- Zaitova et al. *It's Not a Walk in the Park! Challenges of Idiom Translation
  in Speech-to-text Systems.* ACL 2025. <https://aclanthology.org/2025.acl-long.1512/>
- Dankers, Lucas, Titov. *Can Transformer be Too Compositional? Analysing
  Idiom Processing in Neural Machine Translation.* ACL 2022.
- Haagsma et al. *MAGPIE: A Large Corpus of Potentially Idiomatic Expressions.*
  LREC 2020. · Radford et al. *Whisper*, 2022. · NLLB Team, Meta, 2022.

Non-commercial academic project; every pretrained backbone and dataset is
cited and licensed as noted in the repo README.

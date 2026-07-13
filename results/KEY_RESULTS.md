# Key results — everything that matters, in one place

Test set: 250 TTS clips (200 figurative + 50 literal idiom usages), gold Hindi
references annotated by us. All numbers reproducible with the scripts named
in each section.

---

## 1. Idiom-awareness fixes idiom errors — LTE (the headline)

`evaluate.py` · plot: — · data: manifest.csv `literal_trap_hi`

**LTE (Literal Translation Error rate)** = % of figurative sentences whose
Hindi output contains the WRONG literal rendering of the idiom (e.g. बिल्ली
appearing in "raining cats and dogs").

| mode | LTE% (lower = better) |
|---|---|
| baseline (idiom module off) | **48.0%** |
| idiom-aware, gloss appended | 46.0% |
| **idiom-aware, substituted** | **13.5%** |

**Meaning:** the plain cascade mistranslates nearly HALF of all idioms
literally; substituting the idiom's meaning before translation cuts errors
3.6×. Bonus finding: merely *hinting* the meaning (append) barely helps — the
translator still renders the original idiom words literally. The idiom must be
*replaced*, not annotated.

## 2. Idiom-awareness never hurts — the gating result

`evaluate.py` + usage split · detector plot: `detector_cm.png`

chrF++ (translation quality vs gold, higher = better), split by usage:

| mode | figurative (n=200) | literal (n=50) |
|---|---|---|
| baseline (off) | 37.31 | 43.05 |
| substitute | 38.00 | **43.05 (identical)** |
| append | 38.40 | **43.05 (identical)** |

**Meaning:** quality improves on figurative sentences (+0.7 to +1.1 chrF++)
and is BYTE-IDENTICAL on literal ones — our DistilBERT detector (96.5% acc,
F1 0.976 on held-out MAGPIE, see `detector_cm.png`) correctly suppressed
substitution on every literal usage. Zero over-correction. The figurative
chrF++ gain looks small only because the idiom is a small fragment of each
sentence; metric #1 (LTE) isolates the idiom itself.

## 3. The raw signal is fragile — WER vs SNR

`noise_eval.py` · plot: `wer_snr.png` · data: `wer_snr.csv`

ASR word error rate when the audio crosses an AWGN channel:

| SNR | clean | 30 dB | 20 dB | 10 dB | 5 dB | 0 dB | -5 dB |
|---|---|---|---|---|---|---|---|
| WER | 0.10 | 0.12 | 0.15 | 0.37 | 0.60 | 0.83 | 1.05 |

**Meaning:** transmit the waveform over a noisy wireless channel and
transcription quality collapses — >100% WER at -5 dB (output has more errors
than the reference has words). Motivates transmitting MEANING instead of
signal (results 4-5).

## 4. Meaning needs 2418× fewer bits — the bandwidth result

`semcom_eval.py` · plot: `semcom_snr.png` (right panel)

Average cost to send one utterance:

| scheme | what crosses the channel | bits/message |
|---|---|---|
| traditional | waveform as digital PCM bits (16 kHz × 16-bit) over BPSK | 1,692,058 |
| semantic (text) | idiom-resolved English as UTF-8 | **700 (2418× fewer)** |
| semantic (text, rep-3 coded) | same + rate-1/3 repetition code | 2,100 (806× fewer) |
| semantic (our codec) | learned symbols, 8-bit quantized | 2,309 (**733× fewer**) |

**Meaning:** the waveform's megabits carry voice, accent, pauses, room noise —
none of which the receiver needs. Extracting the meaning first (ASR + idiom
resolution at the SENDER) reduces the message to what the receiver actually
consumes. Intelligence at the endpoints replaces bandwidth in the channel —
the core claim of semantic communication.

## 5. Learned semantic coding survives noise — robustness vs SNR

`semcom_eval.py` · plot: `semcom_snr.png` · data: `semcom_snr.csv`

Meaning preservation (chrF, each scheme vs its own clean-channel output —
isolates pure channel robustness):

The traditional scheme is CLASSICAL DIGITAL transmission — the PCM bits cross
the same BPSK channel as the text schemes (literature-standard baseline; a
noise-free channel delivers the waveform bit-perfectly).

| SNR (dB) | BER | traditional | semantic text | text rep-3 coded | our codec |
|---|---|---|---|---|---|
| 10 | 4e-06 | 0.95 | **1.00** | **1.00** | 0.57 |
| 5 | 6e-03 | 0.18 | 0.61 | **1.00** | 0.38 |
| 2 | 4e-02 | 0.13 | 0.14 | **0.74** | 0.28 |
| 0 | 8e-02 | 0.05 | 0.03 | **0.31** | 0.16 |
| -2 | 1e-01 | 0.01 | 0.02 | **0.15** | 0.12 |
| -5 | 2e-01 | 0.01 | 0.01 | 0.02 | **0.10** |

**Meaning:** this is the literature-standard picture (cf.
`literature_comparison.png`). Classical digital transmission is NEAR-PERFECT
on a good channel (0.95 at 10 dB) — and text bits match it at 1/2418 the
bandwidth. Below 10 dB classical cliffs FIRST of all schemes: its 1.7-Mbit
message collects ~10,000 bit errors at 5 dB where the 700-bit text message
collects ~4. The rep-3 code (3× the bits) holds a perfect score down to 5 dB
and dominates the whole mid-range — coding DELAYS the cliff, it does not
remove it (0.02 at -5 dB). Our from-scratch codec has NO cliff: trained with
the noisy channel inside the loop, it degrades gracefully and is the only
scheme still working at -5 dB (0.10). Graceful degradation is the signature
result of learned semantic communication (DeepSC paradigm, Xie et al. 2021).

## 6. Same experiment against gold references — absolute quality

`semcom_eval.py --gold` · plot: `semcom_snr_gold.png` · data: `semcom_snr_gold.csv`

chrF of received Hindi vs OUR gold references (absolute translation quality,
so no scheme reaches 1.0 even on a clean channel):

| SNR (dB) | traditional | semantic text | text rep-3 coded | our codec |
|---|---|---|---|---|
| 10 | 0.33 | **0.35** | **0.35** | 0.19 |
| 5 | 0.14 | 0.29 | **0.35** | 0.16 |
| 2 | 0.12 | 0.11 | **0.30** | 0.14 |
| 0 | 0.05 | 0.03 | **0.20** | 0.12 |
| -2 | 0.01 | 0.01 | **0.11** | 0.10 |
| -5 | 0.01 | 0.01 | 0.02 | **0.08** |

Gold scoring caps EVERY scheme at the translator's own quality (~0.35 even on
a clean channel), so use these tables for *ordering*, not literature
comparison — the robustness tables (§5, §7) are the literature-comparable
view.

**Meaning:** the ordering confirms §5 with real references: traditional ties
text at 10 dB (both at the translator ceiling — but at 2418× the bits), rep-3
dominates the mid-range, and the codec is the last scheme standing at -5 dB.
The codec's lower ceiling at high SNR is its reconstruction paraphrasing —
chrF counts character overlap, so a correct paraphrase scores low; an
embedding-based semantic metric would credit it better (in the
optional-upgrades list).

## 6b. Non-idiomatic sentences only — where traditional catches up

`semcom_eval.py --gold --literal-only` · plot: `semcom_snr_gold_literal.png`
· data: `semcom_snr_gold_literal.csv` (25 literal-usage clips, gold refs)

| SNR (dB) | traditional | semantic text | text rep-3 coded | our codec |
|---|---|---|---|---|
| 10 | 0.42 | **0.43** | **0.43** | 0.25 |
| 5 | 0.19 | 0.36 | **0.43** | 0.22 |
| 2 | 0.10 | 0.12 | **0.36** | 0.17 |
| 0 | 0.05 | 0.03 | **0.22** | 0.13 |
| -2 | 0.01 | 0.02 | **0.10** | **0.10** |
| -5 | 0.01 | 0.01 | 0.02 | **0.08** |

**Meaning:** remove idioms from the test set and traditional TIES semantic
text on a good channel (0.42 vs 0.43 at 10 dB) — they are the same models on
either side of a clean channel. This isolates WHY traditional trails on the
full test set (§6): the gap there is idiom mistranslation, not channel noise.
Semantic's remaining advantages on plain sentences are bandwidth (2418×) and
low-SNR survival (rep-3/codec below 5 dB) — quality-wise, equality is the
honest claim.

## 6c. Bandwidth efficiency at equal quality

plot: `semcom_efficiency.png` (from `semcom_snr.csv`, 10 dB AWGN)

At 10 dB, where traditional (0.95) and text (1.00) deliver ~equal meaning,
efficiency = meaning per kilobit: text **1.43 chrF/kbit**, rep-3 0.48, codec
0.25, traditional **0.0006** — a ~2,500× gap at the same quality. This is §4's
bit table and §5's quality table folded into one number.

## 7. Rayleigh fading — the realistic wireless channel

`semcom_eval.py --channel rayleigh` · plots: `semcom_snr_rayleigh.png`,
`semcom_snr_rayleigh_gold.png` · data: matching `.csv`

Quasi-static flat Rayleigh fading with perfect CSI (one fade per message,
shared by all schemes). Fading is what a real mobile link looks like:
occasional deep fades wreck a message even when the *average* SNR is good.

Meaning preservation (chrF, robustness scoring):

| SNR (dB) | traditional | semantic text | text rep-3 coded | our codec |
|---|---|---|---|---|
| 10 | 0.69 | 0.80 | **0.92** | 0.53 |
| 5 | 0.28 | 0.51 | **0.69** | 0.33 |
| 2 | 0.09 | 0.23 | **0.43** | 0.20 |
| 0 | 0.06 | 0.09 | **0.27** | 0.14 |
| -2 | 0.04 | 0.08 | **0.20** | 0.10 |
| -5 | 0.02 | 0.02 | **0.15** | 0.11 |

**Meaning:** fading punishes every digital scheme earlier than AWGN — deep
fades occasionally shred a whole message even at 10 dB average SNR (text 0.80,
not 1.00; traditional 0.69, not 0.95). **Rep-3 coded text wins at every single
SNR under fading** — redundancy is exactly what survives a fade — making it
the clear practical scheme for realistic channels at 806× compression. The
codec degrades most gracefully at the bottom (0.11 at -5 dB, ~equal to rep-3's
0.15 and 6× the waveform's 0.02), consistent with random-SNR training acting
as fading training. Gold-referenced version confirms the ordering
(`semcom_snr_rayleigh_gold.csv`).

## 8. WER of the received Hindi — same experiments, harsher metric

plots: `semcom_wer.png`, `semcom_wer_rayleigh.png` (+ `_gold` variants)

Every SNR sweep now also reports word error rate of the received Hindi vs the
same references. WER exceeds 1.0 when corrupted text decodes to garbage longer
than the reference — which is exactly what happens to text schemes below their
cliff (uncoded text WER **2.6** at 0 dB AWGN vs codec 0.92; rep-3 hits 6.0 at
-5 dB once its own cliff arrives). The WER view makes the digital cliff *more*
dramatic than chrF: chrF saturates at 0 while WER keeps growing with the
garbage length. Because unbounded insertions make WER>1 an artifact of
hallucinated length rather than lost meaning, chrF (bounded, standard for
Hindi MT) is the primary metric; the WER plots are supporting material.

## 9. Semantic noise — corrupting the meaning, not the channel

`semcom_eval.py --sem-noise 0 0.05 0.1 0.2 0.3` · plots:
`semcom_semnoise.png`, `semcom_semnoise_wer.png` · data: `semcom_semnoise.csv`

Each word of the sender-side semantic message is replaced with probability p
by a random word (clean channel — isolates *semantic* noise from *channel*
noise, a distinction the DeepSC literature draws).

| p (word corruption) | semantic text (chrF) | our codec (chrF) |
|---|---|---|
| 0.00 | 1.00 | 1.00 |
| 0.05 | 0.88 | 0.76 |
| 0.10 | 0.76 | 0.56 |
| 0.20 | 0.58 | 0.42 |
| 0.30 | 0.43 | 0.26 |

**Meaning (an honest negative result):** the codec is trained to survive
*channel* noise, not *semantic* noise — a corrupted word is an
out-of-distribution input whose damage the codec's reconstruction amplifies,
while plain text passes it through untouched. Semantic noise must be fought at
the source (better ASR / idiom resolution), not in the channel code. This
cleanly separates the two noise types: channel noise → codec wins; semantic
noise → nothing downstream can fix it.

---

## 10. Our numbers next to the literature

No paper shares our exact benchmark (EN speech → HI text, our test set), so
these are parallel findings, not same-dataset comparisons. Full comparative
study: `REPORT.md` §6. The canonical published classical-vs-semantic
crossover is redrawn (approximate values) in `literature_comparison.png` —
our robustness curves (§5, §7) reproduce its shape.

| Finding | Published work says | Our result |
|---|---|---|
| NMT translates idioms literally | Baziotis et al. (EACL 2023): idioms are translated literally far more often than ordinary text; they build targeted literal-error evaluation (our LTE metric follows it) | Baseline cascade: **48.0% LTE** on EN→HI |
| Speech systems are even worse at idioms | Zaitova et al. (ACL 2025): SLT systems (Whisper, SeamlessM4T) show a pronounced drop on idioms vs news and "revert to literal translations", DE/RU→EN | Confirms our problem setting; our pipeline is speech-based and idiom failure was ~half of all figurative sentences |
| Injecting the idiom's meaning fixes it | IdiomKB (AAAI 2024): giving figurative meanings to LLM translators considerably boosts idiom translation (KB quality 2.92/3 human-scored); text-only, ZH/EN/JA | Gated gloss **substitution: 48.0% → 13.5% LTE (3.6×)**; hint-only append barely helps (46.0%) — replacement beats hinting |
| Classical is fine at high SNR; semantic wins below a crossover | DeepSC (IEEE TSP 2021), Farsad 2018, DeepJSCC 2019: classical coded transmission near-perfect above ~8-10 dB, collapses below; learned semantic degrades gracefully | Same crossover on our data: digital traditional 0.95 at 10 dB, cliffs below; text ties it at 10 dB at 2418× fewer bits; semantic schemes own everything below |
| Learned semantic coding beats classical coding at low SNR | DeepSC (IEEE TSP 2021): outperforms Huffman/Turbo-coded baselines, ~8× BLEU at low SNR, degrades gracefully instead of a cliff | Same shape reproduced: our codec has **no digital cliff** and beats the 1.7-Mbit waveform at every SNR below 5 dB (e.g. 0.16 vs 0.05 at 0 dB) with 733× fewer bits; uncoded text collapses below ~3 dB |
| Send speech semantics, not speech | DeepSC-SR (2021): transmits text-related features of speech, "much less than the source speech data" | Meaning-as-text: **2,418× fewer bits** than the waveform; our quantized codec **~733× fewer** |
| Learned coding shines under fading | DeepSC (IEEE TSP 2021) evaluates on Rayleigh fading and reports the same advantage as AWGN, larger at low SNR | Same direction: under Rayleigh both robust semantic schemes crush the waveform at -5 dB (rep-3 0.15, codec 0.11, vs 0.02) — and rep-3 coded text wins at every fading SNR, the practical takeaway |

---

## One-paragraph summary

A plain speech-translation cascade destroys idiom meaning (48% literal-error
rate) and its raw signal drowns in channel noise (WER > 100% at -5 dB). We fix
the first with a gated idiom-resolution stage — errors drop to 13.5% with
provably zero harm on literal usages — and the second with semantic
communication: transmitting the extracted meaning instead of the signal
delivers the message perfectly in 2418× fewer bits on a good channel, and our
from-scratch trained semantic codec keeps meaning alive at SNRs where ordinary
digital transmission fails entirely.

# Speaker script — semantitrans (BTech project demo)

## 1. The problem (30 s)

English speech in, Hindi text out. A naive cascade of pretrained models fails
on **idioms**: "it's raining cats and dogs" becomes a literal Hindi sentence
about animals falling from the sky. The words survive; the **meaning** dies.
Everything in this project exists to transmit meaning, not words.

## 2. The pipeline (1 min)

1. **Whisper ASR** (fine-tuned by us on LibriSpeech) — speech → English text.
2. **Idiom resolution — our module.** A DistilBERT classifier **we trained**
   on MAGPIE (96.5% acc, F1 0.976) decides if an idiom is figurative or
   literal; only figurative ones get their meaning substituted from a gloss
   knowledge base *before* translation. The gate prevents "correcting"
   literal uses. [show `detector_cm.png`]
3. **NLLB-200** — idiom-resolved English → Hindi (+ our LoRA adapter).

Live demo, "Translate" tab: say an idiom, toggle idiom mode **off** vs
**substitute** — literal nonsense vs correct meaning.

Gold-reference results (250 annotated items): baseline renders **48%** of
idioms literally; substitution cuts that to **13.5%** (LTE metric). On
literal-usage sentences all modes score identically — the detector gate
causes zero over-correction. [full numbers: `KEY_RESULTS.md`]

## 3. Semantic communication (1 min)

Classical (Shannon) communication reproduces the sender's **bits**; success =
bit error rate. Expensive and brittle — flip a few bits and "10 AM" becomes
"10 PM". **Semantic communication** transmits the **meaning** in the fewest
possible bits; success = did the meaning survive. Our stage-2 output *is* the
extracted meaning — so we transmit that instead of the signal.

Motivation plot [`wer_snr.png`]: WER of the raw signal climbs from 10% (clean)
past 100% at -5 dB SNR. The signal is fragile; the meaning doesn't have to be.

## 4. The experiment (2 min) [show `semcom_snr.png`]

Same utterance, same channel (AWGN or Rayleigh fading), four transmission
schemes:

| Scheme | What crosses the channel | Bits/msg |
|---|---|---|
| Traditional | waveform as digital PCM bits; receiver runs ASR+MT | ~1,692,000 |
| Semantic (text bits) | idiom-resolved English as UTF-8 over BPSK | ~700 (**2418× fewer**) |
| Semantic (text, rep-3 coded) | same + rate-1/3 repetition code | ~2,100 (806× fewer) |
| Semantic (our codec) | learned symbols, 8-bit quantized | ~2,300 (**733× fewer**) |

The codec is a transformer encoder/decoder **trained by us from scratch**
(DeepSC paradigm, Xie et al. 2021) with the noisy channel *inside* the
training loop, at random SNRs.

Results — the literature-standard crossover, reproduced on our data
[`literature_comparison.png`]:
- Good channel (10 dB): classical digital is near-perfect (0.95) — and text
  bits **match it** (1.00) at 1/2418 of the bandwidth
  [`semcom_efficiency.png`: ~2,500× more meaning per kilobit at equal quality].
- Below 10 dB classical cliffs FIRST: its 1.7-Mbit message collects ~10,000
  bit errors at 5 dB where the 700-bit text message collects ~4. Uncoded text
  cliffs below ~3 dB. **Rep-3 coded text holds a perfect score down to 5 dB
  and dominates the mid-range** — the practical sweet spot at 806×
  compression. The **codec degrades gracefully** — the only scheme alive at
  -5 dB. Graceful degradation is the signature result of learned semcom.
- Rayleigh fading [`semcom_snr_rayleigh.png`]: deep fades hurt every digital
  scheme even at 10 dB; **rep-3 wins at every single SNR** — redundancy is
  what survives a fade.
- No idioms, gold refs [`semcom_snr_gold_literal.png`]: on literal-only
  sentences traditional TIES text at 10 dB (0.42 vs 0.43) — the gap on the
  full test set is idiom mistranslation, not channel noise.
- Semantic noise [`semcom_semnoise.png`]: corrupt words at the *sender* and
  the codec does worse than plain text — it fights channel noise, not meaning
  corruption. Honest ablation; meaning errors must be fixed at the source.

Live demo, "Noisy channel" tab: transmit at **10 dB** (text wins perfectly),
drag to **-2 dB**, transmit again (text garbles, codec survives). Then flip
AWGN → Rayleigh at 10 dB and watch text lose its perfect score. Two clicks =
the whole argument.

## 5. What is ours vs cited (30 s)

- **Trained by us:** idiom detector, Whisper fine-tune, LoRA adapter,
  semantic codec (from scratch — code and weights).
- **Written by us:** pipeline, channel simulation, all training/eval code, UI.
- **Cited backbones:** Whisper (OpenAI), NLLB-200 (Meta), DistilBERT,
  DeepSC as the architecture paradigm. Standard practice at every level.

## 6. If asked (honest caveats)

- Traditional is digital but uncoded PCM; with LDPC/turbo coding it would
  hold to ~5 dB before its cliff. rep-3 is likewise a simple code. The
  crossover shape is the standard published result either way.
- Robustness curves score each scheme against its own clean-channel output
  (the literature-standard view); absolute accuracy vs gold Hindi references:
  `semcom_snr_gold.png` — gold caps everything at the translator's quality
  (~0.35), so don't compare those numbers across papers.
- Codec symbols are 8-bit quantized (realistic digital transmission);
  quantization cost no measurable quality vs float32.
- Channels: AWGN + quasi-static flat Rayleigh (perfect CSI); no Doppler or
  frequency selectivity.
- The live demo's traditional scheme adds noise to the analog waveform
  (audibly intuitive); the plots use digital PCM bits over BPSK
  (literature-standard). Same story, different demo affordance.

## Demo checklist

1. `.venv/bin/python app.py` → http://127.0.0.1:7860
2. Translate tab: idiom sentence, mode off vs substitute.
3. Noisy channel tab: 10 dB → transmit; -2 dB → transmit.
4. Plots open: `semcom_snr.png`, `wer_snr.png`, `detector_cm.png`.

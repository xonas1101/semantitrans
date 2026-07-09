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

Same utterance, same AWGN channel, three transmission schemes:

| Scheme | What crosses the channel | Bits/msg |
|---|---|---|
| Traditional | raw waveform; receiver runs ASR+MT | ~1,692,000 |
| Semantic (text bits) | idiom-resolved English as UTF-8 over BPSK | ~700 (**2418× fewer**) |
| Semantic (text, rep-3 coded) | same + rate-1/3 repetition code | ~2,100 (806× fewer) |
| Semantic (our codec) | learned symbols, 8-bit quantized | ~2,300 (**733× fewer**) |

The codec is a transformer encoder/decoder **trained by us from scratch**
(DeepSC paradigm, Xie et al. 2021) with the noisy channel *inside* the
training loop, at random SNRs.

Results:
- Good channel (10 dB): text bits deliver the meaning **perfectly** at 1/2418
  of the bandwidth.
- Bad channel (≤ 2 dB): text bits fall off the "digital cliff" (bit errors
  shred UTF-8: chrF 0.04 at 0 dB). A repetition code delays the cliff a few
  dB but still collapses. The **codec degrades gracefully** — it beats even
  coded text at -2 to -5 dB and roughly matches the full waveform while
  sending 733× fewer bits. Graceful degradation is the signature result of
  learned semantic communication.

Live demo, "Noisy channel" tab: transmit at **10 dB** (text wins perfectly),
drag to **-2 dB**, transmit again (text garbles, codec survives). Two clicks =
the whole argument.

## 5. What is ours vs cited (30 s)

- **Trained by us:** idiom detector, Whisper fine-tune, LoRA adapter,
  semantic codec (from scratch — code and weights).
- **Written by us:** pipeline, channel simulation, all training/eval code, UI.
- **Cited backbones:** Whisper (OpenAI), NLLB-200 (Meta), DistilBERT,
  DeepSC as the architecture paradigm. Standard practice at every level.

## 6. If asked (honest caveats)

- We DO include a coded baseline (rep-3 majority vote); modern LDPC/turbo
  codes would push the cliff further left at similar rate — the shape holds.
- Robustness curves score each scheme against its own clean-channel output;
  absolute accuracy vs gold Hindi references: `semcom_snr_gold.png`.
- Codec symbols are 8-bit quantized (realistic digital transmission);
  quantization cost no measurable quality vs float32.
- Channel is AWGN; fading channels (Rayleigh) are the next step.

## Demo checklist

1. `.venv/bin/python app.py` → http://127.0.0.1:7860
2. Translate tab: idiom sentence, mode off vs substitute.
3. Noisy channel tab: 10 dB → transmit; -2 dB → transmit.
4. Plots open: `semcom_snr.png`, `wer_snr.png`, `detector_cm.png`.

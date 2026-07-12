"""Checkpoint #4 — web UI: drop English audio, get Hindi text.

A thin Gradio front-end over the existing Pipeline. Upload or record English
audio, pick the idiom mode, and see every stage: transcript, literalized
English, detected idioms, and the Hindi output.

A second tab simulates semantic communication over a noisy wireless channel
(AWGN or Rayleigh fading): pick an SNR and see the same utterance arrive four
ways — raw waveform, meaning-as-text-bits (BPSK, uncoded and rep-3 coded), and
through our learned semantic codec — with the bit cost of each. A semantic-noise
slider corrupts the sender-side meaning independently of the channel.

Usage:
  python app.py                 # http://127.0.0.1:7860
  python app.py --lora          # use the LoRA adapter for the whole session
  python app.py --share         # public Gradio link
"""

from __future__ import annotations

import argparse

import config
from semantitrans.idiom_resolver import MODE_APPEND, MODE_OFF, MODE_SUBSTITUTE
from semantitrans.pipeline import Pipeline


def build_ui(pipe: Pipeline):
    import gradio as gr

    codec = None
    codec_dir = config.ROOT_DIR / "models" / "semcodec"
    if codec_dir.exists():
        from semantitrans.semcodec import SemCodec

        codec = SemCodec.load(codec_dir)

    def run_channel(audio_path, snr_db, channel, sem_p):
        import numpy as np
        import librosa
        import soundfile as sf
        import tempfile
        from pathlib import Path

        from noise_eval import add_awgn
        from semcom_eval import (bpsk_ber, fading_snr_db, rayleigh_ber,
                                 semantic_noise, transmit_text)

        if not audio_path:
            return "(no audio given)", "", "", "", "", "", ""
        res = pipe.run(audio_path)
        rng = np.random.default_rng()
        # semantic noise corrupts the sender-side meaning (all semantic schemes)
        sem_msg = semantic_noise(res.intermediate, sem_p, rng)
        # one fade for the whole message, shared by all schemes
        eff = snr_db if channel == "AWGN" else fading_snr_db(snr_db, rng)

        # scheme 1: traditional — waveform through the channel, ASR + translate
        x, _ = librosa.load(audio_path, sr=16000, mono=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, add_awgn(x, eff), 16000)
            tmp = tf.name
        try:
            hi_trad = pipe.translator.translate(pipe.asr.transcribe(tmp))
        finally:
            Path(tmp).unlink(missing_ok=True)

        # scheme 2: semantic — meaning as UTF-8 bits over BPSK (uncoded)
        recv_text = transmit_text(sem_msg, eff, rng)
        hi_text = pipe.translator.translate(recv_text)

        # scheme 3: same text, rep-3 coded (majority vote)
        recv_coded = transmit_text(sem_msg, eff, rng, rep3=True)
        hi_coded = pipe.translator.translate(recv_coded)

        # scheme 4: our learned semantic codec
        recv_codec, hi_codec = "(codec not trained — run train_semcodec.py)", ""
        bits_codec = ""
        if codec:
            recv_codec = codec.reconstruct(sem_msg, eff)
            hi_codec = pipe.translator.translate(recv_codec)
            bits_codec = f" · codec {codec.bits_per_message(sem_msg):,} bits"
        ber_fn = bpsk_ber if channel == "AWGN" else rayleigh_ber
        n_text = len(sem_msg.encode("utf-8")) * 8
        bits = (f"**Bits/message:** waveform {len(x) * 16:,} · "
                f"text {n_text:,} · rep-3 {3 * n_text:,}{bits_codec} · "
                f"{channel} BER at {snr_db:g} dB = {ber_fn(snr_db):.2e}")
        return (sem_msg, hi_trad, recv_text, hi_text,
                f"{recv_coded}\n→ {hi_coded}", f"{recv_codec}\n→ {hi_codec}", bits)

    def run_audio(audio_path, mode):
        if not audio_path:
            return "", "", "", "(no audio given)", ""
        pipe.idiom_mode = mode  # switch mode without reloading models
        res = pipe.run(audio_path)
        idioms = "\n".join(f"• {d.surface!r} → {d.literal!r}" for d in res.detections) or "(none detected)"
        t = res.timings
        timing = f"asr {t['asr']:.1f}s · resolve {t['resolve']:.2f}s · translate {t['translate']:.1f}s"
        return res.hindi, res.english, res.intermediate, idioms, timing

    with gr.Blocks(title="semantitrans — EN speech → HI text") as demo:
        gr.Markdown(
            "# semantitrans\nEnglish **audio** → Hindi **text**, with idiom-aware "
            "resolution and semantic communication over a noisy channel."
        )
        with gr.Tab("Translate"):
            with gr.Row():
                with gr.Column():
                    audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="English audio")
                    mode = gr.Radio(
                        [MODE_SUBSTITUTE, MODE_APPEND, MODE_OFF],
                        value=MODE_SUBSTITUTE,
                        label="Idiom mode",
                        info="substitute/append = idiom-aware · off = plain baseline cascade",
                    )
                    btn = gr.Button("Translate", variant="primary")
                with gr.Column():
                    hindi = gr.Textbox(label="Hindi output", lines=3)
                    english = gr.Textbox(label="Stage 1 — English transcript", lines=2)
                    literal = gr.Textbox(label="Stage 2 — literalized English (fed to translator)", lines=2)
                    idioms = gr.Textbox(label="Idioms resolved", lines=3)
                    timing = gr.Markdown()
            btn.click(run_audio, [audio, mode], [hindi, english, literal, idioms, timing])

        with gr.Tab("Noisy channel"):
            gr.Markdown(
                "Semantic communication demo: the same utterance is sent over a "
                "noisy channel four ways — the raw **waveform**, the extracted "
                "**meaning as text bits** (BPSK, uncoded and rep-3 coded), and "
                "through **our learned semantic codec**. Lower the SNR, switch to "
                "Rayleigh fading, or add semantic noise and watch which survives."
            )
            with gr.Row():
                with gr.Column():
                    ch_audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="English audio")
                    snr = gr.Slider(-6, 15, value=5, step=1, label="Channel SNR (dB)")
                    channel = gr.Radio(["AWGN", "Rayleigh fading"], value="AWGN",
                                       label="Channel model",
                                       info="Rayleigh = flat fading, perfect CSI")
                    sem_p = gr.Slider(0, 0.5, value=0, step=0.05,
                                      label="Semantic noise (word corruption prob)",
                                      info="corrupts the sender-side meaning before transmission")
                    ch_btn = gr.Button("Transmit", variant="primary")
                    bits_md = gr.Markdown()
                with gr.Column():
                    sem_msg = gr.Textbox(label="Semantic message sent (idiom-resolved English)", lines=2)
                    hi_trad = gr.Textbox(label="1) Traditional — waveform through channel → Hindi", lines=2)
                    recv_text = gr.Textbox(label="2) Semantic text bits — received (corrupted) English", lines=2)
                    hi_text = gr.Textbox(label="2) → Hindi", lines=2)
                    coded_out = gr.Textbox(label="3) Rep-3 coded text — received English → Hindi", lines=3)
                    codec_out = gr.Textbox(label="4) Our learned codec — received English → Hindi", lines=3)
            ch_btn.click(run_channel, [ch_audio, snr, channel, sem_p],
                         [sem_msg, hi_trad, recv_text, hi_text, coded_out, codec_out, bits_md])

    return demo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", action="store_true", help="apply LoRA adapter if present")
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    config.configure_logging()
    pipe = Pipeline(idiom_mode=MODE_SUBSTITUTE, use_lora=args.lora)
    demo = build_ui(pipe)
    demo.launch(server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

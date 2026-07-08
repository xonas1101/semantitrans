"""Checkpoint #4 — web UI: drop English audio, get Hindi text.

A thin Gradio front-end over the existing Pipeline. Upload or record English
audio, pick the idiom mode, and see every stage: transcript, literalized
English, detected idioms, and the Hindi output.

A second tab simulates semantic communication over a noisy wireless channel:
pick an SNR and see the same utterance arrive three ways — raw waveform,
meaning-as-text-bits (BPSK), and through our learned semantic codec — with
the bit cost of each.

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

    def run_channel(audio_path, snr_db):
        import numpy as np
        import librosa
        import soundfile as sf
        import tempfile
        from pathlib import Path

        from noise_eval import add_awgn
        from semcom_eval import bpsk_ber, transmit_text

        if not audio_path:
            return "(no audio given)", "", "", "", "", ""
        res = pipe.run(audio_path)
        sem_msg = res.intermediate

        # scheme 1: traditional — waveform through AWGN, ASR + translate at receiver
        x, _ = librosa.load(audio_path, sr=16000, mono=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            sf.write(tf.name, add_awgn(x, snr_db), 16000)
            tmp = tf.name
        try:
            hi_trad = pipe.translator.translate(pipe.asr.transcribe(tmp))
        finally:
            Path(tmp).unlink(missing_ok=True)

        # scheme 2: semantic — meaning as UTF-8 bits over BPSK
        rng = np.random.default_rng()
        recv_text = transmit_text(sem_msg, snr_db, rng)
        hi_text = pipe.translator.translate(recv_text)

        # scheme 3: our learned semantic codec
        recv_codec, hi_codec = "(codec not trained — run train_semcodec.py)", ""
        bits_codec = ""
        if codec:
            recv_codec = codec.reconstruct(sem_msg, snr_db)
            hi_codec = pipe.translator.translate(recv_codec)
            bits_codec = f" · codec {codec.bits_per_message(sem_msg):,} bits"
        bits = (f"**Bits/message:** waveform {len(x) * 16:,} · "
                f"text {len(sem_msg.encode('utf-8')) * 8:,}{bits_codec} · "
                f"BER at {snr_db:g} dB = {bpsk_ber(snr_db):.2e}")
        return sem_msg, hi_trad, recv_text, hi_text, f"{recv_codec}\n→ {hi_codec}", bits

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
                "Semantic communication demo: the same utterance is sent over an "
                "AWGN channel three ways — the raw **waveform**, the extracted "
                "**meaning as text bits** (BPSK), and through **our learned "
                "semantic codec**. Lower the SNR and watch which one survives."
            )
            with gr.Row():
                with gr.Column():
                    ch_audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="English audio")
                    snr = gr.Slider(-6, 15, value=5, step=1, label="Channel SNR (dB)")
                    ch_btn = gr.Button("Transmit", variant="primary")
                    bits_md = gr.Markdown()
                with gr.Column():
                    sem_msg = gr.Textbox(label="Semantic message sent (idiom-resolved English)", lines=2)
                    hi_trad = gr.Textbox(label="1) Traditional — waveform through channel → Hindi", lines=2)
                    recv_text = gr.Textbox(label="2) Semantic text bits — received (corrupted) English", lines=2)
                    hi_text = gr.Textbox(label="2) → Hindi", lines=2)
                    codec_out = gr.Textbox(label="3) Our learned codec — received English → Hindi", lines=3)
            ch_btn.click(run_channel, [ch_audio, snr],
                         [sem_msg, hi_trad, recv_text, hi_text, codec_out, bits_md])

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

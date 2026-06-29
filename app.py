"""Checkpoint #4 — web UI: drop English audio, get Hindi text.

A thin Gradio front-end over the existing Pipeline. Upload or record English
audio, pick the idiom mode, and see every stage: transcript, literalized
English, detected idioms, and the Hindi output.

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
            "resolution. Upload or record English speech below."
        )
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

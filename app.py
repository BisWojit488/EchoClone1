"""Shareable Gradio web interface for the SV2TTS voice-cloning pipeline."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

_requested_device = os.getenv("DEVICE", "auto").lower()
if _requested_device == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

ROOT = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(ROOT / "saved_models")))
MODEL_DIR = MODELS_DIR / "default"
SAMPLE_RATE = Synthesizer.sample_rate
_generation_lock = threading.Lock()
_output_files = []


def _device() -> str:
    if _requested_device in {"cpu", "cuda"}:
        if _requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda was requested, but CUDA is not available.")
        return _requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


class VoiceCloner:
    """Load the existing encoder, synthesizer, and vocoder once."""

    def __init__(self) -> None:
        device = _device()
        ensure_default_models(MODELS_DIR)
        encoder.load_model(MODEL_DIR / "encoder.pt", device=device)
        self.synthesizer = Synthesizer(MODEL_DIR / "synthesizer.pt")
        vocoder.load_model(MODEL_DIR / "vocoder.pt")
        self.device = device

    @staticmethod
    def _reference_embedding(audio_path: str) -> np.ndarray:
        wav = encoder.preprocess_wav(Path(audio_path))
        if wav is None or len(wav) == 0:
            raise ValueError("The reference audio is empty or could not be decoded.")
        return encoder.embed_utterance(wav)

    def generate(self, reference_audio: str, text: str, seed: Optional[float] = None) -> str:
        if not reference_audio:
            raise ValueError("Record or upload reference audio first.")
        if not text or not text.strip():
            raise ValueError("Enter text to synthesize.")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Enter at least one non-empty line of text.")

        with _generation_lock:
            try:
                if seed is not None:
                    torch.manual_seed(int(seed))
                embed = self._reference_embedding(reference_audio)
                specs = self.synthesizer.synthesize_spectrograms(lines, [embed] * len(lines))
                breaks = [spec.shape[1] for spec in specs]
                spec = np.concatenate(specs, axis=1)
                wav = vocoder.infer_waveform(spec)
                ends = np.cumsum(np.asarray(breaks) * Synthesizer.hparams.hop_size)
                starts = np.concatenate(([0], ends[:-1]))
                chunks = [wav[int(start):int(end)] for start, end in zip(starts, ends)]
                pause = np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.float32)
                wav = np.concatenate([part for chunk in chunks for part in (chunk, pause)])
                # Keep the vocoder's native sample rate. Encoder preprocessing resamples to
                # 16 kHz, which is correct for embeddings but would distort generated audio when
                # written at the synthesizer's sample rate.
                wav = np.asarray(wav, dtype=np.float32)
                peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
                if peak:
                    wav = wav / peak * 0.97
                handle, output_path = tempfile.mkstemp(prefix="voice_clone_", suffix=".wav")
                os.close(handle)
                sf.write(output_path, wav.astype(np.float32), SAMPLE_RATE)
                _output_files.append(output_path)
                while len(_output_files) > 3:
                    old = _output_files.pop(0)
                    try:
                        os.unlink(old)
                    except OSError:
                        pass
                return output_path
            except ValueError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Speech generation failed: {exc}") from exc


_cloner: Optional[VoiceCloner] = None


def get_cloner() -> VoiceCloner:
    global _cloner
    if _cloner is None:
        _cloner = VoiceCloner()
    return _cloner


def generate_speech(reference_audio: str, text: str, seed: Optional[float] = None) -> str:
    return get_cloner().generate(reference_audio, text, seed)


def clear_inputs():
    return None, "", None, None


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Real-Time Voice Cloning") as demo:
        gr.Markdown(
            "# Real-Time Voice Cloning\n"
            "Record or upload 5–10 seconds of clear speech, enter text, and generate audio "
            "in that voice. Please only use voices you have permission to clone."
        )
        with gr.Row():
            with gr.Column():
                reference = gr.Audio(source="microphone", type="filepath", label="Record reference voice")
                upload = gr.File(file_types=[".wav", ".mp3", ".flac", ".m4a"], type="file",
                                 label="Or upload an audio file")
                text = gr.Textbox(
                    label="Text to synthesize", lines=8,
                    placeholder="Enter your text here. Use a new line for a short pause.",
                )
                seed = gr.Number(label="Optional random seed", precision=0)
                with gr.Row():
                    generate = gr.Button("Generate speech", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column():
                output = gr.Audio(label="Generated speech", type="filepath", format="wav")
                status = gr.Markdown("Ready.")

        def run_generation(mic_audio, uploaded_audio, prompt, random_seed, progress=gr.Progress()):
            progress(0, desc="Processing audio...")
            audio_path = mic_audio or uploaded_audio
            if hasattr(audio_path, "name"):
                audio_path = audio_path.name
            result = generate_speech(audio_path, prompt, random_seed)
            progress(1, desc="Done")
            return result, "Generated successfully."

        generate.click(run_generation, [reference, upload, text, seed], [output, status])
        clear.click(clear_inputs, outputs=[reference, text, seed, output])
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(concurrency_count=1).launch(
        server_name=os.getenv("HOST", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
        share=os.getenv("SHARE", "false").lower() in {"1", "true", "yes"},
    )

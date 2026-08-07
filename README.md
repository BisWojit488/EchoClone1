# EchoClone

EchoClone is a voice-cloning application that converts a short reference recording into synthesized speech. It contains the encoder, synthesizer, vocoder, desktop toolbox, and browser interface needed for the complete workflow.

Only use reference voices that you own or have permission to clone.

## Features

- Record a reference voice with a microphone.
- Use WAV, MP3, FLAC, or M4A reference audio.
- Create a speaker embedding from the reference recording.
- Convert custom text into speech in the reference voice.
- Play and export generated WAV audio.
- Support CPU and compatible NVIDIA GPU inference.

## Pipeline

1. The speaker encoder extracts a voice embedding from reference audio.
2. The synthesizer converts text and the embedding into a mel-spectrogram.
3. The vocoder converts the spectrogram into the final waveform.

## Project structure

- `encoder/` — speaker embedding model and audio preprocessing.
- `synthesizer/` — text-to-spectrogram model.
- `vocoder/` — waveform generation model.
- `toolbox/` — desktop application interface.
- `app.py` — browser application interface.
- `samples/` — example audio files.
- `tests/` — project smoke and application tests.

## Model files

Pretrained model files are downloaded when the application first needs them. They are intentionally excluded from version control because of their size.

## License

The project includes upstream MIT-licensed components. Their license and copyright notices are retained in `LICENSE`, `synthesizer/LICENSE.txt`, and `vocoder/LICENSE.txt`.

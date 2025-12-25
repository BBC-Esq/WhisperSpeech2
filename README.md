# 🎙️ WhisperSpeech2

An Open Source text-to-speech system built by inverting Whisper. This is a fork of [WhisperSpeech](https://github.com/collabora/WhisperSpeech) optimized for inference that introduces ⚡"cuda graphs"⚡ for faster inference.

## 🚀 Installation

```bash
pip install whisperspeech2
```

> [PyTorch](https://pytorch.org/get-started/locally/)<p>
> [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) libraries if using an Nvidia GPU (tested with CUDA 12.8).

## ✨ Available Models

You can mix and match the models for different quality and compute requirements.  See the [WhisperSpeech Hugging Face repository](https://huggingface.co/WhisperSpeech/WhisperSpeech).

### S2A Models (Semantic to Acoustic)

| Model | Reference |
|-------|-----------|
| Tiny | `WhisperSpeech/WhisperSpeech:s2a-q4-tiny-en+pl.model` |
| Base | `WhisperSpeech/WhisperSpeech:s2a-q4-base-en+pl.model` |
| Small | `WhisperSpeech/WhisperSpeech:s2a-q4-small-en+pl.model` |
| HQ Fast | `WhisperSpeech/WhisperSpeech:s2a-q4-hq-fast-en+pl.model` |
| v1.1 Small | `WhisperSpeech/WhisperSpeech:s2a-v1.1-small-en+pl.model` |

### T2S Models (Text to Semantic)

| Model | Reference |
|-------|-----------|
| Tiny | `WhisperSpeech/WhisperSpeech:t2s-tiny-en+pl.model` |
| Base | `WhisperSpeech/WhisperSpeech:t2s-base-en+pl.model` |
| Small | `WhisperSpeech/WhisperSpeech:t2s-small-en+pl.model` |
| Fast Small | `WhisperSpeech/WhisperSpeech:t2s-fast-small-en+pl.model` |
| Fast Medium | `WhisperSpeech/WhisperSpeech:t2s-fast-medium-en+pl+yt.model` |
| HQ Fast | `WhisperSpeech/WhisperSpeech:t2s-hq-fast-en+pl.model` |

## Benchmark (no cuda graph)

<img width="1871" height="964" alt="image" src="https://github.com/user-attachments/assets/ffba19ae-360c-4bbe-9f25-da8a95fad9ec" />

## Benchmark (with cuda graph)
> People with Nvidia GPUs can set the "use_cuda_graph" parameter to "true" for fastger processing and less VRAM usage.

<img width="1872" height="955" alt="image" src="https://github.com/user-attachments/assets/43352056-b452-4570-a976-a43188184765" />

## Examples

See the `examples/` directory for more usage examples including GUI applications and streaming playback.

## Thanks/Shout Outs

Thanks to [Jakub](https://github.com/jpc) for the inspiration.<p>
The fine folks at the [Dia2](https://github.com/nari-labs/dia2) for forcing me to learn about cuda graph.  Go check theirs out too.<p>

# BIG FUCK YOU

And finally a big "fuck you" to Microsoft for installing shit on my computer that I don't need and/or want for years.<p>Your pathetic VibeVoice project just got pwned!








# WhisperSpeech2

An Open Source text-to-speech system built by inverting Whisper. This is a fork of [WhisperSpeech](https://github.com/collabora/WhisperSpeech) optimized for inference.  The creators of the original project abandoned it hence this fork.

## Installation

```bash
pip install whisperspeech2
```

**Note:** You must also have PyTorch installed. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for installation instructions.

## Quick Start

```python
from whisperspeech2.pipeline import Pipeline

# Initialize the pipeline
pipe = Pipeline(
    s2a_ref='WhisperSpeech/WhisperSpeech:s2a-q4-tiny-en+pl.model',
    t2s_ref='WhisperSpeech/WhisperSpeech:t2s-tiny-en+pl.model'
)

# Generate audio and save to file
pipe.generate_to_file('output.wav', "Hello, world!")

# Or get the audio tensor directly
audio = pipe.generate("Hello, world!")
```

## Available Models

For more details about each model, visit the [WhisperSpeech Hugging Face repository](https://huggingface.co/WhisperSpeech/WhisperSpeech).

### S2A Models (Semantic to Acoustic)

| Model | Reference |
|-------|-----------|
| Tiny (Q4) | `WhisperSpeech/WhisperSpeech:s2a-q4-tiny-en+pl.model` |
| Base (Q4) | `WhisperSpeech/WhisperSpeech:s2a-q4-base-en+pl.model` |
| Small (Q4) | `WhisperSpeech/WhisperSpeech:s2a-q4-small-en+pl.model` |
| HQ Fast (Q4) | `WhisperSpeech/WhisperSpeech:s2a-q4-hq-fast-en+pl.model` |
| v1.1 Small | `WhisperSpeech/WhisperSpeech:s2a-v1.1-small-en+pl.model` |
| v1.95 Small Fast | `WhisperSpeech/WhisperSpeech:s2a-v1.95-small-fast-en.model` |

### T2S Models (Text to Semantic)

| Model | Reference |
|-------|-----------|
| Tiny | `WhisperSpeech/WhisperSpeech:t2s-tiny-en+pl.model` |
| Base | `WhisperSpeech/WhisperSpeech:t2s-base-en+pl.model` |
| Small | `WhisperSpeech/WhisperSpeech:t2s-small-en+pl.model` |
| Fast Small | `WhisperSpeech/WhisperSpeech:t2s-fast-small-en+pl.model` |
| Fast Medium | `WhisperSpeech/WhisperSpeech:t2s-fast-medium-en+pl+yt.model` |
| HQ Fast | `WhisperSpeech/WhisperSpeech:t2s-hq-fast-en+pl.model` |
| v1.1 Small | `WhisperSpeech/WhisperSpeech:t2s-v1.1-small-en+pl.model` |

## Model Recommendations

| Use Case | S2A Model | T2S Model | VRAM | Speed |
|----------|-----------|-----------|------|-------|
| **Lowest Resources** | s2a-q4-tiny | t2s-tiny | ~450 MB | ~16s |
| **Best Speed** | s2a-v1.95-small-fast | t2s-tiny | ~1.7 GB | ~15s |
| **Balanced** | s2a-q4-hq-fast | t2s-tiny | ~1.7 GB | ~15s |
| **Higher Quality** | s2a-q4-hq-fast | t2s-hq-fast | ~2.1 GB | ~16s |

**Avoid:** Combinations using `s2a-q4-small` or `s2a-v1.1-small` with `t2s-fast-medium` result in high VRAM (~4GB) and slow processing (~42s).

<img width="3680" height="1800" alt="image" src="https://github.com/user-attachments/assets/2efc192c-2d1a-4f6d-a5fc-91c3783c161e" />

## Speaker Embedding (Optional)

To use custom speaker embeddings, install the optional dependency:

```bash
pip install whisperspeech2[speaker]
```

Then pass an audio file path to clone a voice:

```python
pipe.generate_to_file('output.wav', "Hello!", speaker='reference.wav')
```

## Examples

See the `examples/` directory for more usage examples including GUI applications and streaming playback.

## License

MIT License
```



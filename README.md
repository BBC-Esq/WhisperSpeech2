# WhisperSpeech2

An Open Source text-to-speech system built by inverting Whisper. This is a fork of [WhisperSpeech](https://github.com/collabora/WhisperSpeech) optimized for inference.  The creators of the original project abandoned it, hence this fork.

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

## Benchmark (no cuda graph)

<img width="3680" height="1800" alt="image" src="https://github.com/user-attachments/assets/2efc192c-2d1a-4f6d-a5fc-91c3783c161e" />

## Benchmark (with cuda graph)
> People with Nvidia GPUs can set the "use_cuda_graph" parameter to "true" and it'll offer the following speedups:

<img width="2424" height="1170" alt="image" src="https://github.com/user-attachments/assets/ba8d404c-1118-47d2-ab6f-17014a651280" />

<details>

<summary>Benchmark Data</summary>

| S2A Model | T2S Model | Original Time (s) | CUDA Graph Time (s) | Speedup | Original VRAM (MB) | CUDA Graph VRAM (MB) | VRAM Reduction |
|-----------|-----------|------------------:|--------------------:|--------:|-------------------:|---------------------:|---------------:|
| s2a-q4-base | t2s-base | 20.66 | 4.43 | **4.66x** | 1056.75 | 767.44 | 27.4% |
| s2a-q4-base | t2s-fast-medium | 24.23 | 4.89 | **4.96x** | 3950.00 | 2481.34 | 37.2% |
| s2a-q4-base | t2s-fast-small | 21.46 | 4.40 | **4.88x** | 1768.00 | 1111.72 | 37.1% |
| s2a-q4-base | t2s-hq-fast | 21.49 | 4.44 | **4.84x** | 1768.00 | 1112.00 | 37.1% |
| s2a-q4-base | t2s-small | 26.74 | 5.04 | **5.31x** | 2100.34 | 1312.12 | 37.5% |
| s2a-q4-base | t2s-tiny | 21.60 | 3.03 | **7.13x** | 746.34 | 525.00 | 29.7% |
| s2a-q4-hq-fast | t2s-base | 19.23 | 3.80 | **5.06x** | 1792.00 | 1488.56 | 16.9% |
| s2a-q4-hq-fast | t2s-fast-medium | 18.77 | 3.20 | **5.87x** | 3814.00 | 2601.32 | 31.8% |
| s2a-q4-hq-fast | t2s-fast-small | 16.18 | 3.78 | **4.28x** | 2135.34 | 2062.00 | 3.4% |
| s2a-q4-hq-fast | t2s-hq-fast | 16.22 | 3.77 | **4.30x** | 2135.34 | 2070.53 | 3.0% |
| s2a-q4-hq-fast | t2s-small | 21.20 | 4.49 | **4.72x** | 2305.34 | 2244.03 | 2.7% |
| s2a-q4-hq-fast | t2s-tiny | 14.63 | 3.67 | **3.99x** | 1684.00 | 1312.00 | 22.1% |
| s2a-q4-small | t2s-base | 41.21 | 8.79 | **4.69x** | 2752.00 | 2494.31 | 9.4% |
| s2a-q4-small | t2s-fast-medium | 42.19 | 9.42 | **4.48x** | 4184.00 | 3165.09 | 24.4% |
| s2a-q4-small | t2s-fast-small | 39.19 | 9.26 | **4.23x** | 3136.34 | 2710.06 | 13.6% |
| s2a-q4-small | t2s-hq-fast | 39.15 | 9.33 | **4.20x** | 3082.00 | 2706.97 | 12.2% |
| s2a-q4-small | t2s-small | 44.42 | 10.15 | **4.38x** | 3259.56 | 3003.03 | 7.9% |
| s2a-q4-small | t2s-tiny | 37.20 | 7.92 | **4.70x** | 2652.00 | 2474.34 | 6.7% |
| s2a-q4-tiny | t2s-base | 17.04 | 2.85 | **5.98x** | 540.34 | 510.00 | 5.6% |
| s2a-q4-tiny | t2s-fast-medium | 15.95 | 3.65 | **4.37x** | 3938.38 | 2394.31 | 39.2% |
| s2a-q4-tiny | t2s-fast-small | 15.97 | 2.99 | **5.34x** | 1690.00 | 998.56 | 40.9% |
| s2a-q4-tiny | t2s-hq-fast | 15.79 | 2.99 | **5.28x** | 1727.72 | 1018.00 | 41.1% |
| s2a-q4-tiny | t2s-small | 20.86 | 3.37 | **6.19x** | 1950.00 | 1340.00 | 31.3% |
| s2a-q4-tiny | t2s-tiny | 16.14 | 2.17 | **7.44x** | 448.34 | 413.88 | 7.7% |
| s2a-v1.1-small | t2s-base | 37.16 | 9.40 | **3.95x** | 2234.25 | 1920.00 | 14.1% |
| s2a-v1.1-small | t2s-fast-medium | 42.37 | 9.31 | **4.55x** | 3824.00 | 3046.41 | 20.3% |
| s2a-v1.1-small | t2s-fast-small | 39.45 | 9.43 | **4.18x** | 2659.34 | 2376.47 | 10.6% |
| s2a-v1.1-small | t2s-hq-fast | 39.10 | 9.31 | **4.20x** | 2602.00 | 2354.91 | 9.5% |
| s2a-v1.1-small | t2s-small | 40.56 | 8.72 | **4.65x** | 2735.75 | 2695.66 | 1.5% |
| s2a-v1.1-small | t2s-tiny | 36.94 | 8.43 | **4.38x** | 2152.00 | 1561.06 | 27.5% |
| s2a-v1.95-small-fast | t2s-base | 16.02 | 3.94 | **4.07x** | 1792.00 | 1486.53 | 17.0% |
| s2a-v1.95-small-fast | t2s-fast-medium | 17.46 | 4.55 | **3.84x** | 3830.59 | 2608.17 | 31.9% |
| s2a-v1.95-small-fast | t2s-fast-small | 16.06 | 3.78 | **4.25x** | 2135.34 | 2062.56 | 3.4% |
| s2a-v1.95-small-fast | t2s-hq-fast | 16.26 | 3.80 | **4.28x** | 2114.00 | 2071.38 | 2.0% |
| s2a-v1.95-small-fast | t2s-small | 21.15 | 4.76 | **4.44x** | 2336.34 | 2242.00 | 4.0% |
| s2a-v1.95-small-fast | t2s-tiny | 15.04 | 3.63 | **4.14x** | 1684.00 | 1366.19 | 18.9% |

</details>
---

**Summary Statistics:**

| Metric | Min | Max | Average |
|--------|----:|----:|--------:|
| **Time Speedup** | 3.84x | 7.44x | **4.72x** |
| **VRAM Reduction** | 1.5% | 41.1% | **18.9%** |

**Note:** `t2s-v1.1-small-en+pl.model` was excluded as it's incompatible with CUDA graph.
> </details>


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







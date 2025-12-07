# WhisperSpeech2

An Open Source text-to-speech system built by inverting Whisper. This is a fork of [WhisperSpeech](https://github.com/collabora/WhisperSpeech) optimized for inference.

## Installation
```bash
pip install whisperspeech2
```

**Note:** You must also have PyTorch installed. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for installation instructions.

## Quick Start
```python
from whisperspeech2.pipeline import Pipeline

# Initialize the pipeline
pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')

# Generate audio and save to file
pipe.generate_to_file('output.wav', "Hello, world!")

# Or get the audio tensor directly
audio = pipe.generate("Hello, world!")
```

## Available Models

| Model | Reference |
|-------|-----------|
| Tiny | `collabora/whisperspeech:s2a-q4-tiny-en+pl.model` |
| Base | `collabora/whisperspeech:s2a-q4-base-en+pl.model` |
| Small | `collabora/whisperspeech:s2a-q4-small-en+pl.model` |

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

### 3. `LICENSE`
```
MIT License

Copyright (c) 2025 Blair Chintella

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
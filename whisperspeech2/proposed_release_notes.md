# WhisperSpeech2 v1.0.0

## Highlights

This release replaces `torchaudio` with [PyAV](https://github.com/PyAV-Org/PyAV) (`av`) as the primary audio backend, adds named speaker presets, updates SpeechBrain compatibility to `>=1.0`, and updates all example scripts to showcase the new speaker feature.

---

## What's Changed

### PyAV replaces torchaudio as the default audio backend

`torchaudio` has been removed from the core dependency list in favor of `av` (PyAV). Both audio saving (`a2wav.py`) and audio loading (`pipeline.py`) now follow a new fallback chain:

1. **PyAV** (preferred)
2. **soundfile**
3. **torchaudio** (save only — kept as a last-resort fallback)

A new standalone `_load_audio()` helper in `pipeline.py` handles all audio file reading, eliminating the direct `torchaudio` dependency from speaker embedding extraction.

### Named speaker presets

A `SPEAKERS` dictionary is now available with three built-in voices:

- **`"default"`** — a new default speaker embedding
- **`"classic"`** — the previous default speaker from v0.9.3
- **`"voice_b"`** — an additional preset voice

You can pass a speaker name as a string directly to `generate()`, `generate_atoks()`, `generate_to_file()`, and `generate_to_notebook()`:

```python
pipe.generate("Hello world", speaker="classic")
```

File paths and raw tensors continue to work as before.

### SpeechBrain >=1.0 compatibility

- The `speechbrain` extra now requires `>=1.0` (previously pinned to `<1.0`).
- The import path has been updated to try `speechbrain.inference.EncoderClassifier` first, falling back to the legacy `speechbrain.pretrained.EncoderClassifier` for older installations.

### T2S progress bar control

- `t2s.generate()` now accepts a `show_progress_bar` parameter to suppress progress output when not needed.

### Updated examples

- All example scripts updated to demonstrate the `speaker` parameter.
- Examples readme updated with speaker preset documentation.

---

## Dependency Changes

| Dependency | v0.9.3 | v1.0.0 |
|---|---|---|
| `torchaudio` | required | removed from core deps |
| `av` (PyAV) | — | required |
| `speechbrain` (extras) | `<1.0` | `>=1.0` |

---

## Upgrading

```bash
pip install --upgrade whisperspeech2
```

If you were relying on `torchaudio` specifically, it will still work as a fallback for saving audio but is no longer installed automatically. To keep it:

```bash
pip install torchaudio
```

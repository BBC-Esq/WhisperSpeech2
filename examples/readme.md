# Example Scripts

Contributions are welcome! Feel free to create an issue or pull request on GitHub.

## Installation

All scripts require:

1. A virtual environment with PyTorch installed ([pytorch.org](https://pytorch.org/get-started/locally/))
2. `pip install whisperspeech2`
3. `pip install sounddevice` (for audio playback scripts)

---

### `minimal.py`

Minimalistic script that takes hardcoded text input and outputs an audio file.

**Additional dependencies:** None

---

### `text_to_playback.py`

Processes a body of text directly into audio playback using the sounddevice library. Designed for minimal script length, but does not include queue management to reduce latency.

**Additional dependencies:** `sounddevice`

---

### `text_to_audio_playback.py`

Processes text one sentence at a time and adds them to a queue for playback. Designed for users who prefer a command-line approach but still want the efficiency of queued playback.

**Additional dependencies:** `sounddevice`

---

### `gui_text_to_audio_playback.py`

Provides a simple GUI where users can enter text to be converted to speech. Text is processed one sentence at a time for low latency.

**Additional dependencies:** `sounddevice`

---

### `gui_file_to_text_to_audio_playback.py`

Provides a graphical user interface allowing users to load a file. The text is then converted into speech, sentence by sentence using queue management in order to reduce latency.

**Additional dependencies:** `sounddevice`, `pypdf`, `python-docx`, `nltk`

---

## Feature Comparison

| Feature | gui_file_to_text_to_audio_playback.py | gui_text_to_audio_playback.py | minimal.py | text_to_audio_playback.py | text_to_playback.py |
|:-------:|:-------------------------------------:|:-----------------------------:|:----------:|:-------------------------:|:-------------------:|
| **GUI** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Input** | File | Text Entry | Predefined Text | Predefined Text | Predefined Text |
| **Output** | Audio Playback | Audio Playback | WAV File | Audio Playback | Audio Playback |
| **Queue Management** | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Load File** | ✅ | ❌ | ❌ | ❌ | ❌ |
'''
DESCRIPTION~

Processes a body of text directly into audio playback using the sounddevice library.

PLEASE NOTE~

If you need more granular control, such as being able to process sentences in one thread
(one sentence at a time) while simultaneously playing them in another thread (reducing latency),
consult the "text_to_audio_playback.py" example.

INSTALLATION INSTRUCTIONS~

(1) Create a virtual environment and activate it
(2) Install pytorch by going to the following website and running the appropriate command for your platform and setup:

https://pytorch.org/get-started/locally/

(3) pip install whisperspeech2
(4) pip install sounddevice
(5) python text_to_playback.py
'''

from whisperspeech2.pipeline import Pipeline
import sounddevice as sd
import numpy as np

# Uncomment the line for the model you want to use
# pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')
pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-base-en+pl.model')
# pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-small-en+pl.model')

text = """
This is some sample text. You would add text here that you want spoken and then only leave one of the above lines uncommented for the model you want to test.
"""

audio_tensor = pipe.generate(text)
audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
if len(audio_np.shape) == 1:
    audio_np = np.expand_dims(audio_np, axis=0)
else:
    audio_np = audio_np.T
sd.play(audio_np, samplerate=24000)
sd.wait()
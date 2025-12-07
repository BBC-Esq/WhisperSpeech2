'''
DESCRIPTION~

Minimalistic script that takes hardcoded text input and outputs an audio file.

INSTALLATION INSTRUCTIONS~

(1) Create a virtual environment and activate it
(2) Install pytorch by going to the following website and running the appropriate command for your platform and setup:

https://pytorch.org/get-started/locally/

(3) pip install whisperspeech2
(4) python minimal.py
'''

from whisperspeech2.pipeline import Pipeline

# Uncomment the line for the model you want to use
tts_pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')
# tts_pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-base-en+pl.model')
# tts_pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-small-en+pl.model')

save_path = 'output.wav'
tts_pipe.generate_to_file(save_path, "This is a test")
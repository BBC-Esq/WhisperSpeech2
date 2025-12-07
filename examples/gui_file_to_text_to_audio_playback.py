'''
DESCRIPTION~

Type text or load a file to be converted TTS, sentence by sentence, to quickstart playback.

INSTALLATION INSTRUCTIONS~

(1) Create a virtual environment and activate it
(2) Install pytorch by going to the following website and running the appropriate command for your platform and setup:

https://pytorch.org/get-started/locally/

(3) pip install whisperspeech2
(4) pip install sounddevice pypdf python-docx nltk
(5) python gui_file_to_text_to_audio_playback.py
'''

from tkinter import *
from tkinter import filedialog
import numpy as np
import re
import threading
import queue
from whisperspeech2.pipeline import Pipeline
import sounddevice as sd
from pypdf import PdfReader
from docx import Document
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

main_bg = "#1B2A2F"
widget_bg = "#303F46"
button_bg = "#263238"
label_bg = "#1E272C"
text_fg = "#FFFFFF"

# Uncomment the line for the model you want to use
# model_ref = 'collabora/whisperspeech:s2a-q4-tiny-en+pl.model'
model_ref = 'collabora/whisperspeech:s2a-q4-base-en+pl.model'
# model_ref = 'collabora/whisperspeech:s2a-q4-small-en+pl.model'

pipe = Pipeline(s2a_ref=model_ref)

audio_queue = queue.Queue()

class TextUtilities:
    def pdf_to_text(self, pdf_file_path):
        reader = PdfReader(pdf_file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ''
        return text

    def docx_to_text(self, docx_file_path):
        doc = Document(docx_file_path)
        return ' '.join([paragraph.text for paragraph in doc.paragraphs])

    def txt_to_text(self, txt_file_path):
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def clean_text(self, text):
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        text = text.replace('\n', ' ')
        sentences = sent_tokenize(text)
        cleaned_text = ' '.join(sentences)
        return cleaned_text

def process_text_to_audio(sentences, pipe):
    for sentence in sentences:
        if sentence:
            audio_tensor = pipe.generate(sentence)
            audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
            if len(audio_np.shape) == 1:
                audio_np = np.expand_dims(audio_np, axis=0)
            else:
                audio_np = audio_np.T
            audio_queue.put(audio_np)
    audio_queue.put(None)

def play_audio_from_queue(audio_queue):
    while True:
        audio_np = audio_queue.get()
        if audio_np is None:
            break
        try:
            sd.play(audio_np, samplerate=24000)
            sd.wait()
        except Exception as e:
            print(f"Error playing audio: {e}")

def start_processing():
    user_input = text_input.get("1.0", "end-1c")
    sentences = sent_tokenize(user_input)
    while not audio_queue.empty():
        audio_queue.get()
    processing_thread = threading.Thread(target=process_text_to_audio, args=(sentences, pipe))
    playback_thread = threading.Thread(target=play_audio_from_queue, args=(audio_queue,))
    processing_thread.start()
    playback_thread.start()

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        text = process_file(file_path)
        text_input.delete("1.0", END)
        text_input.insert("1.0", text)

def process_file(file_path):
    utilities = TextUtilities()
    text = ""
    if file_path.lower().endswith('.pdf'):
        text = utilities.pdf_to_text(file_path)
    elif file_path.lower().endswith('.docx'):
        text = utilities.docx_to_text(file_path)
    elif file_path.lower().endswith(('.txt', '.py', '.html', '.md')):
        text = utilities.txt_to_text(file_path)
    text = utilities.clean_text(text)
    return text

root = Tk()
root.title("Text to Speech")

root.attributes('-topmost', 1)

root.configure(bg=main_bg)

root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

text_input = Text(root, bg=widget_bg, fg=text_fg)
text_input.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

process_button = Button(root, text="Text to Speech", command=start_processing, bg=button_bg, fg=text_fg)
process_button.grid(row=1, column=0, sticky='ew', padx=10)

file_button = Button(root, text="Extract Text from File", command=select_file, bg=button_bg, fg=text_fg)
file_button.grid(row=2, column=0, sticky='ew', padx=10)

support_label = Label(root, text="Supports .pdf (with OCR already done on them), .docx, and .txt files.", bg=label_bg, fg=text_fg)
support_label.grid(row=3, column=0, sticky='ew')

root.mainloop()
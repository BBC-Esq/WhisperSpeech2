__all__ = ['Vocoder']

from vocos import Vocos
from whisperspeech2 import inference
import torch
import numpy as np

class Vocoder:
    def __init__(self, repo_id="charactr/vocos-encodec-24khz", device=None, cache_dir=None):
        if device is None: device = inference.get_compute_device()
        if device == 'mps': device = 'cpu'
        self.device = device
        self.vocos = Vocos.from_pretrained(repo_id).to(device)

    def is_notebook(self):
        try:
            return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
        except:
            return False

    @torch.no_grad()
    def decode(self, atoks):
        if len(atoks.shape) == 3:
            b,q,t = atoks.shape
            atoks = atoks.permute(1,0,2)
        else:
            q,t = atoks.shape
        atoks = atoks.to(self.device)
        features = self.vocos.codes_to_features(atoks)
        bandwidth_id = torch.tensor({2: 0, 4: 1, 8: 2}[q]).to(self.device)
        return self.vocos.decode(features, bandwidth_id=bandwidth_id)

    def _save_audio(self, fname, audio_tensor, sample_rate=24000):
        audio_np = audio_tensor.cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        try:
            import av
            output = av.open(str(fname), mode='w')
            stream = output.add_stream('pcm_s16le', rate=sample_rate, layout='mono')
            audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
            frame = av.AudioFrame.from_ndarray(audio_int16.reshape(1, -1), format='s16', layout='mono')
            frame.sample_rate = sample_rate
            for pkt in stream.encode(frame):
                output.mux(pkt)
            for pkt in stream.encode(None):
                output.mux(pkt)
            output.close()
            return
        except ImportError:
            pass

        try:
            import soundfile as sf
            sf.write(str(fname), audio_np, sample_rate)
            return
        except ImportError:
            pass

        try:
            import torchaudio
            torchaudio.save(str(fname), audio_tensor, sample_rate)
            return
        except (ImportError, RuntimeError):
            pass

        raise ImportError(
            "No audio saving backend available. Please install PyAV or soundfile:\n"
            "  pip install av\n"
            "or\n"
            "  pip install soundfile"
        )

    def decode_to_file(self, fname, atoks):
        audio = self.decode(atoks)
        self._save_audio(fname, audio.cpu(), 24000)
        if self.is_notebook():
            from IPython.display import display, HTML, Audio
            display(HTML(f'<a href="{fname}" target="_blank">Listen to {fname}</a>'))

    def decode_to_notebook(self, atoks):
        from IPython.display import display, HTML, Audio
        audio = self.decode(atoks)
        display(Audio(audio.cpu().numpy(), rate=24000))

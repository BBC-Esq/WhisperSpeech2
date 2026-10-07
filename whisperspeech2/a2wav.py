__all__ = ['Vocoder']

from vocos import Vocos
from vocos.feature_extractors import EncodecFeatures
from huggingface_hub import hf_hub_download
from whisperspeech2 import inference
import torch
import numpy as np

def _load_vocos(repo_id, cache_dir=None):
    # same steps as Vocos.from_pretrained, which has no cache_dir argument
    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml", cache_dir=cache_dir)
    model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", cache_dir=cache_dir)
    model = Vocos.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location="cpu")
    if isinstance(model.feature_extractor, EncodecFeatures):
        state_dict.update({"feature_extractor.encodec." + k: v for k, v in model.feature_extractor.encodec.state_dict().items()})
    model.load_state_dict(state_dict)
    return model.eval()

class Vocoder:
    def __init__(self, repo_id="charactr/vocos-encodec-24khz", device=None, cache_dir=None):
        if device is None: device = inference.get_compute_device()
        if device == 'mps': device = 'cpu'
        self.device = device
        self.vocos = _load_vocos(repo_id, cache_dir).to(device)

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
            with av.open(str(fname), mode='w') as output:
                # pcm only fits in wav, so use the default codec of the container picked by the file extension
                codec = getattr(output, 'default_audio_codec', None) or 'pcm_s16le'
                stream = output.add_stream(codec, rate=sample_rate, layout='mono')
                audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
                frame = av.AudioFrame.from_ndarray(audio_int16.reshape(1, -1), format='s16', layout='mono')
                frame.sample_rate = sample_rate
                for pkt in stream.encode(frame):
                    output.mux(pkt)
                for pkt in stream.encode(None):
                    output.mux(pkt)
            return
        except Exception:
            # PyAV may be missing or unable to encode the format (PyAV 19 has no Vorbis encoder for .ogg), so try soundfile
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

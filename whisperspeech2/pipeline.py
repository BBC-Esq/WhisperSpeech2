__all__ = ['Pipeline']

from os.path import expanduser
import torch
import numpy as np
from whisperspeech2.t2s_up_wds_mlang_enclm import TSARTransformer
from whisperspeech2.s2a_delar_mup_wds_mlang import SADelARTransformer
from whisperspeech2.a2wav import Vocoder
from whisperspeech2 import inference, s2a_delar_mup_wds_mlang_cond
import traceback
from pathlib import Path


def _load_audio(fname, max_seconds=None):
    try:
        import av
        container = av.open(str(fname))
        stream = container.streams.audio[0]
        sample_rate = stream.rate
        frames = []
        samples_collected = 0
        max_samples = int(sample_rate * max_seconds) if max_seconds else None
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.ndim == 2 and arr.shape[0] > 1:
                arr = arr[0:1]
            frames.append(arr)
            samples_collected += arr.shape[-1]
            if max_samples and samples_collected >= max_samples:
                break
        container.close()
        audio = np.concatenate(frames, axis=-1).flatten().astype(np.float32)
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            max_val = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / max_val
        if max_samples:
            audio = audio[:max_samples]
        return audio, sample_rate
    except ImportError:
        pass

    try:
        import soundfile as sf
        audio, sample_rate = sf.read(str(fname), dtype='float32')
        if audio.ndim > 1:
            audio = audio[:, 0]
        if max_seconds:
            max_samples = int(sample_rate * max_seconds)
            audio = audio[:max_samples]
        return audio, sample_rate
    except ImportError:
        pass

    raise ImportError(
        "No audio loading backend available. Please install PyAV or soundfile:\n"
        "  pip install av\n"
        "or\n"
        "  pip install soundfile"
    )


SPEAKERS = {
    "default": torch.tensor(
       [ -2.6272,  26.0722,   8.7710,   4.0312,  -1.0511,  -0.4165, -21.1550, -16.5279,
        -26.5177,  -7.7289,  16.7714,  27.9332,  47.0092,  -3.0851,  18.1486, -11.6979,
        -22.7664, -10.9929,   5.3045,  -8.5002,  -3.5619, -12.1316, -12.5989,  13.4846,
          1.5031,  49.8233, -17.2940,  -4.9682,   3.1641, -11.2652,  26.1352,  -2.7715,
         -4.6180,   9.8383,   7.3885,  43.3685,  28.3884,  16.3251,  40.6940, -12.6687,
          8.5122, -30.8033, -19.9843, -16.7317,  40.8559, -11.8425,  10.5532, -23.2628,
         14.0579,   5.6914,   8.6182,  10.4368,  -1.9560, -40.3709,  22.2701,  -3.2580,
        -14.4522,  35.1736, -21.0287,   2.1883, -24.9578,  27.6887, -10.1931, -26.6430,
         -5.0069, -29.2248,  18.2211,  28.2282,   8.1390, -16.7069, -50.3623, -19.2298,
        -25.1622, -17.4213,   7.9048, -30.6973,  18.2785,  11.1714,   3.8654,  24.5744,
         -4.6835, -18.9038,   8.0004,  -7.0463,  27.7610,  22.7216,  13.9840,  -3.2983,
          3.3448,  15.1766,  14.2691, -21.9645,  12.1105,  -0.0087, -17.4524,  39.3619,
         15.4061, -21.1576,  15.8396, -21.0326, -31.1697,  12.7755,  24.5641,  -0.1242,
          9.6822, -26.1848,  -8.3680, -44.0064, -24.3400, -32.4047,  42.7861,   3.1024,
          2.7695,  -7.3260, -27.7891,  20.0576,  15.9973,  48.6248,  40.0604, -14.0948,
         -3.6877,   1.9057, -12.1939,   8.0454,   5.2518,   2.8762,  44.5568, -13.8598,
        -33.7492,  18.4188, -37.1685,   8.2223, -15.4349, -28.3067,  21.6822, -31.8609,
          4.7230,  -3.7025,  -7.3012,  13.4508,  13.3780, -18.3768, -11.8720, -14.7013,
         14.1594,   8.7369,   3.1048, -16.5940, -35.4444,  16.3970,  -1.7970, -36.6899,
         -7.1282,  -0.4836,  -7.4033,  11.0366, -22.2089,  14.0981, -55.5385,   2.5638,
         -3.3713, -13.4470,  -2.2353, -49.3987,  20.4446, -28.9039,  18.8918,  15.6752,
          8.0099,  11.1582,  -8.8020,  17.4272,  14.7159,  10.2955, -22.4792, -14.8895,
         -3.6067, -16.9126, -17.0436,  31.5830,  29.8893,  10.2815, -10.0778, -12.5186,
         10.3573,  -8.8265,  19.5367, -19.0000,  -3.5584,   6.4846, -16.6129,  -7.1603]
    ),
    "classic": torch.tensor(
       [-0.2929, -0.4503,  0.4155, -0.1417,  0.0473, -0.1624, -0.2322,  0.7071,
         0.4800,  0.5496,  0.0410,  0.6236,  0.4729,  0.0587,  0.2194, -0.0466,
        -0.3036,  0.0497,  0.5028, -0.1703,  0.5039, -0.6464,  0.3857, -0.7350,
        -0.1605,  0.4808,  0.5397, -0.4851,  0.1774, -0.8712,  0.5789,  0.1785,
        -0.1417,  0.3039,  0.4232, -0.0186,  0.2685,  0.6153, -0.3103, -0.5706,
        -0.4494,  0.3394, -0.6184, -0.3617,  1.1041, -0.1178, -0.1885,  0.1997,
         0.5571, -0.2906, -0.0477, -0.4048, -0.1062,  1.4779,  0.1639, -0.3712,
        -0.1776, -0.0568, -0.6162,  0.0110, -0.0207, -0.1319, -0.3854,  0.7248,
         0.0343,  0.5724,  0.0670,  0.0486, -0.3813,  0.1738,  0.3017,  1.0502,
         0.1550,  0.5708,  0.0366,  0.5093,  0.0294, -0.7091, -0.8220, -0.1583,
        -0.2343,  0.1366,  0.7372, -0.0631,  0.1505,  0.4600, -0.1252, -0.5245,
         0.7523, -0.0386, -0.2587,  1.0066, -0.2037,  0.1617, -0.3800,  0.2790,
         0.0184, -0.5111, -0.7291,  0.1627,  0.2367, -0.0192,  0.4822, -0.4458,
         0.1457, -0.5884,  0.1909,  0.2563, -0.2035, -0.0377,  0.7771,  0.2139,
         0.3801,  0.6047, -0.6043, -0.2563, -0.0726,  0.3856,  0.3217,  0.0823,
        -0.1302,  0.3287,  0.5693,  0.2453,  0.8231,  0.0072,  1.0327,  0.6065,
        -0.0620, -0.5572,  0.5220,  0.2485,  0.1520,  0.0222, -0.2179, -0.7392,
        -0.3855,  0.1822,  0.1042,  0.7133,  0.3583,  0.0606, -0.0424, -0.9189,
        -0.4882, -0.5480, -0.5719, -0.1660, -0.3439, -0.5814, -0.2542,  0.0197,
         0.4942,  0.0915, -0.0420, -0.0035,  0.5578,  0.1051, -0.0891,  0.2348,
         0.6876, -0.6685,  0.8215, -0.3692, -0.3150, -0.0462, -0.6806, -0.2661,
        -0.0308, -0.0050,  0.6756, -0.1647,  1.0734,  0.0049,  0.4969,  0.0259,
        -0.8949,  0.0731,  0.0886,  0.3442, -0.1433, -0.6804,  0.2204,  0.1859,
         0.2702,  0.1699, -0.1443, -0.9614,  0.3261,  0.1718,  0.3545, -0.0686]
    ),
    "voice_b": torch.tensor(
       [  0.7755,  31.3930,   7.0757,  -7.5929,   7.7274,  -9.0396, -23.2394,  18.1754,
        -14.5288, -38.5427,   2.9208,  18.8662,  47.3420,   5.2477,  16.2711, -19.1680,
        -20.2626,  -2.1961,  14.2202,  -3.0244,  -3.4595,   1.1656, -38.3262,  24.0523,
         -0.0905, -11.9759, -13.0695,  18.5081,   5.2803,  -2.0612,   0.1653, -12.9101,
         12.4792,   5.5985,   4.2529, -18.5407, -21.2289, -41.9308,  30.9590, -23.5936,
         -6.3067,  11.4646, -17.0195, -22.1615,  28.5130, -17.5434,  21.8328,  12.3399,
        -22.1609,   3.7410,  -5.8843,  31.6574,  24.0201,  -9.0897,   7.6267, -24.7665,
          1.9649,  13.7305,  -4.4996,  16.3443,  21.6231,   6.1629, -23.9669,  20.9412,
         -1.0780, -13.4039, -21.9229,  -4.1445, -19.6287,  -7.0805, -24.0759,  32.0432,
          4.5370, -17.6171,  26.7590,  -4.4865,  14.8651, -19.9182, -20.0921,  -5.7805,
        -29.9526,   4.4824,   2.5327,  -9.2374,  15.5441,  15.0272,  18.1236,   7.9961,
        -25.9573, -11.5329, -10.0390, -22.2816,   7.1748,  -4.1005, -21.5183, -16.4710,
         13.8730,  -8.8661,   3.4853,  -9.9079,   8.7555,   1.1816,   7.9808, -14.2010,
         -5.0044,  -7.0261, -12.9157, -30.5003, -28.6046,  12.7017,   5.8333,   1.9288,
         -9.9668,  -4.3747, -12.6179,  12.2307,  -2.5358,  12.7391,   9.2852,  24.9428,
          4.7972,  -1.7820,  -6.3753, -31.2602,  31.7787,  15.4675,   7.5358,  -5.0418,
          3.9465,   5.1887,  10.0176, -30.2926,  -8.8900, -10.8207,   1.3794,  22.5733,
         22.0768,   4.5023,  10.2391, -16.9981,  -2.1416, -15.2504, -20.3557,  -6.1607,
         19.0860,   0.4307,  20.1297, -38.8662, -33.6556,  35.3069,  26.8719,   3.4390,
          5.9076,  -2.8745,   0.1687,   3.0867,  -1.5797,   7.9500, -24.0141,  -3.9444,
        -42.0746,  24.4441,  -2.0459,  16.3488, -23.7173,   3.5125,  27.4518, -26.9525,
        -13.4511, -26.0014,   0.4319, -10.5715, -13.7528,  -5.9911, -37.4511,  -1.8969,
         18.1645, -23.4319,   1.7097,   6.9217,   0.5416,  -8.5185,   5.9052,  23.6866,
          1.0179, -11.2199,  -6.3670, -36.5379,  11.5156,  -4.7310,   2.4791, -17.0772]
    ),
}


class Pipeline:
    default_speaker = SPEAKERS["default"]

    def __init__(self, t2s_ref=None, s2a_ref=None, optimize=True, torch_compile=False, use_cuda_graph=False, device=None):
        if device is None: device = inference.get_compute_device()
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        args = dict(device = device)
        try:
            if t2s_ref:
                args["ref"] = t2s_ref
            self.t2s = TSARTransformer.load_model(**args)
            if optimize: self.t2s.optimize(torch_compile=torch_compile, use_cuda_graph=use_cuda_graph)
        except:
            print("Failed to load the T2S model:")
            print(traceback.format_exc())
        args = dict(device = device)
        try:
            if s2a_ref:
                spec = inference.load_model(ref=s2a_ref, device=device)
                if [x for x in spec['state_dict'].keys() if x.startswith('cond_embeddings.')]:
                    cls = s2a_delar_mup_wds_mlang_cond.SADelARTransformer
                    args['spec'] = spec
                else:
                    cls = SADelARTransformer
                    args['spec'] = spec
            else:
                cls = SADelARTransformer
            self.s2a = cls.load_model(**args)
            if optimize: self.s2a.optimize(torch_compile=torch_compile, use_cuda_graph=use_cuda_graph)
        except:
            print("Failed to load the S2A model:")
            print(traceback.format_exc())

        self.vocoder = Vocoder(device=device)
        self.encoder = None

    def reset_cuda_graphs(self):
        if hasattr(self, 't2s') and hasattr(self.t2s, 'reset_cuda_graph'):
            self.t2s.reset_cuda_graph()
        if hasattr(self, 's2a') and hasattr(self.s2a, 'reset_cuda_graph'):
            self.s2a.reset_cuda_graph()

    def extract_spk_emb(self, fname):
        if self.encoder is None:
            device = self.device
            if device == 'mps': device = 'cpu'
            try:
                from speechbrain.inference import EncoderClassifier
            except ImportError:
                try:
                    from speechbrain.pretrained import EncoderClassifier
                except ImportError:
                    raise ImportError(
                        "speechbrain is required for speaker embedding extraction.\n"
                        "Install with: pip install speechbrain"
                    )
            self.encoder = EncoderClassifier.from_hparams(
                "speechbrain/spkrec-ecapa-voxceleb",
                savedir=expanduser("~/.cache/speechbrain/"),
                run_opts={"device": device},
            )

        audio_np, sr = _load_audio(fname, max_seconds=30)
        samples = torch.tensor(audio_np, dtype=torch.float32)
        samples = self.encoder.audio_normalizer(samples, sr)
        spk_emb = self.encoder.encode_batch(samples.unsqueeze(0))

        return spk_emb[0,0].to(self.device)

    def generate_atoks(self, text, speaker=None, lang='en', cps=15, step_callback=None):
        if speaker is None: speaker = self.default_speaker
        elif isinstance(speaker, str) and speaker in SPEAKERS: speaker = SPEAKERS[speaker]
        elif isinstance(speaker, (str, Path)): speaker = self.extract_spk_emb(speaker)
        text = text.replace("\n", " ")
        stoks = self.t2s.generate(text, cps=cps, lang=lang, step=step_callback)[0]
        atoks = self.s2a.generate(stoks, speaker.unsqueeze(0), step=step_callback)
        return atoks

    def generate(self, text, speaker=None, lang='en', cps=15, step_callback=None):
        return self.vocoder.decode(self.generate_atoks(text, speaker, lang=lang, cps=cps, step_callback=step_callback))

    def generate_to_file(self, fname, text, speaker=None, lang='en', cps=15, step_callback=None):
        self.vocoder.decode_to_file(fname, self.generate_atoks(text, speaker, lang=lang, cps=cps, step_callback=None))

    def generate_to_notebook(self, text, speaker=None, lang='en', cps=15, step_callback=None):
        self.vocoder.decode_to_notebook(self.generate_atoks(text, speaker, lang=lang, cps=cps, step_callback=None))

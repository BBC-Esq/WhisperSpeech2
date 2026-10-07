__all__ = ['get_compute_device']

import os
import re
import sys
import warnings
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from contextlib import nullcontext

def get_default_compute_device():
    if torch.cuda.is_available() and (torch.version.cuda or torch.version.hip):
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

preferred_device = None

def get_compute_device():
    global preferred_device
    if preferred_device is None: preferred_device = get_default_compute_device()
    return preferred_device

def is_hf_ref(ref):
    # "repo_id:filename" names a Hugging Face file, but a Windows path such as D:\models\t2s.model has a colon too
    if not isinstance(ref, str) or os.path.exists(ref) or re.match(r"[A-Za-z]:[\\/]", ref):
        return False
    return ":" in ref

def load_model(ref=None, spec=None, device='cpu', cache_dir=None):
    if spec is not None: return spec
    if is_hf_ref(ref):
        repo_id, filename = ref.split(":", 1)
        local_filename = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    else:
        local_filename = ref
    return torch.load(local_filename, map_location=device)

def inference_context():
    return nullcontext()

def progress_bar(iterable):
    from tqdm import tqdm
    # tqdm writes to stderr, which is None in GUI apps started with pythonw
    return tqdm(iterable, disable=sys.stderr is None)

def compile_step(fn, device, use_cuda_graph=False):
    if use_cuda_graph:
        warnings.warn("torch_compile is ignored when use_cuda_graph is set because the CUDA graph path never calls the compiled function.")
        return fn
    if device.type == 'cuda' and sys.platform == 'win32':
        try:
            import triton
        except ImportError:
            warnings.warn("torch_compile needs Triton, which on Windows is the triton-windows package, so the model runs without it.")
            return fn
        # before PyTorch 2.14 the static CUDA launcher overflows a 32-bit C long on Windows
        if torch.__version__ < '2.14':
            import torch._inductor.config as inductor_config
            if hasattr(inductor_config, 'use_static_cuda_launcher'):
                inductor_config.use_static_cuda_launcher = False
    return torch.compile(fn, mode="reduce-overhead", fullgraph=True)

def math_attention():
    # for single-token decode steps inside a CUDA graph the math SDPA kernel is the fastest one
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, T=1.0, top_k=None):
    logits = logits / max(T, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, T=1.0, top_k=None):
    probs = logits_to_probs(logits, T, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next
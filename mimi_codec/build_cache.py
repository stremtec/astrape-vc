"""Build full VCTK code cache + Resemblyzer embeddings for all speakers."""
import sys,os; sys.path.insert(0,'/Users/asill/btrv5')
import torch, soundfile as sf, subprocess, time
from moshi.models import loaders; from pathlib import Path
from scipy import signal
import numpy as np
from resemblyzer import VoiceEncoder

SR=24000; STRIDE=1920
base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

# Get all speakers
all_spks = sorted([d for d in os.listdir(base) if os.path.isdir(f'{base}/{d}') and d.startswith('p')])
print(f'Found {len(all_spks)} speakers')

# Step 1: Resemblyzer embeddings (lightweight, do first)
ve = VoiceEncoder()
spk_emb = {}
for s in all_spks:
    utts = sorted([f for f in os.listdir(f'{base}/{s}') if f.endswith('.flac')])[:5]  # 5 utterances each
    embs = []
    for u in utts:
        d,sr = sf.read(f'{base}/{s}/{u}')
        if sr!=16000: d = signal.resample(d, int(len(d)*16000/sr), axis=0)
        if d.ndim>1: d = d.mean(axis=1)
        d = d[:16000*5]
        try:
            embs.append(ve.embed_utterance(d.astype(np.float32)))
        except: pass
    if embs:
        spk_emb[s] = torch.from_numpy(np.stack(embs).mean(0)).float()
        if len(spk_emb) % 20 == 0:
            print(f'  Spk embeddings: {len(spk_emb)}/{len(all_spks)}')

torch.save(spk_emb, '/Users/asill/btrv5/runs/vctk_full_spk.pt')
print(f'Saved {len(spk_emb)} speaker embeddings')

# Step 2: Mimi code cache (heavy — do in batches)
print('Loading Mimi...')
mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)

def load_any(path):
    d,sr = sf.read(path)
    if sr!=SR: d = signal.resample(d, int(len(d)*SR/sr), axis=0)
    L = len(d) - (len(d) % STRIDE)
    d = d[:L]
    if d.ndim>1: d = d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)

cache = {}
t0 = time.time()
for i, s in enumerate(all_spks):
    utts = sorted([f for f in os.listdir(f'{base}/{s}') if f.endswith('.flac')])
    for u in utts:
        try:
            a = load_any(f'{base}/{s}/{u}')
            codes = mimi.encode(a)
            cache[(s,u.replace('.flac',''))] = codes.cpu()
        except: pass
    if (i+1) % 10 == 0:
        print(f'  Codes: {len(cache)} entries ({i+1}/{len(all_spks)} speakers) [{time.time()-t0:.0f}s]')

torch.save(cache, '/Users/asill/btrv5/runs/vctk_full_codes.pt')
print(f'Saved {len(cache)} code entries [{time.time()-t0:.0f}s]')

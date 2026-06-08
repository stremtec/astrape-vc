"""Train Continuous Mimi Latent Converter with token swap teacher."""
import sys, os, random; sys.path.insert(0, '.')
import torch, torch.nn as nn, soundfile as sf, time, subprocess
from moshi.models import loaders; from pathlib import Path
from scipy import signal
import numpy as np
from resemblyzer import VoiceEncoder
from codex_vc.continuous_converter import (
    ContinuousConverter, total_training_loss, convert_continuous
)
from codex_vc.metrics import compute_all_metrics, format_metrics

SR = 24000; STRIDE = 1920
STEPS = 100
LR = 5e-4

def load(path, dur=2):
    d, sr = sf.read(path)
    if sr != SR: d = signal.resample(d, int(len(d)*SR/sr), axis=0)
    L = dur*SR - (dur*SR % STRIDE); d = d[:L]
    if d.ndim > 1: d = d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)

print("Loading Mimi...")
mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)

spk_emb = torch.load('runs/vctk_full_spk.pt', weights_only=True)

base = '/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
spks = [s for s in ['p225','p226','p227','p228','p229'] if s in spk_emb]
utts = ['001','002','003']

print("Caching z_q + teacher...")
cache = {}
teacher_cache = {}

with torch.no_grad():
    for s in spks:
        for u in utts:
            a = load(f'{base}/{s}/{s}_{u}_mic1.flac')
            codes = mimi.encode(a)
            z_q = mimi.quantizer.decode(codes)
            cache[(s, u)] = (z_q, codes)

for u in utts:
    for s in spks:
        for t in spks:
            if s == t: continue
            cs, ct = cache[(s, u)][1], cache[(t, u)][1]
            T = min(cs.shape[2], ct.shape[2])
            c_teacher = cs[:, :, :T].clone()
            c_teacher[:, 1:, :] = ct[:, 1:, :T]
            teacher_cache[(s, t, u)] = mimi.quantizer.decode(c_teacher)

T = min(z.shape[2] for (z, _) in cache.values())
cache = {k: (z[:, :, :T], c[:, :, :T]) for k, (z, c) in cache.items()}
teacher_cache = {k: v[:, :, :T] for k, v in teacher_cache.items()}
print(f"  {len(cache)} latents, {len(teacher_cache)} teachers")

converter = ContinuousConverter()
opt = torch.optim.AdamW(converter.parameters(), lr=LR)
pairs = [(s, t, u) for u in utts for s in spks for t in spks if s != t]
random.shuffle(pairs)
print(f"  {len(pairs)} pairs")

print()
print(f"Training {STEPS} steps (teacher distillation)...")
t0 = time.time()

for step in range(STEPS):
    random.shuffle(pairs)
    tl = 0; ms = {}
    for s, t, u in pairs[:60]:
        z_q_src, _ = cache[(s, u)]
        z_q_tgt, _ = cache[(t, u)]
        z_q_teacher = teacher_cache[(s, t, u)]
        s_tgt = spk_emb[t].unsqueeze(0)
        
        z_q_vc = converter(z_q_src, s_tgt)
        loss, m = total_training_loss(z_q_vc, z_q_src, z_q_tgt, z_q_teacher, mimi, True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(converter.parameters(), 1.0); opt.step()
        tl += loss.item()
        for k, v in m.items(): ms[k] = ms.get(k, 0) + v
    
    if step % 20 == 0 or step == STEPS - 1:
        n = min(len(pairs), 60)
        print(f"  step {step:4d}: loss={tl/n:.4f} "
              f"content={ms.get('content',0)/n:.4f} "
              f"speaker={ms.get('speaker',0)/n:.4f} "
              f"teacher={ms.get('teacher',0)/n:.4f} [{time.time()-t0:.0f}s]")

torch.save(converter.state_dict(), 'runs/continuous_converter.pt')
print(f"Saved [{time.time()-t0:.0f}s]")

# Test
subprocess.run(['ffmpeg', '-y', '-i', '/Users/asill/Downloads/origin.mp3',
                '-ar', '24000', '-ac', '1', '-sample_fmt', 's16', '-t', '3',
                '/tmp/cc2.wav'], capture_output=True)

def load_any(path, dur=None):
    d, sr = sf.read(path)
    if sr != SR: d = signal.resample(d, int(len(d)*SR/sr), axis=0)
    if dur is not None: L = dur*SR - (dur*SR % STRIDE); d = d[:L]
    else: L = len(d) - (len(d) % STRIDE); d = d[:L]
    if d.ndim > 1: d = d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)

tgt_o = load_any('/tmp/cc2.wav', dur=None)

ve = VoiceEncoder()
def get_spk(a):
    a = a[:SR*3]
    if len(a) < 1600: a = np.pad(a, (0, 1600 - len(a)))
    a16 = signal.resample(a, max(1, int(len(a)*16000/SR)))
    return torch.from_numpy(ve.embed_utterance(a16.astype(np.float32))).float()

s_origin = get_spk(tgt_o.numpy())

print()
print("Testing...")
for src_s in ['p225', 'p255']:
    f = f'{base}/{src_s}/{src_s}_001_mic1.flac'
    if not os.path.isfile(f): continue
    src_a = load_any(f)
    
    with torch.no_grad():
        vc = convert_continuous(converter, mimi, src_a, s_origin)
    
    Tc = min(vc.shape[2], src_a.shape[2], tgt_o.shape[2])
    zv = mimi.encode_to_latent(vc[:, :, :Tc], quantize=False)
    zs = mimi.encode_to_latent(src_a[:, :, :Tc], quantize=False)
    zt = mimi.encode_to_latent(tgt_o[:, :, :Tc], quantize=False)
    T2 = min(zv.shape[2], zs.shape[2], zt.shape[2])
    cs = torch.nn.functional.cosine_similarity(
        zv[:, :, :T2].reshape(-1), zs[:, :, :T2].reshape(-1), dim=0)
    ct = torch.nn.functional.cosine_similarity(
        zv[:, :, :T2].reshape(-1), zt[:, :, :T2].reshape(-1), dim=0)
    
    m = compute_all_metrics(
        src_a.squeeze()[:Tc].numpy(),
        tgt_o.squeeze()[:Tc].numpy(),
        vc[0, 0, :Tc].numpy(), SR
    )
    m['cos_src'] = cs.item(); m['cos_tgt'] = ct.item()
    m['delta'] = ct.item() - cs.item()
    
    print(f"=== {src_s} -> origin ===")
    print(format_metrics(m))
    sf.write(f'research5/cc_{src_s}.wav', vc[0, 0, :Tc].numpy(), SR)
    print()

print("Done")

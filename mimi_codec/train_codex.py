"""
Train Codex Architecture on VCTK full cache.

Usage:
    python mimi_codec/train_codex.py
"""
import sys, os, random; sys.path.insert(0, '/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, subprocess, time
from moshi.models import loaders; from pathlib import Path
from scipy import signal
from mimi_codec.codex_vc import CodeGenerator, training_loss, vc_convert

# Config
CACHE_PATH = '/Users/asill/btrv5/runs/vctk_codes_full.pt'
SPK_PATH = '/Users/asill/btrv5/runs/vctk_full_spk.pt'
MODEL_PATH = '/Users/asill/btrv5/runs/codex_model.pt'
SR = 24000
STRIDE = 1920
STEPS = 300
LR = 5e-4

# Load Mimi (only needed for final test, not training)
print('Loading resources...')
spk_emb = torch.load(SPK_PATH)
print(f'  {len(spk_emb)} speaker embeddings')

cache = torch.load(CACHE_PATH)
print(f'  {len(cache)} cached codes')

# Trim all to same T
T = min(c.shape[2] for c in cache.values())
cache = {k: v[:, :, :T] for k, v in cache.items()}
print(f'  T={T}')

# Get available speakers
speakers = sorted(set(k[0] for k in cache.keys()))
utts = sorted(set(k[1] for k in cache.keys()))
speakers = [s for s in speakers if s in spk_emb]
print(f'  {len(speakers)} speakers with both codes + embeddings')

# Build training pairs: same text, different speakers
pairs = []
for u in utts:
    spks_with = [s for s in speakers if (s, u) in cache]
    if len(spks_with) < 2: continue
    for s in spks_with:
        for t in spks_with:
            if s != t:
                pairs.append((s, t, u))
random.shuffle(pairs)
print(f'  {len(pairs)} training pairs (using {min(len(pairs), 2000)} per step)')

# Model
model = CodeGenerator()
opt = torch.optim.AdamW(model.parameters(), lr=LR)
ce = nn.CrossEntropyLoss()

# Train
print(f'\
Training {STEPS} steps...')
t0 = time.time()
best_acc = 0

for step in range(STEPS):
    random.shuffle(pairs)
    batch_pairs = pairs[:min(len(pairs), 2000)]  # limit per step
    lt = 0; total_acc = 0
    
    for s, t, u in batch_pairs:
        lv0 = cache[(s, u)][:, 0, :]           # (1, T)
        lv1_7_gt = cache[(t, u)][:, 1:, :]     # (1, 7, T)
        spk = spk_emb[t].unsqueeze(0)            # (1, 256)
        
        loss = training_loss(model, lv0, lv1_7_gt, spk, ce)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        lt += loss.item()
        with torch.no_grad():
            pred = model.predict(lv0, spk)
            total_acc += (pred == lv1_7_gt).float().mean().item()
    
    avg_loss = lt / len(batch_pairs)
    avg_acc = total_acc / len(batch_pairs)
    
    if avg_acc > best_acc:
        best_acc = avg_acc
        torch.save(model.state_dict(), MODEL_PATH)
    
    if step % 30 == 0:
        print(f'  step {step:4d}: loss={avg_loss:.4f} acc={avg_acc:.4f} best={best_acc:.4f} [{time.time()-t0:.0f}s]')

print(f'\
Done [{time.time()-t0:.0f}s] best_acc={best_acc:.4f}')
print(f'Model saved to {MODEL_PATH}')

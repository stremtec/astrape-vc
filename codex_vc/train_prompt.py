"""Prompt VC: full VCTK training."""
import sys, os, random; sys.path.insert(0, '.')
import torch, torch.nn as nn, soundfile as sf, time
from moshi.models import loaders; from pathlib import Path
from scipy import signal
from codex_vc.prompt_vc import PromptEncoder, PromptConverter, compute_loss

SR = 24000; STRIDE = 1920
STEPS = 200
LR = 5e-4
MAX_PAIRS = 500
VAL_RATIO = 0.2

def load(path, dur=2):
    d, sr = sf.read(path)
    if sr != SR: d = signal.resample(d, int(len(d)*SR/sr), axis=0)
    L = dur*SR - (dur*SR % STRIDE); d = d[:L]
    if d.ndim > 1: d = d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)

print("Loading Mimi...")
mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)

# Load cached codes
print("Loading codes...")
cache = torch.load('runs/vctk_codes_full.pt', weights_only=True)
T = min(c.shape[2] for c in cache.values())
cache = {k: v[:, :, :T] for k, v in cache.items()}
spks = sorted({k[0] for k in cache})
utts = sorted({k[1] for k in cache})
print(f"  {len(cache)} codes, {len(spks)} speakers, {len(utts)} utts, T={T}")

# Load speaker embeddings
spk_emb = torch.load('runs/vctk_full_spk.pt', weights_only=True)
spks = [s for s in spks if s in spk_emb]
print(f"  {len(spks)} speakers with embeddings")

# Pre-compute prompt features for all audios
print("Pre-computing prompt features...")
prompt_enc = PromptEncoder(mimi)
prompt_cache = {}
with torch.no_grad():
    for s in spks:
        for u in utts:
            if (s, u) not in cache: continue
            try:
                a = load(f'/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed/{s}/{s}_{u}_mic1.flac')
                prompt_cache[(s, u)] = prompt_enc(a)
            except: pass
print(f"  {len(prompt_cache)} prompt features")

# Train/val split by utterance
random.seed(42); random.shuffle(utts)
n_val = max(1, int(len(utts) * VAL_RATIO))
val_utts = set(utts[:n_val]); train_utts = set(utts[n_val:])

def make_pairs(utt_set):
    ps = []
    for u in utt_set:
        sw = [s for s in spks if (s, u) in cache and (s, u) in prompt_cache]
        for s in sw:
            for t in sw:
                if s != t: ps.append((s, t, u))
    return ps

train_pairs = make_pairs(train_utts)
val_pairs = make_pairs(val_utts)
random.shuffle(train_pairs)
print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

# Model
converter = PromptConverter()
opt = torch.optim.AdamW(converter.parameters(), lr=LR)
ce = nn.CrossEntropyLoss()

print()
print(f"Training {STEPS} steps...")
t0 = time.time()
best_val_acc = 0.0

for step in range(STEPS):
    random.shuffle(train_pairs)
    step_pairs = train_pairs[:min(len(train_pairs), MAX_PAIRS)]
    tl = 0; ta = 0

    for s, t, u in step_pairs:
        lv0 = cache[(s, u)][:, 0, :]
        lv1_gt = cache[(t, u)][:, 1:, :]
        pf = prompt_cache[(t, u)]
        loss = compute_loss(converter, lv0, lv1_gt, pf, ce)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(converter.parameters(), 1.0); opt.step()
        tl += loss.item()
        with torch.no_grad(): ta += (converter.predict(lv0, pf) == lv1_gt).float().mean().item()

    if step % 20 == 0 or step == STEPS - 1:
        # Validate
        converter.eval()
        vl = 0; va = 0; vn = 0
        for s, t, u in val_pairs[:50]:
            lv0 = cache[(s, u)][:, 0, :]
            lv1_gt = cache[(t, u)][:, 1:, :]
            pf = prompt_cache[(t, u)]
            vl += compute_loss(converter, lv0, lv1_gt, pf, ce).item()
            va += (converter.predict(lv0, pf) == lv1_gt).float().mean().item()
            vn += 1
        converter.train()

        N = len(step_pairs)
        avg_va = va / max(vn, 1)
        if avg_va > best_val_acc:
            best_val_acc = avg_va
            torch.save(converter.state_dict(), 'runs/prompt_vc.pt')

        print(f"  step {step:4d}: train_loss={tl/N:.4f} train_acc={ta/N:.4f} "
              f"val_loss={vl/vn:.4f} val_acc={avg_va:.4f} best={best_val_acc:.4f} "
              f"[{time.time()-t0:.0f}s]")

print()
print(f"Done! Best val_acc={best_val_acc:.4f} [{time.time()-t0:.0f}s]")

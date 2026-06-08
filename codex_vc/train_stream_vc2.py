"""StreamVC scaled training: 30 speakers, HuBERT pre-cache, speaker adversarial."""
import sys, os, random; sys.path.insert(0, '.')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, time, subprocess
from moshi.models import loaders; from pathlib import Path
from transformers import HubertModel
from scipy import signal
import numpy as np
from codex_vc.stream_vc import StreamVC
from codex_vc.metrics import compute_all_metrics, format_metrics

SR = 24000
STEPS = 200
LR = 5e-4

print("Loading models...")
hubert = HubertModel.from_pretrained('facebook/hubert-base-ls960').eval()
mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)

base = '/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
spks = sorted([d for d in os.listdir(base) if os.path.isdir(f'{base}/{d}') and d.startswith('p')])[:30]
utts = ['001','002','003']

spk_to_idx = {s: i for i, s in enumerate(spks)}
n_spk = len(spks)
print(f"  {n_spk} speakers")

# Pre-compute HuBERT layer 0 features + Mimi z_q targets
print("Pre-computing features...")
hubert_feats = {}
zq_targets = {}

for s in spks:
    for u in utts:
        f = f'{base}/{s}/{s}_{u}_mic1.flac'
        if not os.path.isfile(f): continue
        d, sr = sf.read(f)
        if sr != 16000: d = signal.resample(d, int(len(d)*16000/sr), axis=0)
        if d.ndim > 1: d = d.mean(axis=1)
        d = d[:16000*2]
        src_16k = torch.from_numpy(d).float().unsqueeze(0)
        
        with torch.no_grad():
            hs = hubert(src_16k, output_hidden_states=True).hidden_states
            h_avg = (hs[1] + hs[2] + hs[3]) / 3.0  # Ensemble layers 1-3
        
        d_24k = signal.resample(d, int(len(d)*24000/16000), axis=0)
        src_24k = torch.from_numpy(d_24k).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            z = mimi.encode_to_latent(src_24k, quantize=False)
            codes = mimi.quantizer.encode(z)
            z_q = mimi.quantizer.decode(codes).squeeze(0)
        
        hubert_feats[(s, u)] = h_avg  # (1, T_h, 768)
        zq_targets[(s, u)] = z_q   # (512, T_mimi)

# Trim to consistent T
T_h = min(h.shape[1] for h in hubert_feats.values())
T_m = min(z.shape[1] for z in zq_targets.values())
hubert_feats = {k: v[:, :T_h] for k, v in hubert_feats.items()}
zq_targets = {k: v[:, :T_m] for k, v in zq_targets.items()}
print(f"  {len(hubert_feats)} features, T_h={T_h}, T_m={T_m}")

# Model
model = StreamVC(hubert, mimi, n_speakers=n_spk)
spk_emb = nn.Embedding(n_spk, 256)
opt = torch.optim.AdamW(list(model.parameters()) + list(spk_emb.parameters()), lr=LR)
ce_spk = nn.CrossEntropyLoss()

# Training pairs: same-text different-speaker
pairs = []
for u in utts:
    sw = [s for s in spks if (s, u) in hubert_feats]
    for s in sw:
        for t in sw:
            if s != t: pairs.append((s, t, u))
random.shuffle(pairs)
n_train = int(len(pairs) * 0.8)
train_p = pairs[:n_train]; val_p = pairs[n_train:]
print(f"  Train pairs: {len(train_p)}, Val pairs: {len(val_p)}")

print()
print(f"Training {STEPS} steps...")
t0 = time.time()
best_val = float('inf')

for step in range(STEPS):
    random.shuffle(train_p)
    tl = 0; tl_r = 0; tl_a = 0
    max_p = min(len(train_p), 40)
    
    for s, t, u in train_p[:max_p]:
        h0 = hubert_feats[(s, u)]       # (1, T_h, 768)
        z_q_tgt = zq_targets[(t, u)]     # (512, T_m)
        tgt_spk = spk_emb(torch.tensor([spk_to_idx[t]]))
        
        # Forward (skip HuBERT — use pre-computed h0)
        # Manual forward to use pre-computed features
        z_content = model.content_proj(h0)  # (1, 512, T_m)
        
        # Adversarial
        spk_logits = model.spk_adversarial(h0.transpose(1, 2))
        loss_adv = ce_spk(spk_logits, torch.tensor([spk_to_idx[s]]))
        
        # Speaker injection
        gamma = model.spk_gamma(tgt_spk).unsqueeze(-1)
        beta = model.spk_beta(tgt_spk).unsqueeze(-1)
        mean = z_content.mean(dim=2, keepdim=True)
        std = z_content.std(dim=2, keepdim=True) + 1e-5
        z_vc = (z_content - mean) / std * gamma + beta
        
        # Reconstruction
        T = min(z_vc.shape[2], z_q_tgt.shape[1])
        loss_rec = F.mse_loss(z_vc[:, :, :T], z_q_tgt[:, :T].unsqueeze(0))
        
        loss = loss_rec + 1.5 * loss_adv
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += loss.item(); tl_r += loss_rec.item(); tl_a += loss_adv.item()
    
    if step % 30 == 0 or step == STEPS - 1:
        # Quick val
        model.eval()
        vl = 0; n = 0
        for s, t, u in val_p[:10]:
            h0 = hubert_feats[(s, u)]
            z_q_tgt = zq_targets[(t, u)]
            tgt_spk = spk_emb(torch.tensor([spk_to_idx[t]]))
            z_content = model.content_proj(h0)
            gamma = model.spk_gamma(tgt_spk).unsqueeze(-1)
            beta = model.spk_beta(tgt_spk).unsqueeze(-1)
            mean = z_content.mean(dim=2, keepdim=True)
            std = z_content.std(dim=2, keepdim=True) + 1e-5
            z_vc = (z_content - mean) / std * gamma + beta
            T = min(z_vc.shape[2], z_q_tgt.shape[1])
            vl += F.mse_loss(z_vc[:,:,:T], z_q_tgt[:,:T].unsqueeze(0)).item(); n += 1
        model.train()
        
        avg_val = vl / max(n, 1)
        if avg_val < best_val:
            best_val = avg_val
            torch.save({'model': model.state_dict(), 'spk_emb': spk_emb.state_dict()}, 'runs/stream_vc2.pt')
        
        print(f"  step {step:4d}: rec={tl_r/max_p:.4f} adv={tl_a/max_p:.4f} "
              f"val={avg_val:.4f} [{time.time()-t0:.0f}s]")

# Load best
ckpt = torch.load('runs/stream_vc2.pt', weights_only=True)
model.load_state_dict(ckpt['model'])
model.eval()

print(f"Done [{time.time()-t0:.0f}s] best_val={best_val:.4f}")

# Test: p255 -> origin
subprocess.run(['ffmpeg', '-y', '-i', '/Users/asill/Downloads/origin.mp3',
                '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', '-t', '2',
                '/tmp/sv2_test.wav'], capture_output=True)

d_src, sr_src = sf.read(f'{base}/p255/p255_001_mic1.flac')
if sr_src != 16000: d_src = signal.resample(d_src, int(len(d_src)*16000/sr_src), axis=0)
if d_src.ndim > 1: d_src = d_src.mean(axis=1)
src_16k = torch.from_numpy(d_src[:16000*2]).float().unsqueeze(0)

with torch.no_grad():
    # Use p226 speaker embedding as target
    tgt_spk = spk_emb(torch.tensor([1]))  # p226
    vc_audio = model.convert(src_16k, tgt_spk)
    Tc = min(vc_audio.shape[2], len(d_src[:16000*2]))
    vc_24k = signal.resample(vc_audio.squeeze().numpy()[:Tc], int(Tc*24000/16000))
    src_24k = signal.resample(d_src[:16000*2], int(16000*2*24000/16000))
    
    src_t = torch.from_numpy(src_24k).float().unsqueeze(0).unsqueeze(0)
    vc_t = torch.from_numpy(vc_24k).float().unsqueeze(0).unsqueeze(0)
    zs = mimi.encode_to_latent(src_t, quantize=False)
    zv = mimi.encode_to_latent(vc_t, quantize=False)
    T2 = min(zs.shape[2], zv.shape[2])
    cs = F.cosine_similarity(zv[:,:,:T2].reshape(-1), zs[:,:,:T2].reshape(-1), dim=0)
    
    m = compute_all_metrics(src_24k, np.zeros_like(src_24k), vc_24k, 24000)
    m['cos_src'] = cs.item()
    print()
    print("p255->p226 (StreamVC scaled):")
    print(format_metrics(m))
    sf.write('research5/stream_vc2.wav', vc_24k, 24000)
    print("Done")

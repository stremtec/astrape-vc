"""Train StreamVC with speaker adversarial loss."""
import sys, os, random; sys.path.insert(0, '.')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, time, subprocess
from moshi.models import loaders; from pathlib import Path
from transformers import HubertModel
from scipy import signal
import numpy as np
from codex_vc.stream_vc import StreamVC
from codex_vc.metrics import compute_all_metrics, format_metrics

SR = 24000
STEPS = 100
LR = 5e-4

print("Loading models...")
hubert = HubertModel.from_pretrained('facebook/hubert-base-ls960').eval()
mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)

base = '/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
spks = ['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234']
utts = ['001','002','003']

spk_to_idx = {s: i for i, s in enumerate(spks)}
n_spk = len(spks)

print("Preparing data...")
samples = []
for s in spks:
    for u in utts:
        f = f'{base}/{s}/{s}_{u}_mic1.flac'
        if not os.path.isfile(f): continue
        d, sr = sf.read(f)
        if sr != 16000: d = signal.resample(d, int(len(d)*16000/sr), axis=0)
        if d.ndim > 1: d = d.mean(axis=1)
        d = d[:16000*2]
        src_16k = torch.from_numpy(d).float()
        d_24k = signal.resample(d, int(len(d)*24000/16000), axis=0)
        src_24k = torch.from_numpy(d_24k).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            z = mimi.encode_to_latent(src_24k, quantize=False)
            codes = mimi.quantizer.encode(z)
            z_q = mimi.quantizer.decode(codes)
        samples.append((src_16k, z_q.squeeze(0), spk_to_idx[s]))

random.shuffle(samples)
n_train = int(len(samples) * 0.8)
train_s = samples[:n_train]
val_s = samples[n_train:]
print(f"  Train: {len(train_s)}, Val: {len(val_s)}")

spk_emb = nn.Embedding(n_spk, 256)
model = StreamVC(hubert, mimi, n_speakers=n_spk)
opt = torch.optim.AdamW(list(model.parameters()) + list(spk_emb.parameters()), lr=LR)
ce_spk = nn.CrossEntropyLoss()

print()
print(f"Training {STEPS} steps...")
t0 = time.time()

for step in range(STEPS):
    random.shuffle(train_s)
    tl = 0; tl_rec = 0; tl_adv = 0
    for src_16k, z_q_tgt, spk_label in train_s[:30]:
        src_b = src_16k.unsqueeze(0)
        tgt_spk_b = spk_emb(torch.tensor([spk_label]))
        z_vc = model.forward(src_b, tgt_spk_b)
        T = min(z_vc.shape[2], z_q_tgt.shape[1])  # z_q_tgt is (512, T)
        loss_rec = F.mse_loss(z_vc[:, :, :T], z_q_tgt[:, :T].unsqueeze(0))
        with torch.no_grad():
            h0 = model.hubert(src_b, output_hidden_states=True).hidden_states[0]
        spk_logits = model.spk_adversarial(h0.transpose(1, 2))
        loss_adv = ce_spk(spk_logits, torch.tensor([spk_label]))
        loss = loss_rec + 1.5 * loss_adv  # increased adv weight from 0.3
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += loss.item(); tl_rec += loss_rec.item(); tl_adv += loss_adv.item()
    
    if step % 20 == 0 or step == STEPS - 1:
        n = min(len(train_s), 30)
        print(f"  step {step:4d}: loss={tl/n:.4f} rec={tl_rec/n:.4f} "
              f"adv={tl_adv/n:.4f} [{time.time()-t0:.0f}s]")

torch.save(model.state_dict(), 'runs/stream_vc.pt')
print(f"Saved [{time.time()-t0:.0f}s]")

# Test p255 -> origin
subprocess.run(['ffmpeg', '-y', '-i', '/Users/asill/Downloads/origin.mp3',
                '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', '-t', '2',
                '/tmp/sv_test.wav'], capture_output=True)

d2, sr2 = sf.read(f'{base}/p255/p255_001_mic1.flac')
if sr2 != 16000: d2 = signal.resample(d2, int(len(d2)*16000/sr2), axis=0)
if d2.ndim > 1: d2 = d2.mean(axis=1)
src_16k = torch.from_numpy(d2[:16000*2]).float().unsqueeze(0)

with torch.no_grad():
    tgt_spk = spk_emb(torch.tensor([1]))
    vc_audio = model.convert(src_16k, tgt_spk)
    Tc = min(vc_audio.shape[2], len(d2[:16000*2]))
    vc_24k = signal.resample(vc_audio.squeeze().numpy()[:Tc], int(Tc*24000/16000))
    src_24k_np = signal.resample(d2[:16000*2], int(16000*2*24000/16000))
    
    src_t = torch.from_numpy(src_24k_np).float().unsqueeze(0).unsqueeze(0)
    vc_t = torch.from_numpy(vc_24k).float().unsqueeze(0).unsqueeze(0)
    zs = mimi.encode_to_latent(src_t, quantize=False)
    zv = mimi.encode_to_latent(vc_t, quantize=False)
    T2 = min(zs.shape[2], zv.shape[2])
    cs = F.cosine_similarity(zv[:,:,:T2].reshape(-1), zs[:,:,:T2].reshape(-1), dim=0)
    
    m = compute_all_metrics(src_24k_np, np.zeros_like(src_24k_np), vc_24k, 24000)
    print()
    print(f"p255->p226 (StreamVC):")
    print(f"  cos_src: {cs.item():.4f}")
    sf.write('research5/stream_vc_test.wav', vc_24k, 24000)
    print("Done")

"""
Codex Architecture: LV0 + Text-Invariant Speaker → LV1-7 Code Generator.

[Source Audio] → Mimi encode → LV0 codes ──────────────────┐
[Target Audio] → Resemblyzer → spk embedding ──────────────┤
                                                            ↓
                                              Bidirectional Transformer
                                                            ↓
                                              LV1-7 codes (7×T)
                                                            ↓
                                              Mimi decoder → VC Audio
"""

import sys, os, random; sys.path.insert(0,'/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, subprocess, time
from moshi.models import loaders; from pathlib import Path
from scipy import signal
import numpy as np

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)
SR=24000; STRIDE=1920

# Text-invariant speaker encoder (Resemblyzer) — pre-computed separately
spk_emb = torch.load('/Users/asill/btrv5/runs/resemblyzer_spk.pt')
print('Loaded pre-computed speaker embeddings')
for s in spk_emb:
    print(f'  {s}: dim={spk_emb[s].shape[0]}')

def load_any(path,dur=None):
    data,sr=sf.read(path)
    if sr!=SR: data=signal.resample(data,int(len(data)*SR/sr),axis=0)
    if dur is not None: L=dur*SR-(dur*SR%STRIDE); data=data[:L]
    else: L=len(data)-(len(data)%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

class CodeGenerator(nn.Module):
    """
    Bidirectional transformer: LV0 + speaker → LV1-7 codes.
    
    Input: LV0 codes (B, T) + speaker embedding (B, 256)
    Output: LV1-7 codes (B, 7, T)
    """
    def __init__(self, vocab=2048, lv0_dim=256, spk_dim=256, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.vocab = vocab
        
        # Embeddings
        self.lv0_emb = nn.Embedding(vocab, lv0_dim)
        self.spk_proj = nn.Linear(spk_dim, lv0_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        
        # Input projection
        self.input_proj = nn.Linear(lv0_dim * 2, d_model)
        
        # Bidirectional transformer (NO causal mask!)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads: predict code for each LV1-7 level independently
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab) for _ in range(7)])
    
    def forward(self, lv0, spk_emb):
        """
        lv0: (B, T) integers 0..2047
        spk_emb: (B, 256) Resemblyzer embedding
        Returns: (B, 7, T, vocab) logits
        """
        B, T = lv0.shape
        
        # Embed and combine
        lv0_e = self.lv0_emb(lv0)  # (B, T, D)
        spk_e = self.spk_proj(spk_emb).unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
        h = torch.cat([lv0_e, spk_e], dim=-1)  # (B, T, 2D)
        h = self.input_proj(h)  # (B, T, d_model)
        h = h + self.pos_emb[:, :T, :]
        
        # Bidirectional transformer
        h = self.transformer(h)  # (B, T, d_model) — sees ALL positions
        
        # Predict each LV1-7 level
        logits = torch.stack([head(h) for head in self.heads], dim=1)  # (B, 7, T, vocab)
        return logits
    
    def predict(self, lv0, spk_emb):
        """Generate codes (argmax)."""
        logits = self.forward(lv0, spk_emb)
        return logits.argmax(dim=-1)  # (B, 7, T)

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
spks=['p225','p226','p227','p228','p229']; utts=['001','002','003','004','005']

# Cache LV0 codes + LV1-7 ground truth
print('Caching codes...')
cache = {}
with torch.no_grad():
    for s in spks:
        for u in utts:
            a = load_any(f'{base}/{s}/{s}_{u}_mic1.flac')
            codes = mimi.encode(a)  # (1, 8, T)
            cache[(s,u)] = codes
T = min(c.shape[2] for c in cache.values())
cache = {k: v[:,:,:T] for k, v in cache.items()}
print(f'Cached {len(cache)} entries, T={T}')

# Training
gen = CodeGenerator()
opt = torch.optim.AdamW(gen.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()

pairs = [(s,t,u) for s in spks for t in spks for u in utts if s!=t]
random.shuffle(pairs)
print(f'{len(pairs)} training pairs')

print('Training 100 steps...')
t0 = time.time()
for step in range(100):
    random.shuffle(pairs); lt = 0; total_acc = 0
    for s,t,u in pairs:
        lv0 = cache[(s,u)][:, 0, :]    # (1, T) source LV0
        lv1_7_gt = cache[(t,u)][:, 1:, :]  # (1, 7, T) target LV1-7
        
        spk = spk_emb[t].unsqueeze(0)  # (1, 256)
        logits = gen(lv0, spk)  # (1, 7, T, 2048)
        
        # CE loss per level
        loss = sum(ce(logits[0, i].reshape(-1, 2048), lv1_7_gt[0, i].reshape(-1)) for i in range(7))
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0); opt.step()
        lt += loss.item()
        
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == lv1_7_gt).float().mean().item()
        total_acc += acc
    
    if step % 20 == 0:
        n = len(pairs)
        print(f'  step {step:3d}: loss={lt/n:.4f} acc={total_acc/n:.4f} [{time.time()-t0:.1f}s]')

print(f'Done [{time.time()-t0:.1f}s]')

# VC Test
print('\
Testing VC...')
subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1','-sample_fmt','s16','/tmp/tc.wav'],capture_output=True)

def convert(src_a, tgt_a_np):
    with torch.no_grad():
        codes_src = mimi.encode(src_a)
        lv0 = codes_src[:, 0, :]  # (1, T)
        
        spk = spk_emb['p226'].unsqueeze(0)  # (1, 256) use p226 as target
        pred_lv1_7 = gen.predict(lv0, spk)  # (1, 7, T)
        
        codes_vc = torch.cat([lv0.unsqueeze(1), pred_lv1_7], dim=1)
        return mimi.decode(codes_vc)

src_a = load_any(f'{base}/p225/p225_001_mic1.flac')
tgt_p = load_any(f'{base}/p226/p226_001_mic1.flac')
tgt_c = load_any('/tmp/tc.wav', dur=None)

out = '/Users/asill/research5'
for nm, ta in [('parallel', tgt_p), ('cross', tgt_c)]:
    va = convert(src_a, ta)
    Tc = min(va.shape[2], src_a.shape[2], ta.shape[2])
    zv = mimi.encode_to_latent(va[:,:,:Tc], quantize=False)
    zs = mimi.encode_to_latent(src_a[:,:,:Tc], quantize=False)
    zt = mimi.encode_to_latent(ta[:,:,:Tc], quantize=False)
    T2 = min(zv.shape[2], zs.shape[2], zt.shape[2])
    cs = F.cosine_similarity(zv[:,:,:T2].reshape(-1), zs[:,:,:T2].reshape(-1), dim=0)
    ct = F.cosine_similarity(zv[:,:,:T2].reshape(-1), zt[:,:,:T2].reshape(-1), dim=0)
    print(f'{nm}: cos_src={cs:.4f} cos_tgt={ct:.4f} Δ={ct-cs:+.4f}')
    sf.write(f'{out}/codex_{nm}.wav', va[0,0,:Tc].numpy(), SR)
print('✅')

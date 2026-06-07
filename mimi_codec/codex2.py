"""Codex Architecture — scaled training: 10 speakers, 300 steps."""
import sys,os,random; sys.path.insert(0,'/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, subprocess, time
from moshi.models import loaders; from pathlib import Path
from scipy import signal
mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)
SR=24000; STRIDE=1920

spk_emb = torch.load('/Users/asill/btrv5/runs/resemblyzer_spk.pt')
# Add more speakers — compute on the fly if needed
spks = ['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234']
utts = ['001','002','003','004','005']

# For speakers not in spk_emb, use nearest neighbor or just skip
available_spks = [s for s in spks if s in spk_emb]
print(f'Speakers with embeddings: {available_spks}')

class CodeGen(nn.Module):
    def __init__(self, vocab=2048):
        super().__init__()
        self.lv0_emb = nn.Embedding(vocab, 128)
        self.spk_proj = nn.Linear(256, 128)
        self.input_proj = nn.Linear(256, 256)
        self.pos = nn.Parameter(torch.randn(1, 256, 256) * 0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512,
                dropout=0.1, activation='gelu', batch_first=True, norm_first=True),
            num_layers=3)
        self.heads = nn.ModuleList([nn.Linear(256, vocab) for _ in range(7)])
    
    def forward(self, lv0, spk):
        B, T = lv0.shape
        h = torch.cat([self.lv0_emb(lv0), self.spk_proj(spk).unsqueeze(1).expand(-1,T,-1)], dim=-1)
        h = self.input_proj(h) + self.pos[:, :T, :]
        h = self.transformer(h)
        return torch.stack([hd(h) for hd in self.heads], dim=1)

def load_any(path,dur=None):
    data,sr=sf.read(path)
    if sr!=SR: data=signal.resample(data,int(len(data)*SR/sr),axis=0)
    if dur is not None: L=dur*SR-(dur*SR%STRIDE); data=data[:L]
    else: L=len(data)-(len(data)%STRIDE); data=data[:L]
    if data.ndim>1: data=data.mean(axis=1)
    return torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

# Cache codes
print('Caching...')
cache = {}
with torch.no_grad():
    for s in available_spks:
        for u in utts:
            f = f'{base}/{s}/{s}_{u}_mic1.flac'
            if not os.path.isfile(f): continue
            a = load_any(f)
            cache[(s,u)] = mimi.encode(a)
T = min(c.shape[2] for c in cache.values())
cache = {k: v[:,:,:T] for k, v in cache.items()}
print(f'Cached {len(cache)} entries, T={T}')

gen = CodeGen()
opt = torch.optim.AdamW(gen.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()

pairs = [(s,t,u) for s in available_spks for t in available_spks for u in utts if s!=t and (s,u) in cache and (t,u) in cache]
random.shuffle(pairs)
print(f'{len(pairs)} training pairs')

print('Training 200 steps...')
t0 = time.time()
best_acc = 0
for step in range(200):
    random.shuffle(pairs)
    lt = 0; total_acc = 0
    for s,t,u in pairs:
        lv0 = cache[(s,u)][:, 0, :]
        gt = cache[(t,u)][:, 1:, :]
        logits = gen(lv0, spk_emb[t].unsqueeze(0))
        loss = sum(ce(logits[0,i].reshape(-1,2048), gt[0,i].reshape(-1)) for i in range(7))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0); opt.step()
        lt += loss.item()
        total_acc += (logits.argmax(dim=-1) == gt).float().mean().item()
    
    avg_acc = total_acc / len(pairs)
    if avg_acc > best_acc:
        best_acc = avg_acc
        torch.save(gen.state_dict(), '/Users/asill/btrv5/runs/codex_best.pt')
    
    if step % 20 == 0:
        print(f'  step {step:3d}: loss={lt/len(pairs):.4f} acc={avg_acc:.4f} best={best_acc:.4f} [{time.time()-t0:.1f}s]')

print(f'Done [{time.time()-t0:.1f}s], best_acc={best_acc:.4f}')

# Load best
gen.load_state_dict(torch.load('/Users/asill/btrv5/runs/codex_best.pt'))

# Test
subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1','-sample_fmt','s16','/tmp/tcx2.wav'],capture_output=True)
out='/Users/asill/research5'

def test_vc(src_spk, tgt_spk, tgt_audio, label):
    src_a = load_any(f'{base}/{src_spk}/{src_spk}_001_mic1.flac')
    with torch.no_grad():
        cs = mimi.encode(src_a)
        lv0 = cs[:, 0, :]
        spk = spk_emb[tgt_spk].unsqueeze(0)
        pred = gen.forward(lv0, spk).argmax(dim=-1)
        cv = torch.cat([lv0.unsqueeze(1), pred], dim=1)
        va = mimi.decode(cv)
    Tc = min(va.shape[2], src_a.shape[2], tgt_audio.shape[2])
    zv = mimi.encode_to_latent(va[:,:,:Tc], quantize=False)
    zs = mimi.encode_to_latent(src_a[:,:,:Tc], quantize=False)
    zt = mimi.encode_to_latent(tgt_audio[:,:,:Tc], quantize=False)
    T2 = min(zv.shape[2], zs.shape[2], zt.shape[2])
    cs = F.cosine_similarity(zv[:,:,:T2].reshape(-1), zs[:,:,:T2].reshape(-1), dim=0)
    ct = F.cosine_similarity(zv[:,:,:T2].reshape(-1), zt[:,:,:T2].reshape(-1), dim=0)
    print(f'{label}: cos_src={cs:.4f} cos_tgt={ct:.4f} Δ={ct-cs:+.4f}')
    sf.write(f'{out}/codex2_{label}.wav', va[0,0,:Tc].numpy(), SR)

# Parallel: p225 → p226
tgt_p = load_any(f'{base}/p226/p226_001_mic1.flac')
test_vc('p225', 'p226', tgt_p, 'parallel')

# Cross-text: p225 → origin.mp3 (use p226 speaker as proxy since no origin embedding)
tgt_c = load_any('/tmp/tcx2.wav', dur=None)
test_vc('p225', 'p226', tgt_c, 'cross')

# Multi-speaker parallel test
for tgt_s in ['p227','p228','p229','p230']:
    if tgt_s in spk_emb:
        tgt_a = load_any(f'{base}/{tgt_s}/{tgt_s}_001_mic1.flac')
        test_vc('p225', tgt_s, tgt_a, f'p225→{tgt_s}')

print(f'✅ {out}/codex2_*.wav')

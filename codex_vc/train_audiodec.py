"""AudioDec VC — scaled training with Resemblyzer + proper splits."""
import sys, os, random; sys.path.insert(0, '.'); sys.path.insert(0, '/Users/asill/btrvrc0')
import torch, torch.nn.functional as F, soundfile as sf, subprocess, time
from v3lite.codec_audiodec import AudioDecCodec
from scipy import signal
import numpy as np
from codex_vc.audiodec_vc import AudioDecVC

# ── Config ──
SR = 48000; HOP = 300
DUR = 1.5
SPKS = [f'p{i}' for i in range(225, 256)]  # p225-p255
UTTS = ['001','002','003','004','005']
STEPS = 300
VAL_RATIO = 0.2
# ────────────

def load(path, dur=DUR):
    d, sr = sf.read(path)
    if sr != SR: d = signal.resample(d, int(len(d)*SR/sr), axis=0)
    L = int(dur*SR) - (int(dur*SR) % HOP); d = d[:L]
    if d.ndim > 1: d = d.mean(axis=1)
    return torch.from_numpy(d).float()

base = '/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

# ── Load codec ──
print('Loading AudioDec...')
codec = AudioDecCodec(device='cpu')
for p in codec.parameters(): p.requires_grad_(False)
vc = AudioDecVC(codec)
opt = torch.optim.AdamW(vc.parameters(), lr=5e-4)

# ── Cache latents ──
print('Caching...')
cache = {}
available_spks = []
for s in SPKS:
    for u in UTTS:
        f = f'{base}/{s}/{s}_{u}_mic1.flac'
        if not os.path.isfile(f): continue
        try:
            cache[(s, u)] = vc.encode(load(f))
            if s not in available_spks: available_spks.append(s)
        except: pass

T = min(z.shape[0] for z in cache.values())
cache = {k: v[:T] for k, v in cache.items()}
print(f'  {len(cache)} latents from {len(available_spks)} speakers, T={T}')

# ── Resemblyzer speaker embeddings ──
print('Loading Resemblyzer...')
from resemblyzer import VoiceEncoder
ve = VoiceEncoder()
spk_emb = {}
for s in available_spks:
    embs = []
    for u in UTTS[:3]:
        if (s, u) not in cache: continue
        z = cache[(s, u)].numpy()
        # Resemblyzer needs 16kHz raw audio, but we only have latent.
        # Decode to audio first
        a = codec.decode(z)
        a_16k = signal.resample(a, int(len(a)*16000/SR))
        embs.append(ve.embed_utterance(a_16k.astype(np.float32)))
    if embs:
        spk_emb[s] = torch.from_numpy(np.stack(embs).mean(0)).float()

print(f'  {len(spk_emb)} speaker embeddings')

# ── Train/Val split by utterance ──
all_utts = sorted({k[1] for k in cache})
random.seed(42); random.shuffle(all_utts)
n_val = max(1, int(len(all_utts) * VAL_RATIO))
val_utts = set(all_utts[:n_val])
train_utts = set(all_utts[n_val:])

def make_pairs(utt_set):
    ps = []
    for u in utt_set:
        sw = [s for s in available_spks if (s, u) in cache and s in spk_emb]
        for s in sw:
            for t in sw:
                if s != t:
                    ps.append((s, t, u))
    return ps

train_pairs = make_pairs(train_utts)
val_pairs = make_pairs(val_utts)
random.shuffle(train_pairs)
print(f'  Train: {len(train_pairs)}, Val: {len(val_pairs)}')

# ── Training ──
print()
print(f'Training {STEPS} steps...')
t0 = time.time()
best_val = float('inf')
for step in range(STEPS):
    random.shuffle(train_pairs)
    tl = 0; n = 0

    for s, t, u in train_pairs[:200]:  # limit per step for speed
        z_s = cache[(s, u)].unsqueeze(0)
        z_t = cache[(t, u)].unsqueeze(0)
        s_tgt = spk_emb[t]
        loss = vc.training_step(z_s, z_t, s_tgt.unsqueeze(0))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(vc.parameters(), 1.0); opt.step()
        tl += loss.item(); n += 1

    if step % 30 == 0:
        vc.eval()
        vl = 0; vn = 0
        for s, t, u in val_pairs[:50]:
            z_s = cache[(s, u)].unsqueeze(0)
            z_t = cache[(t, u)].unsqueeze(0)
            s_tgt = spk_emb[t]
            vl += vc.training_step(z_s, z_t, s_tgt.unsqueeze(0)).item(); vn += 1
        vc.train()

        avg_val = vl / max(vn, 1)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(vc.state_dict(), 'runs/audiodec_vc.pt')

        print(f'  step {step:4d}: train={tl/n:.4f} val={avg_val:.4f} best={best_val:.4f} [{time.time()-t0:.0f}s]')

print()
print(f'Done! Best val={best_val:.4f} [{time.time()-t0:.0f}s]')

# Load best
vc.load_state_dict(torch.load('runs/audiodec_vc.pt'))

# ── Test ──
subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','48000','-ac','1','-sample_fmt','s16','-t','2','/tmp/ad_final.wav'], capture_output=True)

def load24(path, dur=None):
    d, sr = sf.read(path)
    if sr != SR: d = signal.resample(d, int(len(d)*SR/sr), axis=0)
    if dur is not None: L = int(dur*SR) - (int(dur*SR) % HOP); d = d[:L]
    else: L = len(d) - (len(d) % HOP); d = d[:L]
    if d.ndim > 1: d = d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)

src = load24(f'{base}/p225/p225_001_mic1.flac')
tgt_origin = load24('/tmp/ad_final.wav')

with torch.no_grad():
    s_origin = vc.encode(tgt_origin.squeeze()).mean(0)

print()
print('Results:')
out = '/Users/asill/research5'
for nm, spk in [('p226', spk_emb.get('p226', s_origin)), ('origin', s_origin)]:
    if nm == 'p226' and 'p226' not in spk_emb:
        print(f'  {nm}: skip (no embedding)')
        continue
    vca = vc.convert(src, spk)
    Tc = min(len(vca), src.shape[2])
    zv = vc.encode(vca[:Tc]).unsqueeze(0)
    zs = vc.encode(src.squeeze()[:Tc]).unsqueeze(0)
    tgt_a = load24(f'{base}/p226/p226_001_mic1.flac') if nm == 'p226' else tgt_origin
    zt = vc.encode(tgt_a.squeeze()[:Tc]).unsqueeze(0)
    T2 = min(zv.shape[1], zs.shape[1], zt.shape[1])
    cs = F.cosine_similarity(zv[0,:T2].reshape(-1), zs[0,:T2].reshape(-1), dim=0)
    ct = F.cosine_similarity(zv[0,:T2].reshape(-1), zt[0,:T2].reshape(-1), dim=0)
    print(f'  {nm}: cos_src={cs:.4f} cos_tgt={ct:.4f} Δ={ct-cs:+.4f}')
    sf.write(f'{out}/ad_final_{nm}.wav', vca[:Tc].numpy(), SR)

print(f'✅ {out}/ad_final_*.wav')

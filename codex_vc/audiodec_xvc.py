"""
AudioDec Cross-Text VC: Content extractor + Resemblyzer speaker + differentiable decoder.

Key: AudioDec decode is DIFFERENTIABLE — gradient flows through decoder to converter.
"""
import sys, os, random; sys.path.insert(0, '.'); sys.path.insert(0, '/Users/asill/btrvrc0')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, subprocess, time
from v3lite.codec_audiodec import AudioDecCodec
from scipy import signal
import numpy as np

SR = 48000; HOP = 300
STRIDE = HOP


# ── Model ────────────────────────────────────────────────────────────────

class ContentExtractor(nn.Module):
    """Extract speaker-independent content from AudioDec latent (bottleneck 64→8)."""
    def __init__(self, dim=64, bottleneck=8):
        super().__init__()
        self.compress = nn.Conv1d(dim, bottleneck, 1)
        self.expand = nn.Conv1d(bottleneck, dim, 1)
    def forward(self, z):
        """z: (B, D, T) → (B, D, T)"""
        h = self.compress(z)
        h = F.gelu(h)
        h = self.expand(h)
        return z + h  # residual


class Converter(nn.Module):
    """Content + speaker → VC latent (FiLM conditioning)."""
    def __init__(self, dim=64, spk_dim=256):
        super().__init__()
        self.spk_proj = nn.Sequential(nn.Linear(spk_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        self.refine = nn.Sequential(
            nn.Conv1d(dim, dim*2, 5, padding=2), nn.GELU(),
            nn.Conv1d(dim*2, dim, 5, padding=2),
        )
    def forward(self, c_src, s_tgt):
        """c_src: (B, D, T), s_tgt: (B, spk_dim) → (B, D, T)"""
        sp = self.spk_proj(s_tgt)
        g = self.gamma(sp).unsqueeze(-1)
        b = self.beta(sp).unsqueeze(-1)
        m = c_src.mean(2, keepdim=True)
        st = c_src.std(2, keepdim=True) + 1e-5
        c_mod = (c_src - m) / st * g + b
        return c_src + self.refine(c_mod)


class AudioDecXVC(nn.Module):
    """AudioDec cross-text VC with differentiable decoder."""
    def __init__(self, codec, spk_dim=256):
        super().__init__()
        self.codec = codec
        for p in codec.parameters(): p.requires_grad_(False)
        self.content_ext = ContentExtractor(dim=64, bottleneck=8)
        self.converter = Converter(dim=64, spk_dim=spk_dim)

    def encode(self, audio):
        """(T_audio,) → (T_lat, 64)"""
        with torch.no_grad():
            return self.codec.encode(audio)

    def decode(self, z):
        """(..., T, 64) → (T_audio,) — DIFFERENTIABLE"""
        if z.dim() == 3: z = z.squeeze(0)
        return self.codec.decode(z)

    def forward(self, z_src, s_tgt):
        """z_src: (B, D, T) or (B, T, D), s_tgt: (B, spk_dim) → z_vc: (B, D, T)"""
        # Ensure (B, D, T) for conv
        if z_src.shape[-1] in [64, 8]:  # (B, T, D)
            z_src = z_src.transpose(1, 2)
        c_src = self.content_ext(z_src)
        return self.converter(c_src, s_tgt)

    def training_step(self, z_src, tgt_audio, s_tgt):
        """
        Train with audio reconstruction through differentiable decoder.
        z_src:  (B, D, T) source latent
        tgt_audio: (T_audio,) target audio
        s_tgt:  (B, spk_dim) target speaker embedding
        """
        # Ensure (B, D, T) format
        if z_src.shape[-1] == 64:
            z_src = z_src.transpose(1, 2)

        z_vc = self.forward(z_src, s_tgt)  # (B, D, T)

        # Decode through differentiable decoder
        z_vc_2d = z_vc.squeeze(0).transpose(0, 1)  # (T, 64)
        audio_vc = self.decode(z_vc_2d)  # (T_audio,)

        T = min(len(audio_vc), len(tgt_audio))
        loss_audio = F.mse_loss(audio_vc[:T], tgt_audio[:T])

        return loss_audio

    @torch.no_grad()
    def convert(self, src_audio, s_tgt):
        """Full VC pipeline."""
        z_src = self.encode(src_audio)  # (T, 64)
        z_src_b = z_src.unsqueeze(0)  # (1, T, 64)
        z_vc = self.forward(z_src_b, s_tgt.unsqueeze(0))  # (1, D, T)
        z_vc_2d = z_vc.squeeze(0).transpose(0, 1)  # (T, 64)
        return self.decode(z_vc_2d)


# ── Training ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from resemblyzer import VoiceEncoder
    ve = VoiceEncoder()

    def get_spk(audio_np):
        a = audio_np[:SR*5]
        if len(a) < 1600: a = np.pad(a, (0, 1600 - len(a)))
        a16 = signal.resample(a, max(1, int(len(a)*16000/SR)))
        return torch.from_numpy(ve.embed_utterance(a16.astype(np.float32))).float()

    def load_audio(path, dur=1.5):
        d, sr = sf.read(path)
        if sr != SR: d = signal.resample(d, int(len(d)*SR/sr), axis=0)
        if dur is not None:
            L = int(dur*SR) - (int(dur*SR) % HOP)
            d = d[:L]
        else:
            L = len(d) - (len(d) % HOP)
            d = d[:L]
        if d.ndim > 1: d = d.mean(axis=1)
        return torch.from_numpy(d).float()

    # Load codec
    print("Loading AudioDec...")
    codec = AudioDecCodec(device='cpu')
    vc = AudioDecXVC(codec)
    opt = torch.optim.AdamW(vc.parameters(), lr=5e-4)

    # Data
    base = '/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
    spks = ['p225', 'p226', 'p227', 'p228', 'p229']
    utts = ['001', '002', '003']

    # Resemblyzer speaker embeddings
    print("Resemblyzer embeddings...")
    spk_r = {}
    for s in spks:
        embs = [get_spk(load_audio(f'{base}/{s}/{s}_{u}_mic1.flac').numpy()) for u in utts]
        spk_r[s] = torch.stack(embs).mean(0)

    # Cache latents + audio
    print("Caching...")
    cache = {}
    with torch.no_grad():
        for s in spks:
            for u in utts:
                a = load_audio(f'{base}/{s}/{s}_{u}_mic1.flac')
                cache[(s, u)] = (vc.encode(a), a)

    # Train: same-text different-speaker pairs
    pairs = [(s, t, u) for u in utts for s in spks for t in spks if s != t]
    random.shuffle(pairs)
    print(f"{len(pairs)} training pairs")

    print("Training with audio loss (differentiable decoder)...")
    t0 = time.time()
    for step in range(30):
        random.shuffle(pairs)
        tl = 0; n = 0
        for s, t, u in pairs[:10]:  # limit for speed (decode is heavy)
            z_src, src_a = cache[(s, u)]
            z_src = z_src.unsqueeze(0)
            if z_src.shape[-1] == 64: z_src = z_src.transpose(1, 2)

            tgt_a = cache[(t, u)][1]
            s_tgt = spk_r[t].unsqueeze(0)

            loss = vc.training_step(z_src, tgt_a, s_tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vc.parameters(), 1.0)
            opt.step()
            tl += loss.item(); n += 1

        if step % 5 == 0:
            print(f"  step {step:3d}: loss={tl/n:.4f} [{time.time()-t0:.0f}s]")

    print(f"Done [{time.time()-t0:.0f}s]")

    # Test p255→origin
    subprocess.run(['ffmpeg', '-y', '-i', '/Users/asill/Downloads/origin.mp3',
                    '-ar', '48000', '-ac', '1', '-sample_fmt', 's16', '/tmp/ad_xvc.wav'],
                   capture_output=True)

    tgt_o = load_audio('/tmp/ad_xvc.wav', dur=None)
    s_origin = get_spk(tgt_o.numpy())

    src_s = 'p255' if os.path.isfile(f'{base}/p255/p255_001_mic1.flac') else 'p225'
    src_a = load_audio(f'{base}/{src_s}/{src_s}_001_mic1.flac').unsqueeze(0)

    from codex_vc.metrics import compute_all_metrics, format_metrics

    with torch.no_grad():
        vca = vc.convert(src_a, s_origin)
        Tc = min(len(vca), len(src_a.squeeze()), len(tgt_o))
        vca_np = vca[:Tc].numpy() if torch.is_tensor(vca) else vca[:Tc]

        # Re-encode for cosine metrics (encode expects tensor)
        zv_t = vc.encode(torch.from_numpy(vca_np).float())
        zs_t = vc.encode(src_a.squeeze()[:Tc])
        zt_t = vc.encode(tgt_o[:Tc])
        T2 = min(zv_t.shape[0], zs_t.shape[0], zt_t.shape[0])
        cs = F.cosine_similarity(zv_t[:T2].reshape(-1), zs_t[:T2].reshape(-1), dim=0)
        ct = F.cosine_similarity(zv_t[:T2].reshape(-1), zt_t[:T2].reshape(-1), dim=0)

        m = compute_all_metrics(src_a.squeeze().numpy()[:Tc], tgt_o.numpy()[:Tc], vca_np, SR)
        m['cos_src'] = cs.item(); m['cos_tgt'] = ct.item()
        m['delta'] = ct.item() - cs.item()

        print()
        print(f"=== {src_s} -> origin (AudioDec XVC) ===")
        print(format_metrics(m))
        sf.write(f'research5/ad_xvc_{src_s}.wav', vca_np, SR)
        print("Done")

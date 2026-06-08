"""Test Resemblyzer + Q-Space Converter for cross-text VC."""
import sys, os; sys.path.insert(0, '.')
import torch, torch.nn.functional as F, soundfile as sf, subprocess
from moshi.models import loaders; from pathlib import Path
from scipy import signal
from codex_vc.metrics import compute_all_metrics, format_metrics

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)
SR = 24000; STRIDE = 1920

spk_r = torch.load('runs/resemblyzer_5spk.pt')

class QSConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spk_proj = torch.nn.Linear(256, 512)
        self.gamma = torch.nn.Linear(512, 512)
        self.beta = torch.nn.Linear(512, 512)
        self.ref = torch.nn.Conv1d(512, 512, kernel_size=3, padding=1)
    def forward(self, zq_src, s_tgt):
        sp = self.spk_proj(s_tgt)
        g = self.gamma(sp).unsqueeze(-1)
        b = self.beta(sp).unsqueeze(-1)
        if zq_src.shape[2] % 2 != 0:
            zq_src = F.pad(zq_src, (0, 1))
        m = zq_src.mean(2, keepdim=True)
        st = zq_src.std(2, keepdim=True) + 1e-5
        return zq_src + self.ref((zq_src - m) / st * g + b)

cv = QSConv()
cv.load_state_dict(torch.load('runs/qsconv_res.pt'))
cv.eval()

def load_any(path, dur=None):
    d, sr = sf.read(path)
    if sr != SR: d = signal.resample(d, int(len(d) * SR / sr), axis=0)
    if dur is not None: L = dur * SR - (dur * SR % STRIDE); d = d[:L]
    else: L = len(d) - (len(d) % STRIDE); d = d[:L]
    if d.ndim > 1: d = d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)

subprocess.run(['ffmpeg', '-y', '-i', '/Users/asill/Downloads/origin.mp3',
                '-ar', '24000', '-ac', '1', '-sample_fmt', 's16', '/tmp/rtest.wav'],
               capture_output=True)

base = '/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
src_s = 'p255' if os.path.isfile(f'{base}/p255/p255_001_mic1.flac') else 'p225'
src_a = load_any(f'{base}/{src_s}/{src_s}_001_mic1.flac')
tgt_a = load_any('/tmp/rtest.wav', dur=None)

with torch.no_grad():
    z_src = mimi.encode_to_latent(src_a, quantize=False)
    codes = mimi.quantizer.encode(z_src)
    zq_src = mimi.quantizer.decode(codes)
    zvc = cv(zq_src, spk_r['origin'].unsqueeze(0))
    zu = mimi._to_encoder_framerate(zvc)
    if mimi.decoder_transformer:
        (z_tr,) = mimi.decoder_transformer(zu)
    else:
        z_tr = zu
    vc = mimi.decoder(z_tr)

    Tc = min(vc.shape[2], src_a.shape[2], tgt_a.shape[2])
    zv = mimi.encode_to_latent(vc[:, :, :Tc], quantize=False)
    zs = mimi.encode_to_latent(src_a[:, :, :Tc], quantize=False)
    zt = mimi.encode_to_latent(tgt_a[:, :, :Tc], quantize=False)
    T2 = min(zv.shape[2], zs.shape[2], zt.shape[2])
    cs = F.cosine_similarity(zv[:, :, :T2].reshape(-1), zs[:, :, :T2].reshape(-1), dim=0)
    ct = F.cosine_similarity(zv[:, :, :T2].reshape(-1), zt[:, :, :T2].reshape(-1), dim=0)

    m = compute_all_metrics(
        src_a.squeeze()[:Tc].numpy(),
        tgt_a.squeeze()[:Tc].numpy(),
        vc[0, 0, :Tc].numpy(), SR
    )
    m['cos_src'] = cs.item()
    m['cos_tgt'] = ct.item()
    m['delta'] = ct.item() - cs.item()

    print(f'=== {src_s} -> origin (Resemblyzer + Q-Space) ===')
    print(format_metrics(m))
    sf.write(f'research5/rq_{src_s}_origin.wav', vc[0, 0, :Tc].numpy(), SR)
    print('Done')

"""Test Prompt VC with multi-metrics."""
import sys, os; sys.path.insert(0, '.')
import torch, torch.nn as nn, soundfile as sf, subprocess
from moshi.models import loaders; from pathlib import Path
from scipy import signal
from codex_vc.prompt_vc import PromptEncoder
from codex_vc.metrics import compute_all_metrics, format_metrics

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)

class LightPromptConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab = 2048
        self.lv0_emb = nn.Embedding(2048, 64)
        self.prompt_proj = nn.Linear(512, 64)
        self.pos = nn.Parameter(torch.randn(1, 256, 256) * 0.02)
        self.inp = nn.Linear(128, 256)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512,
                dropout=0.1, activation='gelu', batch_first=True, norm_first=True
            ), num_layers=2)
        self.heads = nn.ModuleList([nn.Linear(256, 2048) for _ in range(7)])

    def forward(self, lv0, pf):
        B, T = lv0.shape
        h = torch.cat([self.lv0_emb(lv0),
                       self.prompt_proj(pf).unsqueeze(1).expand(-1, T, -1)], dim=-1)
        h = self.inp(h) + self.pos[:, :T, :]
        m = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
        return torch.stack([hd(self.transformer(h, mask=m)) for hd in self.heads], dim=1)

    def predict(self, lv0, pf):
        return self.forward(lv0, pf).argmax(-1)

converter = LightPromptConv()
converter.load_state_dict(torch.load('runs/prompt_vc_light.pt', weights_only=True))
converter.eval()

prompt_enc = PromptEncoder(mimi)

SR = 24000; STRIDE = 1920
def load_any(path, dur=None):
    d, sr = sf.read(path)
    if sr != SR: d = signal.resample(d, int(len(d)*SR/sr), axis=0)
    if dur is not None: L = dur*SR - (dur*SR % STRIDE); d = d[:L]
    else: L = len(d) - (len(d) % STRIDE); d = d[:L]
    if d.ndim > 1: d = d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)

subprocess.run(['ffmpeg', '-y', '-i', '/Users/asill/Downloads/origin.mp3',
                '-ar', '24000', '-ac', '1', '-sample_fmt', 's16', '-t', '3',
                '/tmp/p5.wav'], capture_output=True)
tgt_o = load_any('/tmp/p5.wav', dur=None)

base = '/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
out = '/Users/asill/research5'

with torch.no_grad():
    prompt_o = prompt_enc(tgt_o)

    for src_s in ['p225', 'p248', 'p255']:
        f = f'{base}/{src_s}/{src_s}_001_mic1.flac'
        if not os.path.isfile(f): continue
        src_a = load_any(f)

        cs = mimi.encode(src_a)
        lv0 = cs[:, 0, :]
        pred = converter.predict(lv0, prompt_o)
        cv = torch.cat([lv0.unsqueeze(1), pred], dim=1)
        vc = mimi.decode(cv)

        Tc = min(vc.shape[2], src_a.shape[2], tgt_o.shape[2])
        zv = mimi.encode_to_latent(vc[:, :, :Tc], quantize=False)
        zs = mimi.encode_to_latent(src_a[:, :, :Tc], quantize=False)
        zt = mimi.encode_to_latent(tgt_o[:, :, :Tc], quantize=False)
        T2 = min(zv.shape[2], zs.shape[2], zt.shape[2])
        cs2 = torch.nn.functional.cosine_similarity(
            zv[:, :, :T2].reshape(-1), zs[:, :, :T2].reshape(-1), dim=0)
        ct = torch.nn.functional.cosine_similarity(
            zv[:, :, :T2].reshape(-1), zt[:, :, :T2].reshape(-1), dim=0)

        m = compute_all_metrics(
            src_a.squeeze()[:Tc].numpy(),
            tgt_o.squeeze()[:Tc].numpy(),
            vc[0, 0, :Tc].numpy(), SR
        )
        m['cos_src'] = cs2.item()
        m['cos_tgt'] = ct.item()
        m['delta'] = ct.item() - cs2.item()

        print(f'=== {src_s} -> origin ===')
        print(format_metrics(m))
        sf.write(f'{out}/prompt_vc_{src_s}.wav', vc[0, 0, :Tc].numpy(), SR)
        print()

print('Done')

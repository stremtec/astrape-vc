"""HuBERT layer speaker probe — all layers."""
import sys,os,random; sys.path.insert(0,'.')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, time, numpy as np
from transformers import HubertModel
from scipy import signal

hubert = HubertModel.from_pretrained('facebook/hubert-base-ls960', output_hidden_states=True).eval()

base='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
spks = sorted([d for d in os.listdir(base) if os.path.isdir(f'{base}/{d}') and d.startswith('p')])[:30]
spk_to_idx = {s:i for i,s in enumerate(spks)}
n_spk = len(spks)

samples = []
for s in spks:
    uts = sorted([f for f in os.listdir(f'{base}/{s}') if f.endswith('.flac')])[:3]
    for u in uts:
        d,sr = sf.read(f'{base}/{s}/{u}')
        if sr!=16000: d = signal.resample(d, int(len(d)*16000/sr), axis=0)
        if d.ndim>1: d = d.mean(axis=1)
        samples.append((torch.from_numpy(d[:16000*2]).float().unsqueeze(0), spk_to_idx[s]))

random.shuffle(samples)
n_train = int(len(samples)*0.8)
train_s = samples[:n_train]
val_s = samples[n_train:]

class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(768, n_spk))
    def forward(self, x):
        return self.net(x.transpose(1,2))

print(f'Speakers: {n_spk}, Samples: {len(samples)}')
print()
print(f'{"Layer":>6s} {"SpkAcc":>8s} {"Info":>10s}')
print('-'*30)

for layer_idx in range(13):
    probe = Probe()
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        ft = [(hubert(x).hidden_states[layer_idx].mean(dim=1), l) for x,l in train_s]
        fv = [(hubert(x).hidden_states[layer_idx].mean(dim=1), l) for x,l in val_s]
    
    for _ in range(20):
        random.shuffle(ft)
        for f, l in ft[:30]:
            logits = probe(f.unsqueeze(0))
            loss = ce(logits, torch.tensor([l]))
            opt.zero_grad(); loss.backward(); opt.step()
    
    probe.eval()
    acc = sum(probe(f.unsqueeze(0)).argmax(-1).item() == l for f,l in fv) / len(fv)
    
    bar = '█' * int(acc * 20)
    tag = '← SPEAKER' if acc > 0.3 else ('← CONTENT' if acc < 0.1 else '')
    print(f'  {layer_idx:3d}    {acc:.4f}   {bar} {tag}')

print(f'  {"rnd":>3s}    {1/n_spk:.4f}')

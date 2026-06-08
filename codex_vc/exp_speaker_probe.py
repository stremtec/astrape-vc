"""Experiment A: Speaker Leakage Probe on Mimi latent."""
import sys, os, random; sys.path.insert(0, '.')
import torch, torch.nn as nn, torch.nn.functional as F, time
from moshi.models import loaders; from pathlib import Path

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)

cache = torch.load('runs/vctk_codes_full.pt', weights_only=True)
T = min(c.shape[2] for c in cache.values())
cache = {k: v[:, :, :T] for k, v in cache.items()}
spks = sorted({k[0] for k in cache})
spk_to_idx = {s: i for i, s in enumerate(spks)}
n_speakers = len(spks)
print(f'{len(cache)} codes, {n_speakers} speakers')

class SpeakerProbe(nn.Module):
    def __init__(self, dim=512, n_spk=109):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, 256, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, n_spk),
        )
    def forward(self, z_q):
        return self.net(z_q)

print()
print("=== Speaker Leakage Probe ===")

# Build dataset: z_q + speaker labels
samples = []
for (s, u), codes in cache.items():
    z_q = mimi.quantizer.decode(codes)
    samples.append((z_q, spk_to_idx[s]))

random.shuffle(samples)
n_train = int(len(samples) * 0.8)
train_s = samples[:n_train]
val_s = samples[n_train:]
print(f"Train: {len(train_s)}, Val: {len(val_s)}")

probe = SpeakerProbe(n_spk=n_speakers)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()

t0 = time.time()
for step in range(50):
    random.shuffle(train_s)
    tl = 0; ta = 0
    for z_q, label in train_s[:100]:
        logits = probe(z_q)
        loss = ce(logits, torch.tensor([label]))
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item()
        ta += (logits.argmax(-1).item() == label)

    if step % 10 == 0:
        probe.eval()
        va = 0
        for z_q, label in val_s[:50]:
            va += (probe(z_q).argmax(-1).item() == label)
        probe.train()
        print(f"  step {step:3d}: train_loss={tl/100:.4f} "
              f"train_acc={ta/100:.4f} val_acc={va/50:.4f} [{time.time()-t0:.0f}s]")

probe.eval()
correct = sum(probe(z_q).argmax(-1).item() == label for z_q, label in val_s)
final_acc = correct / len(val_s)
chance = 1.0 / n_speakers

print()
print(f">>> z_q speaker probe accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
print(f"    Random chance: {chance:.4f} ({chance*100:.1f}%)")
if final_acc > 0.5:
    print("    !! HIGH speaker leakage — Mimi latent NOT suitable as content!")
elif final_acc > 0.1:
    print("    ~ Moderate leakage")
else:
    print("    OK Low leakage")

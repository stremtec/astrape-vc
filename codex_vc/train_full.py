"""Full VCTK training — 109 speakers, 1000 steps."""
import sys, os, random; sys.path.insert(0, '.')
import torch, torch.nn as nn, torch.nn.functional as F, time
from codex_vc.model import CodeGenerator, compute_loss

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')

# Load data
spk_emb = torch.load('runs/vctk_full_spk.pt', weights_only=True)
cache = torch.load('runs/vctk_codes_full.pt', weights_only=True)
T = min(c.shape[2] for c in cache.values())
cache = {k: v[:, :, :T] for k, v in cache.items()}
speakers = sorted({k[0] for k in cache} & set(spk_emb.keys()))
print(f'{len(spk_emb)} embeddings, {len(cache)} codes, T={T}, {len(speakers)} speakers')

# Build pairs
pairs = []
all_utts = sorted({k[1] for k in cache})
for u in all_utts:
    sw = [s for s in speakers if (s, u) in cache]
    if len(sw) < 2:
        continue
    for s in sw:
        for t in sw:
            if s != t:
                pairs.append((s, t, u))
random.shuffle(pairs)
print(f'{len(pairs)} training pairs')

# Model
model = CodeGenerator().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# Train
STEPS = 1000
BATCH = 48
MAX_PAIRS_PER_STEP = 3000

print()
print(f"Training {STEPS} steps (batch={BATCH}, pairs/step={min(len(pairs), MAX_PAIRS_PER_STEP)})...")
t0 = time.time()
best_acc = 0.0

for step in range(STEPS):
    random.shuffle(pairs)
    step_pairs = pairs[:min(len(pairs), MAX_PAIRS_PER_STEP)]
    total_loss = 0.0
    total_acc = 0.0

    for b_start in range(0, len(step_pairs), BATCH):
        batch = step_pairs[b_start:b_start + BATCH]
        B = len(batch)
        lv0_b, lv1_7_b, spk_b = [], [], []
        for s, t, u in batch:
            lv0_b.append(cache[(s, u)][0, 0])
            lv1_7_b.append(cache[(t, u)][0, 1:])
            spk_b.append(spk_emb[t])

        lv0 = torch.stack(lv0_b).to(device).long()
        lv1_7 = torch.stack(lv1_7_b).to(device).long()
        spk = torch.stack(spk_b).to(device)

        loss = compute_loss(model, lv0, lv1_7, spk, criterion)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * B
        with torch.no_grad():
            total_acc += (model.predict(lv0, spk) == lv1_7).float().mean().item() * B

    N = len(step_pairs)
    avg_loss = total_loss / N
    avg_acc = total_acc / N

    if avg_acc > best_acc:
        best_acc = avg_acc
        torch.save(model.state_dict(), 'runs/codex_model.pt')

    if step % 100 == 0 or step == STEPS - 1:
        elapsed = time.time() - t0
        print(f"  step {step:4d}: loss={avg_loss:.4f} acc={avg_acc:.4f} "
              f"best={best_acc:.4f} [{elapsed:.0f}s]")

print()
print(f"Done! Best acc={best_acc:.4f} [{time.time()-t0:.0f}s]")

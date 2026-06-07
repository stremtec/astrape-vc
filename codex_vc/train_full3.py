"""Full VCTK training — 500 steps with flush."""
import sys, random; sys.path.insert(0, '.')
import torch, torch.nn as nn, time
from codex_vc.model import CodeGenerator, compute_loss

print("Loading...", flush=True)
spk_emb = torch.load('runs/vctk_full_spk.pt', weights_only=True)
cache = torch.load('runs/vctk_codes_full.pt', weights_only=True)
T = min(c.shape[2] for c in cache.values())
cache = {k: v[:, :, :T] for k, v in cache.items()}
speakers = sorted({k[0] for k in cache} & set(spk_emb.keys()))
print(f"  {len(spk_emb)} spk, {len(cache)} codes, T={T}, {len(speakers)} speakers", flush=True)

pairs = [(s,t,u) for u in sorted({k[1] for k in cache})
         for s in speakers for t in speakers
         if s!=t and (s,u) in cache and (t,u) in cache]
random.shuffle(pairs)
print(f"  {len(pairs)} pairs", flush=True)

model = CodeGenerator()
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

STEPS = 500
MAX_P = 500
print(f"Training {STEPS} steps (max_pairs={MAX_P})...", flush=True)
t0 = time.time()
best_acc = 0.0

for step in range(STEPS):
    random.shuffle(pairs)
    sp = pairs[:MAX_P]
    lt = 0.0; ta = 0.0
    for s, t, u in sp:
        lv0 = cache[(s, u)][:, 0, :]
        lv1 = cache[(t, u)][:, 1:, :]
        spk_e = spk_emb[t].unsqueeze(0)
        loss = compute_loss(model, lv0, lv1, spk_e, criterion)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        lt += loss.item()
        with torch.no_grad(): ta += (model.predict(lv0, spk_e) == lv1).float().mean().item()
    
    N = len(sp); al = lt/N; aa = ta/N
    if aa > best_acc:
        best_acc = aa
        torch.save(model.state_dict(), 'runs/codex_model.pt')
    
    if step % 50 == 0 or step == STEPS - 1:
        print(f"  step {step:4d}: loss={al:.4f} acc={aa:.4f} best={best_acc:.4f} [{time.time()-t0:.0f}s]", flush=True)

print(f"Done! Best acc={best_acc:.4f} [{time.time()-t0:.0f}s]", flush=True)

"""Stage A — causal acoustic model, regression (see DECODER_V8_DESIGN.md §3).

content 768@25Hz + speaker 128 → mel80 + logF0 + voicing + energy @150Hz, supervised
against the GROUND-TRUTH acoustic cache.  No teacher, no GAN — stable regression.
The mel head is teacher-forced on GT prosody during training; a short scheduled-sampling
tail (--ss-epochs) switches it to its own predicted prosody to close the exposure gap.

Loss: L1(mel) + 0.5·L1(logF0|voiced) + BCE(voicing) + 0.5·L1(energy).

  .venv/bin/python -m astrape.train_stage_a --device mps --epochs 40 \
      --out-dir /Volumes/UNTITLED/btrv5_checkpoints/v8_stage_a --num-workers 4
"""
import argparse, json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import AcousticDataset, collate_acoustic
from .decoder_v8 import AcousticModel, AcousticModelConfig

S = 44100


def losses(out, batch, device):
    mel = batch["mel"].to(device); logf0 = batch["logf0"].to(device)
    voiced = batch["voiced"].to(device); energy = batch["energy"].to(device)
    T = min(out["mel"].shape[1], mel.shape[1])
    mel_l = F.l1_loss(out["mel"][:, :T], mel[:, :T])
    v = voiced[:, :T]
    f0_l = ((out["logf0"][:, :T] - logf0[:, :T]).abs() * v).sum() / v.sum().clamp(min=1.0)
    voi_l = F.binary_cross_entropy_with_logits(out["voiced_logit"][:, :T], v)
    en_l = F.l1_loss(out["energy"][:, :T], energy[:, :T])
    return mel_l, f0_l, voi_l, en_l


@torch.no_grad()
def evaluate(model, loader, device, n_batches=20):
    model.eval()
    mc, f0e, vf1 = [], [], []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        content = batch["content"].to(device); speaker = batch["speaker"].to(device)
        out = model(content, speaker)                       # predicted prosody path
        T = min(out["mel"].shape[1], batch["mel"].shape[1])
        pm, gm = out["mel"][:, :T].cpu(), batch["mel"][:, :T]
        mc.append(F.cosine_similarity(pm.flatten(1), gm.flatten(1)).mean().item())
        v = batch["voiced"][:, :T]
        f0d = ((out["logf0"][:, :T].cpu() - batch["logf0"][:, :T]).abs() * v).sum() / v.sum().clamp(min=1.0)
        # log-Hz L1 → approx Hz error via exp; report mean |Δ log| in cents-ish
        f0e.append(f0d.item())
        pv = (torch.sigmoid(out["voiced_logit"][:, :T]).cpu() > 0.5).float()
        tp = (pv * v).sum(); fp = (pv * (1 - v)).sum(); fn = ((1 - pv) * v).sum()
        vf1.append((2 * tp / (2 * tp + fp + fn).clamp(min=1.0)).item())
    return float(np.mean(mc)), float(np.mean(f0e)), float(np.mean(vf1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ss-epochs", type=int, default=8,
                    help="Final epochs using PREDICTED prosody in the mel head (scheduled sampling).")
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-dir", type=Path, default=Path("data/mio_vctk_full_compact"))
    ap.add_argument("--content-dir", type=str, default="content_striding_8l_200hz")
    ap.add_argument("--acoustics-dir", type=str, default="acoustics_150hz")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--f0-weight", type=float, default=0.5)
    ap.add_argument("--voiced-weight", type=float, default=1.0)
    ap.add_argument("--energy-weight", type=float, default=0.5)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    # trunk size (default = teacher-replica 82M; trim for tight memory / faster epochs)
    ap.add_argument("--prenet-layers", type=int, default=6)
    ap.add_argument("--decoder-layers", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device); args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = np.load(args.data_dir / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"]); src = meta["source_files"][:n].astype(str)
    spk_names = meta["spk_names"]
    cz = np.load(args.data_dir / "spk_centroids.npz", allow_pickle=False)
    semap = {str(s): torch.from_numpy(e).float() for s, e in zip(cz["speakers"], cz["embeddings"])}
    idx = np.arange(n); np.random.default_rng(args.seed).shuffle(idx)
    split = int(n * 0.95)

    def make(indices, seed):
        ds = AcousticDataset(indices, args.data_dir / args.content_dir,
                             args.data_dir / args.acoustics_dir, src, spk_names, semap,
                             args.max_frames, seed, need_content=True, need_audio=False)
        return DataLoader(ds, args.batch_size, shuffle=True, num_workers=args.num_workers,
                          persistent_workers=args.num_workers > 0,
                          collate_fn=collate_acoustic, drop_last=True)
    loader = make(idx[:split], args.seed)
    val_loader = make(idx[split:], args.seed + 1)

    acfg = AcousticModelConfig(prenet_layers=args.prenet_layers,
                               decoder_layers=args.decoder_layers)
    model = AcousticModel(acfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"]); start = int(ck.get("epoch", -1)) + 1
    print(f"Stage A AcousticModel: {sum(p.numel() for p in model.parameters())/1e6:.1f}M  "
          f"scheduled-sampling last {args.ss_epochs} epochs", flush=True)

    t0 = time.time()
    for ep in range(start, args.epochs):
        teacher_force = ep < args.epochs - args.ss_epochs
        model.train()
        acc = {k: torch.zeros((), device=device) for k in ("mel", "f0", "voi", "en")}
        steps = skipped = 0
        for batch in loader:
            steps += 1
            if steps > args.steps_per_epoch:
                break
            content = batch["content"].to(device); speaker = batch["speaker"].to(device)
            gt_pros = None
            if teacher_force:
                gt_pros = torch.stack([batch["logf0"], batch["voiced"], batch["energy"]],
                                      dim=-1).to(device)
            out = model(content, speaker, gt_prosody=gt_pros)
            mel_l, f0_l, voi_l, en_l = losses(out, batch, device)
            loss = mel_l + args.f0_weight * f0_l + args.voiced_weight * voi_l + args.energy_weight * en_l
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); skipped += 1; continue
            opt.step()
            acc["mel"] += mel_l.detach(); acc["f0"] += f0_l.detach()
            acc["voi"] += voi_l.detach(); acc["en"] += en_l.detach()
            if steps % 100 == 0:
                d = max(steps - skipped, 1)
                print(f"E{ep:02d}[{'TF' if teacher_force else 'SS'}] {steps:04d}/{args.steps_per_epoch} "
                      f"mel={acc['mel'].item()/d:.3f} f0={acc['f0'].item()/d:.3f} "
                      f"voi={acc['voi'].item()/d:.3f} en={acc['en'].item()/d:.3f} "
                      f"{(time.time()-t0)/((ep-start)*args.steps_per_epoch+steps):.3f}s/st", flush=True)

        sch.step()
        mel_cos, f0e, vf1 = evaluate(model, val_loader, device)
        ck = {"state_dict": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(),
              "acoustic_config": acfg.__dict__, "epoch": ep}
        torch.save(ck, args.out_dir / "last.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            torch.save(ck, args.out_dir / f"epoch{ep:03d}.pt")
        (args.out_dir / "summary.json").write_text(json.dumps(
            {"epoch": ep, "tf": teacher_force, "skipped": skipped,
             "val_mel_cos": mel_cos, "val_f0_logL1": f0e, "val_voiced_f1": vf1,
             **{k: v.item() / max(steps - skipped, 1) for k, v in acc.items()}}, indent=2) + "\n")
        print(f"E{ep:02d} done  val_mel_cos={mel_cos:.4f} f0_logL1={f0e:.4f} voiced_f1={vf1:.4f}", flush=True)
    print(f"Done. {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

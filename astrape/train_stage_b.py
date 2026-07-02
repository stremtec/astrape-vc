"""Stage B — conditioned causal vocoder, adversarial (see DECODER_V8_DESIGN.md §4).

Trains ConditionedVocoder on GROUND-TRUTH acoustic conditioning → real waveform
(copy-synthesis).  No content, no speaker, no teacher — just audio + the 150Hz acoustic
cache, so it scales to any 44.1k corpus.  This is the go/no-go on causal 44.1kHz rendering.

Loss: 45·MelL1 + 1·MR-STFT + adv(MPD + MRD, LSGAN) + 2·FM.  Discriminators run on-device
(MPS torch.stft verified stable on this torch build); the finite-grad skip guard is the
backstop, and the disc sees a random ~8k-sample window (recon uses full audio).

  .venv/bin/python -m astrape.train_stage_b --device mps --epochs 60 \
      --out-dir /Volumes/UNTITLED/btrv5_checkpoints/v8_stage_b --num-workers 4
"""
import argparse, json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from .data import AcousticDataset, collate_acoustic
from .decoder_v8 import ConditionedVocoder, VocoderConfig
from .discriminators import (
    VocoderDiscriminator, discriminator_loss, generator_adv_loss, feature_matching_loss)
from .train_decoder import mrstft

S = 44100


def build_cond(batch, device):
    """(B,T,83) conditioning = [mel80 | logf0 | voiced | energy] from GT acoustics."""
    return torch.cat([
        batch["mel"], batch["logf0"][..., None], batch["voiced"][..., None],
        batch["energy"][..., None]], dim=-1).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warmup-epochs", type=int, default=5,
                    help="Recon-only epochs before the GAN turns on.")
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--lr-g", type=float, default=2e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-dir", type=Path, default=Path("data/mio_vctk_full_compact"))
    ap.add_argument("--content-dir", type=str, default="content_striding_8l_200hz")
    ap.add_argument("--acoustics-dir", type=str, default="acoustics_150hz")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--mel-weight", type=float, default=45.0)
    ap.add_argument("--mrstft-weight", type=float, default=1.0)
    ap.add_argument("--adv-weight", type=float, default=1.0)
    ap.add_argument("--fm-weight", type=float, default=2.0)
    ap.add_argument("--nffts", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--disc-window", type=int, default=16384)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--val-n", type=int, default=12, help="held-out clips for per-epoch cosine validation")
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--lr-warmup-steps", type=int, default=500,
                    help="Linear lr warmup from 0 over N steps. The iSTFT head trains from a wild "
                         "initial output → log-spectral loss gradients explode → NaN weights by "
                         "step ~5 at any fixed lr. Warmup takes near-zero early steps so the output "
                         "settles before the loss gradients bite.")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device); args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──
    meta = np.load(args.data_dir / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"]); src = meta["source_files"][:n].astype(str)
    spk_names = meta["spk_names"]
    cz = np.load(args.data_dir / "spk_centroids.npz", allow_pickle=False)
    semap = {str(s): torch.from_numpy(e).float() for s, e in zip(cz["speakers"], cz["embeddings"])}
    idx = np.arange(n); np.random.default_rng(args.seed).shuffle(idx)
    split = int(n * 0.95)
    ds = AcousticDataset(idx[:split], args.data_dir / args.content_dir,
                         args.data_dir / args.acoustics_dir, src, spk_names, semap,
                         args.max_frames, args.seed, need_content=False, need_audio=True)
    loader = DataLoader(ds, args.batch_size, shuffle=True, num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0, collate_fn=collate_acoustic,
                        drop_last=True)
    # held-out val set for per-epoch copy-synth cosine metrics (vs SOURCE/GT audio).
    val_ds = AcousticDataset(idx[split:], args.data_dir / args.content_dir,
                             args.data_dir / args.acoustics_dir, src, spk_names, semap,
                             args.max_frames, args.seed + 1, need_content=False, need_audio=True)
    val_loader = DataLoader(val_ds, 1, shuffle=False, collate_fn=collate_acoustic)

    # ── models ── (discriminators fully on-device; MPS stft verified stable)
    voc = ConditionedVocoder(VocoderConfig()).to(device)
    disc = VocoderDiscriminator().to(device)
    opt_g = torch.optim.AdamW(voc.parameters(), lr=args.lr_g, betas=(0.8, 0.99), eps=args.adam_eps)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.8, 0.99), eps=args.adam_eps)
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)
    sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs)
    mel_fn = torchaudio.transforms.MelSpectrogram(
        S, n_fft=2048, hop_length=512, n_mels=80, f_min=0, f_max=S / 2, power=1).to(device)

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        voc.load_state_dict(ck["state_dict"]); disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        sch_g.load_state_dict(ck["sch_g"]); sch_d.load_state_dict(ck["sch_d"])
        start = int(ck.get("epoch", -1)) + 1

    print(f"Stage B ConditionedVocoder: {sum(p.numel() for p in voc.parameters())/1e6:.1f}M  "
          f"disc {sum(p.numel() for p in disc.parameters())/1e6:.1f}M (MPD+MRD on-device)  "
          f"warmup<{args.warmup_epochs}", flush=True)

    t0 = time.time()
    gstep = 0                                             # global step (for lr warmup)
    for ep in range(start, args.epochs):
        adversarial = ep >= args.warmup_epochs
        voc.train(); disc.train()
        acc = {k: torch.zeros((), device=device) for k in ("recon", "adv", "fm", "d")}
        steps = skipped = 0
        for batch in loader:
            steps += 1
            if steps > args.steps_per_epoch:
                break
            gstep += 1
            if gstep <= args.lr_warmup_steps:             # linear lr warmup from 0
                w = gstep / max(1, args.lr_warmup_steps)
                for g in opt_g.param_groups: g["lr"] = args.lr_g * w
                for g in opt_d.param_groups: g["lr"] = args.lr_d * w
            cond = build_cond(batch, device)              # (B,T,83)
            audio = batch["audio"].to(device)
            pred = voc(cond)
            t_len = min(pred.shape[1], audio.shape[1])
            pred, tgt = pred[:, :t_len], audio[:, :t_len]

            # ── reconstruction: mel on-device, MR-STFT on CPU (cheap, extra safety) ──
            recon = (args.mel_weight * F.l1_loss(
                        mel_fn(pred).clamp_min(1e-5).log(), mel_fn(tgt).clamp_min(1e-5).log())
                     + args.mrstft_weight * mrstft(pred.float().cpu(), tgt.float().cpu(),
                                                    args.nffts).to(device))

            if adversarial:
                W = min(args.disc_window, pred.shape[1])
                st = torch.randint(0, pred.shape[1] - W + 1, (1,)).item()
                pred_w, tgt_w = pred[:, st:st + W], tgt[:, st:st + W]

                # ── D step ──
                real_lg, _ = disc(tgt_w)
                fake_lg, _ = disc(pred_w.detach())
                d_loss = discriminator_loss(real_lg, fake_lg)
                opt_d.zero_grad(set_to_none=True); d_loss.backward()
                dn = torch.nn.utils.clip_grad_norm_(disc.parameters(), args.clip_grad)
                if torch.isfinite(dn): opt_d.step()

                # ── G step ──
                fake_lg, fake_fm = disc(pred_w)
                real_lg, real_fm = disc(tgt_w)
                adv = generator_adv_loss(fake_lg)
                fm = feature_matching_loss(real_fm, fake_fm)
                g_loss = recon + args.adv_weight * adv + args.fm_weight * fm
            else:
                adv = fm = d_loss = pred.new_zeros(())
                g_loss = recon

            opt_g.zero_grad(set_to_none=True); g_loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(voc.parameters(), args.clip_grad)
            if not torch.isfinite(gn):
                opt_g.zero_grad(set_to_none=True); skipped += 1; continue
            opt_g.step()
            acc["recon"] += recon.detach(); acc["adv"] += adv.detach()
            acc["fm"] += fm.detach(); acc["d"] += d_loss.detach()
            if steps % 100 == 0:
                d = max(steps - skipped, 1)
                tag = "ADV" if adversarial else "REC"
                print(f"E{ep:02d}[{tag}] {steps:04d}/{args.steps_per_epoch} "
                      f"recon={acc['recon'].item()/d:.3f} adv={acc['adv'].item()/d:.3f} "
                      f"fm={acc['fm'].item()/d:.3f} d={acc['d'].item()/d:.3f} skip={skipped} "
                      f"{(time.time()-t0)/((ep-start)*args.steps_per_epoch+steps):.3f}s/st", flush=True)

        sch_g.step(); sch_d.step()
        val = validate(voc, val_loader, device, args.val_n)   # copy-synth cosines vs SOURCE
        ck = {"state_dict": voc.state_dict(), "disc": disc.state_dict(),
              "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
              "sch_g": sch_g.state_dict(), "sch_d": sch_d.state_dict(),
              "vocoder_config": VocoderConfig().__dict__, "epoch": ep}
        torch.save(ck, args.out_dir / "last.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            torch.save(ck, args.out_dir / f"epoch{ep:03d}.pt")
        (args.out_dir / "summary.json").write_text(json.dumps(
            {"epoch": ep, "phase": "adv" if adversarial else "warmup", "skipped": skipped,
             **val, **{k: v.item() / max(steps - skipped, 1) for k, v in acc.items()}}, indent=2) + "\n")
        print(f"E{ep:02d} done ({'ADV' if adversarial else 'REC'})  "
              f"mel_sor_cos_val={val['mel_sor_cos_val']:.3f}  wav_sor_cos_val={val['wav_sor_cos_val']:+.3f}  "
              f"frame_buzz_val={val['frame_buzz_val']:.1f}dB (gt≈{val['frame_buzz_gt']:.1f})", flush=True)
    print(f"Done. {time.time()-t0:.0f}s", flush=True)


@torch.no_grad()
def validate(voc, val_loader, device, n_clips):
    """Copy-synth on held-out clips → cosine similarity to the SOURCE (GT) waveform.
    mel_sor_cos = phase-invariant magnitude fidelity (the gate); wav_sor_cos = time-domain
    (phase-sensitive, ~0 even when good — kept for reference); frame_buzz = the 150Hz
    frame-rate artifact level (target ≈ the GT's own frame_buzz)."""
    from .eval_perceptual import mel_cos, wave_cos, frame_mod_db
    voc.eval()
    mc, wc, fb, fg = [], [], [], []
    for i, batch in enumerate(val_loader):
        if i >= n_clips:
            break
        cond = torch.cat([batch["mel"], batch["logf0"][..., None],
                          batch["voiced"][..., None], batch["energy"][..., None]], dim=-1).to(device)
        wav = voc(cond)[0].cpu(); gt = batch["audio"][0]
        mc.append(mel_cos(wav, gt)); wc.append(wave_cos(wav, gt))
        fb.append(frame_mod_db(wav.float())); fg.append(frame_mod_db(gt.float()))
    voc.train()
    return {"mel_sor_cos_val": float(np.mean(mc)), "wav_sor_cos_val": float(np.mean(wc)),
            "frame_buzz_val": float(np.mean(fb)), "frame_buzz_gt": float(np.mean(fg))}


if __name__ == "__main__":
    main()

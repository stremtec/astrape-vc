"""Train CausalDecoderMCS — teacher ISTFT head distillation.

No GAN, no CPU spectral losses. Directly distill the teacher's
ISTFT head output (mag_log, phase) which is on MPS — fast.

Losses:
  - L1(pred_mag_log, teacher_mag_log)
  - anti-wrap phase loss (IP + group delay)

Usage:
  .venv/bin/python -m astrape.train_decoder_mcs --device mps --epochs 80 \
      --content-dir content_striding_8l_200hz --num-workers 4 \
      --out-dir /Volumes/UNTITLED/btrv5_checkpoints/decoder_mcs
"""
import argparse, json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import Phase0Dataset, collate_phase0
from .decoder_mcs import CausalDecoderMCS, CausalDecoderMCSConfig

S = 44100


@torch.no_grad()
def teacher_istft_head(mio, content, speaker, stft_length):
    """Capture the teacher's ISTFT head output: (mag_log, phase) before iSTFT."""
    cap = {}
    def hook(m, inp):
        cap["x"] = inp[0]  # (B, T_stft, 512) — input to ISTFTHead
    handle = mio.istft_head.register_forward_pre_hook(hook)
    try:
        mio.forward_wave(content, speaker, stft_length=stft_length)
    finally:
        handle.remove()
    x = cap["x"]  # (B, T_stft, 512)
    # Replicate ISTFTHead: Linear(512→n_fft+2) → chunk
    xo = mio.istft_head.out(x).transpose(1, 2)  # (B, n_fft+2, T_stft)
    mag_log, phase = xo.chunk(2, dim=1)          # (B, n_freq, T_stft)
    return mag_log, phase


def anti_wrap_phase(pred_phase, tgt_phase, mag_weight):
    """Anti-wrapping phase loss: IP + group-delay, magnitude-weighted."""
    aw = lambda x: torch.atan2(torch.sin(x), torch.cos(x)).abs()  # wrap → [0, π]
    # Instantaneous phase
    ip = (aw(pred_phase - tgt_phase) * mag_weight).sum() / mag_weight.sum().clamp(min=1e-7)
    # Group delay (frequency-axis derivative)
    dgd = aw((pred_phase[:, 1:] - pred_phase[:, :-1]) -
             (tgt_phase[:, 1:] - tgt_phase[:, :-1]))
    gd = (dgd * mag_weight[:, 1:]).sum() / mag_weight[:, 1:].sum().clamp(min=1e-7)
    return ip + gd


@torch.no_grad()
def eval_cos(dec, mio, loader, device, num_batches=10):
    """Waveform cosine vs teacher."""
    wave_cos_all = []
    for i, (content, audio, speaker, _) in enumerate(loader):
        if i >= num_batches: break
        content, audio, speaker = content.to(device), audio.to(device), speaker.to(device)
        tch_len = mio._calculate_target_stft_length(audio.shape[1])
        teacher = mio.forward_wave(content, speaker, stft_length=tch_len)
        pred = dec(content, speaker)
        tl = min(pred.shape[1], teacher.shape[1])
        p, t = pred[0, :tl].float(), teacher[0, :tl].float()
        wave_cos_all.append(F.cosine_similarity(p, t, dim=0).item())
    return float(np.mean(wave_cos_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-dir", type=Path, default=Path("data/mio_vctk_full_compact"))
    ap.add_argument("--wavlm-dir", type=str, default="wavlm_L4_200hz")
    ap.add_argument("--wavlm-rate", type=int, default=200)
    ap.add_argument("--content-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--phase-weight", type=float, default=0.5)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--prenet-layers", type=int, default=7)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device); args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──
    meta = np.load(args.data_dir / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"]); src = meta["source_files"][:n].astype(str); spk_names = meta["spk_names"]
    cz = np.load(args.data_dir / "spk_centroids.npz", allow_pickle=False)
    semap = {str(s): torch.from_numpy(e).float() for s, e in zip(cz["speakers"], cz["embeddings"])}
    idx = np.arange(n); np.random.default_rng(args.seed).shuffle(idx)
    split = int(len(idx) * 0.95)
    ds = Phase0Dataset(idx[:split], args.data_dir / args.wavlm_dir, src, None, spk_names,
                       args.max_frames, args.seed, wavlm_rate=args.wavlm_rate,
                       speaker_emb_map=semap,
                       content_dir=args.data_dir / args.content_dir, load_audio=False)
    loader = DataLoader(ds, args.batch_size, shuffle=True, num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0,
                        collate_fn=collate_phase0, drop_last=True)

    # ── model ──
    from .miocodec import load_mio
    mio = load_mio(args.device).eval()
    dec_cfg = CausalDecoderMCSConfig(prenet_layers=args.prenet_layers)
    dec = CausalDecoderMCS(dec_cfg).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, betas=(0.8, 0.99))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        dec.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"]); start = int(ck.get("epoch", -1)) + 1

    dec_params = sum(p.numel() for p in dec.parameters())
    algo = (dec_cfg.n_fft - dec_cfg.hop_length) / 2 / S * 1000
    print(f"CausalDecoderMCS: {dec_params/1e6:.2f}M  prenet={dec_cfg.prenet_layers}L  "
          f"latency={algo:.1f}ms  (L1 mag + {args.phase_weight}*phase)", flush=True)

    t0 = time.time()
    for ep in range(start, args.epochs):
        dec.train()
        acc = {"mag": torch.zeros((), device=device),
               "phase": torch.zeros((), device=device)}
        steps = skipped = 0
        for content, audio, speaker, _i in loader:
            steps += 1
            if steps > args.steps_per_epoch: break
            content, audio, speaker = content.to(device), audio.to(device), speaker.to(device)

            # Teacher ISTFT head target: mag_log, phase from the teacher
            with torch.no_grad():
                tch_len = mio._calculate_target_stft_length(audio.shape[1])
                t_mag_log, t_phase = teacher_istft_head(mio, content, speaker, tch_len)

            # Student: forward through decoder, get mag_log + phase from ISTFT head
            pred = dec(content, speaker, return_spec=True)
            if isinstance(pred, tuple):
                _, p_mag_log, p_phase = pred
            else:
                wav = pred; continue  # shouldn't happen with return_spec=True

            # Align lengths
            Ts = min(p_mag_log.shape[-1], t_mag_log.shape[-1])
            p_mag_log, p_phase = p_mag_log[..., :Ts], p_phase[..., :Ts]
            t_mag_log, t_phase = t_mag_log[..., :Ts], t_phase[..., :Ts]

            # Magnitude loss: L1 on log-mag
            mag_loss = F.l1_loss(p_mag_log, t_mag_log)

            # Phase loss: anti-wrap IP + group delay
            w = torch.exp(t_mag_log).clamp(max=1e2)  # teacher magnitude as weight
            w = w / w.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-7)
            p_phase = p_phase.float(); t_phase = t_phase.float()
            phase_loss = anti_wrap_phase(p_phase, t_phase, w)

            loss = mag_loss + args.phase_weight * phase_loss
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(dec.parameters(), args.clip_grad)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); skipped += 1; continue
            opt.step()
            acc["mag"] += mag_loss.detach(); acc["phase"] += phase_loss.detach()

            if steps % 100 == 0:
                d = max(steps - skipped, 1)
                print(f"E{ep:02d} {steps:04d}/{args.steps_per_epoch} "
                      f"mag={acc['mag'].item()/d:.3f} ph={acc['phase'].item()/d:.3f} "
                      f"skip={skipped}", flush=True)

        sch.step()
        ckpt = {"state_dict": dec.state_dict(), "opt": opt.state_dict(),
                "sch": sch.state_dict(), "decoder_config": dec_cfg.__dict__, "epoch": ep}
        torch.save(ckpt, args.out_dir / "last.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            torch.save(ckpt, args.out_dir / f"epoch{ep:03d}.pt")
        (args.out_dir / "summary.json").write_text(json.dumps(
            {"epoch": ep, "skipped": skipped,
             **{k: v.item() / max(steps - skipped, 1) for k, v in acc.items()}}, indent=2) + "\n")

        # Eval
        dec.eval()
        wave_cos = eval_cos(dec, mio, loader, device)
        print(f"E{ep:02d} done  wave_cos={wave_cos:.4f}", flush=True)
        (args.out_dir / "eval.json").write_text(json.dumps(
            {"epoch": ep, "wave_cos": wave_cos}, indent=2) + "\n")

    print(f"Done. {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

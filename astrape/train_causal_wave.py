"""Train the CausalWaveDecoder (MioCodec-structure causal replica) by distillation.

Two signals, both from the frozen MioCodec teacher run on the SAME content+speaker:
  • OUTPUT distill   — mrstft + mel + complex-STFT on the waveform (CPU; the proven signal)
  • FEATURE distill  — L1 between the student's per-module activations and the teacher's
                       corresponding intermediate features (modules align 1:1). This is the
                       point of replicating the structure: rich, direct supervision.
No GAN, no NSF. Strictly causal (3.3ms iSTFT lookahead only).

  .venv/bin/python -m astrape.train_causal_wave --device mps --epochs 80 \
      --content-dir /Users/asill/btrv5_content --num-workers 4 \
      --out-dir /Volumes/UNTITLED/btrv5_checkpoints/causal_wave
"""
import argparse, json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from .data import Phase0Dataset, collate_phase0
from .train_decoder import mrstft, complex_stft_loss
from .causal_wave_decoder import CausalWaveDecoder, CausalWaveDecoderConfig

S = 44100
# teacher module → student feat key (1:1 aligned), with the time axis of each activation
TEACHER_MODULES = {
    "wave_prenet": ("prenet", 1), "wave_prior_net": ("prior", 2),
    "wave_decoder": ("decoder", 1), "wave_post_net": ("post", 2),
    "wave_upsampler": ("upsampler", 1),
}


def feat_l1(s, t, time_dim):
    L = min(s.shape[time_dim], t.shape[time_dim])
    return F.l1_loss(s.narrow(time_dim, 0, L), t.narrow(time_dim, 0, L).detach())


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
    ap.add_argument("--nffts", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--feat-weight", type=float, default=1.0, help="intermediate feature distill weight")
    ap.add_argument("--cstft-weight", type=float, default=1.0)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device); args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = np.load(args.data_dir / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"]); src = meta["source_files"][:n].astype(str); spk_names = meta["spk_names"]
    cz = np.load(args.data_dir / "spk_centroids.npz", allow_pickle=False)
    semap = {str(s): torch.from_numpy(e).float() for s, e in zip(cz["speakers"], cz["embeddings"])}
    idx = np.arange(n); np.random.default_rng(args.seed).shuffle(idx)
    ds = Phase0Dataset(idx[:int(n * 0.95)], args.data_dir / args.wavlm_dir, src, None, spk_names,
                       args.max_frames, args.seed, wavlm_rate=args.wavlm_rate, speaker_emb_map=semap,
                       content_dir=args.data_dir / args.content_dir)
    loader = DataLoader(ds, args.batch_size, shuffle=True, num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0, collate_fn=collate_phase0, drop_last=True)

    from .miocodec import load_mio
    mio = load_mio(args.device).eval()
    dec = CausalWaveDecoder(CausalWaveDecoderConfig()).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, betas=(0.8, 0.99))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mel_fn = torchaudio.transforms.MelSpectrogram(S, n_fft=2048, hop_length=512, n_mels=80, f_min=0, f_max=S/2, power=1)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        dec.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sch"])
        start = int(ck.get("epoch", -1)) + 1
    print(f"CausalWaveDecoder: {sum(p.numel() for p in dec.parameters())/1e6:.2f}M  "
          f"(distill: output + {len(TEACHER_MODULES)} intermediate features)", flush=True)

    # teacher feature hooks
    tfeats = {}
    for mod_name, (key, _) in TEACHER_MODULES.items():
        getattr(mio, mod_name).register_forward_hook(
            lambda m, i, o, k=key: tfeats.__setitem__(k, (o[0] if isinstance(o, tuple) else o)))

    t0 = time.time()
    for ep in range(start, args.epochs):
        dec.train()
        acc = {k: torch.zeros((), device=device) for k in ("out", "feat")}
        steps = skipped = 0
        for content, audio, speaker, _i in loader:
            steps += 1
            if steps > args.steps_per_epoch:
                break
            content, audio, speaker = content.to(device), audio.to(device), speaker.to(device)
            with torch.no_grad():
                tch_len = mio._calculate_target_stft_length(audio.shape[1])
                tfeats.clear()
                teacher_wav = mio.forward_wave(content, speaker, stft_length=tch_len)
            wav, mag, phase, sfeats = dec(content, speaker, return_feats=True)

            tl = min(wav.shape[1], teacher_wav.shape[1])
            pc, tc = wav[:, :tl].float().cpu(), teacher_wav[:, :tl].float().cpu()
            out_loss = (mrstft(pc, tc, args.nffts)
                        + F.l1_loss(mel_fn(pc).clamp_min(1e-5).log(), mel_fn(tc).clamp_min(1e-5).log())
                        + args.cstft_weight * complex_stft_loss(pc, tc, args.nffts)).to(device)
            feat_loss = wav.new_zeros(())
            for _mod, (key, td) in TEACHER_MODULES.items():
                if key in tfeats and key in sfeats:
                    feat_loss = feat_loss + feat_l1(sfeats[key], tfeats[key], td)

            loss = out_loss + args.feat_weight * feat_loss
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(dec.parameters(), args.clip_grad)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); skipped += 1; continue
            opt.step()
            acc["out"] += out_loss.detach(); acc["feat"] += feat_loss.detach()
            if steps % 100 == 0:
                d = max(steps - skipped, 1)
                print(f"E{ep:02d} {steps:04d}/{args.steps_per_epoch} out={acc['out'].item()/d:.3f} "
                      f"feat={acc['feat'].item()/d:.3f} skip={skipped} "
                      f"{(time.time()-t0)/((ep-start)*args.steps_per_epoch+steps):.3f}s/step", flush=True)
        sch.step()
        ckpt = {"state_dict": dec.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(),
                "decoder_config": CausalWaveDecoderConfig().__dict__, "epoch": ep}
        torch.save(ckpt, args.out_dir / "last.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            torch.save(ckpt, args.out_dir / f"epoch{ep:03d}.pt")
        (args.out_dir / "summary.json").write_text(json.dumps(
            {"epoch": ep, "skipped": skipped,
             **{k: v.item() / max(steps - skipped, 1) for k, v in acc.items()}}, indent=2) + "\n")
        print(f"E{ep:02d} done", flush=True)
    print(f"Done. {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

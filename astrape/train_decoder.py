"""Decoder v5 training — 2-phase adversarial curriculum.

Curriculum
  Phase A  (epochs < --warmup-epochs):  RECONSTRUCTION ONLY
      MR-STFT (CPU, grad-preserving) + Mel-L1 on Gaussian-blurred target.
      Gets the generator to a stable spectral baseline before the GAN.
  Phase B  (epochs >= --warmup-epochs): ADVERSARIAL
      + MPD/MSD discriminators (LSGAN) + feature matching, recon kept as anchor.
      (Optionally enable NSF via --use-nsf for the harmonic source.)

Reuses the frozen encoder + data pipeline from astrape.data / astrape.encoder.

Usage:
  .venv/bin/python -m astrape.train_decoder --device mps --epochs 60 --warmup-epochs 10 \
      --encoder-ckpt /Volumes/UNTITLED/btrv5_checkpoints/striding_8l_200hz/striding_8l_200hz.best.pt \
      --wavlm-dir wavlm_L4_200hz --num-workers 6
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from .data import Phase0Dataset, collate_phase0, gaussian_blur_wave
from .decoder import CausalDecoderV5, CausalDecoderV5Config
from .discriminators import (
    CombinedDiscriminator, discriminator_loss, generator_adv_loss, feature_matching_loss,
)

S = 44100


def mrstft(pred: torch.Tensor, tgt: torch.Tensor, nffts) -> torch.Tensor:
    """Batched multi-resolution STFT loss, computed on the input's device.

    NOTE: call this with CPU tensors. torch.stft's BACKWARD on MPS intermittently
    emits non-finite gradients (the forward matches CPU to ~4e-5, but the backward
    does NOT) — this destabilised decoder training (accelerating skipped steps /
    recon→nan), so the caller moves pred/tgt to CPU. Device-agnostic and batched;
    correct on CPU.
    """
    loss = pred.new_zeros(())
    for n_fft in nffts:
        win = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
        ps = torch.stft(pred, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
                        window=win, return_complex=True).abs().clamp_min(1e-7)
        ts = torch.stft(tgt, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
                        window=win, return_complex=True).abs().clamp_min(1e-7)
        sc = torch.linalg.vector_norm(ps - ts) / torch.linalg.vector_norm(ts).clamp_min(1e-7)
        loss = loss + sc + F.l1_loss(ps.log(), ts.log())
    return loss / len(nffts)


def complex_stft_loss(pred: torch.Tensor, tgt: torch.Tensor, nffts) -> torch.Tensor:
    """Multi-resolution complex-STFT L1 (real+imag) — jointly supervises magnitude AND
    phase with smooth gradients (no angle() singularity). For teacher distillation: the
    teacher target carries correct phase, so matching its complex spectrum transfers it.
    CPU only (MPS torch.stft backward bug — see mrstft)."""
    loss = pred.new_zeros(())
    for n_fft in nffts:
        win = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
        P = torch.stft(pred, n_fft, n_fft // 4, n_fft, win, return_complex=True)
        T = torch.stft(tgt, n_fft, n_fft // 4, n_fft, win, return_complex=True)
        loss = loss + F.l1_loss(torch.view_as_real(P), torch.view_as_real(T))
    return loss / len(nffts)


def anti_wrap_phase_loss(pred: torch.Tensor, tgt: torch.Tensor,
                         n_fft: int = 1024, hop: int = 256) -> torch.Tensor:
    """APNet2-style anti-wrapping phase loss: instantaneous-phase (IP) + group-delay (GD),
    each wrapped to [0, pi] and magnitude-weighted by the (clean teacher) target so phase
    is supervised only where there is energy. Directly teaches phase the GAN learns slowly.
    CPU only."""
    win = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
    P = torch.stft(pred, n_fft, hop, n_fft, win, return_complex=True)
    T = torch.stft(tgt, n_fft, hop, n_fft, win, return_complex=True)
    w = T.abs(); w = w / w.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    pp, pt = P.angle(), T.angle()
    aw = lambda x: torch.atan2(torch.sin(x), torch.cos(x)).abs()           # wrap → [0, pi]
    ip = (aw(pp - pt) * w).sum() / w.sum().clamp_min(1e-7)
    dgd = aw((pp[..., 1:, :] - pp[..., :-1, :]) - (pt[..., 1:, :] - pt[..., :-1, :]))
    gd = (dgd * w[..., 1:, :]).sum() / w[..., 1:, :].sum().clamp_min(1e-7)  # along freq
    return ip + gd


def extract_f0(wave: torch.Tensor, sr: int = S, hop: int = 252, frame: int = 2048,
               fmin: float = 50.0, fmax: float = 600.0, vthresh: float = 0.35):
    """Vectorized FFT-autocorrelation F0 + voicing per `hop` samples (175Hz, the NSF/STFT
    frame rate). Non-differentiable TARGET for the NSF F0 head (extracted from the teacher
    waveform). Returns f0 (B, T) Hz and voiced (B, T) in {0,1}. CPU."""
    w = F.pad(wave, (frame // 2, frame // 2))
    fr = w.unfold(1, frame, hop) * torch.hann_window(frame, device=wave.device)  # (B,T,frame)
    nfft = 2 * frame
    ac = torch.fft.irfft(torch.fft.rfft(fr, n=nfft).abs().pow(2), n=nfft)[..., :frame]
    ac = ac / ac[..., :1].clamp_min(1e-9)                       # normalize by lag-0
    lmin, lmax = int(sr / fmax), int(sr / fmin)
    peak, k = ac[..., lmin:lmax].max(dim=-1)                    # (B, T)
    f0 = sr / (lmin + k).float()
    energy = fr.pow(2).mean(-1).sqrt()
    voiced = ((peak > vthresh) & (energy > 1e-3)).float()
    return f0, voiced


def teacher_spec(mio, content, speaker, stft_length):
    """The teacher's predicted (magnitude, phase) at its iSTFT head — the intermediate value
    fed to the iSTFT (the user's distillation target). Captured via a forward-pre-hook on the
    head; with v5 on the SAME 392/98 grid the student head is distilled against it directly.
    No grad (frozen target). mag,phase: (B, n_freq, T_stft)."""
    cap = {}
    handle = mio.istft_head.register_forward_pre_hook(lambda m, inp: cap.__setitem__("x", inp[0]))
    try:
        with torch.no_grad():
            mio.forward_wave(content, speaker, stft_length=stft_length)
    finally:
        handle.remove()
    xo = mio.istft_head.out(cap["x"]).transpose(1, 2)              # (B, n_fft+2, T)
    mag_log, phase = xo.chunk(2, dim=1)
    return torch.exp(mag_log).clamp(max=1e2), phase


def spec_distill_loss(smag, sphase, tmag, tphase):
    """Direct istft-head distillation: amplitude (log-mag L1) + anti-wrapping phase (IP +
    group-delay), magnitude-weighted by the teacher. student/teacher (B, n_freq, T) on the
    SAME 392/98 grid → no torch.stft (MPS-safe, on-device). Returns (amp_loss, phase_loss)."""
    Ts = min(smag.shape[-1], tmag.shape[-1])
    smag, sphase, tmag, tphase = smag[..., :Ts], sphase[..., :Ts], tmag[..., :Ts], tphase[..., :Ts]
    amp = F.l1_loss(smag.clamp_min(1e-5).log(), tmag.clamp_min(1e-5).log())
    w = tmag / tmag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    aw = lambda x: torch.atan2(torch.sin(x), torch.cos(x)).abs()   # wrap → [0, pi]
    ip = (aw(sphase - tphase) * w).sum() / w.sum().clamp_min(1e-7)
    gd = (aw((sphase[:, 1:] - sphase[:, :-1]) - (tphase[:, 1:] - tphase[:, :-1]))
          * w[:, 1:]).sum() / w[:, 1:].sum().clamp_min(1e-7)
    return amp, ip + gd


def load_encoder(checkpoint_path, device="cpu"):
    """Load the frozen Q2D2 encoder from a checkpoint (Phase 2: → astrape.encoder)."""
    from .encoder import MCSTransQ2D2Config, MCSTransQ2D2
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scfg = ck.get("config", {})
    scfg2 = {k: tuple(v) if isinstance(v, list) else v
             for k, v in scfg.items() if not k.startswith("_")}
    scfg2["use_wavlm_frontend"] = True
    known = set(MCSTransQ2D2Config.__dataclass_fields__.keys())
    scfg2 = {k: v for k, v in scfg2.items() if k in known}
    config = MCSTransQ2D2Config(**scfg2)
    model = MCSTransQ2D2(config).to(device).eval()
    model.load_state_dict(ck["state_dict"], strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warmup-epochs", type=int, default=10,
                    help="Reconstruction-only epochs before the GAN turns on.")
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--lr-g", type=float, default=1e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--encoder-ckpt", type=Path, default=None,
                    help="frozen encoder checkpoint (required unless --content-dir is given)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/Volumes/UNTITLED/btrv5_checkpoints/decoder_v5"))
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=Path("data/mio_vctk_full_compact"))
    ap.add_argument("--wavlm-dir", type=str, default="wavlm_L4_200hz")
    ap.add_argument("--wavlm-rate", type=int, default=200,
                    help="WavLM cache rate (must match the encoder; 200 for wavlm_L4_200hz).")
    ap.add_argument("--content-dir", type=str, default=None,
                    help="Pre-cached FULL-context content subdir (e.g. content_striding_8l_200hz). "
                         "Skips the per-step frozen-encoder forward and matches streaming context.")
    # losses / curriculum
    ap.add_argument("--nffts", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--blur-sigma-ms", type=float, default=0.0,
                    help="Time-domain Gaussian blur of the TARGET. KEEP 0: even 2ms is an "
                         "~80Hz low-pass that strips ~94%% of speech energy → trains the "
                         "decoder to a near-silent target (the old quiet-output bug).")
    ap.add_argument("--mrstft-weight", type=float, default=1.0)
    ap.add_argument("--mel-l1-weight", type=float, default=1.0)
    ap.add_argument("--adv-weight", type=float, default=1.0)
    ap.add_argument("--fm-weight", type=float, default=2.0)
    ap.add_argument("--disc-window", type=int, default=16384,
                    help="Samples the discriminator sees per step (random crop, "
                         "HiFi-GAN-style ~0.37s). Reconstruction still uses full audio. "
                         "Cuts discriminator compute ~5x with no quality loss.")
    ap.add_argument("--use-nsf", action="store_true", help="Enable Phase 2b NSF harmonic source.")
    # ── GAN-free teacher distillation (alternative to the adversarial curriculum) ──
    ap.add_argument("--teacher-distill", action="store_true",
                    help="GAN-free: distill the (non-causal) MioCodec teacher decoder's output "
                         "(correct phase/voicing/harmonics — the ceiling for our content) instead "
                         "of adversarial training. Loss vs teacher = mrstft+mel (amplitude) + "
                         "complex-STFT + anti-wrap-phase. No discriminator → ~warmup speed. Fixes "
                         "the buzz (over-voicing/pitch/noisy-highs) the GAN learns only slowly.")
    ap.add_argument("--cstft-weight", type=float, default=1.0)
    ap.add_argument("--phase-weight", type=float, default=0.3)
    ap.add_argument("--f0-weight", type=float, default=1.0,
                    help="With --use-nsf + --teacher-distill: supervise the NSF F0 head "
                         "(log-Hz L1 on voiced frames) against F0 extracted from the teacher. "
                         "Directly fixes the weak-fundamental / octave-up pitch instability.")
    ap.add_argument("--voiced-weight", type=float, default=0.5)
    ap.add_argument("--spec-distill", action="store_true",
                    help="With --teacher-distill: distill the teacher's iSTFT-head magnitude & "
                         "phase DIRECTLY (needs v5 on the teacher's 392/98 grid). Loss = "
                         "amp_weight·log-mag-L1 + specphase_weight·anti-wrap-phase. The crisp "
                         "8.9ms window + exact teacher phase target fix the fizz.")
    ap.add_argument("--amp-weight", type=float, default=1.0)
    ap.add_argument("--specphase-weight", type=float, default=1.0)
    ap.add_argument("--n-fft", type=int, default=392)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──
    data_dir = args.data_dir
    meta = np.load(data_dir / "meta.npz", allow_pickle=False)
    n_samples = int(meta["n_samples"])
    source_files = meta["source_files"][:n_samples].astype(str)
    spk_names = meta["spk_names"]
    # Per-speaker centroids (run `astrape.cache --what speakers` first) — covers all
    # speakers, unlike spk_1k.npy which only has the first ~4 (VCTK is grouped).
    cz_path = data_dir / "spk_centroids.npz"
    if not cz_path.exists():
        raise SystemExit(
            f"Missing {cz_path}.\n  Run first: .venv/bin/python -m astrape.cache --what speakers")
    cz = np.load(cz_path, allow_pickle=False)
    speaker_emb_map = {str(s): torch.from_numpy(e).float()
                       for s, e in zip(cz["speakers"], cz["embeddings"])}
    print(f"Loaded {len(speaker_emb_map)} speaker centroids")

    idx = np.arange(n_samples)              # ALL samples → all speakers (VCTK is grouped)
    np.random.default_rng(args.seed).shuffle(idx)
    split = int(len(idx) * 0.95)
    train_idx = idx[:split]
    train_ds = Phase0Dataset(train_idx, data_dir / args.wavlm_dir, source_files,
                             None, spk_names, args.max_frames, args.seed,
                             wavlm_rate=args.wavlm_rate, speaker_emb_map=speaker_emb_map,
                             content_dir=(data_dir / args.content_dir) if args.content_dir else None)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              persistent_workers=args.num_workers > 0,
                              collate_fn=collate_phase0, drop_last=True)

    # ── encoder (frozen — only if content isn't pre-cached), generator (v5), discriminators ──
    if args.content_dir:
        encoder = None
        print(f"Using pre-cached full-context content: {args.content_dir}", flush=True)
    else:
        if not args.encoder_ckpt:
            raise SystemExit("--encoder-ckpt is required unless --content-dir is given")
        encoder, _ = load_encoder(args.encoder_ckpt, device)
    dec_cfg = CausalDecoderV5Config(use_nsf=args.use_nsf, n_fft=args.n_fft)
    decoder = CausalDecoderV5(dec_cfg).to(device)
    disc = CombinedDiscriminator().to(device)

    mio = None
    if args.teacher_distill:
        from .miocodec import load_mio
        mio = load_mio(args.device).eval()
        print("Teacher distillation (GAN-free): MioCodec decoder output is the target", flush=True)

    opt_g = torch.optim.AdamW(decoder.parameters(), lr=args.lr_g, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.8, 0.99))
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)
    sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Shape-filtered load so a non-NSF checkpoint can warm-start an --use-nsf model:
        # the NSF head + widened istft_bridge are re-initialised, everything else transfers.
        sd = decoder.state_dict()
        compat = {k: v for k, v in ck["state_dict"].items() if k in sd and sd[k].shape == v.shape}
        decoder.load_state_dict(compat, strict=False)
        reinit = [k for k in sd if k not in compat]
        # Preserve the learned istft_bridge when widening for NSF: copy the old input
        # channels, zero the new NSF channels → the decoder starts IDENTICAL to the non-NSF
        # checkpoint (NSF is a no-op), so no spectral quality is lost; the F0 head still gets
        # direct F0 supervision and the bridge learns to use the comb from zero.
        bw = ck["state_dict"].get("istft_bridge.weight")
        if bw is not None and tuple(bw.shape) != tuple(sd["istft_bridge.weight"].shape):
            with torch.no_grad():
                decoder.istft_bridge.weight.zero_()
                decoder.istft_bridge.weight[:, :bw.shape[1]].copy_(bw)
            print(f"  resume: preserved istft_bridge ({bw.shape[1]} ch copied, NSF ch zeroed)", flush=True)
        if reinit:
            print(f"  resume: {len(compat)}/{len(sd)} tensors loaded, {len(reinit)} re-init "
                  f"(NSF/bridge): {reinit[:3]}", flush=True)
        disc.load_state_dict(ck.get("disc", {}), strict=False)
        if not reinit:                        # same architecture → restore optimiser/schedule
            if "opt_g" in ck: opt_g.load_state_dict(ck["opt_g"])
            if "opt_d" in ck: opt_d.load_state_dict(ck["opt_d"])
            if "sch_g" in ck: sch_g.load_state_dict(ck["sch_g"])
            if "sch_d" in ck: sch_d.load_state_dict(ck["sch_d"])
        start_epoch = int(ck.get("epoch", -1)) + 1

    # CPU: the spectral reconstruction loss is computed on CPU (MPS torch.stft's
    # backward intermittently emits non-finite grads — see the recon block), so its
    # mel transform lives on CPU too.
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=S, n_fft=2048, hop_length=512, n_mels=80, f_min=0, f_max=S / 2, power=1,
    )

    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder v5: {dec_params/1e6:.2f}M  (n_fft={dec_cfg.n_fft}, nsf={dec_cfg.use_nsf}, "
          f"algo-latency={(dec_cfg.n_fft-dec_cfg.hop_length)/2/S*1000:.1f}ms)", flush=True)
    print(f"Curriculum: warmup(recon) epochs 0..{args.warmup_epochs-1}, "
          f"adversarial epochs {args.warmup_epochs}..{args.epochs-1}", flush=True)

    t0 = time.time()
    for ep in range(start_epoch, args.epochs):
        adversarial = ep >= args.warmup_epochs
        decoder.train(); disc.train()
        # On-device accumulators → no per-step MPS↔CPU sync (only .item() at log).
        acc = {k: torch.zeros((), device=device) for k in ("recon", "adv", "fm", "d")}
        steps = 0; skipped = 0
        for wavlm, audio, speaker, _idx in train_loader:
            steps += 1
            if steps > args.steps_per_epoch:
                break
            wavlm, audio, speaker = wavlm.to(device), audio.to(device), speaker.to(device)

            if encoder is None:                       # pre-cached full-context content (B, T, 768)
                content = wavlm
            else:
                with torch.no_grad():
                    mask = torch.ones(wavlm.shape[0], wavlm.shape[1] // 2, dtype=torch.bool, device=device)
                    content = encoder(wavlm.transpose(1, 2), padding_mask=mask)["projected"].transpose(1, 2)
            stft_len = decoder._compute_stft_length(content.shape[1])
            pred_f0 = pred_voiced = smag = sphase = None
            if args.spec_distill:
                pred, smag, sphase = decoder(content, speaker, stft_length=stft_len, return_spec=True)
            elif args.use_nsf:
                pred, pred_f0, pred_voiced = decoder(content, speaker, stft_length=stft_len, return_aux=True)
            else:
                pred = decoder(content, speaker, stft_length=stft_len)

            t_len = min(pred.shape[1], audio.shape[1])
            pred, tgt = pred[:, :t_len], audio[:, :t_len]
            if args.teacher_distill and args.spec_distill:
                # ── direct iSTFT-head distillation: match the teacher's predicted mag/phase ──
                # (the user's "distil the inter-module values" idea, at the one aligned point:
                # both heads now on the 392/98 grid). On-device, no torch.stft.
                with torch.no_grad():
                    tch_len = mio._calculate_target_stft_length(tgt.shape[1])
                    tmag, tphase = teacher_spec(mio, content, speaker, tch_len)
                amp, ph = spec_distill_loss(smag, sphase, tmag, tphase)
                recon = args.amp_weight * amp + args.specphase_weight * ph
                adv, fm = amp.detach(), ph.detach()          # log slots: adv=amp, fm=phase
                d_loss = pred.new_zeros(())
                g_loss = recon
            elif args.teacher_distill:
                # ── GAN-free: distill the (non-causal) MioCodec teacher decoder ──
                # The teacher is the achievable ceiling for OUR content and gets phase /
                # voicing / harmonics right — matching it directly removes the buzz
                # (over-voicing, octave-up pitch, noisy highs) the GAN fixes only slowly.
                with torch.no_grad():
                    tch_len = mio._calculate_target_stft_length(tgt.shape[1])
                    teacher = mio.forward_wave(content, speaker, stft_length=tch_len)
                tl = min(pred.shape[1], teacher.shape[1])
                pred_c, tgt_c = pred[:, :tl].float().cpu(), teacher[:, :tl].float().cpu()
                recon = (args.mrstft_weight * mrstft(pred_c, tgt_c, args.nffts)
                         + args.mel_l1_weight * F.l1_loss(
                             mel_fn(pred_c).clamp_min(1e-5).log(),
                             mel_fn(tgt_c).clamp_min(1e-5).log())
                         + args.cstft_weight * complex_stft_loss(pred_c, tgt_c, args.nffts)
                         + args.phase_weight * anti_wrap_phase_loss(pred_c, tgt_c)).to(device)
                adv = fm = d_loss = pred.new_zeros(())
                g_loss = recon
                if args.use_nsf and pred_f0 is not None:
                    # ── explicit F0 supervision: pull the NSF F0 head to the teacher's pitch ──
                    # (the spectral loss alone leaves the fundamental weak → octave-up jitter).
                    with torch.no_grad():
                        tgt_f0, tgt_voiced = extract_f0(tgt_c)            # from teacher waveform
                    Tf = min(pred_f0.shape[1], tgt_f0.shape[1])
                    pf, pv = pred_f0[:, :Tf].float().cpu(), pred_voiced[:, :Tf].float().cpu()
                    tf0, tv = tgt_f0[:, :Tf], tgt_voiced[:, :Tf]
                    f0_loss = ((pf.clamp_min(1.0).log() - tf0.clamp_min(1.0).log()).abs() * tv
                               ).sum() / tv.sum().clamp_min(1.0)
                    voiced_loss = F.binary_cross_entropy(pv.clamp(1e-4, 1 - 1e-4), tv)
                    g_loss = g_loss + (args.f0_weight * f0_loss
                                       + args.voiced_weight * voiced_loss).to(device)
                    adv, fm = f0_loss.detach().to(device), voiced_loss.detach().to(device)  # log slots
            else:
                tgt_blur = gaussian_blur_wave(tgt, args.blur_sigma_ms) if args.blur_sigma_ms > 0 else tgt

                # ── reconstruction (always) — spectral loss computed on CPU ──
                # MPS torch.stft's BACKWARD intermittently emits non-finite gradients,
                # which (pre-guard) poisoned the weights → recon=nan, and (post-guard)
                # showed up as accelerating skipped steps. CPU stft is exact; autograd
                # flows the grads back to the MPS decoder. The decoder fwd/iSTFT stays on
                # MPS (verified always finite). Per-step transfer is ~0.7MB → negligible.
                pred_c, tgt_c = pred.float().cpu(), tgt_blur.float().cpu()
                recon = (args.mrstft_weight * mrstft(pred_c, tgt_c, args.nffts)
                         + args.mel_l1_weight * F.l1_loss(
                             mel_fn(pred_c).clamp_min(1e-5).log(),
                             mel_fn(tgt_c).clamp_min(1e-5).log())).to(device)

                if adversarial:
                    # Discriminate on a random short window (HiFi-GAN segment style):
                    # same crop for real+fake, full audio still used for reconstruction.
                    # Discriminators are local/shift-invariant → quality-neutral, ~5x cheaper.
                    W = min(args.disc_window, pred.shape[1])
                    st = torch.randint(0, pred.shape[1] - W + 1, (1,)).item()
                    pred_w, tgt_w = pred[:, st:st + W], tgt[:, st:st + W]

                    # ── D step ──
                    real_lg, _ = disc(tgt_w)
                    fake_lg, _ = disc(pred_w.detach())
                    d_loss = discriminator_loss(real_lg, fake_lg)
                    opt_d.zero_grad(set_to_none=True)
                    d_loss.backward()
                    dnorm = torch.nn.utils.clip_grad_norm_(disc.parameters(), args.clip_grad)
                    if torch.isfinite(dnorm):
                        opt_d.step()

                    # ── G step ──
                    fake_lg, fake_fm = disc(pred_w)
                    real_lg, real_fm = disc(tgt_w)
                    adv = generator_adv_loss(fake_lg)
                    fm = feature_matching_loss(real_fm, fake_fm)
                    g_loss = args.adv_weight * adv + args.fm_weight * fm + recon
                else:
                    adv = fm = d_loss = pred.new_zeros(())
                    g_loss = recon

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_grad)
            if not torch.isfinite(gnorm):
                # Rare non-finite gradient (corrupt cache sample / numerical edge):
                # skip the update so ONE bad batch can't permanently poison the
                # weights — the failure mode behind recon→nan-forever.
                opt_g.zero_grad(set_to_none=True)
                skipped += 1
                continue
            opt_g.step()
            acc["recon"] += recon.detach(); acc["adv"] += adv.detach()
            acc["fm"] += fm.detach(); acc["d"] += d_loss.detach()

            if steps % 100 == 0:
                d = max(steps - skipped, 1)
                tag = "DST" if args.teacher_distill else ("ADV" if adversarial else "REC")
                print(f"E{ep:02d}[{tag}] {steps:04d}/{args.steps_per_epoch} "
                      f"recon={acc['recon'].item()/d:.3f} adv={acc['adv'].item()/d:.3f} "
                      f"fm={acc['fm'].item()/d:.3f} d={acc['d'].item()/d:.3f} skip={skipped} "
                      f"{(time.time()-t0)/((ep-start_epoch)*args.steps_per_epoch + steps):.3f}s/step",
                      flush=True)

        sch_g.step(); sch_d.step()
        tot = {k: v.item() for k, v in acc.items()}
        ckpt = {"state_dict": decoder.state_dict(), "disc": disc.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "sch_g": sch_g.state_dict(), "sch_d": sch_d.state_dict(),
                "decoder_config": dec_cfg.__dict__, "epoch": ep}
        torch.save(ckpt, args.out_dir / "last.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            torch.save(ckpt, args.out_dir / f"epoch{ep:03d}.pt")
        (args.out_dir / "summary.json").write_text(json.dumps(
            {"epoch": ep, "phase": "adversarial" if adversarial else "warmup",
             "skipped": skipped,
             **{k: v / max(steps - skipped, 1) for k, v in tot.items()}}, indent=2) + "\n")
        print(f"E{ep:02d} done ({'ADV' if adversarial else 'REC'})", flush=True)

    print(f"Done. elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

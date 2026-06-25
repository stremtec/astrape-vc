"""Test Path A (asymmetric window) + Path C (frame-skipped distillation).

Path A: Replace symmetric Hann window with right-aligned half-Hann.
  Gives full weight to the most recent sample.
Path C: student(t) → teacher(t-1) alignment.
  Student at time t has seen up to t; teacher at t-1 needed less info.
"""
import sys, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)
sys.path.insert(0, "external/MioCodec/src")
sys.path.insert(0, ".")

import torch, torchaudio, numpy as np, argparse, random, time, json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from mcs_common import (
    Batch, split_by_speaker, speaker_balanced_subset,
    move_batch, save_checkpoint, DEFAULT_DATA_DIR,
    _voiced_weights,
)
from mcs_q2d2 import Q2D2Projection, Q2D2Quantizer, compute_q2d2_perplexity
from train_mcs_q2d2 import (
    MCSTransQ2D2Config, MCSTransQ2D2, q2d2_losses,
)

SAMPLE_RATE = 44100
MEL_HOP = 882
N_FFT = 2048


def asymmetric_hann_window(n_fft: int) -> torch.Tensor:
    """Right-aligned half-Hann: peaks at the rightmost sample, decays left."""
    # Standard Hann: 0.5 * (1 - cos(2π*n/(N-1))), symmetric
    # Asymmetric: take the RIGHT half of a 2*N_FFT Hann
    full = torch.hann_window(2 * n_fft, periodic=False)
    return full[n_fft:]  # right half: peaks at index 0 (rightmost recent sample)


class PathADataset(Dataset):
    """Mel dataset with asymmetric window and frame-skipped teacher."""

    def __init__(self, indices, speakers, source_files, max_seconds=3.0,
                 frame_skip: int = 0):
        self.indices = [int(i) for i in indices]
        self.speakers = speakers
        self.source_files = source_files
        self.max_samples = int(max_seconds * SAMPLE_RATE)
        self.rng = random.Random(42)
        self.frame_skip = frame_skip  # 0 = normal, 1 = Path C

        # Custom asymmetric window
        self.window = asymmetric_hann_window(N_FFT)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        import soundfile as sf
        idx = self.indices[item]
        src = Path(str(self.source_files[idx]))
        wav, sr = sf.read(str(src), dtype="float32")
        wav = torch.from_numpy(np.asarray(wav))
        if wav.ndim == 2:
            wav = wav.mean(dim=1)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)
        if wav.shape[0] > self.max_samples:
            start = self.rng.randint(0, wav.shape[0] - self.max_samples)
            wav = wav[start:start + self.max_samples]
        elif wav.shape[0] < self.max_samples:
            wav = F.pad(wav, (0, self.max_samples - wav.shape[0]))

        # Mel with asymmetric window, center=False
        stft_spec = torch.stft(
            wav, n_fft=N_FFT, hop_length=MEL_HOP,
            win_length=N_FFT, window=self.window,
            center=False, return_complex=True,
        )
        mag = stft_spec.abs().clamp_min(1e-7)

        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=N_FFT // 2 + 1, f_min=0.0, f_max=SAMPLE_RATE / 2.0,
            n_mels=80, sample_rate=SAMPLE_RATE,
        ).T.to(mag.device)
        mel = torch.matmul(mel_fb, mag)  # (80, T)
        logmel = torch.log(torch.clamp(mel, min=1e-5))

        # Teacher content from cache (with optional frame skip)
        cache_dir = Path("data/mio_vctk_full_compact")
        npz = np.load(cache_dir / f"s_{idx:05d}.npz", allow_pickle=False)
        teacher_full = torch.from_numpy(npz["ce_768"].astype(np.float32))

        # Path C: frame-skip distillation
        if self.frame_skip > 0:
            teacher_full = F.pad(teacher_full, (0, 0, self.frame_skip, 0))[:-self.frame_skip]

        return {"mel": logmel.float(), "content": teacher_full.float(),
                "speaker": str(self.speakers[idx]), "idx": idx}


class MelCollator:
    def __init__(self, max_seconds):
        self.max_mel_frames = int(max_seconds * 50)
        self.max_content_frames = int(max_seconds * 25)

    def __call__(self, samples):
        B = len(samples)
        mels = torch.zeros(B, 80, self.max_mel_frames)
        contents = torch.zeros(B, self.max_content_frames, 768)
        masks = torch.zeros(B, self.max_content_frames, dtype=torch.bool)
        speakers, indices = [], []
        crop_starts = torch.zeros(B, dtype=torch.long)

        for i, s in enumerate(samples):
            m = s["mel"]
            mf = min(m.shape[1], self.max_mel_frames)
            mels[i, :, :mf] = m[:, :mf]

            c = s["content"]
            cf = min(c.shape[0], self.max_content_frames)
            contents[i, :cf] = c[:cf]
            masks[i, :cf] = True
            speakers.append(s["speaker"])
            indices.append(s["idx"])

        return Batch(
            mel=mels, content=contents,
            tokens=torch.zeros(B, self.max_content_frames, dtype=torch.long),
            mask=masks, speakers=speakers,
            indices=torch.tensor(indices, dtype=torch.long),
            crop_starts=crop_starts,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-seconds", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame-skip", type=int, default=0, help="Path C: 0=normal, 1=student(t)→teacher(t-1)")
    p.add_argument("--out-dir", type=Path, default=Path("/Volumes/UNTITLED/btrv5_checkpoints/path_ac"))
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    meta = np.load(DEFAULT_DATA_DIR / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"])
    speakers = meta["spk_names"][:n].astype(str)
    source_files = meta["source_files"][:n].astype(str)
    train_idx, val_idx = split_by_speaker(speakers, 0.05, args.seed)
    probe_idx = speaker_balanced_subset(val_idx, speakers, 256, args.seed)

    train_ds = PathADataset(train_idx, speakers, source_files, args.max_seconds, args.frame_skip)
    probe_ds = PathADataset(probe_idx, speakers, source_files, args.max_seconds, args.frame_skip)
    collator = MelCollator(args.max_seconds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    probe_loader = DataLoader(probe_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    config = MCSTransQ2D2Config(
        n_layers=4, trans_dim=512, n_heads=8, ffn_dim=1024, window=256,
        use_rope=True, use_swiglu=True,
        q2d2_dim=6, q2d2_levels=(9,9,9,9,9,9), q2d2_grid="rhombic",
    )
    model = MCSTransQ2D2(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args_ns = argparse.Namespace(
        content_cos_weight=1.0, content_l1_weight=0.5, delta_weight=0.04,
        voiced_boost=1.0, grl_weight=0.0,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    quantizer = model.q2d2.quantizer
    best_cos = -1.0
    t0 = time.time()

    label = f"Path A (asym window) + Path C (skip={args.frame_skip})"
    print(f"Config: {label}", flush=True)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    for epoch in range(args.epochs):
        model.train()
        totals = {}
        for step, batch in enumerate(train_loader, start=1):
            if step > args.steps_per_epoch:
                break
            batch = move_batch(batch, device)
            output = model(batch.mel, padding_mask=batch.mask)
            loss, metrics = q2d2_losses(output, batch, args_ns, quantizer)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            if step % 200 == 0:
                d = max(step, 1)
                print(f"E{epoch:03d} step={step:04d} loss={totals['loss']/d:.4f} "
                      f"cos768={totals['cos768']/d:.4f}", flush=True)
        scheduler.step()

        model.eval()
        pb = {}
        for batch in probe_loader:
            batch = move_batch(batch, device)
            output = model(batch.mel, padding_mask=batch.mask)
            _, m = q2d2_losses(output, batch, args_ns, quantizer)
            for k, v in m.items():
                pb.setdefault(k, []).append(v)
        model.train()
        probe = {k: float(np.mean(v)) for k, v in pb.items()}
        current = probe.get("cos768", 0)
        print(f"E{epoch:03d} probe cos768={current:.4f}", flush=True)

        metrics_full = {"epoch": epoch, "global_step": (epoch+1)*args.steps_per_epoch,
                        "probe": probe, "elapsed_seconds": time.time()-t0}
        save_checkpoint(args.out_dir / "last.pt", model, optimizer, scheduler,
                        epoch, metrics_full, args, best_cos)
        if current > best_cos:
            best_cos = current
            save_checkpoint(args.out_dir / "best.pt", model, optimizer, scheduler,
                            epoch, metrics_full, args, best_cos)
        (args.out_dir / "summary.json").write_text(json.dumps(metrics_full, indent=2)+"\n")

    print(f"done best_cos768={best_cos:.4f}  ({label})", flush=True)


if __name__ == "__main__":
    main()

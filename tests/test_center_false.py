"""Quick test: train mel-based MCS-Q2D2 with center=False (strictly causal mel)."""
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


class CenterFalseMelDataset(Dataset):
    """Load raw audio, compute mel with center=False on-the-fly."""

    def __init__(self, indices, speakers, source_files, max_seconds=3.0):
        self.indices = [int(i) for i in indices]
        self.speakers = speakers
        self.source_files = source_files
        self.max_samples = int(max_seconds * SAMPLE_RATE)
        self.rng = random.Random(42)

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

        # Compute mel with center=False (strictly causal!)
        mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=2048, hop_length=MEL_HOP,
            n_mels=80, f_min=0.0, f_max=SAMPLE_RATE/2.0,
            power=1, center=False,
        )
        mel = mel_fn(wav.unsqueeze(0))
        logmel = torch.log(torch.clamp(mel, min=1e-5))

        # Load teacher content from cache
        cache_dir = Path("data/mio_vctk_full_compact")
        npz = np.load(cache_dir / f"s_{idx:05d}.npz", allow_pickle=False)
        teacher = torch.from_numpy(npz["ce_768"].astype(np.float32))

        return {"mel": logmel[0].float(), "content": teacher.float(),
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
        speakers = []
        indices = []
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
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-seconds", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("/Volumes/UNTITLED/btrv5_checkpoints/mel_center_false"))
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Data
    meta = np.load(DEFAULT_DATA_DIR / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"])
    speakers = meta["spk_names"][:n].astype(str)
    source_files = meta["source_files"][:n].astype(str)
    train_idx, val_idx = split_by_speaker(speakers, 0.05, args.seed)
    probe_idx = speaker_balanced_subset(val_idx, speakers, 256, args.seed)

    train_ds = CenterFalseMelDataset(train_idx, speakers, source_files, args.max_seconds)
    probe_ds = CenterFalseMelDataset(probe_idx, speakers, source_files, args.max_seconds)
    collator = MelCollator(args.max_seconds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator)
    probe_loader = DataLoader(probe_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collator)

    # Model
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

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, center=False, device={device}", flush=True)

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

        # Probe
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
        metrics_full = {"epoch": epoch, "global_step": (epoch+1)*args.steps_per_epoch,
                        "probe": probe, "elapsed_seconds": time.time()-t0}
        current = probe.get("cos768", 0)
        print(f"E{epoch:03d} probe cos768={current:.4f}", flush=True)

        save_checkpoint(args.out_dir / "last.pt", model, optimizer, scheduler,
                        epoch, metrics_full, args, best_cos)
        if current > best_cos:
            best_cos = current
            save_checkpoint(args.out_dir / "best.pt", model, optimizer, scheduler,
                            epoch, metrics_full, args, best_cos)
        (args.out_dir / "summary.json").write_text(json.dumps(metrics_full, indent=2)+"\n")

    print(f"done best_cos768={best_cos:.4f}", flush=True)


if __name__ == "__main__":
    main()

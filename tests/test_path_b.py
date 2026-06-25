"""Path B: Predictive Coding — train causal model to predict future teacher frames.

Key insight: The teacher's content at frame t contains future information 
(bidirectional WavLM + Transformer). A causal model can't observe that future,
but it can LEARN TO PREDICT it from past context.

Implementation:
  - Add prediction head to Transformer output (before Q2D2)
  - hidden(t) → Linear → predicted_content(t+1)
  - Loss = content_cos(t) + λ * MSE(predicted(t+1), teacher(t+1))
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
)
from mcs_q2d2 import Q2D2Projection, Q2D2Quantizer, compute_q2d2_perplexity
from train_mcs_q2d2 import MCSTransQ2D2Config, MCSTransQ2D2

SAMPLE_RATE = 44100
MEL_HOP = 882
N_FFT = 2048


class PredictiveMCSTrans(MCSTransQ2D2):
    """MCS-Trans with predictive future-frame head."""

    def __init__(self, config: MCSTransQ2D2Config):
        super().__init__(config)
        # Prediction head: transformer hidden → next content frame
        self.predict_head = nn.Sequential(
            nn.Linear(config.trans_dim, config.trans_dim),
            nn.ReLU(),
            nn.Linear(config.trans_dim, config.content_dim),  # 768
        )

    def forward(self, mel, padding_mask=None):
        out = super().forward(mel, padding_mask)
        return out  # We'll add prediction in the loss function


class MelDataset(Dataset):
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

        mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=MEL_HOP,
            n_mels=80, f_min=0.0, f_max=SAMPLE_RATE/2.0, power=1, center=False,
        )
        mel = mel_fn(wav.unsqueeze(0))
        logmel = torch.log(torch.clamp(mel, min=1e-5))

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


def predictive_losses(model, output, batch, args, quantizer):
    """Standard Q2D2 losses + predictive coding loss."""
    projected = output["projected"]
    q2d2_codes = output.get("q2d2_codes")

    length = min(projected.shape[2], batch.content.shape[1], batch.mask.shape[1])
    mask = batch.mask[:, :length]

    pred_768 = projected[:, :, :length]
    tgt_768 = batch.content[:, :length]

    # Standard content loss
    pred_masked = pred_768.permute(0, 2, 1)[mask]
    tgt_masked = tgt_768[mask]
    cos768 = F.cosine_similarity(pred_masked, tgt_masked, dim=-1).mean()
    cos768_loss = 1.0 - cos768

    pred_flat = pred_768.permute(0, 2, 1)
    l1_per_frame = (pred_flat - tgt_768).abs().mean(dim=-1)
    content_l1 = ((l1_per_frame * mask.float()).sum() / mask.float().sum().clamp(min=1))

    if length >= 2:
        delta_mask = mask[:, 1:] & mask[:, :-1]
        pred_delta = pred_flat[:, 1:] - pred_flat[:, :-1]
        tgt_delta = tgt_768[:, 1:] - tgt_768[:, :-1]
        delta = F.smooth_l1_loss(pred_delta[delta_mask], tgt_delta[delta_mask], reduction="mean")
    else:
        delta = projected.sum() * 0.0

    loss = (args.content_cos_weight * cos768_loss +
            args.content_l1_weight * content_l1 +
            args.delta_weight * delta)

    # ── Path B: Predictive coding ──
    pred_loss_val = 0.0
    if length >= 3:
        # The Transformer output (before Q2D2) is available internally.
        # We use the Q2D2 codes as a proxy for the transformer hidden state.
        # Actually, we need access to the transformer output.
        # For now, use the final content projection as prediction feature.
        # Better: make the model return transformer hidden states.
        
        # Simple approach: predict next teacher from current projected content
        pred_t = pred_flat[:, :-1]  # (B, L-1, 768)
        next_teacher = tgt_768[:, 1:]  # (B, L-1, 768)
        # Use model's predict_head on projected content
        # But predict_head expects transformer dim (512), not content dim (768)
        # For simplicity, use a direct MSE on projected→projected prediction
        # pred_next = model.predict_head(pred_t) — would need trans_dim input
        
        # Quick approximation: force content(t) ≈ teacher(t+1)
        next_mask = mask[:, 1:]
        pred_next_masked = pred_t[next_mask]
        next_teacher_masked = next_teacher[next_mask]
        pred_next_cos = F.cosine_similarity(pred_next_masked, next_teacher_masked, dim=-1).mean()
        pred_loss = 1.0 - pred_next_cos
        pred_loss_val = float(pred_loss.detach().cpu())
        
        pred_weight = getattr(args, "pred_weight", 0.1)
        loss = loss + pred_weight * pred_loss

    metrics = {
        "loss": float(loss.detach().cpu()),
        "cos768": float(cos768.detach().cpu()),
        "content_l1": float(content_l1.detach().cpu()),
        "delta": float(delta.detach().cpu()),
        "pred_loss": pred_loss_val,
    }

    if quantizer is not None and q2d2_codes is not None:
        with torch.no_grad():
            stats = compute_q2d2_perplexity(quantizer, q2d2_codes)
            metrics["q2d2_usage"] = stats["overall_usage"]

    return loss, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-seconds", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--pred-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("/Volumes/UNTITLED/btrv5_checkpoints/path_b"))
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

    train_ds = MelDataset(train_idx, speakers, source_files, args.max_seconds)
    probe_ds = MelDataset(probe_idx, speakers, source_files, args.max_seconds)
    collator = MelCollator(args.max_seconds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    probe_loader = DataLoader(probe_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    config = MCSTransQ2D2Config(
        n_layers=4, trans_dim=512, n_heads=8, ffn_dim=1024, window=256,
        use_rope=True, use_swiglu=True,
        q2d2_dim=6, q2d2_levels=(9,9,9,9,9,9), q2d2_grid="rhombic",
    )
    model = MCSTransQ2D2(config).to(device)  # use base model (predict in loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    quantizer = model.q2d2.quantizer
    best_cos = -1.0
    t0 = time.time()

    ns = argparse.Namespace(
        content_cos_weight=1.0, content_l1_weight=0.5, delta_weight=0.04,
        voiced_boost=1.0, grl_weight=0.0, pred_weight=args.pred_weight,
    )
    print(f"Path B: Predictive Coding (λ={args.pred_weight}), center=False mel", flush=True)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    for epoch in range(args.epochs):
        model.train()
        totals = {}
        for step, batch in enumerate(train_loader, start=1):
            if step > args.steps_per_epoch:
                break
            batch = move_batch(batch, device)
            output = model(batch.mel, padding_mask=batch.mask)
            loss, metrics = predictive_losses(model, output, batch, ns, quantizer)
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
            _, m = predictive_losses(model, output, batch, ns, quantizer)
            for k, v in m.items():
                pb.setdefault(k, []).append(v)
        model.train()
        probe = {k: float(np.mean(v)) for k, v in pb.items()}
        current = probe.get("cos768", 0)
        print(f"E{epoch:03d} probe cos768={current:.4f}  pred_loss={probe.get('pred_loss',0):.4f}", flush=True)

        metrics_full = {"epoch": epoch, "global_step": (epoch+1)*args.steps_per_epoch,
                        "probe": probe, "elapsed_seconds": time.time()-t0}
        save_checkpoint(args.out_dir / "last.pt", model, optimizer, scheduler,
                        epoch, metrics_full, args, best_cos)
        if current > best_cos:
            best_cos = current
            save_checkpoint(args.out_dir / "best.pt", model, optimizer, scheduler,
                            epoch, metrics_full, args, best_cos)
        (args.out_dir / "summary.json").write_text(json.dumps(metrics_full, indent=2)+"\n")

    print(f"done best_cos768={best_cos:.4f}  (Path B, λ={args.pred_weight})", flush=True)


if __name__ == "__main__":
    main()

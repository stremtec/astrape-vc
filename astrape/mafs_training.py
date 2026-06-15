"""Training loop for manifold-aligned factorized student (MAFS).

The critical design decision: all content losses operate in the 5-dimensional
continuous code space, not in the 768-dimensional projected space. This
eliminates the metric inflation problem identified in the plateau analysis
(768d cosine ≈ 0.907 corresponds to only 0.835 true 5d-cosine).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .data import ContentBatch, ContentCollator, MioContentDataset, speaker_disjoint_split
from .flat_ctc_training import (
    _ctc_loss, _edit_distance, _greedy_ctc_sequences, _split_targets,
    speaker_balanced_subset,
)
from .fsq import indices_to_codes, masked_fsq_cross_entropy
from .mafs_model import (
    MafsConfig, MafsModel, MafsOutput, load_mafs_checkpoint, save_mafs_checkpoint,
)
from .training import seed_everything


@dataclass(frozen=True)
class MafsTrainingConfig:
    data_dir: Path
    projection_path: Path
    output_dir: Path
    run_name: str = "mafs_384x4_512x8"
    device: str = "mps"
    batch_size: int = 2
    epochs: int = 30
    steps_per_epoch: int = 1000
    learning_rate: float = 2e-4
    scheduler_t_max: int = 30
    weight_decay: float = 1e-5
    validation_fraction: float = 0.15
    supervised_mel_frames: int = 300
    history_mel_frames: int = 100
    pad_mel_multiple: int = 64
    ctc_weight: float = 0.05
    future_weight: float = 0.1
    delta_weight: float = 0.03
    initial_axis_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.4, 1.5)
    seed: int = 42
    num_workers: int = 0
    probe_samples: int = 1024
    full_validation_every: int = 5
    target_cosine: float = 0.92
    log_every: int = 50
    resume: Optional[Path] = None


# ── helpers ─────────────────────────────────────────────────────────────────────

def _move(batch: ContentBatch, device: torch.device) -> ContentBatch:
    return ContentBatch(
        mel=batch.mel.to(device),
        content=batch.content.to(device),
        pre_fsq=None,
        token_indices=batch.token_indices.to(device) if batch.token_indices is not None else None,
        input_lengths=batch.input_lengths.to(device),
        target_lengths=batch.target_lengths.to(device),
        target_mask=batch.target_mask.to(device),
        transcripts=batch.transcripts.to(device) if batch.transcripts is not None else None,
        transcript_lengths=batch.transcript_lengths.to(device) if batch.transcript_lengths is not None else None,
    )


def _code_delta_loss(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    pd = pred[:, 1:] - pred[:, :-1]
    td = tgt[:, 1:] - tgt[:, :-1]
    pm = mask[:, 1:] & mask[:, :-1]
    if not pm.any():
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pd[pm].contiguous(), td[pm].contiguous())


def _future_loss(pred_future: torch.Tensor, target_codes: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    """pred_future: [B, T, H=5, 5]  —  H future tokens × 5 axes"""
    horizon = pred_future.shape[2]
    losses = []
    for h in range(horizon):
        if pred_future.shape[1] <= h + 1:
            continue
        valid = mask[:, :-h-1] & mask[:, h+1:]
        if valid.any():
            losses.append(
                F.smooth_l1_loss(
                    pred_future[:, :-(h+1), h][valid].contiguous(),
                    target_codes[:, h+1:][valid].contiguous(),
                )
            )
    if not losses:
        return pred_future.sum() * 0.0
    return torch.stack(losses).mean()


# ── loss ────────────────────────────────────────────────────────────────────────

def mafs_loss(output: MafsOutput, batch: ContentBatch, config: MafsTrainingConfig,
              axis_weights: torch.Tensor, fsq_levels: tuple[int, ...]) -> tuple[torch.Tensor, dict[str, float]]:
    if batch.token_indices is None:
        raise RuntimeError("MAFS training requires cached FSQ tokens")
    L = min(output.codes.shape[1], batch.content.shape[1],
            batch.token_indices.shape[1], batch.target_mask.shape[1])
    pred_codes = output.codes[:, :L]
    mask = batch.target_mask[:, :L]

    tgt_codes = indices_to_codes(batch.token_indices[:, :L], fsq_levels).to(
        device=pred_codes.device, dtype=pred_codes.dtype)

    # --- 5d content loss (THE core loss) ---
    # Cosine in 5d space
    frame_cos = F.cosine_similarity(pred_codes, tgt_codes, dim=-1)
    cos_loss = (1.0 - frame_cos[mask]).mean()

    # Axis-weighted L1 in 5d space
    diff = (pred_codes - tgt_codes).abs()
    weighted_diff = diff * axis_weights.to(diff.device)
    code_l1 = weighted_diff[mask].mean()

    # --- Ordinal (per-axis FSQ cross-entropy) ---
    ord_loss, ord_acc, exact_acc = masked_fsq_cross_entropy(
        tuple(logits[:, :, :L] for logits in output.fsq_logits),
        batch.token_indices[:, :L],
        mask,
        fsq_levels,
    )

    # --- Delta (temporal smoothness) ---
    delta = _code_delta_loss(pred_codes, tgt_codes, mask)

    # --- Future prediction auxiliary ---
    future = _future_loss(output.future_codes[:, :L], tgt_codes, mask)

    # --- CTC (from edge — detached! Only when transcripts available) ---
    ctc = torch.tensor(0.0, device=pred_codes.device)
    if output.text_logits is not None and batch.transcripts is not None:
        ctc = _ctc_loss(output.text_logits, batch,
                        torch.nn.CTCLoss(blank=0, zero_infinity=True))

    loss = cos_loss + code_l1 \
        + 0.1 * ord_loss \
        + config.delta_weight * delta \
        + config.future_weight * future \
        + config.ctc_weight * ctc

    return loss, {
        "cos_loss": cos_loss.item(),
        "code_l1": code_l1.item(),
        "frame_cosine": frame_cos[mask].mean().item(),
        "ord_loss": ord_loss.item(),
        "ord_accuracy": ord_acc.item(),
        "exact_accuracy": exact_acc.item(),
        "delta_loss": delta.item(),
        "future_loss": future.item(),
        "ctc_loss": ctc.item() if isinstance(ctc, torch.Tensor) else ctc,
    }


def update_axis_weights(grad_norms: list[float], current_weights: torch.Tensor) -> torch.Tensor:
    """Adapt axis weights based on gradient norms, bounded to [0.5, 2.0]."""
    if len(grad_norms) != len(current_weights):
        return current_weights
    norms = torch.tensor(grad_norms, dtype=torch.float32)
    if norms.sum() == 0:
        return current_weights
    target = len(norms) * norms / norms.sum()  # normalize to mean=1
    new = 0.9 * current_weights + 0.1 * target
    return new.clamp(0.5, 2.0)


# ── evaluation ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate_mafs(model: MafsModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    frame_cos_5d: list[torch.Tensor] = []
    frame_cos_768: list[torch.Tensor] = []
    seq_cos_5d: list[float] = []
    axis_correct: list[torch.Tensor] = []
    exact_correct: list[torch.Tensor] = []
    total_ctc = 0.0
    total_edits = 0
    total_chars = 0
    batches = 0

    for raw_batch in loader:
        batch = _move(raw_batch, device)
        if batch.token_indices is None:
            continue
        out = model(batch.mel, batch.input_lengths)
        L = min(out.codes.shape[1], batch.content.shape[1],
                batch.token_indices.shape[1], batch.target_mask.shape[1])
        mask = batch.target_mask[:, :L]
        pred = out.codes[:, :L]
        tgt = indices_to_codes(batch.token_indices[:, :L], model.config.fsq_levels).to(
            device=pred.device, dtype=pred.dtype)

        # 5d cosine
        cos5 = F.cosine_similarity(pred, tgt, dim=-1)
        frame_cos_5d.append(cos5[mask].cpu())

        # 768d projected cosine (for reference only)
        pred768 = out.projected[:, :, :L].transpose(1, 2)
        tgt768 = batch.content[:, :L]
        cos768 = F.cosine_similarity(pred768, tgt768, dim=-1)
        frame_cos_768.append(cos768[mask].cpu())

        # Sequence cosine in 5d
        for b in range(pred.shape[0]):
            valid = mask[b]
            if valid.any():
                seq_cos_5d.append(
                    F.cosine_similarity(
                        pred[b, valid].reshape(1, -1),
                        tgt[b, valid].reshape(1, -1),
                    ).item()
                )

        # FSQ accuracy
        if out.fsq_logits:
            tgt_levels = _token_to_levels(batch.token_indices[:, :L], model.config.fsq_levels, device)
            pred_levels = torch.stack([lg[:, :, :L].argmax(dim=1) for lg in out.fsq_logits], dim=-1)
            correct = pred_levels == tgt_levels
            axis_correct.append(correct[mask].float().cpu())
            exact_correct.append(correct.all(dim=-1)[mask].float().cpu())

        # CTC
        if out.text_logits is not None and batch.transcripts is not None:
            crit = torch.nn.CTCLoss(blank=0, zero_infinity=True)
            total_ctc += _ctc_loss(out.text_logits, batch, crit).item()
            exp = _split_targets(batch.transcripts.cpu(), batch.transcript_lengths.cpu())
            pred_seq = _greedy_ctc_sequences(out.text_logits, batch.input_lengths)
            for h, r in zip(pred_seq, exp):
                total_edits += _edit_distance(h, r)
                total_chars += len(r)
        batches += 1

    c5 = torch.cat(frame_cos_5d)
    c768 = torch.cat(frame_cos_768) if frame_cos_768 else c5
    result = {
        "val_5d_cosine": c5.mean().item(),
        "val_5d_p05": c5.quantile(0.05).item(),
        "val_768_cosine": c768.mean().item(),
        "val_768_p05": c768.quantile(0.05).item(),
        "val_seq_cosine": float(np.mean(seq_cos_5d)) if seq_cos_5d else 0.0,
    }
    if axis_correct:
        ac = torch.cat(axis_correct)
        ec = torch.cat(exact_correct)
        result["val_ordinal_accuracy"] = ac.mean().item()
        result["val_exact_accuracy"] = ec.mean().item()
        for a in range(ac.shape[1]):
            result[f"val_axis_{a}_acc"] = ac[:, a].mean().item()
    if batches:
        result["val_ctc_loss"] = total_ctc / batches
        result["val_cer"] = total_edits / max(total_chars, 1)
    return result


def _token_to_levels(tokens: torch.Tensor, levels: tuple[int, ...], device: torch.device) -> torch.Tensor:
    out = []
    d = 1
    for L in levels:
        out.append(((tokens // d) % L).to(device))
        d *= L
    return torch.stack(out, dim=-1)


# ── main training loop ──────────────────────────────────────────────────────────

def train_mafs(model_config: MafsConfig, train_config: MafsTrainingConfig) -> None:
    seed_everything(train_config.seed)
    device = torch.device(train_config.device)

    # Data
    with np.load(train_config.data_dir / "meta.npz") as meta:
        fmt = str(meta["cache_format"].item())
        if fmt != "compact-fp16-ctc-v2":
            raise ValueError(f"MAFS requires compact-fp16-ctc-v2 cache (got {fmt})")
        speakers = meta["spk_names"][:int(meta["n_samples"])].astype(str)
    train_idx, val_idx = speaker_disjoint_split(speakers, train_config.validation_fraction, train_config.seed)
    probe_idx = speaker_balanced_subset(val_idx, speakers, train_config.probe_samples, train_config.seed)

    # Train loader: no transcripts (CTC evaluated only during probe/full validation
    # because transcript alignment requires full utterances, not random crops).
    collator = ContentCollator(train_config.supervised_mel_frames, train_config.seed,
                               history_mel_frames=train_config.history_mel_frames,
                               pad_mel_multiple=train_config.pad_mel_multiple,
                               include_transcripts=False)
    train_loader = DataLoader(
        MioContentDataset(train_config.data_dir, train_config.data_dir, train_idx),
        batch_size=train_config.batch_size, shuffle=True,
        num_workers=train_config.num_workers, collate_fn=collator,
        generator=torch.Generator().manual_seed(train_config.seed),
    )
    probe_collator = ContentCollator(None, train_config.seed,
                                     pad_mel_multiple=train_config.pad_mel_multiple,
                                     include_transcripts=True)
    probe_loader = DataLoader(
        MioContentDataset(train_config.data_dir, train_config.data_dir, probe_idx),
        batch_size=train_config.batch_size, shuffle=False,
        num_workers=train_config.num_workers, collate_fn=probe_collator,
    )
    full_loader = DataLoader(
        MioContentDataset(train_config.data_dir, train_config.data_dir, val_idx),
        batch_size=train_config.batch_size, shuffle=False,
        num_workers=train_config.num_workers, collate_fn=probe_collator,
    )

    # Model
    if train_config.resume is not None:
        model, metadata = load_mafs_checkpoint(train_config.resume, device=device)
        start_epoch = int(metadata["epoch"]) + 1
        best_probe = float(metadata["metrics"].get("best_probe_5d_cosine", -1))
        best_full = float(metadata["metrics"].get("best_full_5d_cosine", -1))
    else:
        model = MafsModel(model_config).to(device)
        proj = torch.load(train_config.projection_path, map_location=device)
        model.load_fsq_projection(proj)
        start_epoch = 0
        best_probe = -1.0
        best_full = -1.0

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config.scheduler_t_max)
    if train_config.resume is not None:
        payload = torch.load(train_config.resume, map_location=device)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    axis_weights = torch.tensor(train_config.initial_axis_weights, dtype=torch.float32)

    train_config.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = train_config.output_dir / f"{train_config.run_name}.last.pt"
    best_path = train_config.output_dir / f"{train_config.run_name}.best.pt"
    pbest_path = train_config.output_dir / f"{train_config.run_name}.probe-best.pt"
    steps = min(train_config.steps_per_epoch, len(train_loader))

    print(f"Train={len(train_idx)} Probe={len(probe_idx)} FullVal={len(val_idx)}", flush=True)
    print(f"Params={sum(p.numel() for p in model.parameters()):,} Device={device} "
          f"crop={train_config.supervised_mel_frames} hist={train_config.history_mel_frames} "
          f"epochs={train_config.epochs}x{steps} target_5d={train_config.target_cosine}", flush=True)

    for epoch in range(start_epoch, train_config.epochs):
        model.train()
        totals: dict[str, float] = {}
        axis_grads: list[float] = [0.0] * 5
        started = time.perf_counter()
        for step, raw_batch in enumerate(train_loader, start=1):
            if step > steps:
                break
            batch = _move(raw_batch, device)
            out = model(batch.mel, batch.input_lengths)

            # Collect axis gradient norms for adaptive weighting
            if step % 50 == 0:
                for a in range(5):
                    g = out.codes[:, :, a].abs().sum()
                    g.backward(retain_graph=(a < 4))
                    axis_grads[a] += model.code_head.weight.grad.abs().sum().item() if model.code_head.weight.grad is not None else 0
                    model.zero_grad(set_to_none=True)
                out = model(batch.mel, batch.input_lengths)  # re-forward

            loss, parts = mafs_loss(out, batch, train_config, axis_weights,
                                    model.config.fsq_levels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v

            if step % 50 == 0 and sum(axis_grads) > 0:
                axis_weights = update_axis_weights(axis_grads, axis_weights)
                axis_grads = [0.0] * 5

            if step % train_config.log_every == 0 or step == steps:
                elapsed = time.perf_counter() - started
                n = step
                print(f"E{epoch:03d} step={step}/{steps} "
                      f"loss={totals.get('cos_loss',0)/n + totals.get('code_l1',0)/n:.4f} "
                      f"cos5={totals.get('frame_cosine',0)/n:.4f} "
                      f"ord={totals.get('ord_accuracy',0)/n:.4f} "
                      f"exact={totals.get('exact_accuracy',0)/n:.4f} "
                      f"ctc={totals.get('ctc_loss',0)/n:.4f} "
                      f"aw={','.join(f'{w:.1f}' for w in axis_weights.tolist())} "
                      f"{elapsed/step:.3f}s/step", flush=True)

        scheduler.step()

        # Probe evaluation
        probe = evaluate_mafs(model, probe_loader, device)
        metrics = {f"probe_{k}": v for k, v in probe.items()}
        metrics.update({
            "train_loss": sum(totals.get(k, 0) for k in ("cos_loss", "code_l1")) / steps,
            "train_5d_cosine": totals.get("frame_cosine", 0) / steps,
            "train_ord_accuracy": totals.get("ord_accuracy", 0) / steps,
            "train_ctc_loss": totals.get("ctc_loss", 0) / steps,
            "best_probe_5d_cosine": max(best_probe, probe.get("val_5d_cosine", -1)),
        })

        full_due = (epoch + 1) % train_config.full_validation_every == 0 or epoch + 1 == train_config.epochs
        if full_due:
            full = evaluate_mafs(model, full_loader, device)
            metrics.update(full)
            metrics["best_full_5d_cosine"] = max(best_full, full.get("val_5d_cosine", -1))

        probe_better = probe.get("val_5d_cosine", -1) > best_probe
        full_better = full_due and metrics.get("val_5d_cosine", -1) > best_full
        if probe_better: best_probe = probe["val_5d_cosine"]
        if full_better: best_full = metrics["val_5d_cosine"]

        save_mafs_checkpoint(last_path, model, epoch=epoch, metrics=metrics,
                             optimizer=optimizer, scheduler=scheduler)
        if probe_better:
            save_mafs_checkpoint(pbest_path, model, epoch=epoch, metrics=metrics)
        if full_better:
            save_mafs_checkpoint(best_path, model, epoch=epoch, metrics=metrics)

        print(f"E{epoch:03d} probe_5d={probe['val_5d_cosine']:.4f} "
              f"p05={probe['val_5d_p05']:.4f} "
              f"768_ref={probe['val_768_cosine']:.4f} "
              f"ord={probe.get('val_ordinal_accuracy', 0):.4f} "
              f"exact={probe.get('val_exact_accuracy', 0):.4f} "
              f"gap={max(0, train_config.target_cosine - probe['val_5d_cosine']):.4f}", flush=True)
        if full_due:
            print(f"E{epoch:03d} full_5d={metrics['val_5d_cosine']:.4f} "
                  f"768={metrics.get('val_768_cosine', 0):.4f} "
                  f"cer={metrics.get('val_cer', 0):.4f}", flush=True)

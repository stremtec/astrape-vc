from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .data import (
    ContentBatch,
    ContentCollator,
    MioContentDataset,
    speaker_disjoint_split,
)
from .flat_ctc_training import speaker_balanced_subset
from .fsq import indices_to_codes, masked_fsq_cross_entropy
from .token_student import (
    TokenStudentConfig,
    TokenSynchronousStudent,
    load_token_checkpoint,
    save_token_checkpoint,
)
from .training import seed_everything


@dataclass(frozen=True)
class TokenPhase0Config:
    data_dir: Path
    projection_path: Path
    output_dir: Path
    run_name: str = "content_student_token_sync_phase0"
    device: str = "mps"
    batch_size: int = 2
    epochs: int = 3
    steps_per_epoch: int = 1000
    learning_rate: float = 2e-4
    scheduler_t_max_epochs: int = 10
    weight_decay: float = 1e-5
    validation_fraction: float = 0.15
    supervised_mel_frames: int = 300
    history_mel_frames: int = 100
    pad_mel_multiple: int = 64
    ordinal_weight: float = 0.1
    delta_weight: float = 0.03
    future_weight: float = 0.0
    seed: int = 42
    num_workers: int = 0
    probe_samples: int = 1024
    full_validation_every: int = 3
    target_cosine: float = 0.885
    log_every: int = 50
    resume: Path | None = None


def validate_phase0_config(config: TokenPhase0Config) -> None:
    if config.batch_size <= 0 or config.epochs <= 0:
        raise ValueError("batch_size and epochs must be positive")
    if config.steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.scheduler_t_max_epochs <= 0:
        raise ValueError("scheduler_t_max_epochs must be positive")
    if config.supervised_mel_frames <= 0 or config.supervised_mel_frames % 2:
        raise ValueError("supervised_mel_frames must be a positive even number")
    if config.history_mel_frames < 0 or config.history_mel_frames % 2:
        raise ValueError("history_mel_frames must be a non-negative even number")
    if config.pad_mel_multiple <= 0:
        raise ValueError("pad_mel_multiple must be positive")
    if config.ordinal_weight < 0 or config.delta_weight < 0:
        raise ValueError("auxiliary loss weights must be non-negative")
    if config.future_weight < 0:
        raise ValueError("future_weight must be non-negative")
    if config.probe_samples <= 0 or config.full_validation_every <= 0:
        raise ValueError("validation settings must be positive")
    if not 0.0 <= config.target_cosine <= 1.0:
        raise ValueError("target_cosine must be between zero and one")


def move_token_batch(
    batch: ContentBatch,
    device: torch.device,
) -> ContentBatch:
    return ContentBatch(
        mel=batch.mel.to(device),
        content=batch.content.to(device),
        pre_fsq=None,
        token_indices=(
            batch.token_indices.to(device)
            if batch.token_indices is not None
            else None
        ),
        input_lengths=batch.input_lengths.to(device),
        target_lengths=batch.target_lengths.to(device),
        target_mask=batch.target_mask.to(device),
        transcripts=None,
        transcript_lengths=None,
    )


def _masked_delta_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape[1] < 2:
        return prediction.sum() * 0.0
    prediction_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    pair_mask = mask[:, 1:] & mask[:, :-1]
    if not pair_mask.any():
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(
        prediction_delta[pair_mask].contiguous(),
        target_delta[pair_mask].contiguous(),
    )


def _future_code_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for horizon in range(1, prediction.shape[2] + 1):
        if prediction.shape[1] <= horizon:
            continue
        valid = mask[:, :-horizon] & mask[:, horizon:]
        if valid.any():
            losses.append(
                F.smooth_l1_loss(
                    prediction[:, :-horizon, horizon - 1][valid].contiguous(),
                    target[:, horizon:][valid].contiguous(),
                )
            )
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def token_phase0_loss(
    model: TokenSynchronousStudent,
    batch: ContentBatch,
    config: TokenPhase0Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    if batch.token_indices is None:
        raise RuntimeError("Token-synchronous training requires cached FSQ tokens")
    output = model(batch.mel, batch.input_lengths)
    length = min(
        output.codes.shape[1],
        batch.content.shape[1],
        batch.token_indices.shape[1],
        batch.target_mask.shape[1],
    )
    codes = output.codes[:, :length]
    target_codes = indices_to_codes(
        batch.token_indices[:, :length],
        model.config.fsq_levels,
    ).to(codes.dtype)
    mask = batch.target_mask[:, :length]
    predicted_content = output.content[:, :, :length].transpose(1, 2)
    target_content = batch.content[:, :length]
    frame_cosine = F.cosine_similarity(
        predicted_content,
        target_content,
        dim=-1,
    )
    cosine = frame_cosine[mask].mean()
    axis_weights = codes.new_tensor([1.0, 1.0, 1.0, 1.4, 1.5])
    code_loss = (
        F.smooth_l1_loss(
            codes[mask].contiguous(),
            target_codes[mask].contiguous(),
            reduction="none",
        )
        * axis_weights
    ).mean()
    ordinal_loss, ordinal_accuracy, exact_accuracy = masked_fsq_cross_entropy(
        tuple(logits[:, :, :length] for logits in output.fsq_logits),
        batch.token_indices[:, :length],
        mask,
        model.config.fsq_levels,
    )
    delta_loss = _masked_delta_loss(codes, target_codes, mask)
    future_loss = _future_code_loss(
        output.future_codes[:, :length],
        target_codes,
        mask,
    )
    loss = (
        1.0
        - cosine
        + code_loss
        + config.ordinal_weight * ordinal_loss
        + config.delta_weight * delta_loss
        + config.future_weight * future_loss
    )
    return loss, {
        "content_cosine": cosine.item(),
        "code_loss": code_loss.item(),
        "ordinal_loss": ordinal_loss.item(),
        "ordinal_accuracy": ordinal_accuracy.item(),
        "exact_accuracy": exact_accuracy.item(),
        "delta_loss": delta_loss.item(),
        "future_loss": future_loss.item(),
    }


@torch.inference_mode()
def evaluate_token_student(
    model: TokenSynchronousStudent,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    frame_cosines = []
    sequence_cosines = []
    axis_errors = []
    ordinal_correct = []
    exact_correct = []
    for raw_batch in loader:
        batch = move_token_batch(raw_batch, device)
        if batch.token_indices is None:
            raise RuntimeError("Validation requires cached FSQ tokens")
        output = model(batch.mel, batch.input_lengths)
        length = min(
            output.codes.shape[1],
            batch.content.shape[1],
            batch.token_indices.shape[1],
            batch.target_mask.shape[1],
        )
        mask = batch.target_mask[:, :length]
        predicted_content = output.content[:, :, :length].transpose(1, 2)
        target_content = batch.content[:, :length]
        cosines = F.cosine_similarity(
            predicted_content,
            target_content,
            dim=-1,
        )
        frame_cosines.append(cosines[mask].cpu())
        target_codes = indices_to_codes(
            batch.token_indices[:, :length],
            model.config.fsq_levels,
        ).to(output.codes.dtype)
        axis_errors.append(
            (output.codes[:, :length][mask] - target_codes[mask]).abs().cpu()
        )
        target_levels = []
        divisor = 1
        for level in model.config.fsq_levels:
            target_levels.append(
                (batch.token_indices[:, :length] // divisor) % level
            )
            divisor *= level
        target_levels_tensor = torch.stack(target_levels, dim=-1)
        predicted_levels = torch.stack(
            [
                logits[:, :, :length].argmax(dim=1)
                for logits in output.fsq_logits
            ],
            dim=-1,
        )
        correct = predicted_levels == target_levels_tensor
        ordinal_correct.append(correct[mask].float().cpu())
        exact_correct.append(correct.all(dim=-1)[mask].float().cpu())
        for item in range(predicted_content.shape[0]):
            valid = mask[item]
            if valid.any():
                sequence_cosines.append(
                    F.cosine_similarity(
                        predicted_content[item, valid].reshape(1, -1),
                        target_content[item, valid].reshape(1, -1),
                    ).item()
                )
    cosines = torch.cat(frame_cosines)
    errors = torch.cat(axis_errors)
    correct = torch.cat(ordinal_correct)
    exact = torch.cat(exact_correct)
    metrics = {
        "val_frame_cosine": cosines.mean().item(),
        "val_frame_cosine_p05": torch.quantile(cosines, 0.05).item(),
        "val_sequence_cosine": float(np.mean(sequence_cosines)),
        "val_ordinal_accuracy": correct.mean().item(),
        "val_exact_accuracy": exact.mean().item(),
    }
    for axis, value in enumerate(errors.mean(dim=0).tolist()):
        metrics[f"val_axis_{axis}_mae"] = value
    return metrics


def train_token_phase0(
    model_config: TokenStudentConfig,
    config: TokenPhase0Config,
) -> None:
    validate_phase0_config(config)
    seed_everything(config.seed)
    device = torch.device(config.device)
    projection = torch.load(config.projection_path, map_location="cpu")
    with np.load(config.data_dir / "meta.npz") as meta:
        if str(meta["cache_format"].item()) != "compact-fp16-ctc-v2":
            raise ValueError("Phase 0 requires compact-fp16-ctc-v2 cache")
        speakers = meta["spk_names"][: int(meta["n_samples"])].astype(str)
    train_indices, validation_indices = speaker_disjoint_split(
        speakers,
        config.validation_fraction,
        config.seed,
    )
    probe_indices = speaker_balanced_subset(
        validation_indices,
        speakers,
        config.probe_samples,
        config.seed,
    )
    train_loader = DataLoader(
        MioContentDataset(config.data_dir, config.data_dir, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=ContentCollator(
            config.supervised_mel_frames,
            config.seed,
            history_mel_frames=config.history_mel_frames,
            pad_mel_multiple=config.pad_mel_multiple,
        ),
        generator=torch.Generator().manual_seed(config.seed),
    )
    probe_loader = DataLoader(
        MioContentDataset(config.data_dir, config.data_dir, probe_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=ContentCollator(
            None,
            config.seed,
            pad_mel_multiple=config.pad_mel_multiple,
        ),
    )
    full_loader = DataLoader(
        MioContentDataset(config.data_dir, config.data_dir, validation_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=ContentCollator(
            None,
            config.seed,
            pad_mel_multiple=config.pad_mel_multiple,
        ),
    )
    start_epoch = 0
    best_probe = -float("inf")
    best_full = -float("inf")
    if config.resume is None:
        model = TokenSynchronousStudent(model_config).to(device)
        model.load_fsq_projection(projection)
    else:
        model, metadata = load_token_checkpoint(config.resume, device=device)
        if model.config != model_config:
            raise ValueError("Resume checkpoint config does not match")
        start_epoch = int(metadata["epoch"]) + 1
        best_probe = float(metadata["metrics"].get("best_probe_cosine", best_probe))
        best_full = float(metadata["metrics"].get("best_full_cosine", best_full))
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.scheduler_t_max_epochs,
    )
    if config.resume is not None:
        payload = torch.load(config.resume, map_location=device)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = config.output_dir / f"{config.run_name}.last.pt"
    probe_best_path = config.output_dir / f"{config.run_name}.probe-best.pt"
    best_path = config.output_dir / f"{config.run_name}.best.pt"
    steps_per_epoch = min(config.steps_per_epoch, len(train_loader))
    print(
        f"Train={len(train_indices)} Probe={len(probe_indices)} "
        f"FullVal={len(validation_indices)}",
        flush=True,
    )
    print(
        f"Params={sum(p.numel() for p in model.parameters()):,} "
        f"Device={device} crop={config.supervised_mel_frames} "
        f"history={config.history_mel_frames} "
        f"pad_multiple={config.pad_mel_multiple} "
        f"epochs={config.epochs}x{steps_per_epoch} "
        f"scheduler_tmax={config.scheduler_t_max_epochs} "
        f"target={config.target_cosine:.3f}",
        flush=True,
    )
    for epoch in range(start_epoch, config.epochs):
        model.train()
        totals: dict[str, float] = {}
        started = time.perf_counter()
        for step, raw_batch in enumerate(train_loader, start=1):
            if step > steps_per_epoch:
                break
            batch = move_token_batch(raw_batch, device)
            loss, parts = token_phase0_loss(model, batch, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            totals["loss"] = totals.get("loss", 0.0) + loss.item()
            totals["gradient_norm"] = (
                totals.get("gradient_norm", 0.0) + float(gradient_norm)
            )
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            if step % config.log_every == 0 or step == steps_per_epoch:
                elapsed = time.perf_counter() - started
                print(
                    f"E{epoch:03d} step={step}/{steps_per_epoch} "
                    f"loss={totals['loss'] / step:.4f} "
                    f"cos={totals['content_cosine'] / step:.4f} "
                    f"code={totals['code_loss'] / step:.4f} "
                    f"ord={totals['ordinal_accuracy'] / step:.4f} "
                    f"exact={totals['exact_accuracy'] / step:.4f} "
                    f"grad={totals['gradient_norm'] / step:.3f} "
                    f"{elapsed / step:.3f}s/step",
                    flush=True,
                )
        scheduler.step()
        probe = evaluate_token_student(model, probe_loader, device)
        metrics = {
            key.replace("val_", "probe_", 1): value
            for key, value in probe.items()
        }
        metrics.update(
            {
                "train_loss": totals["loss"] / steps_per_epoch,
                "train_content_cosine": totals["content_cosine"] / steps_per_epoch,
                "best_probe_cosine": max(
                    best_probe,
                    probe["val_frame_cosine"],
                ),
            }
        )
        full_due = (
            (epoch + 1) % config.full_validation_every == 0
            or epoch + 1 == config.epochs
        )
        if full_due:
            full = evaluate_token_student(model, full_loader, device)
            metrics.update(full)
            metrics["best_full_cosine"] = max(
                best_full,
                full["val_frame_cosine"],
            )
        probe_improved = probe["val_frame_cosine"] > best_probe
        full_improved = full_due and metrics["val_frame_cosine"] > best_full
        best_probe = max(best_probe, probe["val_frame_cosine"])
        if full_improved:
            best_full = metrics["val_frame_cosine"]
        save_token_checkpoint(
            last_path,
            model,
            epoch=epoch,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if probe_improved:
            save_token_checkpoint(
                probe_best_path,
                model,
                epoch=epoch,
                metrics=metrics,
            )
        if full_improved:
            save_token_checkpoint(
                best_path,
                model,
                epoch=epoch,
                metrics=metrics,
            )
        print(
            f"E{epoch:03d} probe_cos={probe['val_frame_cosine']:.6f} "
            f"p05={probe['val_frame_cosine_p05']:.6f} "
            f"seq={probe['val_sequence_cosine']:.6f} "
            f"ord={probe['val_ordinal_accuracy']:.4f} "
            f"exact={probe['val_exact_accuracy']:.4f} "
            f"gap={max(0.0, config.target_cosine - probe['val_frame_cosine']):.6f}",
            flush=True,
        )
        if full_due:
            print(
                f"E{epoch:03d} val_cos={metrics['val_frame_cosine']:.6f} "
                f"p05={metrics['val_frame_cosine_p05']:.6f} "
                f"seq={metrics['val_sequence_cosine']:.6f} "
                f"gap={max(0.0, config.target_cosine - metrics['val_frame_cosine']):.6f}",
                flush=True,
            )

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

from .checkpoint import load_content_checkpoint, save_checkpoint
from .data import (
    ContentBatch,
    ContentCollator,
    MioContentDataset,
    masked_content_loss,
    speaker_disjoint_split,
)
from .fsq import indices_to_codes, masked_fsq_cross_entropy
from .model import ContentStudent, ContentStudentConfig
from .training import seed_everything


@dataclass(frozen=True)
class HybridTrainingConfig:
    data_dir: Path
    mel_dir: Path
    fsq_projection: Path
    output_dir: Path
    run_name: str = "content_student_hybrid_fsq"
    device: str = "mps"
    batch_size: int = 4
    epochs: int = 60
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.15
    supervised_mel_frames: int = 80
    history_mel_frames: int = 160
    seed: int = 42
    num_workers: int = 0
    resume: Path | None = None
    init_checkpoint: Path | None = None
    initialized_backbone_lr_scale: float = 0.25
    direct_weight: float = 1.0
    soft_fsq_weight: float = 0.25
    fsq_code_weight: float = 0.25
    fsq_ce_weight: float = 0.1
    pre_fsq_weight: float = 0.2
    delta_weight: float = 0.1
    target_cosine: float = 0.99
    log_every: int = 50


def validate_hybrid_config(config: HybridTrainingConfig) -> None:
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if config.supervised_mel_frames <= 0 or config.supervised_mel_frames % 2:
        raise ValueError("supervised_mel_frames must be a positive even number")
    if config.history_mel_frames < 0 or config.history_mel_frames % 2:
        raise ValueError("history_mel_frames must be a non-negative even number")
    if config.log_every <= 0:
        raise ValueError("log_every must be positive")
    if not 0.0 <= config.initialized_backbone_lr_scale <= 1.0:
        raise ValueError("initialized_backbone_lr_scale must be between 0 and 1")
    if not 0.0 <= config.target_cosine <= 1.0:
        raise ValueError("target_cosine must be between 0 and 1")


def _move(batch: ContentBatch, device: torch.device) -> ContentBatch:
    return ContentBatch(
        mel=batch.mel.to(device),
        content=batch.content.to(device),
        pre_fsq=batch.pre_fsq.to(device) if batch.pre_fsq is not None else None,
        token_indices=(
            batch.token_indices.to(device) if batch.token_indices is not None else None
        ),
        input_lengths=batch.input_lengths.to(device),
        target_lengths=batch.target_lengths.to(device),
        target_mask=batch.target_mask.to(device),
    )


def _masked_code_loss(
    prediction: torch.Tensor,
    token_indices: torch.Tensor,
    mask: torch.Tensor,
    levels: tuple[int, ...],
) -> torch.Tensor:
    length = min(prediction.shape[1], token_indices.shape[1], mask.shape[1])
    prediction = prediction[:, :length]
    target = indices_to_codes(token_indices[:, :length], levels).to(prediction)
    valid = mask[:, :length].unsqueeze(-1).expand_as(prediction)
    return F.smooth_l1_loss(prediction[valid], target[valid])


def _masked_delta_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.transpose(1, 2)
    length = min(prediction.shape[1], target.shape[1], mask.shape[1])
    if length < 2:
        return prediction.sum() * 0.0
    prediction_delta = prediction[:, 1:length] - prediction[:, : length - 1]
    target_delta = target[:, 1:length] - target[:, : length - 1]
    pair_mask = mask[:, 1:length] & mask[:, : length - 1]
    if not pair_mask.any():
        return prediction.sum() * 0.0
    valid = pair_mask.unsqueeze(-1).expand_as(prediction_delta)
    return F.smooth_l1_loss(prediction_delta[valid], target_delta[valid])


def hybrid_teacher_loss(
    model: ContentStudent,
    batch: ContentBatch,
    config: HybridTrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    if batch.token_indices is None:
        raise RuntimeError("Hybrid training requires cached Mio FSQ token indices")
    output = model(batch.mel, batch.input_lengths)
    if (
        output.fsq_logits is None
        or output.soft_fsq_codes is None
        or output.soft_fsq_content is None
    ):
        raise RuntimeError("Hybrid model did not produce its structured FSQ outputs")

    direct_loss, direct_cosine = masked_content_loss(
        output.content,
        batch.content,
        batch.target_mask,
        l1_weight=0.2,
    )
    soft_fsq_loss, soft_fsq_cosine = masked_content_loss(
        output.soft_fsq_content,
        batch.content,
        batch.target_mask,
        l1_weight=0.1,
    )
    fsq_code_loss = _masked_code_loss(
        output.soft_fsq_codes,
        batch.token_indices,
        batch.target_mask,
        model.config.fsq_levels,
    )
    fsq_ce_loss, axis_accuracy, exact_accuracy = masked_fsq_cross_entropy(
        output.fsq_logits,
        batch.token_indices,
        batch.target_mask,
        model.config.fsq_levels,
    )
    delta_loss = _masked_delta_loss(
        output.content,
        batch.content,
        batch.target_mask,
    )
    loss = (
        config.direct_weight * direct_loss
        + config.soft_fsq_weight * soft_fsq_loss
        + config.fsq_code_weight * fsq_code_loss
        + config.fsq_ce_weight * fsq_ce_loss
        + config.delta_weight * delta_loss
    )

    pre_fsq_loss = output.content.sum() * 0.0
    if output.pre_fsq is not None and batch.pre_fsq is not None:
        pre_fsq_loss, _ = masked_content_loss(
            output.pre_fsq,
            batch.pre_fsq,
            batch.target_mask,
            l1_weight=0.0,
        )
        loss = loss + config.pre_fsq_weight * pre_fsq_loss

    return loss, {
        "direct_cosine": direct_cosine.item(),
        "soft_fsq_cosine": soft_fsq_cosine.item(),
        "axis_accuracy": axis_accuracy.item(),
        "exact_accuracy": exact_accuracy.item(),
        "direct_loss": direct_loss.item(),
        "soft_fsq_loss": soft_fsq_loss.item(),
        "fsq_code_loss": fsq_code_loss.item(),
        "fsq_ce_loss": fsq_ce_loss.item(),
        "pre_fsq_loss": pre_fsq_loss.item(),
        "delta_loss": delta_loss.item(),
    }


def _frame_cosines(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.transpose(1, 2)
    length = min(prediction.shape[1], target.shape[1], mask.shape[1])
    return F.cosine_similarity(
        prediction[:, :length],
        target[:, :length],
        dim=-1,
    )[mask[:, :length]]


@torch.inference_mode()
def evaluate_hybrid(
    model: ContentStudent,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    direct_cosines = []
    soft_fsq_cosines = []
    hard_fsq_cosines = []
    sequence_cosines = []
    axis_correct = 0.0
    axis_total = 0
    exact_correct = 0.0
    exact_total = 0

    for raw_batch in loader:
        batch = _move(raw_batch, device)
        output = model(batch.mel, batch.input_lengths)
        if (
            output.soft_fsq_content is None
            or output.hard_content is None
            or output.fsq_logits is None
            or batch.token_indices is None
        ):
            raise RuntimeError("Hybrid validation requires all FSQ outputs")
        direct_cosines.append(
            _frame_cosines(output.content, batch.content, batch.target_mask).cpu()
        )
        soft_fsq_cosines.append(
            _frame_cosines(
                output.soft_fsq_content,
                batch.content,
                batch.target_mask,
            ).cpu()
        )
        hard_fsq_cosines.append(
            _frame_cosines(
                output.hard_content,
                batch.content,
                batch.target_mask,
            ).cpu()
        )

        prediction = output.content.transpose(1, 2)
        length = min(prediction.shape[1], batch.content.shape[1])
        mask = batch.target_mask[:, :length]
        for item in range(prediction.shape[0]):
            valid = mask[item]
            sequence_cosines.append(
                F.cosine_similarity(
                    prediction[item, :length][valid].reshape(1, -1),
                    batch.content[item, :length][valid].reshape(1, -1),
                ).item()
            )

        _, axis_accuracy, exact_accuracy = masked_fsq_cross_entropy(
            output.fsq_logits,
            batch.token_indices,
            batch.target_mask,
            model.config.fsq_levels,
        )
        frames = int(batch.target_mask.sum().item())
        axes = frames * len(model.config.fsq_levels)
        axis_correct += axis_accuracy.item() * axes
        axis_total += axes
        exact_correct += exact_accuracy.item() * frames
        exact_total += frames

    direct = torch.cat(direct_cosines)
    soft_fsq = torch.cat(soft_fsq_cosines)
    hard_fsq = torch.cat(hard_fsq_cosines)
    return {
        "val_frame_cosine": direct.mean().item(),
        "val_frame_cosine_p05": torch.quantile(direct, 0.05).item(),
        "val_sequence_cosine": float(np.mean(sequence_cosines)),
        "val_soft_fsq_cosine": soft_fsq.mean().item(),
        "val_hard_fsq_cosine": hard_fsq.mean().item(),
        "val_axis_accuracy": axis_correct / max(axis_total, 1),
        "val_exact_token_accuracy": exact_correct / max(exact_total, 1),
    }


def _initialize_model(
    model_config: ContentStudentConfig,
    config: HybridTrainingConfig,
    device: torch.device,
) -> tuple[ContentStudent, int, float]:
    if config.resume is not None:
        model, metadata = load_content_checkpoint(config.resume, device=device)
        if model.config != model_config:
            raise ValueError("Resume checkpoint config does not match")
        return (
            model,
            int(metadata.get("epoch", -1)) + 1,
            float(metadata.get("metrics", {}).get("val_frame_cosine", -float("inf"))),
        )

    model = ContentStudent(model_config).to(device)
    if config.init_checkpoint is None:
        projection = torch.load(config.fsq_projection, map_location=device)
        model.load_fsq_projection(projection)
        return model, 0, -float("inf")

    payload = torch.load(config.init_checkpoint, map_location=device)
    if not (
        isinstance(payload, dict)
        and payload.get("model_type") == "content_student"
        and "state_dict" in payload
    ):
        raise ValueError("init_checkpoint must be a versioned content checkpoint")
    source_state = payload["state_dict"]
    target_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if "fsq_projection.weight" in missing or "fsq_projection.bias" in missing:
        projection = torch.load(config.fsq_projection, map_location=device)
        model.load_fsq_projection(projection)
    print(
        f"Initialized {len(compatible)}/{len(target_state)} tensors from "
        f"{config.init_checkpoint}; new={len(missing)} ignored={len(unexpected)}"
    )
    return model, 0, -float("inf")


def train_hybrid_student(
    model_config: ContentStudentConfig,
    config: HybridTrainingConfig,
) -> None:
    if not model_config.structured_fsq or not model_config.hybrid_content:
        raise ValueError("Hybrid training requires structured_fsq and hybrid_content")
    validate_hybrid_config(config)
    seed_everything(config.seed)
    device = torch.device(config.device)

    meta = np.load(config.data_dir / "meta.npz")
    speakers = meta["spk_names"][: int(meta["n_samples"])].astype(str)
    train_indices, validation_indices = speaker_disjoint_split(
        speakers,
        config.validation_fraction,
        config.seed,
    )
    train_speakers = set(speakers[train_indices])
    validation_speakers = set(speakers[validation_indices])
    if train_speakers & validation_speakers:
        raise RuntimeError("Speaker-disjoint split failed")

    train_loader = DataLoader(
        MioContentDataset(config.data_dir, config.mel_dir, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=ContentCollator(
            config.supervised_mel_frames,
            config.seed,
            history_mel_frames=config.history_mel_frames,
        ),
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation_loader = DataLoader(
        MioContentDataset(config.data_dir, config.mel_dir, validation_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=ContentCollator(None, config.seed),
    )
    print(
        f"Train: {len(train_indices)} samples/{len(train_speakers)} speakers | "
        f"Val: {len(validation_indices)} samples/{len(validation_speakers)} speakers"
    )

    model, start_epoch, best_cosine = _initialize_model(
        model_config,
        config,
        device,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    direct_parameters = list(model.content_head.parameters())
    direct_ids = {id(parameter) for parameter in direct_parameters}
    backbone_parameters = [
        parameter for parameter in trainable if id(parameter) not in direct_ids
    ]
    backbone_scale = (
        config.initialized_backbone_lr_scale
        if config.init_checkpoint is not None and config.resume is None
        else 1.0
    )
    optimizer = AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": config.learning_rate * backbone_scale,
            },
            {"params": direct_parameters, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.resume is not None:
        payload = torch.load(config.resume, map_location=device)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.output_dir / f"{config.run_name}.best.pt"
    last_path = config.output_dir / f"{config.run_name}.last.pt"
    print(
        f"Params: {sum(parameter.numel() for parameter in model.parameters()):,} | "
        f"Device: {device} | supervised={config.supervised_mel_frames} mel | "
        f"history<={config.history_mel_frames} mel",
        flush=True,
    )

    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss = 0.0
        total_direct_cosine = 0.0
        total_soft_fsq_cosine = 0.0
        epoch_started = time.perf_counter()
        for step, raw_batch in enumerate(train_loader, start=1):
            batch = _move(raw_batch, device)
            loss, parts = hybrid_teacher_loss(model, batch, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_direct_cosine += parts["direct_cosine"]
            total_soft_fsq_cosine += parts["soft_fsq_cosine"]
            if step % config.log_every == 0 or step == len(train_loader):
                elapsed = time.perf_counter() - epoch_started
                print(
                    f"E{epoch:03d} step={step}/{len(train_loader)} "
                    f"loss={total_loss / step:.4f} "
                    f"direct_cos={total_direct_cosine / step:.4f} "
                    f"soft_fsq_cos={total_soft_fsq_cosine / step:.4f} "
                    f"{elapsed / step:.3f}s/step",
                    flush=True,
                )
        scheduler.step()

        metrics = evaluate_hybrid(model, validation_loader, device)
        metrics.update(
            {
                "train_loss": total_loss / max(len(train_loader), 1),
                "train_direct_cosine": total_direct_cosine
                / max(len(train_loader), 1),
                "train_soft_fsq_cosine": total_soft_fsq_cosine
                / max(len(train_loader), 1),
                "target_gap": max(
                    0.0,
                    config.target_cosine - metrics["val_frame_cosine"],
                ),
            }
        )
        save_checkpoint(
            last_path,
            model,
            epoch=epoch,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if metrics["val_frame_cosine"] > best_cosine:
            best_cosine = metrics["val_frame_cosine"]
            save_checkpoint(best_path, model, epoch=epoch, metrics=metrics)
        print(
            f"E{epoch:03d} val_direct={metrics['val_frame_cosine']:.4f} "
            f"p05={metrics['val_frame_cosine_p05']:.4f} "
            f"seq={metrics['val_sequence_cosine']:.4f} "
            f"soft_fsq={metrics['val_soft_fsq_cosine']:.4f} "
            f"hard_fsq={metrics['val_hard_fsq_cosine']:.4f} "
            f"axis={metrics['val_axis_accuracy']:.3f} "
            f"token={metrics['val_exact_token_accuracy']:.3f} "
            f"gap={metrics['target_gap']:.4f}",
            flush=True,
        )

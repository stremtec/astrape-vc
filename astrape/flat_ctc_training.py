from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CTCLoss
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
from .model import ContentStudent, ContentStudentConfig
from .training import seed_everything


@dataclass(frozen=True)
class FlatCtcTrainingConfig:
    data_dir: Path
    output_dir: Path
    run_name: str = "content_student_flat_ctc_512x10"
    device: str = "mps"
    batch_size: int = 2
    epochs: int = 30
    steps_per_epoch: int | None = 1000
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.15
    ctc_weight: float = 0.05
    delta_weight: float = 0.1
    seed: int = 42
    num_workers: int = 0
    resume: Path | None = None
    init_checkpoint: Path | None = None
    probe_samples: int = 1024
    full_validation_every: int = 5
    target_cosine: float = 0.99
    log_every: int = 100


def validate_flat_ctc_config(config: FlatCtcTrainingConfig) -> None:
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if config.steps_per_epoch is not None and config.steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive when set")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.ctc_weight <= 0:
        raise ValueError("ctc_weight must be positive")
    if config.delta_weight < 0:
        raise ValueError("delta_weight must be non-negative")
    if config.log_every <= 0:
        raise ValueError("log_every must be positive")
    if config.probe_samples <= 0:
        raise ValueError("probe_samples must be positive")
    if config.full_validation_every <= 0:
        raise ValueError("full_validation_every must be positive")
    if config.resume is not None and config.init_checkpoint is not None:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    if not 0.0 <= config.target_cosine <= 1.0:
        raise ValueError("target_cosine must be between 0 and 1")


def speaker_balanced_subset(
    indices: np.ndarray,
    speakers: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if len(indices) <= max_samples:
        return indices.copy()

    rng = np.random.default_rng(seed)
    queues: dict[str, list[int]] = {}
    for speaker in sorted(set(speakers[indices].astype(str))):
        speaker_indices = indices[speakers[indices].astype(str) == speaker].copy()
        rng.shuffle(speaker_indices)
        queues[speaker] = speaker_indices.tolist()

    selected = []
    active = list(queues)
    while active and len(selected) < max_samples:
        next_active = []
        for speaker in active:
            if queues[speaker]:
                selected.append(queues[speaker].pop())
                if len(selected) == max_samples:
                    break
            if queues[speaker]:
                next_active.append(speaker)
        active = next_active
    return np.asarray(selected, dtype=np.int64)


def move_content_batch(
    batch: ContentBatch,
    device: torch.device,
) -> ContentBatch:
    return ContentBatch(
        mel=batch.mel.to(device),
        content=batch.content.to(device),
        pre_fsq=None,
        token_indices=None,
        input_lengths=batch.input_lengths.to(device),
        target_lengths=batch.target_lengths.to(device),
        target_mask=batch.target_mask.to(device),
        transcripts=(
            batch.transcripts.to(device) if batch.transcripts is not None else None
        ),
        transcript_lengths=(
            batch.transcript_lengths.to(device)
            if batch.transcript_lengths is not None
            else None
        ),
    )


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


def _ctc_loss(
    text_logits: torch.Tensor,
    batch: ContentBatch,
    criterion: CTCLoss,
) -> torch.Tensor:
    if batch.transcripts is None or batch.transcript_lengths is None:
        raise RuntimeError("Flat CTC training requires cached transcripts")
    ctc_device = (
        torch.device("cpu")
        if text_logits.device.type == "mps"
        else text_logits.device
    )
    return criterion(
        text_logits.log_softmax(dim=-1).transpose(0, 1).to(ctc_device),
        batch.transcripts.to(ctc_device),
        batch.input_lengths.to(ctc_device),
        batch.transcript_lengths.to(ctc_device),
    )


def flat_ctc_loss(
    model: ContentStudent,
    batch: ContentBatch,
    config: FlatCtcTrainingConfig,
    criterion: CTCLoss,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(batch.mel, batch.input_lengths)
    if output.text_logits is None:
        raise RuntimeError("Flat CTC model did not produce text logits")
    content_loss, cosine = masked_content_loss(
        output.content,
        batch.content,
        batch.target_mask,
        l1_weight=0.2,
    )
    delta_loss = _masked_delta_loss(
        output.content,
        batch.content,
        batch.target_mask,
    )
    ctc_loss = _ctc_loss(output.text_logits, batch, criterion)
    loss = (
        content_loss
        + config.delta_weight * delta_loss
        + config.ctc_weight * ctc_loss
    )
    return loss, {
        "content_loss": content_loss.item(),
        "content_cosine": cosine.item(),
        "delta_loss": delta_loss.item(),
        "ctc_loss": ctc_loss.item(),
    }


def _split_targets(
    flat_targets: torch.Tensor,
    lengths: torch.Tensor,
) -> list[list[int]]:
    result = []
    offset = 0
    for length in lengths.tolist():
        result.append(flat_targets[offset : offset + length].tolist())
        offset += length
    return result


def _greedy_ctc_sequences(
    logits: torch.Tensor,
    lengths: torch.Tensor,
) -> list[list[int]]:
    predictions = logits.argmax(dim=-1).cpu()
    result = []
    for row, length in zip(predictions, lengths.cpu().tolist()):
        sequence = []
        previous = -1
        for token in row[:length].tolist():
            if token != 0 and token != previous:
                sequence.append(token)
            previous = token
        result.append(sequence)
    return result


def _edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


@torch.inference_mode()
def evaluate_flat_ctc(
    model: ContentStudent,
    loader: DataLoader,
    device: torch.device,
    criterion: CTCLoss,
) -> dict[str, float]:
    model.eval()
    frame_cosines = []
    sequence_cosines = []
    total_ctc_loss = 0.0
    total_edits = 0
    total_characters = 0
    batches = 0

    for raw_batch in loader:
        batch = move_content_batch(raw_batch, device)
        output = model(batch.mel, batch.input_lengths)
        if output.text_logits is None:
            raise RuntimeError("Flat CTC validation requires text logits")
        prediction = output.content.transpose(1, 2)
        length = min(prediction.shape[1], batch.content.shape[1])
        mask = batch.target_mask[:, :length]
        prediction = prediction[:, :length]
        target = batch.content[:, :length]
        frame_cosines.append(
            F.cosine_similarity(prediction, target, dim=-1)[mask].cpu()
        )
        for item in range(prediction.shape[0]):
            valid = mask[item]
            sequence_cosines.append(
                F.cosine_similarity(
                    prediction[item, valid].reshape(1, -1),
                    target[item, valid].reshape(1, -1),
                ).item()
            )

        total_ctc_loss += _ctc_loss(output.text_logits, batch, criterion).item()
        expected = _split_targets(
            batch.transcripts.cpu(),
            batch.transcript_lengths.cpu(),
        )
        predicted = _greedy_ctc_sequences(
            output.text_logits,
            batch.input_lengths,
        )
        for hypothesis, reference in zip(predicted, expected):
            total_edits += _edit_distance(hypothesis, reference)
            total_characters += len(reference)
        batches += 1

    cosines = torch.cat(frame_cosines)
    return {
        "val_frame_cosine": cosines.mean().item(),
        "val_frame_cosine_p05": torch.quantile(cosines, 0.05).item(),
        "val_sequence_cosine": float(np.mean(sequence_cosines)),
        "val_ctc_loss": total_ctc_loss / max(batches, 1),
        "val_character_error_rate": total_edits / max(total_characters, 1),
    }


def train_flat_ctc_student(
    model_config: ContentStudentConfig,
    config: FlatCtcTrainingConfig,
) -> None:
    if (
        model_config.structured_fsq
        or model_config.hybrid_content
        or model_config.auxiliary_prefsq
        or model_config.text_vocab_size <= 0
    ):
        raise ValueError("Flat CTC training requires only direct content and text heads")
    validate_flat_ctc_config(config)
    seed_everything(config.seed)
    device = torch.device(config.device)

    with np.load(config.data_dir / "meta.npz") as meta:
        cache_format = (
            str(meta["cache_format"].item())
            if "cache_format" in meta.files
            else ""
        )
        if cache_format != "compact-fp16-ctc-v2":
            raise ValueError("Flat CTC training requires compact-fp16-ctc-v2 cache")
        speakers = meta["spk_names"][: int(meta["n_samples"])].astype(str)
    train_indices, validation_indices = speaker_disjoint_split(
        speakers,
        config.validation_fraction,
        config.seed,
    )
    train_speakers = set(speakers[train_indices])
    validation_speakers = set(speakers[validation_indices])
    probe_indices = speaker_balanced_subset(
        validation_indices,
        speakers,
        config.probe_samples,
        config.seed,
    )
    probe_speakers = set(speakers[probe_indices])

    collator = ContentCollator(
        None,
        config.seed,
        include_transcripts=True,
    )
    train_loader = DataLoader(
        MioContentDataset(config.data_dir, config.data_dir, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
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
            include_transcripts=True,
        ),
    )
    full_validation_loader = DataLoader(
        MioContentDataset(config.data_dir, config.data_dir, validation_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=ContentCollator(
            None,
            config.seed,
            include_transcripts=True,
        ),
    )
    print(
        f"Train: {len(train_indices)} samples/{len(train_speakers)} speakers | "
        f"Probe: {len(probe_indices)} samples/{len(probe_speakers)} speakers | "
        f"Full val: {len(validation_indices)} samples"
    )

    start_epoch = 0
    best_probe_cosine = -float("inf")
    best_full_cosine = -float("inf")
    if config.resume is not None:
        model, metadata = load_content_checkpoint(config.resume, device=device)
        if model.config != model_config:
            raise ValueError("Resume checkpoint config does not match")
        start_epoch = int(metadata.get("epoch", -1)) + 1
        checkpoint_metrics = metadata.get("metrics", {})
        best_probe_cosine = float(
            checkpoint_metrics.get(
                "best_probe_cosine",
                checkpoint_metrics.get("probe_frame_cosine", best_probe_cosine),
            )
        )
        best_full_cosine = float(
            checkpoint_metrics.get(
                "best_full_cosine",
                checkpoint_metrics.get("val_frame_cosine", best_full_cosine),
            )
        )
    elif config.init_checkpoint is not None:
        model, metadata = load_content_checkpoint(
            config.init_checkpoint,
            device=device,
        )
        if model.config != model_config:
            raise ValueError("Initial checkpoint config does not match")
        source_metrics = metadata.get("metrics", {})
        print(
            f"Warm start: {config.init_checkpoint} "
            f"source_epoch={metadata.get('epoch', '?')} "
            f"source_val_cos={source_metrics.get('val_frame_cosine', 'n/a')}",
            flush=True,
        )
    else:
        model = ContentStudent(model_config).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.resume is not None:
        payload = torch.load(config.resume, map_location=device)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    criterion = CTCLoss(blank=0, zero_infinity=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.output_dir / f"{config.run_name}.best.pt"
    probe_best_path = config.output_dir / f"{config.run_name}.probe-best.pt"
    last_path = config.output_dir / f"{config.run_name}.last.pt"
    steps_per_epoch = min(
        len(train_loader),
        config.steps_per_epoch or len(train_loader),
    )
    print(
        f"Params: {sum(parameter.numel() for parameter in model.parameters()):,} | "
        f"Device: {device} | full utterances | ctc_weight={config.ctc_weight} | "
        f"{steps_per_epoch} steps/epoch | full_val_every={config.full_validation_every}",
        flush=True,
    )
    if config.init_checkpoint is not None:
        initial_probe = evaluate_flat_ctc(
            model,
            probe_loader,
            device,
            criterion,
        )
        print(
            f"INIT probe_cos={initial_probe['val_frame_cosine']:.4f} "
            f"p05={initial_probe['val_frame_cosine_p05']:.4f} "
            f"seq={initial_probe['val_sequence_cosine']:.4f} "
            f"ctc={initial_probe['val_ctc_loss']:.4f} "
            f"cer={initial_probe['val_character_error_rate']:.4f} "
            f"gap={max(0.0, config.target_cosine - initial_probe['val_frame_cosine']):.4f}",
            flush=True,
        )

    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss = 0.0
        total_cosine = 0.0
        total_ctc = 0.0
        started = time.perf_counter()
        for step, raw_batch in enumerate(train_loader, start=1):
            if step > steps_per_epoch:
                break
            batch = move_content_batch(raw_batch, device)
            loss, parts = flat_ctc_loss(model, batch, config, criterion)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_cosine += parts["content_cosine"]
            total_ctc += parts["ctc_loss"]
            if step % config.log_every == 0 or step == steps_per_epoch:
                elapsed = time.perf_counter() - started
                print(
                    f"E{epoch:03d} step={step}/{steps_per_epoch} "
                    f"loss={total_loss / step:.4f} "
                    f"content_cos={total_cosine / step:.4f} "
                    f"ctc={total_ctc / step:.4f} "
                    f"{elapsed / step:.3f}s/step",
                    flush=True,
                )
        scheduler.step()

        probe_metrics = evaluate_flat_ctc(
            model,
            probe_loader,
            device,
            criterion,
        )
        metrics = {
            key.replace("val_", "probe_", 1): value
            for key, value in probe_metrics.items()
        }
        metrics.update(
            {
                "train_loss": total_loss / steps_per_epoch,
                "train_content_cosine": total_cosine / steps_per_epoch,
                "train_ctc_loss": total_ctc / steps_per_epoch,
                "probe_target_gap": max(
                    0.0,
                    config.target_cosine - probe_metrics["val_frame_cosine"],
                ),
            }
        )
        full_validation_due = (
            (epoch + 1) % config.full_validation_every == 0
            or epoch + 1 == config.epochs
        )
        if full_validation_due:
            full_metrics = evaluate_flat_ctc(
                model,
                full_validation_loader,
                device,
                criterion,
            )
            metrics.update(full_metrics)
            metrics["target_gap"] = max(
                0.0,
                config.target_cosine - full_metrics["val_frame_cosine"],
            )
        probe_improved = (
            probe_metrics["val_frame_cosine"] > best_probe_cosine
        )
        full_improved = (
            full_validation_due
            and metrics["val_frame_cosine"] > best_full_cosine
        )
        if probe_improved:
            best_probe_cosine = probe_metrics["val_frame_cosine"]
        if full_improved:
            best_full_cosine = metrics["val_frame_cosine"]
        metrics["best_probe_cosine"] = best_probe_cosine
        if best_full_cosine > -float("inf"):
            metrics["best_full_cosine"] = best_full_cosine
        save_checkpoint(
            last_path,
            model,
            epoch=epoch,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if probe_improved:
            save_checkpoint(
                probe_best_path,
                model,
                epoch=epoch,
                metrics=metrics,
            )
        if full_improved:
            save_checkpoint(best_path, model, epoch=epoch, metrics=metrics)
        print(
            f"E{epoch:03d} probe_cos={metrics['probe_frame_cosine']:.4f} "
            f"p05={metrics['probe_frame_cosine_p05']:.4f} "
            f"seq={metrics['probe_sequence_cosine']:.4f} "
            f"ctc={metrics['probe_ctc_loss']:.4f} "
            f"cer={metrics['probe_character_error_rate']:.4f} "
            f"gap={metrics['probe_target_gap']:.4f}",
            flush=True,
        )
        if full_validation_due:
            print(
                f"E{epoch:03d} val_cos={metrics['val_frame_cosine']:.4f} "
                f"p05={metrics['val_frame_cosine_p05']:.4f} "
                f"seq={metrics['val_sequence_cosine']:.4f} "
                f"ctc={metrics['val_ctc_loss']:.4f} "
                f"cer={metrics['val_character_error_rate']:.4f} "
                f"gap={metrics['target_gap']:.4f}",
                flush=True,
            )

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import CTCLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .checkpoint import load_content_checkpoint, save_checkpoint
from .curriculum import original_loss
from .data import ContentCollator, MioContentDataset, speaker_disjoint_split
from .flat_ctc_training import (
    FlatCtcTrainingConfig,
    evaluate_flat_ctc,
    flat_ctc_loss,
    move_content_batch,
    speaker_balanced_subset,
)
from .model import ContentStudent, ContentStudentConfig
from .original_data import OriginalCollator, OriginalVCTKDataset, scan_vctk
from .training import seed_everything


@dataclass(frozen=True)
class TwoPhaseTrainingConfig:
    data_dir: Path
    audio_root: Path
    transcript_root: Path
    output_dir: Path
    run_name: str = "content_student_mio_causal_two_phase"
    device: str = "mps"
    batch_size: int = 2
    phase1_epochs: int = 10
    phase2_epochs: int = 20
    steps_per_epoch: int = 1000
    phase1_learning_rate: float = 2e-4
    phase2_learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.15
    teacher_probability: float = 0.5
    teacher_ctc_weight: float = 0.05
    original_ctc_weight: float = 0.05
    delta_weight: float = 0.1
    seed: int = 42
    num_workers: int = 0
    resume: Path | None = None
    probe_samples: int = 1024
    full_validation_every: int = 5
    target_cosine: float = 0.99
    log_every: int = 100

    @property
    def epochs(self) -> int:
        return self.phase1_epochs + self.phase2_epochs


def validate_two_phase_config(config: TwoPhaseTrainingConfig) -> None:
    if config.phase1_epochs <= 0 or config.phase2_epochs <= 0:
        raise ValueError("Both training phases must contain at least one epoch")
    if config.batch_size <= 0 or config.steps_per_epoch <= 0:
        raise ValueError("batch_size and steps_per_epoch must be positive")
    if config.phase1_learning_rate <= 0 or config.phase2_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")
    if not 0.0 < config.teacher_probability < 1.0:
        raise ValueError("teacher_probability must be between zero and one")
    if config.teacher_ctc_weight <= 0 or config.original_ctc_weight <= 0:
        raise ValueError("CTC weights must be positive")
    if config.delta_weight < 0:
        raise ValueError("delta_weight must be non-negative")
    if config.probe_samples <= 0 or config.full_validation_every <= 0:
        raise ValueError("Validation intervals and sizes must be positive")
    if config.log_every <= 0:
        raise ValueError("log_every must be positive")
    if not 0.0 <= config.target_cosine <= 1.0:
        raise ValueError("target_cosine must be between zero and one")


def phase_for_epoch(epoch: int, config: TwoPhaseTrainingConfig) -> str:
    if not 0 <= epoch < config.epochs:
        raise ValueError("epoch is outside the configured training range")
    return "teacher" if epoch < config.phase1_epochs else "teacher_original"


def phase2_source_schedule(
    steps: int,
    teacher_probability: float,
    seed: int,
) -> list[bool]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 <= teacher_probability <= 1.0:
        raise ValueError("teacher_probability must be between zero and one")
    teacher_steps = int(round(steps * teacher_probability))
    schedule = [True] * teacher_steps + [False] * (steps - teacher_steps)
    random.Random(seed).shuffle(schedule)
    return schedule


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _make_optimizer(
    model: ContentStudent,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    return AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def _flat_loss_config(config: TwoPhaseTrainingConfig) -> FlatCtcTrainingConfig:
    return FlatCtcTrainingConfig(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        run_name=config.run_name,
        device=config.device,
        batch_size=config.batch_size,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        learning_rate=config.phase1_learning_rate,
        weight_decay=config.weight_decay,
        validation_fraction=config.validation_fraction,
        ctc_weight=config.teacher_ctc_weight,
        delta_weight=config.delta_weight,
        seed=config.seed,
        num_workers=config.num_workers,
        probe_samples=config.probe_samples,
        full_validation_every=config.full_validation_every,
        target_cosine=config.target_cosine,
        log_every=config.log_every,
    )


def train_two_phase_student(
    model_config: ContentStudentConfig,
    config: TwoPhaseTrainingConfig,
) -> None:
    if model_config.architecture != "mio_causal":
        raise ValueError("Two-phase training requires the mio_causal architecture")
    if (
        model_config.structured_fsq
        or model_config.hybrid_content
        or model_config.auxiliary_prefsq
        or model_config.text_vocab_size <= 0
    ):
        raise ValueError("Two-phase training requires direct content and text heads")
    validate_two_phase_config(config)
    seed_everything(config.seed)
    device = torch.device(config.device)

    with np.load(config.data_dir / "meta.npz") as meta:
        cache_format = (
            str(meta["cache_format"].item())
            if "cache_format" in meta.files
            else ""
        )
        if cache_format != "compact-fp16-ctc-v2":
            raise ValueError(
                "Two-phase training requires compact-fp16-ctc-v2 teacher cache"
            )
        speakers = meta["spk_names"][: int(meta["n_samples"])].astype(str)
    train_indices, validation_indices = speaker_disjoint_split(
        speakers,
        config.validation_fraction,
        config.seed,
    )
    train_speakers = sorted(set(speakers[train_indices]))
    validation_speakers = set(speakers[validation_indices])
    if set(train_speakers) & validation_speakers:
        raise RuntimeError("Speaker-disjoint split failed")
    probe_indices = speaker_balanced_subset(
        validation_indices,
        speakers,
        config.probe_samples,
        config.seed,
    )

    teacher_train = DataLoader(
        MioContentDataset(config.data_dir, config.data_dir, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=ContentCollator(
            None,
            config.seed,
            include_transcripts=True,
        ),
        generator=torch.Generator().manual_seed(config.seed),
    )
    probe_validation = DataLoader(
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
    full_validation = DataLoader(
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
    original_records = scan_vctk(
        config.audio_root,
        config.transcript_root,
        allowed_speakers=train_speakers,
    )
    if not original_records:
        raise RuntimeError("No original VCTK utterances matched teacher train speakers")
    original_train = DataLoader(
        OriginalVCTKDataset(original_records),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=OriginalCollator(),
        generator=torch.Generator().manual_seed(config.seed + 1),
    )
    print(
        f"Teacher train: {len(train_indices)} samples/{len(train_speakers)} speakers | "
        f"Original: {len(original_records)} utterances | "
        f"Probe: {len(probe_indices)} | Full val: {len(validation_indices)} samples/"
        f"{len(validation_speakers)} speakers",
        flush=True,
    )

    start_epoch = 0
    best_probe_cosine = -float("inf")
    best_full_cosine = -float("inf")
    best_phase1_probe_cosine = -float("inf")
    if config.resume is not None:
        model, metadata = load_content_checkpoint(config.resume, device=device)
        if model.config != model_config:
            raise ValueError("Resume checkpoint config does not match")
        start_epoch = int(metadata.get("epoch", -1)) + 1
        checkpoint_metrics = metadata.get("metrics", {})
        best_probe_cosine = float(
            checkpoint_metrics.get("best_probe_cosine", best_probe_cosine)
        )
        best_full_cosine = float(
            checkpoint_metrics.get("best_full_cosine", best_full_cosine)
        )
        best_phase1_probe_cosine = float(
            checkpoint_metrics.get(
                "best_phase1_probe_cosine",
                best_phase1_probe_cosine,
            )
        )
    else:
        model = ContentStudent(model_config).to(device)

    in_phase1 = start_epoch < config.phase1_epochs
    optimizer = _make_optimizer(
        model,
        (
            config.phase1_learning_rate
            if in_phase1
            else config.phase2_learning_rate
        ),
        config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.phase1_epochs if in_phase1 else config.phase2_epochs,
    )
    if config.resume is not None:
        payload = torch.load(config.resume, map_location=device)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    criterion = CTCLoss(blank=0, zero_infinity=True)
    flat_config = _flat_loss_config(config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    phase1_best_path = config.output_dir / f"{config.run_name}.phase1-best.pt"
    probe_best_path = config.output_dir / f"{config.run_name}.probe-best.pt"
    best_path = config.output_dir / f"{config.run_name}.best.pt"
    last_path = config.output_dir / f"{config.run_name}.last.pt"
    print(
        f"Params: {sum(parameter.numel() for parameter in model.parameters()):,} | "
        f"Device: {device} | P1={config.phase1_epochs}x{config.steps_per_epoch} "
        f"teacher-only @ {config.phase1_learning_rate:g} | "
        f"P2={config.phase2_epochs}x{config.steps_per_epoch} "
        f"teacher/original={config.teacher_probability:.0%}/"
        f"{1.0 - config.teacher_probability:.0%} @ "
        f"{config.phase2_learning_rate:g}",
        flush=True,
    )

    for epoch in range(start_epoch, config.epochs):
        phase = phase_for_epoch(epoch, config)
        if epoch == config.phase1_epochs:
            if phase1_best_path.exists():
                payload = torch.load(phase1_best_path, map_location=device)
                model.load_state_dict(payload["state_dict"], strict=True)
                print(
                    f"PHASE2 start: restored best phase-1 probe checkpoint "
                    f"(cos={best_phase1_probe_cosine:.4f})",
                    flush=True,
                )
            else:
                print(
                    "PHASE2 start: phase-1 best checkpoint missing; using current weights",
                    flush=True,
                )
            optimizer = _make_optimizer(
                model,
                config.phase2_learning_rate,
                config.weight_decay,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=config.phase2_epochs)

        if phase == "teacher":
            sources = [True] * config.steps_per_epoch
        else:
            sources = phase2_source_schedule(
                config.steps_per_epoch,
                config.teacher_probability,
                config.seed + epoch,
            )

        model.train()
        teacher_iterator = iter(teacher_train)
        original_iterator = iter(original_train)
        total_loss = 0.0
        teacher_loss_total = 0.0
        teacher_cosine_total = 0.0
        teacher_ctc_total = 0.0
        original_ctc_total = 0.0
        teacher_steps = 0
        original_steps = 0
        started = time.perf_counter()
        for step, use_teacher in enumerate(sources, start=1):
            if use_teacher:
                raw_batch, teacher_iterator = _next_batch(
                    teacher_iterator,
                    teacher_train,
                )
                batch = move_content_batch(raw_batch, device)
                loss, parts = flat_ctc_loss(
                    model,
                    batch,
                    flat_config,
                    criterion,
                )
                teacher_loss_total += loss.item()
                teacher_cosine_total += parts["content_cosine"]
                teacher_ctc_total += parts["ctc_loss"]
                teacher_steps += 1
            else:
                batch, original_iterator = _next_batch(
                    original_iterator,
                    original_train,
                )
                raw_ctc_loss = original_loss(model, batch, device, criterion)
                loss = config.original_ctc_weight * raw_ctc_loss
                original_ctc_total += raw_ctc_loss.item()
                original_steps += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += loss.item()
            if step % config.log_every == 0 or step == config.steps_per_epoch:
                elapsed = time.perf_counter() - started
                teacher_cosine = (
                    teacher_cosine_total / teacher_steps if teacher_steps else float("nan")
                )
                original_ctc = (
                    original_ctc_total / original_steps if original_steps else float("nan")
                )
                print(
                    f"E{epoch:03d} {phase} step={step}/{config.steps_per_epoch} "
                    f"loss={total_loss / step:.4f} "
                    f"teacher_cos={teacher_cosine:.4f} "
                    f"original_ctc={original_ctc:.4f} "
                    f"teacher={teacher_steps} original={original_steps} "
                    f"{elapsed / step:.3f}s/step",
                    flush=True,
                )
        scheduler.step()

        probe_metrics = evaluate_flat_ctc(
            model,
            probe_validation,
            device,
            criterion,
        )
        metrics = {
            key.replace("val_", "probe_", 1): value
            for key, value in probe_metrics.items()
        }
        metrics.update(
            {
                "phase_id": 1.0 if phase == "teacher" else 2.0,
                "train_loss": total_loss / config.steps_per_epoch,
                "teacher_steps": float(teacher_steps),
                "original_steps": float(original_steps),
                "train_teacher_loss": (
                    teacher_loss_total / teacher_steps if teacher_steps else 0.0
                ),
                "train_teacher_content_cosine": (
                    teacher_cosine_total / teacher_steps if teacher_steps else 0.0
                ),
                "train_teacher_ctc_loss": (
                    teacher_ctc_total / teacher_steps if teacher_steps else 0.0
                ),
                "train_original_ctc_loss": (
                    original_ctc_total / original_steps if original_steps else 0.0
                ),
                "probe_target_gap": max(
                    0.0,
                    config.target_cosine - probe_metrics["val_frame_cosine"],
                ),
            }
        )
        full_validation_due = (
            (epoch + 1) % config.full_validation_every == 0
            or epoch + 1 == config.phase1_epochs
            or epoch + 1 == config.epochs
        )
        if full_validation_due:
            full_metrics = evaluate_flat_ctc(
                model,
                full_validation,
                device,
                criterion,
            )
            metrics.update(full_metrics)
            metrics["target_gap"] = max(
                0.0,
                config.target_cosine - full_metrics["val_frame_cosine"],
            )

        probe_improved = probe_metrics["val_frame_cosine"] > best_probe_cosine
        phase1_improved = (
            phase == "teacher"
            and probe_metrics["val_frame_cosine"] > best_phase1_probe_cosine
        )
        full_improved = (
            full_validation_due
            and metrics["val_frame_cosine"] > best_full_cosine
        )
        if probe_improved:
            best_probe_cosine = probe_metrics["val_frame_cosine"]
        if phase1_improved:
            best_phase1_probe_cosine = probe_metrics["val_frame_cosine"]
        if full_improved:
            best_full_cosine = metrics["val_frame_cosine"]
        metrics["best_probe_cosine"] = best_probe_cosine
        metrics["best_phase1_probe_cosine"] = best_phase1_probe_cosine
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
            save_checkpoint(probe_best_path, model, epoch=epoch, metrics=metrics)
        if phase1_improved:
            save_checkpoint(phase1_best_path, model, epoch=epoch, metrics=metrics)
        if full_improved:
            save_checkpoint(best_path, model, epoch=epoch, metrics=metrics)

        print(
            f"E{epoch:03d} {phase} probe_cos={metrics['probe_frame_cosine']:.4f} "
            f"p05={metrics['probe_frame_cosine_p05']:.4f} "
            f"seq={metrics['probe_sequence_cosine']:.4f} "
            f"ctc={metrics['probe_ctc_loss']:.4f} "
            f"cer={metrics['probe_character_error_rate']:.4f} "
            f"gap={metrics['probe_target_gap']:.4f}",
            flush=True,
        )
        if full_validation_due:
            print(
                f"E{epoch:03d} {phase} val_cos={metrics['val_frame_cosine']:.4f} "
                f"p05={metrics['val_frame_cosine_p05']:.4f} "
                f"seq={metrics['val_sequence_cosine']:.4f} "
                f"ctc={metrics['val_ctc_loss']:.4f} "
                f"cer={metrics['val_character_error_rate']:.4f} "
                f"gap={metrics['target_gap']:.4f}",
                flush=True,
            )

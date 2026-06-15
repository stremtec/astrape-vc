"""Training loop and helpers for the FFL-enabled mio_ffl ContentStudent.

The FFL model produces, in addition to the regular CTC/content outputs, a
`false_future_*` family of tensors that describe how each Mio block's hidden
state was corrected by a confidence-gated residual derived from the slot
generator. This module supervises those tensors alongside the standard
direct content distillation and CTC objective.

Teacher hidden targets required by the full `L_hidden` and `L_slot` losses are
not cached on disk. Instead, this trainer derives a final-output effect target
from two passes of the same student:

  - A causal pass with FFL disabled defines the current baseline.
  - The teacher residual is `teacher_content - causal_baseline`.
  - The FFL residual is `ffl_content - causal_baseline`.
  - `L_output_effect` directly matches those two residuals.
  - The auxiliary character CTC head at 50 Hz.
  - Direct content and smooth-delta losses.

Each epoch additionally records:

  - `gate_mean`: how open the FFL gate is on the validation set.
  - `effect_ratio`: fraction of magnitude that survives the gate (mean of
    `corrections / effect`), giving a quantitative read on how often FFL
    activates.
  - `gate_sparsity_frac`: share of frames with `gate < 0.05` (initial value
    should approach 1.0; later training should reduce this).

If `effect_target_path` is provided, an additional cached oracle target is
expected in `data_dir`'s `d_*.npz` files under key `oracle_effect`, of shape
[batch, layers, time, hidden]. Until the cache is built, callers should leave
this unset.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
from .flat_ctc_training import (  # noqa: F401  (re-exported for convenience)
    _ctc_loss,
    _edit_distance,
    _greedy_ctc_sequences,
    _masked_delta_loss,
    _split_targets,
    move_content_batch,
    speaker_balanced_subset,
)
from .model import ContentStudent, ContentStudentConfig
from .training import seed_everything


@dataclass(frozen=True)
class FflTrainingConfig:
    data_dir: Path
    output_dir: Path
    run_name: str = "content_student_mio_ffl_768x6"
    device: str = "mps"
    batch_size: int = 2
    epochs: int = 30
    steps_per_epoch: int | None = 1000
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.15
    ctc_weight: float = 0.05
    delta_weight: float = 0.1
    output_effect_weight: float = 0.5
    output_effect_cosine_weight: float = 0.1
    gate_l2_weight: float = 0.0
    causal_warmup_epochs: int = 1
    effect_warmup_epochs: int = 1
    effect_warmup_gate: float = 0.25
    pad_mel_multiple: int = 64
    mps_empty_cache_every: int = 100
    seed: int = 42
    num_workers: int = 0
    resume: Path | None = None
    init_checkpoint: Path | None = None
    probe_samples: int = 1024
    full_validation_every: int = 5
    target_cosine: float = 0.99
    log_every: int = 100


def validate_ffl_config(config: FflTrainingConfig) -> None:
    if config.batch_size <= 0 or config.epochs <= 0:
        raise ValueError("batch_size and epochs must be positive")
    if config.steps_per_epoch is not None and config.steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive when set")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.ctc_weight <= 0:
        raise ValueError("ctc_weight must be positive")
    if config.delta_weight < 0:
        raise ValueError("delta_weight must be non-negative")
    if config.output_effect_weight < 0:
        raise ValueError("output_effect_weight must be non-negative")
    if config.output_effect_cosine_weight < 0:
        raise ValueError("output_effect_cosine_weight must be non-negative")
    if config.gate_l2_weight < 0:
        raise ValueError("gate_l2_weight must be non-negative")
    if config.causal_warmup_epochs < 0 or config.effect_warmup_epochs < 0:
        raise ValueError("warmup epochs must be non-negative")
    if config.causal_warmup_epochs + config.effect_warmup_epochs > config.epochs:
        raise ValueError("warmup epochs cannot exceed total epochs")
    if not 0.0 < config.effect_warmup_gate <= 1.0:
        raise ValueError("effect_warmup_gate must be in (0, 1]")
    if config.pad_mel_multiple <= 0:
        raise ValueError("pad_mel_multiple must be positive")
    if config.mps_empty_cache_every < 0:
        raise ValueError("mps_empty_cache_every must be non-negative")
    if config.probe_samples <= 0:
        raise ValueError("probe_samples must be positive")
    if config.full_validation_every <= 0:
        raise ValueError("full_validation_every must be positive")
    if config.resume is not None and config.init_checkpoint is not None:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    if not 0.0 <= config.target_cosine <= 1.0:
        raise ValueError("target_cosine must be between 0 and 1")


def _masked_output_effect_loss(
    prediction: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    cosine_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = prediction.transpose(1, 2)
    baseline = baseline.transpose(1, 2)
    length = min(
        prediction.shape[1],
        baseline.shape[1],
        target.shape[1],
        mask.shape[1],
    )
    prediction = prediction[:, :length]
    baseline = baseline[:, :length]
    target = target[:, :length]
    mask = mask[:, :length]
    predicted_effect = prediction - baseline
    target_effect = target - baseline
    expanded_mask = mask.unsqueeze(-1).expand_as(predicted_effect)
    smooth_l1 = F.smooth_l1_loss(
        predicted_effect.masked_select(expanded_mask),
        target_effect.masked_select(expanded_mask),
    )
    valid_prediction = predicted_effect[mask]
    valid_target = target_effect[mask]
    effect_cosine = F.cosine_similarity(
        valid_prediction,
        valid_target,
        dim=-1,
        eps=1e-6,
    ).mean()
    return smooth_l1 + cosine_weight * (1 - effect_cosine), effect_cosine


def ffl_loss(
    model_output,
    batch: ContentBatch,
    config: FflTrainingConfig,
    criterion: CTCLoss,
    baseline_content: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compose direct losses and optional causal-to-teacher output-effect loss."""
    content_loss, cosine = masked_content_loss(
        model_output.content,
        batch.content,
        batch.target_mask,
        l1_weight=0.2,
    )
    delta_loss = _masked_delta_loss(
        model_output.content,
        batch.content,
        batch.target_mask,
    )
    ctc_loss = _ctc_loss(model_output.text_logits, batch, criterion)
    if model_output.false_future_gates is None:
        gate_l2 = model_output.content.sum() * 0.0
        gate_mean = 0.0
        effect_ratio = 0.0
    else:
        gate_l2 = model_output.false_future_gates.pow(2).mean()
        gate_mean = float(model_output.false_future_gates.mean().item())
        effect_ratio = float(
            (
                model_output.false_future_corrections.abs().mean()
                / (model_output.false_future_effects.abs().mean() + 1e-8)
            ).item()
        )
    output_effect_loss = model_output.content.sum() * 0.0
    output_effect_cosine = 0.0
    if baseline_content is not None:
        output_effect_loss, effect_cosine = _masked_output_effect_loss(
            model_output.content,
            baseline_content,
            batch.content,
            batch.target_mask,
            config.output_effect_cosine_weight,
        )
        output_effect_cosine = float(effect_cosine.item())
    loss = (
        content_loss
        + config.delta_weight * delta_loss
        + config.ctc_weight * ctc_loss
        + config.output_effect_weight * output_effect_loss
        + config.gate_l2_weight * gate_l2
    )
    return loss, {
        "content_loss": float(content_loss.item()),
        "content_cosine": float(cosine.item()),
        "delta_loss": float(delta_loss.item()),
        "ctc_loss": float(ctc_loss.item()),
        "output_effect_loss": float(output_effect_loss.item()),
        "output_effect_cosine": output_effect_cosine,
        "gate_l2": float(gate_l2.item()),
        "gate_mean": gate_mean,
        "effect_ratio": effect_ratio,
    }


def _training_phase(epoch: int, config: FflTrainingConfig) -> str:
    if epoch < config.causal_warmup_epochs:
        return "causal"
    if epoch < config.causal_warmup_epochs + config.effect_warmup_epochs:
        return "effect"
    return "joint"


def _set_phase_trainability(model: ContentStudent, phase: str) -> None:
    for name, parameter in model.named_parameters():
        is_ffl = name.startswith("false_future_")
        if phase == "causal":
            parameter.requires_grad_(not is_ffl)
        elif phase == "effect":
            parameter.requires_grad_(is_ffl)
        else:
            parameter.requires_grad_(True)


@torch.inference_mode()
def evaluate_ffl(
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
    gate_means: list[float] = []
    effect_ratios: list[float] = []
    sparse_fracs: list[float] = []
    batches = 0

    for raw_batch in loader:
        batch = move_content_batch(raw_batch, device)
        output = model(batch.mel, batch.input_lengths)
        if output.text_logits is None:
            raise RuntimeError("FFL validation requires text logits")
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
            batch.transcripts.cpu(), batch.transcript_lengths.cpu()
        )
        predicted = _greedy_ctc_sequences(
            output.text_logits, batch.input_lengths
        )
        for hypothesis, reference in zip(predicted, expected):
            total_edits += _edit_distance(hypothesis, reference)
            total_characters += len(reference)

        gate = output.false_future_gates
        effect = output.false_future_effects
        correction = output.false_future_corrections
        gate_means.append(float(gate.mean().item()))
        effect_ratios.append(
            float(
                (correction.abs().mean() / (effect.abs().mean() + 1e-8)).item()
            )
        )
        sparse_fracs.append(float((gate < 0.05).float().mean().item()))
        batches += 1

    cosines = torch.cat(frame_cosines)
    return {
        "val_frame_cosine": cosines.mean().item(),
        "val_frame_cosine_p05": torch.quantile(cosines, 0.05).item(),
        "val_sequence_cosine": float(np.mean(sequence_cosines)),
        "val_ctc_loss": total_ctc_loss / max(batches, 1),
        "val_character_error_rate": total_edits / max(total_characters, 1),
        "val_gate_mean": float(np.mean(gate_means)),
        "val_effect_ratio": float(np.mean(effect_ratios)),
        "val_gate_sparsity_frac": float(np.mean(sparse_fracs)),
    }


def train_ffl_student(
    model_config: ContentStudentConfig,
    config: FflTrainingConfig,
) -> None:
    if model_config.architecture != "mio_ffl":
        raise ValueError("FFL training requires architecture='mio_ffl'")
    if (
        model_config.structured_fsq
        or model_config.hybrid_content
        or model_config.auxiliary_prefsq
    ):
        raise ValueError("FFL training expects a direct + CTC head only")
    if model_config.text_vocab_size <= 0:
        raise ValueError("FFL training requires text_vocab_size > 0")
    validate_ffl_config(config)
    seed_everything(config.seed)
    device = torch.device(config.device)

    with np.load(config.data_dir / "meta.npz") as meta:
        cache_format = (
            str(meta["cache_format"].item())
            if "cache_format" in meta.files
            else ""
        )
        if cache_format != "compact-fp16-ctc-v2":
            raise ValueError("FFL training requires compact-fp16-ctc-v2 cache")
        speakers = meta["spk_names"][: int(meta["n_samples"])].astype(str)
    train_indices, validation_indices = speaker_disjoint_split(
        speakers, config.validation_fraction, config.seed
    )
    probe_indices = speaker_balanced_subset(
        validation_indices, speakers, config.probe_samples, config.seed
    )

    collator = ContentCollator(
        None,
        config.seed,
        include_transcripts=True,
        pad_mel_multiple=config.pad_mel_multiple,
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
            pad_mel_multiple=config.pad_mel_multiple,
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
            pad_mel_multiple=config.pad_mel_multiple,
        ),
    )
    print(
        f"Train: {len(train_indices)} | Probe: {len(probe_indices)} | "
        f"Full val: {len(validation_indices)}",
        flush=True,
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
            config.init_checkpoint, device=device
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

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.resume is not None:
        payload = torch.load(config.resume, map_location=device)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    criterion = CTCLoss(blank=0, zero_infinity=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.output_dir / f"{config.run_name}.best.pt"
    probe_best_path = config.output_dir / f"{config.run_name}.probe-best.pt"
    last_path = config.output_dir / f"{config.run_name}.last.pt"
    steps_per_epoch = min(
        len(train_loader),
        config.steps_per_epoch or len(train_loader),
    )
    print(
        f"Params: {sum(p.numel() for p in model.parameters()):,} | "
        f"Device: {device} | ffl=on | ctc_weight={config.ctc_weight} | "
        f"effect_weight={config.output_effect_weight} | "
        f"gate_l2={config.gate_l2_weight} | "
        f"pad_multiple={config.pad_mel_multiple} | "
        f"{steps_per_epoch} steps/epoch | full_val_every={config.full_validation_every}",
        flush=True,
    )

    for epoch in range(start_epoch, config.epochs):
        phase = _training_phase(epoch, config)
        _set_phase_trainability(model, phase)
        trainable = [p for p in model.parameters() if p.requires_grad]
        model.train()
        total_loss = 0.0
        total_cosine = 0.0
        total_ctc = 0.0
        total_output_effect = 0.0
        total_output_effect_cosine = 0.0
        total_gate = 0.0
        total_effect_ratio = 0.0
        started = time.perf_counter()
        for step, raw_batch in enumerate(train_loader, start=1):
            if step > steps_per_epoch:
                break
            batch = move_content_batch(raw_batch, device)
            baseline_content = None
            if phase == "causal":
                output = model(
                    batch.mel,
                    batch.input_lengths,
                    enable_false_future=False,
                )
            else:
                with torch.no_grad():
                    baseline_content = model(
                        batch.mel,
                        batch.input_lengths,
                        enable_false_future=False,
                    ).content
                output = model(
                    batch.mel,
                    batch.input_lengths,
                    false_future_gate_override=(
                        config.effect_warmup_gate if phase == "effect" else None
                    ),
                )
            loss, parts = ffl_loss(
                output,
                batch,
                config,
                criterion,
                baseline_content,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_cosine += parts["content_cosine"]
            total_ctc += parts["ctc_loss"]
            total_output_effect += parts["output_effect_loss"]
            total_output_effect_cosine += parts["output_effect_cosine"]
            total_gate += parts["gate_mean"]
            total_effect_ratio += parts["effect_ratio"]
            if (
                device.type == "mps"
                and config.mps_empty_cache_every
                and step % config.mps_empty_cache_every == 0
            ):
                torch.mps.empty_cache()
            if step % config.log_every == 0 or step == steps_per_epoch:
                elapsed = time.perf_counter() - started
                print(
                    f"E{epoch:03d} {phase} step={step}/{steps_per_epoch} "
                    f"loss={total_loss/step:.4f} "
                    f"content_cos={total_cosine/step:.4f} "
                    f"ctc={total_ctc/step:.4f} "
                    f"out_eff={total_output_effect/step:.4f} "
                    f"out_eff_cos={total_output_effect_cosine/step:.4f} "
                    f"gate={total_gate/step:.4f} "
                    f"eff_ratio={total_effect_ratio/step:.3f} "
                    f"{elapsed/step:.3f}s/step",
                    flush=True,
                )
        scheduler.step()

        probe_metrics = evaluate_ffl(model, probe_loader, device, criterion)
        metrics = {
            key.replace("val_", "probe_", 1): value
            for key, value in probe_metrics.items()
        }
        metrics.update(
            {
                "train_loss": total_loss / steps_per_epoch,
                "train_content_cosine": total_cosine / steps_per_epoch,
                "train_ctc_loss": total_ctc / steps_per_epoch,
                "train_output_effect_loss": total_output_effect / steps_per_epoch,
                "train_output_effect_cosine": (
                    total_output_effect_cosine / steps_per_epoch
                ),
                "train_gate_mean": total_gate / steps_per_epoch,
                "train_effect_ratio": total_effect_ratio / steps_per_epoch,
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
            full_metrics = evaluate_ffl(
                model, full_validation_loader, device, criterion
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
            last_path, model, epoch=epoch, metrics=metrics,
            optimizer=optimizer, scheduler=scheduler,
        )
        if probe_improved:
            save_checkpoint(
                probe_best_path, model, epoch=epoch, metrics=metrics,
            )
        if full_improved:
            save_checkpoint(best_path, model, epoch=epoch, metrics=metrics)
        print(
            f"E{epoch:03d} {phase} "
            f"probe_cos={metrics['probe_frame_cosine']:.4f} "
            f"p05={metrics['probe_frame_cosine_p05']:.4f} "
            f"seq={metrics['probe_sequence_cosine']:.4f} "
            f"ctc={metrics['probe_ctc_loss']:.4f} "
            f"cer={metrics['probe_character_error_rate']:.4f} "
            f"gate={metrics['probe_gate_mean']:.4f} "
            f"eff={metrics['probe_effect_ratio']:.3f} "
            f"sparse={metrics['probe_gate_sparsity_frac']:.3f} "
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
                f"gate={metrics['val_gate_mean']:.4f} "
                f"eff={metrics['val_effect_ratio']:.3f} "
                f"sparse={metrics['val_gate_sparsity_frac']:.3f} "
                f"gap={metrics['target_gap']:.4f}",
                flush=True,
            )
        if device.type == "mps":
            torch.mps.empty_cache()

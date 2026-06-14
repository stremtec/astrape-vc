#!/usr/bin/env python3
"""Measure alignment, FSQ-subspace leakage, and loss conflicts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CTCLoss
from torch.utils.data import DataLoader

from astrape.checkpoint import load_content_checkpoint
from astrape.data import (
    ContentCollator,
    MioContentDataset,
    masked_content_loss,
    speaker_disjoint_split,
)
from astrape.flat_ctc_training import (
    _ctc_loss,
    _masked_delta_loss,
    _move,
    speaker_balanced_subset,
)
from astrape.fsq import indices_to_level_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/content_student_flat_ctc_512x10_probe1k.best.pt"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mio_vctk_full_compact"),
    )
    parser.add_argument(
        "--fsq-projection",
        type=Path,
        default=Path("checkpoints/teacher_fsq_proj_out.pt"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-offset", type=int, default=4)
    parser.add_argument("--gradient-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def nearest_fsq_codes(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    levels = (8, 8, 8, 5, 5)
    quantized = []
    indices = []
    for axis, level in enumerate(levels):
        values = (
            torch.arange(level, device=codes.device, dtype=codes.dtype)
            - level // 2
        ) / (level // 2)
        distance = (codes[:, axis, None] - values[None, :]).abs()
        axis_indices = distance.argmin(dim=-1)
        indices.append(axis_indices)
        quantized.append(values[axis_indices])
    return torch.stack(quantized, dim=-1), torch.stack(indices, dim=-1)


def cosine_stats(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> tuple[float, float]:
    values = torch.cat(
        [
            F.cosine_similarity(prediction, target, dim=-1)
            for prediction, target in zip(predictions, targets)
        ]
    )
    return values.mean().item(), torch.quantile(values, 0.05).item()


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.gradient_batches < 0:
        raise SystemExit("--samples must be positive and --gradient-batches non-negative")

    device = torch.device(args.device)
    model, metadata = load_content_checkpoint(args.checkpoint, device=device)
    model.eval()
    projection = torch.load(args.fsq_projection, map_location=device)
    weight = projection["weight"].to(device)
    bias = projection["bias"].to(device)
    inverse = torch.linalg.pinv(weight.T)

    with np.load(args.data_dir / "meta.npz") as meta:
        speakers = meta["spk_names"][: int(meta["n_samples"])].astype(str)
    _, validation_indices = speaker_disjoint_split(speakers, 0.15, args.seed)
    diagnostic_indices = speaker_balanced_subset(
        validation_indices,
        speakers,
        args.samples,
        args.seed,
    )
    loader = DataLoader(
        MioContentDataset(args.data_dir, args.data_dir, diagnostic_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ContentCollator(None, args.seed, include_transcripts=True),
    )

    raw_predictions: list[torch.Tensor] = []
    projected_predictions: list[torch.Tensor] = []
    quantized_predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    target_tokens: list[torch.Tensor] = []
    predicted_levels: list[torch.Tensor] = []
    offset_cosines = {
        offset: [] for offset in range(-args.max_offset, args.max_offset + 1)
    }
    target_autocorrelation = {
        offset: [] for offset in range(1, args.max_offset + 1)
    }

    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move(raw_batch, device)
            output = model(batch.mel, batch.input_lengths)
            prediction = output.content.transpose(1, 2)
            for item, length in enumerate(batch.target_lengths.tolist()):
                predicted = prediction[item, :length]
                target = batch.content[item, :length]
                codes = (predicted - bias) @ inverse
                quantized_codes, level_indices = nearest_fsq_codes(codes)
                projected = codes @ weight.T + bias
                quantized = quantized_codes @ weight.T + bias

                raw_predictions.append(predicted.cpu())
                projected_predictions.append(projected.cpu())
                quantized_predictions.append(quantized.cpu())
                targets.append(target.cpu())
                predicted_levels.append(level_indices.cpu())
                if raw_batch.token_indices is not None:
                    target_tokens.append(
                        raw_batch.token_indices[item, :length].cpu()
                    )

                for offset in offset_cosines:
                    if abs(offset) >= length:
                        continue
                    if offset >= 0:
                        left = predicted[: length - offset]
                        right = target[offset:length]
                    else:
                        left = predicted[-offset:length]
                        right = target[: length + offset]
                    offset_cosines[offset].append(
                        F.cosine_similarity(left, right, dim=-1).cpu()
                    )
                for offset in target_autocorrelation:
                    if offset < length:
                        target_autocorrelation[offset].append(
                            F.cosine_similarity(
                                target[:-offset],
                                target[offset:],
                                dim=-1,
                            ).cpu()
                        )

    raw_mean, raw_p05 = cosine_stats(raw_predictions, targets)
    projected_mean, projected_p05 = cosine_stats(
        projected_predictions,
        targets,
    )
    quantized_mean, quantized_p05 = cosine_stats(
        quantized_predictions,
        targets,
    )
    print(
        f"checkpoint_epoch={metadata.get('epoch')} samples={len(diagnostic_indices)}"
    )
    print(f"raw_cos={raw_mean:.6f} raw_p05={raw_p05:.6f}")
    print(
        f"affine_projected_cos={projected_mean:.6f} "
        f"affine_projected_p05={projected_p05:.6f}"
    )
    print(
        f"hard_fsq_cos={quantized_mean:.6f} hard_fsq_p05={quantized_p05:.6f}"
    )

    if target_tokens:
        expected_levels = torch.cat(
            [indices_to_level_indices(tokens) for tokens in target_tokens]
        )
        actual_levels = torch.cat(predicted_levels)
        axis_accuracy = (actual_levels == expected_levels).float().mean(dim=0)
        exact_accuracy = (
            (actual_levels == expected_levels).all(dim=-1).float().mean()
        )
        print(
            "fsq_axis_accuracy="
            + ",".join(f"{value:.4f}" for value in axis_accuracy.tolist())
            + f" exact={exact_accuracy.item():.4f}"
        )

    print("prediction_target_offset_cosine:")
    for offset, chunks in offset_cosines.items():
        values = torch.cat(chunks)
        print(
            f"  offset={offset:+d} mean={values.mean().item():.6f} "
            f"p05={torch.quantile(values, 0.05).item():.6f}"
        )
    print("teacher_target_autocorrelation:")
    for offset, chunks in target_autocorrelation.items():
        values = torch.cat(chunks)
        print(f"  offset={offset:+d} mean={values.mean().item():.6f}")

    if args.gradient_batches == 0:
        return
    criterion = CTCLoss(blank=0, zero_infinity=True)
    shared_parameters = list(model.blocks[-1].parameters())
    gradient_cosines = []
    weighted_norm_ratios = []
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= args.gradient_batches:
            break
        batch = _move(raw_batch, device)
        output = model(batch.mel, batch.input_lengths)
        content_loss, _ = masked_content_loss(
            output.content,
            batch.content,
            batch.target_mask,
            l1_weight=0.2,
        )
        content_objective = content_loss + 0.1 * _masked_delta_loss(
            output.content,
            batch.content,
            batch.target_mask,
        )
        ctc_objective = _ctc_loss(output.text_logits, batch, criterion)
        content_gradients = torch.autograd.grad(
            content_objective,
            shared_parameters,
            retain_graph=True,
        )
        ctc_gradients = torch.autograd.grad(
            ctc_objective,
            shared_parameters,
        )
        dot = 0.0
        content_squared_norm = 0.0
        ctc_squared_norm = 0.0
        for left, right in zip(content_gradients, ctc_gradients):
            left_cpu = left.detach().float().cpu().double()
            right_cpu = right.detach().float().cpu().double()
            dot += (left_cpu * right_cpu).sum().item()
            content_squared_norm += left_cpu.square().sum().item()
            ctc_squared_norm += right_cpu.square().sum().item()
        content_norm = content_squared_norm**0.5
        ctc_norm = ctc_squared_norm**0.5
        gradient_cosines.append(dot / (content_norm * ctc_norm))
        weighted_norm_ratios.append(0.05 * ctc_norm / content_norm)
    print(
        f"gradient_cosine_mean={np.mean(gradient_cosines):.6f} "
        f"min={np.min(gradient_cosines):.6f} "
        f"negative_fraction={np.mean(np.asarray(gradient_cosines) < 0):.4f}"
    )
    print(
        f"weighted_ctc_to_content_grad_norm="
        f"{np.mean(weighted_norm_ratios):.6f}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe how 50 Hz frames should be paired for a 25 Hz Mio target."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from astrape.data import MioContentDataset, speaker_disjoint_split
from astrape.flat_ctc_training import speaker_balanced_subset
from astrape.fsq import indices_to_codes


@dataclass
class ProbeSample:
    mel: torch.Tensor
    codes: torch.Tensor


class PairingProbe(nn.Module):
    def __init__(self, mode: str, hidden: int):
        super().__init__()
        if mode not in {"immediate", "delayed", "complete", "predicted"}:
            raise ValueError(f"unsupported pairing mode: {mode}")
        self.mode = mode
        self.input = nn.Sequential(
            nn.LayerNorm(80),
            nn.Linear(80, hidden),
            nn.SiLU(),
        )
        self.encoder = nn.GRU(
            hidden,
            hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.predictor = (
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 2),
                nn.SiLU(),
                nn.Linear(hidden * 2, hidden),
            )
            if mode == "predicted"
            else None
        )
        fusion_input = hidden * 2 if mode in {"complete", "predicted"} else hidden
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input),
            nn.Linear(fusion_input, hidden),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden, 5)

    def forward(
        self,
        mel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden, _ = self.encoder(self.input(mel.transpose(1, 2)))
        even = hidden[:, 0::2]
        if self.mode == "immediate":
            fused = self.fusion(even)
            consistency = None
        elif self.mode == "delayed":
            fused = self.fusion(hidden[:, 1::2])
            consistency = None
        else:
            odd = hidden[:, 1::2]
            length = min(even.shape[1], odd.shape[1])
            even = even[:, :length]
            odd = odd[:, :length]
            if self.mode == "complete":
                partner = odd
                consistency = None
            else:
                partner = self.predictor(even)
                consistency = F.smooth_l1_loss(
                    partner.contiguous(),
                    odd.detach().contiguous(),
                )
            fused = self.fusion(torch.cat((even, partner), dim=-1))
        return self.output(fused), consistency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--train-samples", type=int, default=192)
    parser.add_argument("--validation-samples", type=int, default=48)
    parser.add_argument("--max-mel-frames", type=int, default=200)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("immediate", "delayed", "complete", "predicted"),
        default=("immediate", "delayed", "complete", "predicted"),
    )
    return parser.parse_args()


def load_samples(
    dataset: MioContentDataset,
    indices: np.ndarray,
    max_mel_frames: int,
) -> list[ProbeSample]:
    samples = []
    for index in indices:
        sample = dataset[int(index)]
        length = min(sample.mel.shape[1], max_mel_frames)
        length -= length % 2
        target_length = min(length // 2, sample.content.shape[0])
        if sample.token_indices is None or target_length == 0:
            continue
        samples.append(
            ProbeSample(
                mel=sample.mel[:, : length].contiguous(),
                codes=indices_to_codes(
                    sample.token_indices[:target_length],
                    (8, 8, 8, 5, 5),
                ).float(),
            )
        )
    return samples


def collate(
    samples: list[ProbeSample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mel_lengths = torch.tensor([sample.mel.shape[1] for sample in samples])
    target_lengths = torch.tensor(
        [min(sample.mel.shape[1] // 2, len(sample.codes)) for sample in samples]
    )
    max_mel = int(mel_lengths.max())
    max_target = int(target_lengths.max())
    mel = torch.stack(
        [F.pad(sample.mel, (0, max_mel - sample.mel.shape[1])) for sample in samples]
    )
    codes = torch.stack(
        [
            F.pad(sample.codes[:length], (0, 0, 0, max_target - length))
            for sample, length in zip(samples, target_lengths.tolist())
        ]
    )
    mask = torch.arange(max_target).unsqueeze(0) < target_lengths.unsqueeze(1)
    return mel, codes, mask


def projected_cosines(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    length = min(prediction.shape[1], target.shape[1])
    prediction = prediction[:, :length]
    target = target[:, :length]
    mask = mask[:, :length]
    predicted_content = prediction @ weight.T + bias
    target_content = target @ weight.T + bias
    return F.cosine_similarity(
        predicted_content,
        target_content,
        dim=-1,
    )[mask]


def run_probe(
    mode: str,
    train_samples: list[ProbeSample],
    validation_samples: list[ProbeSample],
    args: argparse.Namespace,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, float]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = PairingProbe(mode, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    axis_weights = torch.tensor([1.0, 0.8, 0.8, 1.7, 2.0], device=device)
    model.train()
    for step in range(1, args.steps + 1):
        selected = rng.sample(train_samples, args.batch_size)
        mel, target, mask = collate(selected)
        mel = mel.to(device)
        target = target.to(device)
        mask = mask.to(device)
        prediction, consistency = model(mel)
        length = min(prediction.shape[1], target.shape[1])
        prediction = prediction[:, :length]
        target = target[:, :length]
        mask = mask[:, :length]
        code_loss = (
            F.smooth_l1_loss(
                prediction[mask],
                target[mask],
                reduction="none",
            )
            * axis_weights
        ).mean()
        cosine = projected_cosines(
            prediction,
            target,
            mask,
            weight,
            bias,
        ).mean()
        loss = code_loss + 1.0 - cosine
        if consistency is not None:
            loss = loss + 0.1 * consistency
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0:
            print(
                f"{mode} step={step}/{args.steps} "
                f"loss={loss.item():.5f} cos={cosine.item():.6f}",
                flush=True,
            )

    model.eval()
    all_cosines = []
    axis_errors = []
    with torch.inference_mode():
        for start in range(0, len(validation_samples), args.batch_size):
            mel, target, mask = collate(
                validation_samples[start : start + args.batch_size]
            )
            mel = mel.to(device)
            target = target.to(device)
            mask = mask.to(device)
            prediction, _ = model(mel)
            length = min(prediction.shape[1], target.shape[1])
            prediction = prediction[:, :length]
            target = target[:, :length]
            mask = mask[:, :length]
            all_cosines.append(
                projected_cosines(
                    prediction,
                    target,
                    mask,
                    weight,
                    bias,
                ).cpu()
            )
            axis_errors.append((prediction[mask] - target[mask]).abs().cpu())
    cosines = torch.cat(all_cosines)
    errors = torch.cat(axis_errors)
    result = {
        "cosine": cosines.mean().item(),
        "p05": torch.quantile(cosines, 0.05).item(),
    }
    print(
        f"{mode} validation_cos={result['cosine']:.6f} "
        f"p05={result['p05']:.6f} axis_mae="
        + ",".join(f"{value:.4f}" for value in errors.mean(dim=0).tolist()),
        flush=True,
    )
    return result


def main() -> None:
    args = parse_args()
    with np.load(args.data_dir / "meta.npz") as meta:
        count = int(meta["n_samples"])
        speakers = meta["spk_names"][:count].astype(str)
    train_indices, validation_indices = speaker_disjoint_split(
        speakers,
        0.15,
        args.seed,
    )
    train_indices = speaker_balanced_subset(
        train_indices,
        speakers,
        args.train_samples,
        args.seed,
    )
    validation_indices = speaker_balanced_subset(
        validation_indices,
        speakers,
        args.validation_samples,
        args.seed,
    )
    dataset = MioContentDataset(args.data_dir, args.data_dir)
    train_samples = load_samples(dataset, train_indices, args.max_mel_frames)
    validation_samples = load_samples(
        dataset,
        validation_indices,
        args.max_mel_frames,
    )
    projection = torch.load(args.fsq_projection, map_location="cpu")
    weight = projection["weight"].to(args.device)
    bias = projection["bias"].to(args.device)
    print(
        f"train={len(train_samples)} validation={len(validation_samples)} "
        f"frames<={args.max_mel_frames}",
        flush=True,
    )
    results = {}
    for mode in args.modes:
        results[mode] = run_probe(
            mode,
            train_samples,
            validation_samples,
            args,
            weight,
            bias,
        )
    if "immediate" in results:
        for mode, result in results.items():
            if mode == "immediate":
                continue
            print(
                f"{mode}_minus_immediate="
                f"{result['cosine'] - results['immediate']['cosine']:+.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

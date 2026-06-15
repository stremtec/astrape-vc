#!/usr/bin/env python3
"""Probe whether past SSL features can stand in for Mio's missing future context."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from extract_content_cache import TEACHER_SAMPLE_RATE, resample


METHODS = (
    "no_future",
    "zero",
    "repeat",
    "replay",
    "reverse",
    "extrapolate",
    "oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mio_vctk_full_compact"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--future-frames", type=int, default=62)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=(0.3, 0.5, 0.7),
    )
    return parser.parse_args()


def raw_ssl(teacher, waveform: torch.Tensor) -> torch.Tensor:
    padding = teacher._calculate_waveform_padding(waveform.shape[-1])
    padded = waveform.unsqueeze(0)
    if padding > 0:
        padded = F.pad(padded, (padding, padding))
    layers = teacher.ssl_feature_extractor(padded)
    return teacher._process_ssl_features(layers, teacher.local_ssl_layers)


def normalize(features: torch.Tensor) -> torch.Tensor:
    return (features - features.mean(dim=1, keepdim=True)) / (
        features.std(dim=1, keepdim=True) + 1e-8
    )


def make_false_future(
    observed: torch.Tensor,
    full: torch.Tensor,
    future_frames: int,
    method: str,
) -> torch.Tensor:
    available_future = full[:, observed.shape[1] : observed.shape[1] + future_frames]
    horizon = available_future.shape[1]
    if horizon == 0:
        return observed[:, :0]
    if method == "oracle":
        return available_future
    if method == "no_future":
        return observed[:, :0]
    if method == "zero":
        return observed.new_zeros(observed.shape[0], horizon, observed.shape[2])
    if method == "repeat":
        return observed[:, -1:].expand(-1, horizon, -1)

    history = observed[:, max(0, observed.shape[1] - horizon) :]
    if history.shape[1] < horizon:
        history = F.pad(history, (0, 0, horizon - history.shape[1], 0))
    if method == "replay":
        return history
    if method == "reverse":
        return history.flip(1)
    if method == "extrapolate":
        velocity = observed[:, -1:] - observed[:, -2:-1]
        steps = torch.arange(
            1,
            horizon + 1,
            device=observed.device,
            dtype=observed.dtype,
        ).view(1, -1, 1)
        damping = torch.exp(-steps / 8.0)
        return observed[:, -1:] + steps * damping * velocity
    raise ValueError(f"Unknown false-future method: {method}")


def encode_content(teacher, features: torch.Tensor) -> torch.Tensor:
    encoded = teacher.local_encoder(features)
    encoded = teacher.conv_downsample(encoded.transpose(1, 2)).transpose(1, 2)
    content, _ = teacher.local_quantizer.encode(encoded)
    return content


def content_at(
    teacher,
    full_features: torch.Tensor,
    observed_features: torch.Tensor,
    target_index: int,
    future_frames: int,
    method: str,
) -> torch.Tensor | None:
    fake = make_false_future(
        observed_features,
        full_features,
        future_frames,
        method,
    )
    candidate = torch.cat((observed_features, fake), dim=1)
    content = encode_content(teacher, candidate)
    if content.shape[1] <= target_index:
        return None
    return content[:, target_index]


def summarize(values: dict[str, list[float]], label: str) -> None:
    print(label)
    baseline = np.asarray(values["no_future"])
    for method in METHODS:
        scores = values[method]
        if not scores:
            continue
        array = np.asarray(scores)
        delta = array - baseline if len(array) == len(baseline) else np.asarray([])
        difficult = baseline <= np.quantile(baseline, 0.25) if baseline.size else None
        comparison = (
            f" delta={delta.mean():+.6f} win={(delta > 0).mean():.1%} "
            f"hard25_delta={delta[difficult].mean():+.6f}"
            if delta.size
            else ""
        )
        print(
            f"  {method:11s} n={len(array):3d} "
            f"mean={array.mean():.6f} "
            f"p05={np.quantile(array, 0.05):.6f} "
            f"p50={np.quantile(array, 0.50):.6f} "
            f"exact={(array > 0.99999).mean():.1%}"
            f"{comparison}"
        )


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.future_frames <= 0:
        raise SystemExit("--samples and --future-frames must be positive")
    if any(not 0.0 < fraction < 1.0 for fraction in args.fractions):
        raise SystemExit("--fractions must be between zero and one")
    try:
        from miocodec.model import MioCodecModel
    except ModuleNotFoundError as error:
        raise SystemExit("Run with the dedicated Mio Python environment") from error

    with np.load(args.data_dir / "meta.npz") as meta:
        source_files = meta["source_files"].astype(str)
        sample_count = min(args.samples, int(meta["n_samples"]))
    selected = np.linspace(
        0,
        len(source_files) - 1,
        sample_count,
        dtype=np.int64,
    )

    device = torch.device(args.device)
    teacher = MioCodecModel.from_pretrained(
        "Aratako/MioCodec-25Hz-44.1kHz-v2"
    ).eval().to(device)
    local_only: dict[str, list[float]] = defaultdict(list)
    prefix_end_to_end: dict[str, list[float]] = defaultdict(list)

    with torch.inference_mode():
        for cache_index in selected.tolist():
            audio, sample_rate = sf.read(source_files[cache_index], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = resample(
                audio,
                sample_rate,
                TEACHER_SAMPLE_RATE,
            ).astype(np.float32, copy=False)
            waveform = torch.from_numpy(audio).to(device)
            full_raw = raw_ssl(teacher, waveform)
            full_normalized = normalize(full_raw)
            full_content = encode_content(teacher, full_normalized)

            for fraction in args.fractions:
                target_index = min(
                    full_content.shape[1] - 2,
                    max(2, round(full_content.shape[1] * fraction)),
                )
                observed_ssl_frames = target_index * 2 + 2
                if full_normalized.shape[1] <= observed_ssl_frames:
                    continue
                target = full_content[:, target_index]

                local_observed = full_normalized[:, :observed_ssl_frames]
                for method in METHODS:
                    prediction = content_at(
                        teacher,
                        full_normalized,
                        local_observed,
                        target_index,
                        args.future_frames,
                        method,
                    )
                    if prediction is not None:
                        local_only[method].append(
                            F.cosine_similarity(prediction, target, dim=-1).item()
                        )

                available_seconds = (
                    observed_ssl_frames * 320 + 400
                ) / 16000
                prefix_samples = min(
                    len(audio),
                    round(available_seconds * TEACHER_SAMPLE_RATE),
                )
                prefix_raw = raw_ssl(teacher, waveform[:prefix_samples])
                if prefix_raw.shape[1] < observed_ssl_frames:
                    continue
                prefix_observed = normalize(prefix_raw)[:, :observed_ssl_frames]
                for method in METHODS:
                    prediction = content_at(
                        teacher,
                        full_normalized,
                        prefix_observed,
                        target_index,
                        args.future_frames,
                        method,
                    )
                    if prediction is not None:
                        prefix_end_to_end[method].append(
                            F.cosine_similarity(prediction, target, dim=-1).item()
                        )

    print(
        f"samples={sample_count} points_per_sample={len(args.fractions)} "
        f"future_frames={args.future_frames} "
        f"future_ms={args.future_frames * 20}"
    )
    summarize(local_only, "local_encoder_only:")
    summarize(prefix_end_to_end, "prefix_wavlm_end_to_end:")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure how Mio content targets change as future audio is revealed."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from extract_content_cache import TEACHER_SAMPLE_RATE, resample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mio_vctk_full_compact"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument(
        "--lookahead-ms",
        type=int,
        nargs="+",
        default=(0, 40, 80, 160, 320, 640, 1280),
    )
    return parser.parse_args()


def extract_stages(teacher, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
    padding = teacher._calculate_waveform_padding(waveform.shape[-1])
    padded = waveform.unsqueeze(0)
    if padding > 0:
        padded = F.pad(padded, (padding, padding), mode="constant")
    ssl_layers = teacher.ssl_feature_extractor(padded)
    raw_ssl = teacher._process_ssl_features(
        ssl_layers,
        teacher.local_ssl_layers,
    )
    normalized_ssl = teacher._normalize_ssl_features(raw_ssl)
    local_encoded = teacher.local_encoder(normalized_ssl)
    if teacher.downsample_factor > 1:
        if teacher.config.use_conv_downsample:
            pre_fsq = teacher.conv_downsample(
                local_encoded.transpose(1, 2)
            ).transpose(1, 2)
        else:
            pre_fsq = F.avg_pool1d(
                local_encoded.transpose(1, 2),
                kernel_size=teacher.downsample_factor,
                stride=teacher.downsample_factor,
            ).transpose(1, 2)
    else:
        pre_fsq = local_encoded
    content, _ = teacher.local_quantizer.encode(pre_fsq)
    return {
        "raw_ssl": raw_ssl.squeeze(0).cpu(),
        "normalized_ssl": normalized_ssl.squeeze(0).cpu(),
        "local_encoded": local_encoded.squeeze(0).cpu(),
        "pre_fsq": pre_fsq.squeeze(0).cpu(),
        "content": content.squeeze(0).cpu(),
    }


def frame_cosine(
    prefix: dict[str, torch.Tensor],
    full: dict[str, torch.Tensor],
    target_index: int,
) -> dict[str, float] | None:
    ssl_start = target_index * 2
    ssl_end = ssl_start + 2
    if (
        prefix["content"].shape[0] <= target_index
        or prefix["raw_ssl"].shape[0] < ssl_end
    ):
        return None
    result = {}
    for name in ("raw_ssl", "normalized_ssl", "local_encoded"):
        result[name] = F.cosine_similarity(
            prefix[name][ssl_start:ssl_end],
            full[name][ssl_start:ssl_end],
            dim=-1,
        ).mean().item()
    for name in ("pre_fsq", "content"):
        result[name] = F.cosine_similarity(
            prefix[name][target_index],
            full[name][target_index],
            dim=-1,
        ).item()
    return result


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
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
    collected: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with torch.inference_mode():
        for cache_index in selected.tolist():
            path = Path(source_files[cache_index])
            audio, sample_rate = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = resample(
                audio,
                sample_rate,
                TEACHER_SAMPLE_RATE,
            ).astype(np.float32, copy=False)
            waveform = torch.from_numpy(audio).to(device)
            full = extract_stages(teacher, waveform)
            with np.load(args.data_dir / f"s_{cache_index:05d}.npz") as cached:
                cached_content = torch.from_numpy(cached["ce_768"]).float()
            common = min(cached_content.shape[0], full["content"].shape[0])
            cache_cosine = F.cosine_similarity(
                cached_content[:common],
                full["content"][:common],
                dim=-1,
            ).mean().item()
            print(
                f"sample={cache_index} seconds={len(audio) / TEACHER_SAMPLE_RATE:.2f} "
                f"tokens={full['content'].shape[0]} cache_cos={cache_cosine:.6f}"
            )

            token_count = full["content"].shape[0]
            for fraction in (0.35, 0.65):
                target_index = min(
                    token_count - 2,
                    max(2, round(token_count * fraction)),
                )
                student_available_seconds = (
                    target_index * 2 * 320 + 512
                ) / 16000
                for lookahead_ms in args.lookahead_ms:
                    prefix_samples = min(
                        len(audio),
                        round(
                            (
                                student_available_seconds
                                + lookahead_ms / 1000
                            )
                            * TEACHER_SAMPLE_RATE
                        ),
                    )
                    prefix = extract_stages(
                        teacher,
                        waveform[:prefix_samples],
                    )
                    scores = frame_cosine(prefix, full, target_index)
                    if scores is None:
                        continue
                    for name, value in scores.items():
                        collected[lookahead_ms][name].append(value)

    print("future_context_stability:")
    for lookahead_ms in args.lookahead_ms:
        metrics = collected[lookahead_ms]
        if not metrics:
            continue
        values = " ".join(
            f"{name}={np.mean(metrics[name]):.6f}"
            for name in (
                "raw_ssl",
                "normalized_ssl",
                "local_encoded",
                "pre_fsq",
                "content",
            )
        )
        print(
            f"  lookahead_ms={lookahead_ms:4d} points={len(metrics['content'])} "
            f"{values}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract a compact full-VCTK cache for causal content distillation."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from astrape.audio import StreamingLogMel
from astrape.text import encode_transcript


TEACHER_SAMPLE_RATE = 44100
MEL_SAMPLE_RATE = 16000


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(
        audio,
        target_rate // divisor,
        source_rate // divisor,
    )


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vctk-root",
        type=Path,
        default=Path(
            "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mio_vctk_full_compact"),
    )
    parser.add_argument(
        "--transcript-root",
        type=Path,
        default=Path("/Users/asill/asill/research2/datasets/vctk/txt"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def discover_audio(
    root: Path,
    transcript_root: Path,
) -> list[tuple[str, Path, Path]]:
    selected = []
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir() or not speaker_dir.name.startswith("p"):
            continue
        for path in sorted(speaker_dir.glob(f"{speaker_dir.name}_*_mic1.flac")):
            utterance = path.name.removesuffix("_mic1.flac")
            transcript_path = transcript_root / speaker_dir.name / f"{utterance}.txt"
            if transcript_path.exists():
                selected.append((speaker_dir.name, path, transcript_path))
    return selected


def extract_local_targets(
    teacher,
    waveform: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padding = teacher._calculate_waveform_padding(waveform.shape[-1])
    local_ssl, _ = teacher.forward_ssl_features(
        waveform.unsqueeze(0),
        padding=padding,
    )
    device_type = local_ssl.device.type
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device_type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with autocast:
        pre_fsq = teacher.local_encoder(local_ssl)
        if teacher.downsample_factor > 1:
            if teacher.config.use_conv_downsample:
                pre_fsq = teacher.conv_downsample(
                    pre_fsq.transpose(1, 2)
                ).transpose(1, 2)
            else:
                pre_fsq = F.avg_pool1d(
                    pre_fsq.transpose(1, 2),
                    kernel_size=teacher.downsample_factor,
                    stride=teacher.downsample_factor,
                ).transpose(1, 2)
        content, token_indices = teacher.local_quantizer.encode(pre_fsq)
    return content.squeeze(0), token_indices.squeeze(0), pre_fsq.squeeze(0)


def main() -> None:
    args = parse_args()
    if not args.vctk_root.is_dir():
        raise SystemExit(f"VCTK root does not exist: {args.vctk_root}")
    if not args.transcript_root.is_dir():
        raise SystemExit(f"Transcript root does not exist: {args.transcript_root}")
    if args.log_every <= 0:
        raise SystemExit("--log-every must be positive")

    try:
        from miocodec.model import MioCodecModel
    except ModuleNotFoundError as error:
        raise SystemExit(
            "MioCodec is required. Run this script with the dedicated Mio runtime."
        ) from error

    selected = discover_audio(args.vctk_root, args.transcript_root)
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No VCTK mic1 FLAC files found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    speakers = np.asarray([speaker for speaker, _, _ in selected])
    source_files = np.asarray([str(path) for _, path, _ in selected])
    transcript_files = np.asarray(
        [str(path) for _, _, path in selected]
    )
    utterance_ids = np.asarray(
        [path.stem.split("_")[1] for _, path, _ in selected]
    )
    atomic_savez(
        args.output_dir / "meta.npz",
        spk_names=speakers,
        utterance_ids=utterance_ids,
        source_files=source_files,
        transcript_files=transcript_files,
        n_samples=np.asarray(len(selected), dtype=np.int64),
        cache_format=np.asarray("compact-fp16-ctc-v2"),
    )

    device = torch.device(args.device)
    teacher = MioCodecModel.from_pretrained(
        "Aratako/MioCodec-25Hz-44.1kHz-v2"
    ).eval().to(device)
    mel_extractor = StreamingLogMel()
    started = time.perf_counter()
    completed = 0
    skipped = 0
    cache_bytes = sum(
        file.stat().st_size for file in args.output_dir.glob("s_*.npz")
    )

    for index, (_, path, transcript_path) in enumerate(selected):
        output_path = args.output_dir / f"s_{index:05d}.npz"
        if args.resume and output_path.exists():
            skipped += 1
            continue

        audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = resample(audio, sample_rate, TEACHER_SAMPLE_RATE).astype(
            np.float32,
            copy=False,
        )
        if args.max_seconds is not None:
            audio = audio[: int(TEACHER_SAMPLE_RATE * args.max_seconds)]
        waveform = torch.from_numpy(audio).to(device)

        with torch.inference_mode():
            content, tokens, pre_fsq = extract_local_targets(teacher, waveform)
            audio_16k = resample(
                audio,
                TEACHER_SAMPLE_RATE,
                MEL_SAMPLE_RATE,
            ).astype(np.float32, copy=False)
            logmel = mel_extractor(torch.from_numpy(audio_16k)).squeeze(0)

        expected_frames = (logmel.shape[1] + 1) // 2
        aligned_frames = min(
            expected_frames,
            content.shape[0],
            tokens.shape[0],
            pre_fsq.shape[0],
        )
        if aligned_frames <= 0:
            raise RuntimeError(f"No aligned frames extracted from {path}")
        logmel = logmel[:, : 2 * aligned_frames]
        atomic_savez(
            output_path,
            logmel=logmel.numpy().astype(np.float16),
            ce_768=content[:aligned_frames].float().cpu().numpy().astype(np.float16),
            pre_fsq_768=pre_fsq[:aligned_frames]
            .float()
            .cpu()
            .numpy()
            .astype(np.float16),
            ct=tokens[:aligned_frames].cpu().numpy().astype(np.uint16),
            transcript=encode_transcript(
                transcript_path.read_text(encoding="utf-8")
            ).numpy().astype(np.uint8),
        )
        cache_bytes += output_path.stat().st_size
        completed += 1
        processed = index + 1
        if processed % args.log_every == 0 or processed == len(selected):
            elapsed = time.perf_counter() - started
            rate = completed / max(elapsed, 1e-6)
            remaining = (len(selected) - processed) / max(rate, 1e-6)
            size_gib = cache_bytes / 2**30
            print(
                f"{processed}/{len(selected)} extracted={completed} skipped={skipped} "
                f"{rate:.2f} files/s eta={remaining / 3600:.1f}h "
                f"cache={size_gib:.2f}GiB",
                flush=True,
            )

    print(
        f"Done: {len(selected)} samples, {len(set(speakers.tolist()))} speakers",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the Mio-like causal content student in two quality-first phases."""

import argparse
from pathlib import Path

from astrape.model import ContentStudentConfig
from astrape.text import VOCAB_SIZE
from astrape.two_phase_training import (
    TwoPhaseTrainingConfig,
    train_two_phase_student,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mio_vctk_full_compact"),
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path(
            "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
        ),
    )
    parser.add_argument(
        "--transcript-root",
        type=Path,
        default=Path("/Users/asill/asill/research2/datasets/vctk/txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--run-name",
        default="content_student_mio_causal_two_phase",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--phase1-epochs", type=int, default=10)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--phase1-learning-rate", type=float, default=2e-4)
    parser.add_argument("--phase2-learning-rate", type=float, default=5e-6)
    parser.add_argument("--teacher-probability", type=float, default=0.5)
    parser.add_argument("--teacher-ctc-weight", type=float, default=0.05)
    parser.add_argument("--original-ctc-weight", type=float, default=0.05)
    parser.add_argument("--delta-weight", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--attention-context-frames", type=int, default=125)
    parser.add_argument("--mio-ff-hidden", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--probe-samples", type=int, default=1024)
    parser.add_argument("--full-validation-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = ContentStudentConfig(
        architecture="mio_causal",
        hidden=args.hidden,
        n_layers=args.layers,
        n_heads=args.heads,
        text_vocab_size=VOCAB_SIZE,
        max_attention_context=args.attention_context_frames,
        mio_ff_hidden=args.mio_ff_hidden,
    )
    train_two_phase_student(
        model_config,
        TwoPhaseTrainingConfig(
            data_dir=args.data_dir,
            audio_root=args.audio_root,
            transcript_root=args.transcript_root,
            output_dir=args.output_dir,
            run_name=args.run_name,
            device=args.device,
            batch_size=args.batch_size,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            steps_per_epoch=args.steps_per_epoch,
            phase1_learning_rate=args.phase1_learning_rate,
            phase2_learning_rate=args.phase2_learning_rate,
            teacher_probability=args.teacher_probability,
            teacher_ctc_weight=args.teacher_ctc_weight,
            original_ctc_weight=args.original_ctc_weight,
            delta_weight=args.delta_weight,
            seed=args.seed,
            num_workers=args.num_workers,
            resume=args.resume,
            probe_samples=args.probe_samples,
            full_validation_every=args.full_validation_every,
            log_every=args.log_every,
        ),
    )


if __name__ == "__main__":
    main()

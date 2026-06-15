#!/usr/bin/env python3
"""Run the disposable token-synchronous content-student Phase 0."""

from __future__ import annotations

import argparse
from pathlib import Path

from astrape.text import VOCAB_SIZE
from astrape.token_student import TokenStudentConfig
from astrape.token_student_training import TokenPhase0Config, train_token_phase0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mio_vctk_full_compact"),
    )
    parser.add_argument(
        "--projection",
        type=Path,
        default=Path("checkpoints/teacher_fsq_proj_out.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--run-name",
        default="content_student_token_sync_phase0",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--scheduler-t-max-epochs", type=int, default=10)
    parser.add_argument("--supervised-mel-frames", type=int, default=300)
    parser.add_argument("--history-mel-frames", type=int, default=100)
    parser.add_argument("--pad-mel-multiple", type=int, default=64)
    parser.add_argument("--probe-samples", type=int, default=1024)
    parser.add_argument("--full-validation-every", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = TokenStudentConfig(text_vocab_size=VOCAB_SIZE)
    config = TokenPhase0Config(
        data_dir=args.data_dir,
        projection_path=args.projection,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        scheduler_t_max_epochs=args.scheduler_t_max_epochs,
        supervised_mel_frames=args.supervised_mel_frames,
        history_mel_frames=args.history_mel_frames,
        pad_mel_multiple=args.pad_mel_multiple,
        probe_samples=args.probe_samples,
        full_validation_every=args.full_validation_every,
        log_every=args.log_every,
        seed=args.seed,
        num_workers=args.num_workers,
        resume=args.resume,
    )
    train_token_phase0(model, config)


if __name__ == "__main__":
    main()

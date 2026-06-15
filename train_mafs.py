#!/usr/bin/env python3
"""Train the Manifest-Aligned Factorized Student (MAFS)."""

import argparse
from pathlib import Path

from astrape.mafs_model import MafsConfig
from astrape.mafs_training import MafsTrainingConfig, train_mafs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mio_vctk_full_compact"))
    parser.add_argument("--projection-path", type=Path, default=Path("checkpoints/teacher_fsq_proj_out.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", default="mafs_384x4_512x8")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--ctc-weight", type=float, default=0.05)
    parser.add_argument("--future-weight", type=float, default=0.1)
    parser.add_argument("--delta-weight", type=float, default=0.03)
    parser.add_argument("--probe-samples", type=int, default=1024)
    parser.add_argument("--full-validation-every", type=int, default=5)
    parser.add_argument("--target-cosine", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = MafsConfig()
    train_mafs(
        model,
        MafsTrainingConfig(
            data_dir=args.data_dir,
            projection_path=args.projection_path,
            output_dir=args.output_dir,
            run_name=args.run_name,
            device=args.device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            ctc_weight=args.ctc_weight,
            future_weight=args.future_weight,
            delta_weight=args.delta_weight,
            probe_samples=args.probe_samples,
            full_validation_every=args.full_validation_every,
            target_cosine=args.target_cosine,
            seed=args.seed,
            num_workers=args.num_workers,
            resume=args.resume,
            log_every=args.log_every,
        ),
    )


if __name__ == "__main__":
    main()

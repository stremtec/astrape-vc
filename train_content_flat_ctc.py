#!/usr/bin/env python3
"""Train a strict-causal direct 768d content student with auxiliary CTC."""

import argparse
from pathlib import Path

from astrape.flat_ctc_training import (
    FlatCtcTrainingConfig,
    train_flat_ctc_student,
)
from astrape.model import ContentStudentConfig
from astrape.text import VOCAB_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mio_vctk_full_compact"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--architecture",
        choices=("legacy", "mio_causal", "mio_ffl"),
        default="legacy",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--ctc-weight", type=float, default=0.05)
    parser.add_argument("--delta-weight", type=float, default=0.1)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--attention-context-frames", type=int)
    parser.add_argument("--mio-ff-hidden", type=int)
    parser.add_argument("--ffl-hidden", type=int, default=256)
    parser.add_argument("--ffl-horizon", type=int, default=16)
    parser.add_argument("--ffl-history", type=int, default=64)
    parser.add_argument("--ffl-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--probe-samples", type=int, default=1024)
    parser.add_argument("--full-validation-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.architecture in {"mio_causal", "mio_ffl"}:
        hidden = args.hidden or 768
        layers = args.layers or 6
        heads = args.heads or 12
        attention_context = args.attention_context_frames or 125
        default_name = (
            "content_student_mio_ffl_768x6"
            if args.architecture == "mio_ffl"
            else "content_student_mio_causal_768x6"
        )
        run_name = args.run_name or default_name
    else:
        hidden = args.hidden or 512
        layers = args.layers or 10
        heads = args.heads or 8
        attention_context = args.attention_context_frames or 200
        run_name = args.run_name or "content_student_flat_ctc_512x10"
    model = ContentStudentConfig(
        architecture=args.architecture,
        hidden=hidden,
        n_layers=layers,
        n_heads=heads,
        text_vocab_size=VOCAB_SIZE,
        max_attention_context=attention_context,
        mio_ff_hidden=args.mio_ff_hidden,
        ffl_hidden=args.ffl_hidden,
        ffl_horizon=args.ffl_horizon,
        ffl_history=args.ffl_history,
        ffl_heads=args.ffl_heads,
    )
    train_flat_ctc_student(
        model,
        FlatCtcTrainingConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            run_name=run_name,
            device=args.device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            ctc_weight=args.ctc_weight,
            delta_weight=args.delta_weight,
            seed=args.seed,
            num_workers=args.num_workers,
            resume=args.resume,
            init_checkpoint=args.init_checkpoint,
            probe_samples=args.probe_samples,
            full_validation_every=args.full_validation_every,
            log_every=args.log_every,
        ),
    )


if __name__ == "__main__":
    main()

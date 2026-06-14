#!/usr/bin/env python3
"""Train the strict-causal continuous-content student with a five-axis FSQ head."""

import argparse
from pathlib import Path

from astrape.hybrid_training import HybridTrainingConfig, train_hybrid_student
from astrape.model import ContentStudentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mio_4k"))
    parser.add_argument("--mel-dir", type=Path, default=Path("data/mio_4k_mel"))
    parser.add_argument(
        "--fsq-projection",
        type=Path,
        default=Path("checkpoints/teacher_fsq_proj_out.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", default="content_student_hybrid_fsq")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--supervised-mel-frames", type=int, default=80)
    parser.add_argument("--history-mel-frames", type=int, default=160)
    parser.add_argument("--attention-context-frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = ContentStudentConfig(
        hidden=args.hidden,
        n_layers=args.layers,
        n_heads=args.heads,
        auxiliary_prefsq=True,
        structured_fsq=True,
        hybrid_content=True,
        max_attention_context=args.attention_context_frames,
    )
    train_hybrid_student(
        model,
        HybridTrainingConfig(
            data_dir=args.data_dir,
            mel_dir=args.mel_dir,
            fsq_projection=args.fsq_projection,
            output_dir=args.output_dir,
            run_name=args.run_name,
            device=args.device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            supervised_mel_frames=args.supervised_mel_frames,
            history_mel_frames=args.history_mel_frames,
            seed=args.seed,
            num_workers=args.num_workers,
            resume=args.resume,
            init_checkpoint=args.init_checkpoint,
            log_every=args.log_every,
        ),
    )


if __name__ == "__main__":
    main()

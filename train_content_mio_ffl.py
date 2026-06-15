#!/usr/bin/env python3
"""Train the mio_ffl ContentStudent on compact VCTK with FFL supervision.

The first iteration supervises only what is locally available: the direct 768d
content, character CTC at 50 Hz, and a smooth delta loss. The full L_effect,
L_hidden, and L_slot objectives need a teacher causal-effect cache that does
not yet exist on disk; future revisions can layer them on top.

Usage:

    .venv/bin/python train_content_mio_ffl.py \\
        --data-dir data/mio_vctk_full_compact \\
        --run-name content_student_mio_ffl_768x6 \\
        --epochs 30 --steps-per-epoch 1000 \\
        --device mps
"""

import argparse
from pathlib import Path

from astrape.ffl_training import FflTrainingConfig, train_ffl_student
from astrape.model import ContentStudentConfig
from astrape.text import VOCAB_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mio_vctk_full_compact"),
    )
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--transcript-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--run-name", default="content_student_mio_ffl_768x6"
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--ctc-weight", type=float, default=0.05)
    parser.add_argument("--delta-weight", type=float, default=0.1)
    parser.add_argument("--output-effect-weight", type=float, default=0.5)
    parser.add_argument("--output-effect-cosine-weight", type=float, default=0.1)
    parser.add_argument("--gate-l2-weight", type=float, default=0.0)
    parser.add_argument("--causal-warmup-epochs", type=int, default=1)
    parser.add_argument("--effect-warmup-epochs", type=int, default=1)
    parser.add_argument("--effect-warmup-gate", type=float, default=0.25)
    parser.add_argument("--pad-mel-multiple", type=int, default=64)
    parser.add_argument("--mps-empty-cache-every", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--attention-context-frames", type=int, default=125)
    parser.add_argument("--mio-ff-hidden", type=int)
    parser.add_argument("--ffl-hidden", type=int, default=256)
    parser.add_argument("--ffl-horizon", type=int, default=16)
    parser.add_argument("--ffl-history", type=int, default=64)
    parser.add_argument("--ffl-heads", type=int, default=4)
    parser.add_argument("--ffl-coarse-layers", type=int, default=2)
    parser.add_argument("--ffl-reverse-layers", type=int, default=2)
    parser.add_argument("--ffl-summary-layers", type=int, default=3)
    parser.add_argument("--ffl-gate-bias", type=float, default=-4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--probe-samples", type=int, default=1024)
    parser.add_argument("--full-validation-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--target-cosine", type=float, default=0.99,
        help="Diagnostic only; achievement of this is not a stop criterion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.audio_root is not None
        or args.transcript_root is not None
    ):
        # The FFL trainer only consumes the compact teacher cache, but we
        # mirror the flat_ctc CLI for users who copy/paste between scripts.
        pass

    if args.hidden % args.heads:
        raise SystemExit("--hidden must be divisible by --heads")

    model = ContentStudentConfig(
        architecture="mio_ffl",
        hidden=args.hidden,
        n_layers=args.layers,
        n_heads=args.heads,
        text_vocab_size=VOCAB_SIZE,
        max_attention_context=args.attention_context_frames,
        mio_ff_hidden=args.mio_ff_hidden,
        ffl_hidden=args.ffl_hidden,
        ffl_horizon=args.ffl_horizon,
        ffl_history=args.ffl_history,
        ffl_heads=args.ffl_heads,
        ffl_coarse_layers=args.ffl_coarse_layers,
        ffl_reverse_layers=args.ffl_reverse_layers,
        ffl_summary_layers=args.ffl_summary_layers,
        ffl_gate_bias=args.ffl_gate_bias,
    )
    train_ffl_student(
        model,
        FflTrainingConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            run_name=args.run_name,
            device=args.device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            ctc_weight=args.ctc_weight,
            delta_weight=args.delta_weight,
            output_effect_weight=args.output_effect_weight,
            output_effect_cosine_weight=args.output_effect_cosine_weight,
            gate_l2_weight=args.gate_l2_weight,
            causal_warmup_epochs=args.causal_warmup_epochs,
            effect_warmup_epochs=args.effect_warmup_epochs,
            effect_warmup_gate=args.effect_warmup_gate,
            pad_mel_multiple=args.pad_mel_multiple,
            mps_empty_cache_every=args.mps_empty_cache_every,
            seed=args.seed,
            num_workers=args.num_workers,
            resume=args.resume,
            init_checkpoint=args.init_checkpoint,
            probe_samples=args.probe_samples,
            full_validation_every=args.full_validation_every,
            target_cosine=args.target_cosine,
            log_every=args.log_every,
        ),
    )


if __name__ == "__main__":
    main()

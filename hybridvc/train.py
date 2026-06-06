"""
HybridVC Training Pipeline.

4-Phase training as per design doc:

Phase 0: Weight Transfer — btrv3lite checkpoint → HybridVC converter
Phase 1: RAF Vocoder Training — BigVGAN-base + RAF loss
Phase 2: SSL Distillation — WavLM teacher for encoder
Phase 3: End-to-End + BWE — joint fine-tune with BWE

Usage:
    # Phase 0: weight transfer
    python -m hybridvc.train --phase 0 \
        --btrv3lite-ckpt /path/to/btrv3lite_converter.pt \
        --output-dir runs/phase0

    # Phase 1: RAF vocoder training
    python -m hybridvc.train --phase 1 \
        --cache-dir runs/cache --bank runs/bank.pt \
        --output-dir runs/phase1 --device cuda --batch-size 8

    # Phase 3: E2E + BWE
    python -m hybridvc.train --phase 3 \
        --resume runs/phase1/converter_step010000.pt \
        --output-dir runs/phase3 --device mps --batch-size 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ConverterConfig, TrainConfig
from .converter import CausalConvNeXtConverter, make_converter


# ── Phase 0: Weight Transfer ───────────────────────────────────

def _transfer_weights_btrv3lite(
    hybridvc_ckpt_path: str,
    output_path: str,
    n_blocks: int = 10,
    content_dim: int = 768,
    speaker_dim: int = 128,
) -> CausalConvNeXtConverter:
    """
    Transfer weights from btrv3lite checkpoint to HybridVC converter.

    Strategy:
    - First 8 blocks → direct copy (same architecture)
    - Blocks 9-10 → identity-init (new blocks)
    - Cross-attn layers → random init (new)
    - in_proj, out_proj, out_gate → copy from v1
    - AdaLN-Zero MLPs → copy for blocks 0-7, identity for 8-9
    """
    print(f"[Phase 0] Loading btrv3lite checkpoint: {hybridvc_ckpt_path}")
    ckpt = torch.load(hybridvc_ckpt_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "converter" in ckpt:
        state_dict = ckpt["converter"]
    else:
        state_dict = ckpt

    # Create HybridVC converter
    cfg = ConverterConfig(
        content_dim=content_dim,
        speaker_dim=speaker_dim,
        n_blocks=n_blocks,
        use_cross_attn=True,
    )
    converter = CausalConvNeXtConverter(cfg)

    # Map weights
    hv_state = converter.state_dict()
    transferred = 0
    skipped = 0
    new_init = 0

    for key in hv_state:
        if key in state_dict:
            # Direct transfer if shapes match
            if hv_state[key].shape == state_dict[key].shape:
                hv_state[key] = state_dict[key]
                transferred += 1
            else:
                print(f"  Shape mismatch: {key} {hv_state[key].shape} vs {state_dict[key].shape}")
                skipped += 1
        else:
            # New parameters (cross-attn, extra blocks) → keep init
            new_init += 1

    converter.load_state_dict(hv_state)

    print(f"  Transferred: {transferred}, Skipped: {skipped}, New-init: {new_init}")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({"converter": converter.state_dict(), "config": cfg}, output_path)
    print(f"  Saved: {output_path}")

    return converter


# ── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HybridVC Training")
    parser.add_argument("--phase", type=int, default=0,
                        help="Training phase: 0=transfer, 1=RAF-vocoder, 2=SSL-distill, 3=E2E+BWE")
    parser.add_argument("--btrv3lite-ckpt", type=str, default="",
                        help="Path to btrv3lite converter checkpoint (Phase 0)")
    parser.add_argument("--output-dir", type=str, default="./runs",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, cuda, mps")
    parser.add_argument("--n-blocks", type=int, default=10)
    parser.add_argument("--content-dim", type=int, default=768)
    parser.add_argument("--speaker-dim", type=int, default=128)
    args = parser.parse_args()

    if args.phase == 0:
        if not args.btrv3lite_ckpt:
            print("ERROR: --btrv3lite-ckpt required for Phase 0")
            sys.exit(1)

        output_path = os.path.join(args.output_dir, "converter_transferred.pt")
        _transfer_weights_btrv3lite(
            hybridvc_ckpt_path=args.btrv3lite_ckpt,
            output_path=output_path,
            n_blocks=args.n_blocks,
            content_dim=args.content_dim,
            speaker_dim=args.speaker_dim,
        )

    elif args.phase == 1:
        print("[Phase 1] RAF Vocoder Training — NOT YET IMPLEMENTED")
        print("  TODO: Load BigVGAN-base, integrate RAF loss, train")

    elif args.phase == 2:
        print("[Phase 2] SSL Distillation — NOT YET IMPLEMENTED")

    elif args.phase == 3:
        print("[Phase 3] E2E + BWE — NOT YET IMPLEMENTED")

    else:
        print(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    main()

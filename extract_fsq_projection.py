#!/usr/bin/env python3
"""Recover MioCodec's frozen FSQ-to-content projection from cached examples."""

import argparse
from pathlib import Path

import numpy as np
import torch

from astrape.fsq import fit_fsq_projection, indices_to_codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mio_4k"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/teacher_fsq_proj_out.pt"),
    )
    parser.add_argument("--samples", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = np.load(args.data_dir / "meta.npz")
    count = min(int(meta["n_samples"]), args.samples)
    if count <= 0:
        raise SystemExit("--samples must select at least one cached example")
    tokens = []
    embeddings = []
    for index in range(count):
        with np.load(args.data_dir / f"s_{index:05d}.npz") as data:
            tokens.append(torch.from_numpy(data["ct"]).long())
            embeddings.append(torch.from_numpy(data["ce_768"]).float())
    token_tensor = torch.cat(tokens)
    embedding_tensor = torch.cat(embeddings)
    projection = fit_fsq_projection(token_tensor, embedding_tensor)
    reconstructed = torch.nn.functional.linear(
        indices_to_codes(token_tensor),
        projection["weight"],
        projection["bias"],
    )
    max_error = (reconstructed - embedding_tensor).abs().max().item()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projection, args.output)
    print(
        f"Saved {args.output} from {len(token_tensor)} frames "
        f"| reconstruction max error={max_error:.3e}"
    )


if __name__ == "__main__":
    main()

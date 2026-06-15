#!/usr/bin/env python3
"""Bulk-migrate legacy ``.npz`` voice banks to the new ``.astrape`` layout.

Usage::

    .venv/bin/python migrate_voicebanks.py                  # copy every *.npz next to its .astrape sibling
    .venv/bin/python migrate_voicebanks.py --move           # delete the .npz after a successful copy
    .venv/bin/python migrate_voicebanks.py --voicebanks voicebanks/

The migration is **lossless by construction**: ``VoiceBank.load(npz)`` is
fed straight into ``VoiceBank.save(.astrape)``. Each conversion is
round-trip-verified by comparing the embedded tensor element-by-element and
the JSON metadata field-by-field before the script touches the next file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

from astrape.voicebank import (
    ASTRAPE_EMBEDDING_BYTES,
    MIO_GLOBAL_MODEL,
    VoiceBank,
    detect_format,
    parse_astrape_header,
)


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voicebanks", type=Path, default=Path("voicebanks"))
    parser.add_argument("--move", action="store_true",
                        help="delete legacy .npz files after a successful copy")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-existing", action="store_true",
                        help="if an .astrape sibling already exists, skip migration")
    return parser.parse_args()


def _iter_npz(voicebanks_dir: Path) -> Iterable[Path]:
    return sorted(voicebanks_dir.glob("*.npz"))


def _verify_round_trip(before: VoiceBank, after_path: Path) -> None:
    """Re-open the written .astrape and assert lossless equality."""
    if not after_path.exists():
        raise FileNotFoundError(f"expected new file at {after_path}")
    size = after_path.stat().st_size
    if size < ASTRAPE_EMBEDDING_BYTES + 48 + 16:
        raise ValueError(f"new file looks too small: {size} bytes")
    with after_path.open("rb") as handle:
        header = parse_astrape_header(handle.read(48))
    if header["embedding_length"] != ASTRAPE_EMBEDDING_BYTES:
        raise ValueError(
            f"unexpected embedding_length={header['embedding_length']}"
        )
    if detect_format(after_path) != "astrape":
        raise ValueError("written file is not recognised as .astrape")
    reloaded = VoiceBank.load(after_path)
    if not _tensor_equal(before.global_embedding, reloaded.global_embedding):
        raise AssertionError("embedding differs after round-trip")
    _compare_metadata(before, reloaded)


def _tensor_equal(left: "torch.Tensor", right: "torch.Tensor") -> bool:
    if left.shape != right.shape:
        return False
    import torch
    return torch.equal(left, right) and left.dtype == right.dtype


def _compare_metadata(before: VoiceBank, after: VoiceBank) -> None:
    fields = (
        "duration_seconds",
        "source_sample_rate",
        "source_path",
        "embedding_model",
        "reference_sha256",
        "created_utc",
        "quality_warnings",
    )
    for name in fields:
        if getattr(before, name) != getattr(after, name):
            raise AssertionError(
                f"metadata mismatch on field {name!r}: "
                f"{getattr(before, name)!r} vs {getattr(after, name)!r}"
            )
    for name in (
        "peak_amplitude",
        "rms_dbfs",
        "clipping_fraction",
        "active_speech_ratio",
        "dc_offset",
    ):
        a = getattr(before, name)
        b = getattr(after, name)
        if a != a and b != b:  # both NaN
            continue
        if abs(a - b) > 1e-6:
            raise AssertionError(
                f"metadata field {name!r} differs: {a} vs {b}"
            )


def migrate_all(voicebanks_dir: Path, *, move: bool,
                quiet: bool, dry_run: bool,
                keep_existing: bool) -> int:
    if not voicebanks_dir.is_dir():
        raise SystemExit(f"voicebanks directory does not exist: {voicebanks_dir}")
    migrated = 0
    skipped = 0
    failed = 0
    started = time.perf_counter()
    for legacy in _iter_npz(voicebanks_dir):
        target = legacy.with_suffix(".astrape")
        if keep_existing and target.exists() and target.stat().st_size > 64:
            skipped += 1
            if not quiet:
                print(f"  skip (target exists): {legacy.name}")
            continue
        if not quiet:
            print(f"  migrate: {legacy.name} -> {target.name}")
        try:
            bank = VoiceBank.load(legacy)
            if dry_run:
                if not quiet:
                    print("    [dry-run] no write performed")
                migrated += 1
                continue
            bank.save(target)
            _verify_round_trip(bank, target)
            if move:
                legacy.unlink()
                if not quiet:
                    print(f"    removed legacy {legacy.name}")
            migrated += 1
        except Exception as error:
            failed += 1
            if not quiet:
                print(f"  FAIL {legacy.name}: {error}", file=sys.stderr)
    elapsed = time.perf_counter() - started
    print(
        f"\nDone: migrated={migrated}, skipped={skipped}, failed={failed} "
        f"in {elapsed:.2f}s"
    )
    return 0 if failed == 0 else 1


def main() -> None:
    args = _arg_parser()
    raise SystemExit(
        migrate_all(
            args.voicebanks,
            move=args.move,
            quiet=args.quiet,
            dry_run=args.dry_run,
            keep_existing=args.keep_existing,
        )
    )


if __name__ == "__main__":
    main()

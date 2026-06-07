"""
Proper train/val split training for Codex VC.

Key improvements over train_full3.py:
- Utterance-level train/val split (no data leakage)
- Validation-based checkpoint saving (best_val_acc)
- Batch training support
- Speaker embedding ablation testing
- JSONL metric logging
- Random segment crop to prevent position memorization
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codex_vc.model import CodeGenerator, compute_loss


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Codex VC training with proper splits")
    p.add_argument("--cache", default="runs/vctk_codes_full.pt",
                   help="Path to cached Mimi codes")
    p.add_argument("--spk-emb", default="runs/vctk_full_spk.pt",
                   help="Path to Resemblyzer speaker embeddings")
    p.add_argument("--out", default="runs/codex_model.pt",
                   help="Output checkpoint path")
    p.add_argument("--log", default="runs/codex_train_metrics.jsonl",
                   help="JSONL metrics log path")
    p.add_argument("--steps", type=int, default=500,
                   help="Training steps")
    p.add_argument("--pairs-per-step", type=int, default=500,
                   help="Training pairs per step")
    p.add_argument("--val-pairs", type=int, default=200,
                   help="Validation pairs per eval")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size")
    p.add_argument("--val-interval", type=int, default=50,
                   help="Validate every N steps")
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="Fraction of utterances for validation")
    p.add_argument("--split-mode", choices=["utterance", "speaker"], default="utterance",
                   help="Split mode: utterance-based or speaker-based")
    p.add_argument("--segment-frames", type=int, default=0,
                   help="Random crop segment length (0 = full sequence)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Learning rate")
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto",
                   help="Device")
    p.add_argument("--ablation", action="store_true",
                   help="Run speaker embedding ablation at each validation")
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────

def load_data(cache_path, spk_path, val_ratio, split_mode, seed):
    """Load cached codes and speaker embeddings, create train/val splits."""
    random.seed(seed)

    spk_emb = torch.load(spk_path, weights_only=True)
    cache = torch.load(cache_path, weights_only=True)

    # Normalize time dimension
    T = min(c.shape[2] for c in cache.values())
    cache = {k: v[:, :, :T] for k, v in cache.items()}

    speakers = sorted({k[0] for k in cache} & set(spk_emb.keys()))
    all_utts = sorted({k[1] for k in cache})

    if split_mode == "utterance":
        # Split by utterance ID
        random.shuffle(all_utts)
        n_val = max(1, int(len(all_utts) * val_ratio))
        val_utts = set(all_utts[:n_val])
        train_utts = set(all_utts[n_val:])

        train_pairs = _build_pairs(cache, speakers, train_utts)
        val_pairs = _build_pairs(cache, speakers, val_utts)

        split_summary = {
            "mode": "utterance",
            "train_utts": len(train_utts),
            "val_utts": len(val_utts),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
        }

    elif split_mode == "speaker":
        # Split by speaker
        random.shuffle(speakers)
        n_val = max(1, int(len(speakers) * val_ratio))
        val_spks = set(speakers[:n_val])
        train_spks = set(speakers[n_val:])

        train_pairs = _build_pairs_filter_spk(cache, train_spks, all_utts)
        val_pairs = _build_pairs_filter_spk(cache, val_spks, all_utts)

        split_summary = {
            "mode": "speaker",
            "train_spks": len(train_spks),
            "val_spks": len(val_spks),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
        }

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    return spk_emb, cache, T, speakers, train_pairs, val_pairs, split_summary


def _build_pairs(cache, speakers, utt_set):
    """Build (src_spk, tgt_spk, utt) pairs for given utterance set."""
    pairs = []
    for u in utt_set:
        sw = [s for s in speakers if (s, u) in cache]
        if len(sw) < 2:
            continue
        for s in sw:
            for t in sw:
                if s != t:
                    pairs.append((s, t, u))
    return pairs


def _build_pairs_filter_spk(cache, spk_set, utts):
    """Build pairs where target speaker is in spk_set."""
    pairs = []
    for u in utts:
        sw = [s for s in spk_set if (s, u) in cache]
        if len(sw) < 2:
            continue
        for s in sw:
            for t in sw:
                if s != t:
                    pairs.append((s, t, u))
    return pairs


# ── Batch construction ───────────────────────────────────────────────────

def make_batch(pairs_batch, cache, spk_emb, segment_frames, device):
    """Create batched tensors from pair list."""
    lv0_b, lv1_b, spk_b = [], [], []

    for s, t, u in pairs_batch:
        lv0_full = cache[(s, u)][0, 0]       # (T,)
        lv1_full = cache[(t, u)][0, 1:]      # (7, T)
        spk = spk_emb[t]                      # (256,)

        if segment_frames > 0 and lv0_full.shape[0] > segment_frames:
            start = random.randint(0, lv0_full.shape[0] - segment_frames)
            lv0_b.append(lv0_full[start:start + segment_frames])
            lv1_b.append(lv1_full[:, start:start + segment_frames])
        else:
            lv0_b.append(lv0_full)
            lv1_b.append(lv1_full)

        spk_b.append(spk)

    # Stack — works if all same length (segment_frames) or padded
    lv0 = torch.stack(lv0_b).to(device).long()
    lv1 = torch.stack(lv1_b).to(device).long()
    spk = torch.stack(spk_b).to(device)

    return lv0, lv1, spk


# ── Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, pairs, cache, spk_emb, n_pairs, segment_frames, device, criterion):
    """Compute loss and accuracy on validation set."""
    if n_pairs > len(pairs):
        n_pairs = len(pairs)
    eval_pairs = random.sample(pairs, n_pairs)

    total_loss = 0.0
    total_acc = 0.0
    N = len(eval_pairs)

    for i in range(0, N, 16):  # batch size 16 for eval
        batch = eval_pairs[i:i + 16]
        lv0, lv1, spk = make_batch(batch, cache, spk_emb, segment_frames, device)
        loss = compute_loss(model, lv0, lv1, spk, criterion)
        total_loss += loss.item() * lv0.shape[0]
        total_acc += (model.predict(lv0, spk) == lv1).float().mean().item() * lv0.shape[0]

    return total_loss / N, total_acc / N


@torch.no_grad()
def ablation_eval(model, val_pairs, cache, spk_emb, n_pairs, segment_frames, device, criterion):
    """Run embedding ablation: normal, shuffled, zero, source."""
    if n_pairs > len(val_pairs):
        n_pairs = len(val_pairs)
    pairs = random.sample(val_pairs, n_pairs)

    results = {}
    for mode in ["target", "shuffled", "zero", "source"]:
        total_acc = 0.0
        for i in range(0, len(pairs), 16):
            batch = pairs[i:i + 16]
            if mode == "target":
                # Normal: use target speaker embedding
                lv0, lv1, spk = make_batch(batch, cache, spk_emb, segment_frames, device)
            elif mode == "shuffled":
                # Shuffle speaker embeddings within batch
                lv0, lv1, spk = make_batch(batch, cache, spk_emb, segment_frames, device)
                idx = torch.randperm(spk.shape[0])
                spk = spk[idx]
            elif mode == "zero":
                lv0, lv1, _ = make_batch(batch, cache, spk_emb, segment_frames, device)
                spk = torch.zeros_like(lv0[:, :1].float().repeat(1, spk_emb[list(spk_emb.keys())[0]].shape[0]))
                spk = spk.to(device)
            elif mode == "source":
                # Use source speaker embedding
                lv0_b, lv1_b, spk_b = [], [], []
                for s, t, u in batch:
                    lv0_b.append(cache[(s, u)][0, 0] if segment_frames == 0
                                 else cache[(s, u)][0, 0][:segment_frames])
                    lv1_b.append(cache[(t, u)][0, 1:] if segment_frames == 0
                                 else cache[(t, u)][0, 1:, :segment_frames])
                    spk_b.append(spk_emb[s])  # source speaker, not target!
                lv0 = torch.stack(lv0_b).to(device).long()
                lv1 = torch.stack(lv1_b).to(device).long()
                spk = torch.stack(spk_b).to(device)

            total_acc += (model.predict(lv0, spk) == lv1).float().mean().item() * lv0.shape[0]

        results[f"abl_{mode}_acc"] = total_acc / len(pairs)

    return results


# ── Training loop ────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    print("Loading data...", flush=True)
    spk_emb, cache, T, speakers, train_pairs, val_pairs, split_summary = load_data(
        args.cache, args.spk_emb, args.val_ratio, args.split_mode, args.seed
    )
    print(f"  {len(spk_emb)} speakers, {len(cache)} codes, T={T}", flush=True)
    print(f"  Split: {split_summary}", flush=True)

    # Model
    model = CodeGenerator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Logging
    log_file = open(args.log, "a") if args.log else None

    print()
    print(f"Training {args.steps} steps...", flush=True)
    t0 = time.time()
    best_val_acc = 0.0
    train_losses = []

    for step in range(args.steps):
        # Train
        random.shuffle(train_pairs)
        step_pairs = train_pairs[:min(len(train_pairs), args.pairs_per_step)]
        total_loss = 0.0
        total_acc = 0.0

        for b_start in range(0, len(step_pairs), args.batch_size):
            batch = step_pairs[b_start:b_start + args.batch_size]
            lv0, lv1, spk = make_batch(batch, cache, spk_emb, args.segment_frames, device)
            loss = compute_loss(model, lv0, lv1, spk, criterion)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * lv0.shape[0]
            with torch.no_grad():
                total_acc += (model.predict(lv0, spk) == lv1).float().mean().item() * lv0.shape[0]

        N = len(step_pairs)
        train_loss = total_loss / N
        train_acc = total_acc / N
        train_losses.append(train_loss)

        # Validate periodically
        metrics = {"step": step, "train_loss": train_loss, "train_acc": train_acc,
                   "elapsed": time.time() - t0}

        if step % args.val_interval == 0 or step == args.steps - 1:
            val_loss, val_acc = evaluate(
                model, val_pairs, cache, spk_emb, args.val_pairs,
                args.segment_frames, device, criterion
            )
            metrics.update({"val_loss": val_loss, "val_acc": val_acc})

            # Ablation
            if args.ablation:
                abl = ablation_eval(
                    model, val_pairs, cache, spk_emb, args.val_pairs,
                    args.segment_frames, device, criterion
                )
                metrics.update(abl)

            # Best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step,
                    "best_val_acc": best_val_acc,
                    "config": vars(args),
                    "split": split_summary,
                }
                torch.save(ckpt, args.out)

            print(f"  step {step:4d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"best_val={best_val_acc:.4f} [{time.time()-t0:.0f}s]", flush=True)
        else:
            if step % 50 == 0:
                print(f"  step {step:4d}: train_loss={train_loss:.4f} "
                      f"train_acc={train_acc:.4f} [{time.time()-t0:.0f}s]", flush=True)

        # Log
        if log_file:
            log_file.write(json.dumps(metrics))
            log_file.write(chr(10))  # newline
            log_file.flush()

    print()
    print(f"Done! Best val_acc={best_val_acc:.4f} [{time.time()-t0:.0f}s]", flush=True)
    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()

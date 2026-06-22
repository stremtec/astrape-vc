"""Astrape Raw — Waveform Frontend MCS-Trans.

Replaces mel-spectrogram frontend with direct CausalConv1d on raw PCM.
Same Q2D2 quantizer + transformer backbone as train_mcs_q2d2.py.

Key change:
  mel (80-bin, 50Hz) → Conv1d(80→320, k=5)
  ↓ replaced by
  raw PCM → CausalConv1d(1→320, k=2048, stride=441) @100Hz → stride-2 → 50Hz

Latency reduction: 46ms (mel STFT buffer) → 0ms (raw conv, 10ms stride)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
sys.path.insert(0, "external/MioCodec/src")

from mcs_common import (
    Batch, MioCompactDataset, ContentCollator,
    split_by_speaker, speaker_balanced_subset,
    move_batch, save_checkpoint,
    CausalConv1d, ResidualConvBlock, CellDownsample,
    DEFAULT_DATA_DIR, DEFAULT_PROJECTION,
    _voiced_weights, multi_resolution_stft_loss,
)
from mcs_q2d2 import Q2D2Projection, Q2D2Quantizer, compute_q2d2_perplexity

DEFAULT_OUT_DIR = Path("astrape_raw/checkpoints")
DEFAULT_Q2D2_LEVELS = (7, 7, 7, 7, 7, 7)

# Import RoPE + GRL helpers from the Q2D2 training script
from train_mcs_q2d2 import (
    MCSTransQ2D2Config,
    GradientReversal, grad_reverse, SpeakerClassifier,
    _precompute_rope_freqs, _apply_rope, _rotate_half,
    _causal_window_mask, TransformerBlock,
)


# ─────────────────────────────────────────────
# Raw Waveform Frontend
# ─────────────────────────────────────────────

class RawWaveformFrontend(nn.Module):
    """Replace mel frontend with causal conv on raw PCM.

    PCM (1ch, T_audio) → CausalConv1d(k=2048, stride=441) → (320, T_100Hz)
    → ResidualBlocks → skip connections → stride-2 downsample → (320, T_50Hz)
    → Linear(320→512) → transformer

    This matches the existing transformer input rate (50Hz) and dimension (512).
    """

    def __init__(self, config: MCSTransQ2D2Config):
        super().__init__()
        self.config = config
        dim = config.conv_dim  # 320

        # Raw audio conv: 2048 samples ≈ 46ms context at 44.1kHz
        # stride=441 → 100Hz frame rate (44100/441=100)
        self.raw_conv = CausalConv1d(1, dim, kernel_size=2048, stride=441)

        # Residual blocks (same as original)
        self.blocks = nn.ModuleList([
            ResidualConvBlock(dim, config.conv_kernel, d, config.dropout)
            for d in config.stem_dilations
        ])

        # Skip connections from raw audio (need to match rate)
        self.skips = nn.ModuleList([
            CausalConv1d(1, dim, kernel_size=2048, stride=441, dilation=d)
            for d in config.skip_dilations
        ])
        self.skip_gates = nn.ParameterList([
            nn.Parameter(torch.full((1, dim, 1), -2.0))
            for _ in config.skip_dilations
        ])

        # Stride-2: 100Hz → 50Hz
        self.downsample = CellDownsample(dim)

        # Project to transformer dim
        self.proj_in = (
            nn.Linear(dim, config.trans_dim, bias=False)
            if dim != config.trans_dim else nn.Identity()
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, 1, T_audio)
        h = F.silu(self.raw_conv(waveform))  # (B, 320, T_100Hz)

        for block in self.blocks:
            h = block(h)

        for skip, gate in zip(self.skips, self.skip_gates):
            h = h + torch.sigmoid(gate) * F.silu(skip(waveform))

        h = self.downsample(h).transpose(1, 2)  # (B, T_50Hz, 320)
        h = self.proj_in(h)  # (B, T_50Hz, trans_dim)
        return h


# ─────────────────────────────────────────────
# MCS-Trans with Raw Waveform Input
# ─────────────────────────────────────────────

class MCSTransRaw(nn.Module):
    """Same as MCSTransQ2D2 but with RawWaveformFrontend instead of mel conv."""

    def __init__(self, config: MCSTransQ2D2Config):
        super().__init__()
        self.config = config

        # Raw waveform frontend (replaces mel conv)
        self.frontend = RawWaveformFrontend(config)

        # Transformer (same as before)
        self.trans_layers = nn.ModuleList([
            TransformerBlock(config.trans_dim, config.n_heads,
                             config.ffn_dim, config.dropout,
                             use_rope=config.use_rope,
                             use_swiglu=config.use_swiglu)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.trans_dim)
        self.smooth = CausalConv1d(
            config.trans_dim, config.trans_dim, kernel_size=3,
            groups=config.trans_dim,
        )

        # Q2D2 quantizer
        self.q2d2 = Q2D2Projection(
            encoder_dim=config.trans_dim,
            q2d2_dim=config.q2d2_dim,
            content_dim=config.content_dim,
            levels=list(config.q2d2_levels),
            vq_type=config.q2d2_grid,
        )

        # GRL speaker classifier
        self.speaker_classifier: SpeakerClassifier | None = None
        if config.grl_weight > 0 and config.grl_num_speakers > 0:
            self.speaker_classifier = SpeakerClassifier(
                dim=config.content_dim,
                num_speakers=config.grl_num_speakers,
            )

    def forward(self, waveform: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> dict:
        # Frontend
        h = self.frontend(waveform)  # (B, T50, trans_dim)

        # Causal transformer
        T = h.shape[1]
        attn_mask = _causal_window_mask(T, self.config.window, h.device)
        kpm = (~padding_mask[:, :T]).float() * -1e4 if padding_mask is not None else None
        for layer in self.trans_layers:
            h = layer(h, attn_mask, kpm)
        h = self.norm(h)
        h = h + self.smooth(h.transpose(1, 2)).transpose(1, 2)

        # Q2D2
        content, q2d2_codes = self.q2d2(h, return_codes=True)

        return {
            "projected": content.transpose(1, 2),
            "q2d2_codes": q2d2_codes,
            "ordinal": None,
        }


# ─────────────────────────────────────────────
# Data pipeline: load waveform instead of mel
# ─────────────────────────────────────────────

class WaveformDataset(Dataset):
    """Load raw waveform + teacher content from compact cache."""

    def __init__(self, root: Path, indices: np.ndarray, speakers: np.ndarray,
                 source_files: np.ndarray, max_seconds: float = 3.0):
        self.root = root
        self.indices = [int(i) for i in indices.tolist()]
        self.speakers = speakers
        self.source_files = source_files
        self.max_samples = int(max_seconds * 44100)
        self._rng = random.Random(42)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        import soundfile as sf
        idx = self.indices[item]
        src = Path(str(self.source_files[idx]))
        wav, sr = sf.read(str(src), dtype="float32")
        wav = torch.from_numpy(np.asarray(wav))
        if wav.ndim == 2:
            wav = wav.mean(dim=1)
        if sr != 44100:
            import torchaudio
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 44100).squeeze(0)

        # Crop or pad
        if wav.shape[0] > self.max_samples:
            start = self._rng.randint(0, wav.shape[0] - self.max_samples)
            wav = wav[start:start + self.max_samples]
        elif wav.shape[0] < self.max_samples:
            wav = F.pad(wav, (0, self.max_samples - wav.shape[0]))

        # Load teacher content from cache
        npz = np.load(self.root / f"s_{idx:05d}.npz", allow_pickle=False)
        content = torch.from_numpy(npz["ce_768"].astype(np.float32))

        return {
            "waveform": wav,
            "content": content,
            "speaker": str(self.speakers[idx]),
            "idx": idx,
            "crop_start": 0,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--probe-samples", type=int, default=256)
    p.add_argument("--max-seconds", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--run-name", default="astrape_raw")

    # Transformer
    p.add_argument("--conv-dim", type=int, default=320)
    p.add_argument("--trans-dim", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--rope", action="store_true")
    p.add_argument("--swiglu", action="store_true")

    # Q2D2
    p.add_argument("--q2d2-dim", type=int, default=6)
    p.add_argument("--q2d2-levels", type=str, default="7,7,7,7,7,7")
    p.add_argument("--q2d2-grid", default="rhombic")

    # Loss
    p.add_argument("--content-cos-weight", type=float, default=1.0)
    p.add_argument("--content-l1-weight", type=float, default=0.5)
    p.add_argument("--delta-weight", type=float, default=0.04)
    p.add_argument("--grl-weight", type=float, default=0.0)
    p.add_argument("--voiced-boost", type=float, default=1.0)

    return p.parse_args()


if __name__ == "__main__":
    print("Astrape Raw — Waveform Frontend")
    print("TODO: implement full training loop (see train_mcs_q2d2.py for reference)")
    print("Ready to train!")

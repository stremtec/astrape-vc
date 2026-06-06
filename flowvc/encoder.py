"""
F³-Encoder for FlowVC.

Fully causal ConvNeXt v2 encoder. KL-free (no VQ, no commitment loss).
Noise regularization: z_reg = z + σ·ε during training only.

Architecture:
  waveform (44.1kHz) → 6-stage ConvNeXt v2 (strides: 2,2,3,3,7,7)
  → z_raw (768-dim @ 25Hz)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CausalConv1d, ConvNeXtV2Block
from .config import EncoderConfig


class F3Encoder(nn.Module):
    """
    Causal Continuous Encoder.
    
    Total downsample: 2×2×3×3×7×7 = 1764 → 44100/1764 = 25Hz.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.noise_sigma = cfg.noise_sigma

        in_ch = 1
        stages = []
        for i, (out_ch, stride) in enumerate(zip(cfg.stages, cfg.strides)):
            # Strided conv
            stages.append(
                CausalConv1d(in_ch, out_ch, kernel_size=stride * 3, stride=stride)
            )
            # ConvNeXt v2 blocks
            for _ in range(cfg.blocks_per_stage):
                stages.append(
                    ConvNeXtV2Block(
                        out_ch,
                        kernel_size=cfg.kernel_size,
                        mlp_expansion=cfg.mlp_expansion,
                        use_grn=cfg.use_grn,
                    )
                )
            in_ch = out_ch

        self.stages = nn.Sequential(*stages)

        # Final normalization
        self.norm = nn.LayerNorm(cfg.stages[-1])

    def forward(self, wav: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Args:
            wav: (B, 1, T_audio) waveform @ 44.1kHz
            training: if True, add noise regularization
        Returns:
            z: (B, T_lat, content_dim) @ 25Hz
        """
        # Channel-first for conv pipeline
        x = self.stages(wav)  # (B, C_out, T_lat)

        # → (B, T_lat, C_out)
        z = x.transpose(1, 2)
        z = self.norm(z)

        # Noise regularization (F³-Tokenizer style) — training only
        if training and self.noise_sigma > 0:
            z = z + torch.randn_like(z) * self.noise_sigma

        return z

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Inference-mode encode (no noise)."""
        return self.forward(wav, training=False)


def make_encoder(**kwargs) -> F3Encoder:
    cfg = EncoderConfig(**kwargs)
    return F3Encoder(cfg)

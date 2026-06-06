"""
F³-Decoder for FlowVC.

Causal ConvNeXt v2 decoder with MRF (Multi-Receptive Field) upsampler.
Mirrors F³-Encoder: reverses strides with TransposedConv.
FiLM conditioning from speaker embedding at each stage.

Output: 44.1kHz waveform.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    CausalConv1d, CausalConvTranspose1d, ConvNeXtV2Block, FiLM
)
from .config import DecoderConfig


class MRFBlock(nn.Module):
    """
    Multi-Receptive Field block (HiFi-GAN style).
    Parallel Conv1d paths with different kernel sizes and dilations.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: tuple[int, ...] = (3, 7, 11),
        dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.paths = nn.ModuleList()
        for ks, dils in zip(kernel_sizes, dilations):
            path = nn.ModuleList()
            for d in dils:
                path.append(
                    nn.Sequential(
                        CausalConv1d(dim, dim, ks, dilation=d, groups=dim),
                        nn.LeakyReLU(0.1),
                    )
                )
            self.paths.append(path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residuals = []
        for path in self.paths:
            h = x
            for layer in path:
                h = layer(h)
            residuals.append(h)

        h = sum(residuals) / len(residuals)
        return x + h  # residual


class DecoderStage(nn.Module):
    """One upsampling stage: TransposedConv → MRF ×2 → FiLM."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        kernel_size: int = 7,
        mrf_config: dict | None = None,
    ):
        super().__init__()
        stride_kernel = stride * 2 + 1  # ensure good coverage
        self.upsample = CausalConvTranspose1d(
            in_ch, out_ch, kernel_size=stride_kernel, stride=stride
        )
        self.mrf1 = MRFBlock(out_ch, **(mrf_config or {}))
        self.mrf2 = MRFBlock(out_ch, **(mrf_config or {}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.mrf1(x)
        x = self.mrf2(x)
        return x


class F3Decoder(nn.Module):
    """
    Causal Decoder: latent → waveform.
    
    Architecture:
      z (T_lat, 768) → FiLM → ConvNeXt blocks
      → 6-stage upsampling (strides: 7,7,3,3,2,2 = ×1764)
      → Conv1d → tanh → waveform
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg

        latent_dim = cfg.latent_dim
        speaker_dim = 192  # matches speaker encoder output

        # Input FiLM
        self.film_in = FiLM(latent_dim, speaker_dim)

        # Pre-upsampling ConvNeXt blocks
        self.pre_blocks = nn.ModuleList([
            ConvNeXtV2Block(
                latent_dim,
                kernel_size=cfg.kernel_size,
                mlp_expansion=4,
                use_grn=cfg.use_grn,
            )
            for _ in range(cfg.pre_upsample_blocks)
        ])

        # Upsampling stages (reverse of encoder)
        in_ch = latent_dim
        mrf_cfg = {
            "kernel_sizes": cfg.mrf_kernel_sizes,
            "dilations": cfg.mrf_dilations,
        }
        stages = []
        for out_ch, stride in zip(cfg.stages, cfg.strides):
            stages.append(DecoderStage(in_ch, out_ch, stride, mrf_config=mrf_cfg))
            # FiLM per stage
            stages.append(FiLM(out_ch, speaker_dim))
            in_ch = out_ch

        self.upsample_stages = nn.ModuleList(stages)

        # Final projection
        self.final_conv = CausalConv1d(cfg.stages[-1], 1, kernel_size=7)

    def forward(self, z: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T_lat, latent_dim) latent @ 25Hz
            speaker_emb: (B, speaker_dim) target speaker
        Returns:
            wav: (B, 1, T_audio) waveform @ 44.1kHz
        """
        # Input FiLM
        x = z.transpose(1, 2)  # (B, dim, T)
        x = self.film_in(x, speaker_emb)

        # Pre-upsampling refinement
        for block in self.pre_blocks:
            x = block(x)

        # Upsampling stages
        for i, stage in enumerate(self.upsample_stages):
            if isinstance(stage, FiLM):
                x = stage(x, speaker_emb)
            else:
                x = stage(x)

        # Final
        x = self.final_conv(x)
        x = torch.tanh(x)

        return x


def make_decoder(**kwargs) -> F3Decoder:
    cfg = DecoderConfig(**kwargs)
    return F3Decoder(cfg)

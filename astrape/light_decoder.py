"""Lightweight Causal Decoder — transformer-free, FiLM-conditioned upsampling.

~8M params. CausalConv1d stack with speaker-conditioned FiLM modulation.
Content(768d, 25Hz) → upsample(882x) → 44.1kHz waveform.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from mcs_common import CausalConv1d


@dataclass
class LightDecoderConfig:
    content_dim: int = 768
    condition_dim: int = 128
    sample_rate: int = 44100
    content_rate: int = 25  # Hz

    # Upsampling stages: channel dimensions at each stage
    stage_channels: tuple = (384, 256, 192, 128, 96, 64)
    upsample_factors: tuple = (7, 7, 3, 3, 4)  # 7*7*3*3*4 = 1764x (25Hz → 44.1kHz)
    # Or equivalently: 1764/25 → need 882x total upsample from 25Hz to 44100Hz
    # 25Hz → upsample → 44100Hz = 1764x. Wait, 1764/25 = 70.56? No.
    # Actually: 1 content frame = 40ms = 1764 samples
    # So we need 1764x upsample from 25Hz frame rate to 44.1kHz
    # 1764 = 7*7*3*3*4 = 1764. Let me fix factors.
    # With factors (7,7,3,3,2) = 882. Need 1764.
    # Use (7,7,3,3,4) = 1764 or (7,7,7,3,2) ≈ 2058

    kernel_size: int = 7
    dropout: float = 0.1

    # FiLM conditioning
    film_hidden: int = 256


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: speaker embedding → scale + shift."""
    def __init__(self, cond_dim: int, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, feat_dim * 2),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), condition: (B, cond_dim)
        scale_shift = self.net(condition)  # (B, C*2)
        scale, shift = scale_shift.chunk(2, dim=1)  # (B, C) each
        return x * (1.0 + scale.unsqueeze(2)) + shift.unsqueeze(2)


class SnakeBeta(nn.Module):
    """Snake activation with trainable alpha/beta (from BigVGAN)."""
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, dim, 1))
        self.beta = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.beta + 1e-6)) * torch.sin(self.alpha * x).pow(2)


class UpsampleBlock(nn.Module):
    """Causal transposed conv + FiLM + Snake."""
    def __init__(self, in_c: int, out_c: int, factor: int, cond_dim: int,
                 kernel: int = 7, film_hidden: int = 256):
        super().__init__()
        # MPS workaround: interpolate + conv instead of ConvTranspose1d
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=factor, mode='linear'),
            CausalConv1d(in_c, out_c, kernel_size=kernel),
        )
        self.conv = CausalConv1d(out_c, out_c, kernel_size=kernel)
        self.film = FiLM(cond_dim, out_c, film_hidden)
        self.snake = SnakeBeta(out_c)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Trim causal padding from upsample
        x = self.conv(x)
        x = self.snake(x)
        x = self.film(x, cond)
        return x


class LightDecoder(nn.Module):
    """Transformer-free causal waveform decoder.

    content(768d,25Hz) → pre_conv → upsample_blocks → final_conv → waveform
    speaker_emb(128d) ─────────────────→ FiLM at each block
    """

    def __init__(self, config: LightDecoderConfig):
        super().__init__()
        self.config = config
        dim = config.stage_channels[0]

        # Input projection: content → first stage dim
        self.pre_conv = nn.Sequential(
            CausalConv1d(config.content_dim, dim, kernel_size=3),
            SnakeBeta(dim),
            FiLM(config.condition_dim, dim, config.film_hidden),
        )

        # Upsample blocks
        channels = [dim] + list(config.stage_channels[1:])
        self.upsample_blocks = nn.ModuleList()
        for i, factor in enumerate(config.upsample_factors):
            self.upsample_blocks.append(
                UpsampleBlock(
                    channels[i], channels[i + 1], factor,
                    config.condition_dim, config.kernel_size, config.film_hidden,
                )
            )

        # Final output: 1 channel audio
        self.final_conv = nn.Sequential(
            CausalConv1d(config.stage_channels[-1], 1, kernel_size=config.kernel_size),
            nn.Tanh(),
        )

    def forward(self, content: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        # content: (B, T, 768) → (B, 768, T)
        x = content.transpose(1, 2)
        x = F.silu(self.pre_conv[0](x))
        x = self.pre_conv[1](x)
        x = self.pre_conv[2](x, speaker_emb)
        for block in self.upsample_blocks:
            x = block(x, speaker_emb)
        x = self.final_conv(x)
        return x.squeeze(1)  # (B, T_audio)


# ── Test ──
if __name__ == "__main__":
    cfg = LightDecoderConfig()
    cfg.upsample_factors = (7, 7, 3, 3, 4)  # 1764x total
    cfg.stage_channels = (256, 192, 128, 96, 64, 48)
    m = LightDecoder(cfg)
    c = torch.randn(2, 50, 768)  # 2s at 25Hz
    s = torch.randn(2, 128)
    out = m(c, s)
    params = sum(p.numel() for p in m.parameters())
    print(f"Content: {c.shape}, Speaker: {s.shape}")
    print(f"Output: {out.shape} (expected: 2, {50*1764})")
    print(f"Params: {params:,}")

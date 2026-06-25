"""V2: Dual-path LightDecoder — discrete (Q2D2) + continuous (projected) skip.

content_768 (Q2D2 quantized) ──→ FiLM_pre ─┐
projected_768 (continuous) ────→ FiLM_pre ─┤→ merge → upsample → wave
speaker_128 ──────────────────────────────→ FiLM at every stage

~3.5M params. Transformer-free. Phase 0: both paths = teacher ce_768.
Phase 2: discrete=Q2D2, continuous=projected (skip-connection).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from mcs_common import CausalConv1d


@dataclass
class DualDecoderConfig:
    content_dim: int = 768       # discrete (Q2D2 output)
    projected_dim: int = 768     # continuous (pre-Q2D2)
    condition_dim: int = 128
    sample_rate: int = 44100
    content_rate: int = 25

    stage_channels: tuple = (256, 192, 128, 96, 64)  # 5 stages → final output 64ch
    upsample_factors: tuple = (7, 7, 3, 3, 4)  # 1764x

    kernel_size: int = 7
    dropout: float = 0.1
    film_hidden: int = 256
    merge_dim: int = 512  # after merging discrete+continuous


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, feat_dim * 2),
        )
    def forward(self, x, condition):
        ss = self.net(condition)
        scale, shift = ss.chunk(2, dim=1)
        return x * (1.0 + scale.unsqueeze(2)) + shift.unsqueeze(2)


class SnakeBeta(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, dim, 1))
        self.beta = nn.Parameter(torch.ones(1, dim, 1))
    def forward(self, x):
        return x + (1.0 / (self.beta + 1e-6)) * torch.sin(self.alpha * x).pow(2)


class DualPathInput(nn.Module):
    """Merge discrete + continuous paths with speaker conditioning."""
    def __init__(self, disc_dim, cont_dim, merge_dim, cond_dim, film_hidden=256):
        super().__init__()
        self.disc_proj = nn.Sequential(
            CausalConv1d(disc_dim, merge_dim // 2, kernel_size=3),
            SnakeBeta(merge_dim // 2),
        )
        self.cont_proj = nn.Sequential(
            CausalConv1d(cont_dim, merge_dim // 2, kernel_size=3),
            SnakeBeta(merge_dim // 2),
        )
        self.film = FiLM(cond_dim, merge_dim, film_hidden)

    def forward(self, disc, cont, spk):
        # disc: (B, T, D), cont: (B, T, D), spk: (B, C)
        d = self.disc_proj(disc.transpose(1, 2))
        c = self.cont_proj(cont.transpose(1, 2))
        x = torch.cat([d, c], dim=1)  # (B, merge_dim, T)
        x = self.film(x, spk)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, out_c, factor, cond_dim, kernel=7, film_hidden=256):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=factor, mode='linear'),
            CausalConv1d(in_c, out_c, kernel_size=kernel),
        )
        self.snake = SnakeBeta(out_c)
        self.film = FiLM(cond_dim, out_c, film_hidden)

    def forward(self, x, spk):
        x = self.upsample(x)
        x = self.snake(x)
        x = self.film(x, spk)
        return x


class DualLightDecoder(nn.Module):
    """Dual-path causal waveform decoder.

    Input:  discrete(768d, 25Hz) + continuous(768d, 25Hz)
    Output: waveform (44.1kHz)
    """

    def __init__(self, config: DualDecoderConfig):
        super().__init__()
        self.cfg = config
        self.input_merge = DualPathInput(
            config.content_dim, config.projected_dim,
            config.merge_dim, config.condition_dim, config.film_hidden,
        )
        channels = [config.merge_dim] + list(config.stage_channels)
        self.upsample_blocks = nn.ModuleList()
        for i, factor in enumerate(config.upsample_factors):
            self.upsample_blocks.append(
                UpsampleBlock(channels[i], channels[i+1], factor,
                              config.condition_dim, config.kernel_size, config.film_hidden)
            )
        self.final_conv = nn.Sequential(
            CausalConv1d(config.stage_channels[-1], 1, kernel_size=config.kernel_size),
            nn.Tanh(),
        )

    def forward(self, discrete, continuous, speaker_emb):
        x = self.input_merge(discrete, continuous, speaker_emb)
        for blk in self.upsample_blocks:
            x = blk(x, speaker_emb)
        x = self.final_conv(x)
        return x.squeeze(1)


# ── Test ──
if __name__ == "__main__":
    cfg = DualDecoderConfig()
    m = DualLightDecoder(cfg)
    disc = torch.randn(2, 50, 768)
    cont = torch.randn(2, 50, 768)
    spk = torch.randn(2, 128)
    out = m(disc, cont, spk)
    p = sum(p.numel() for p in m.parameters())
    print(f"Input: disc={disc.shape}, cont={cont.shape}, spk={spk.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {p:,}")

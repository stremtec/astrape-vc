"""MCS-Decoder v3 — all-conv causal decoder, MioCodec 7-stage flow.

content 768d @25Hz + speaker 128d →
  ① Content Smoothing   ConvNeXt-v2(causal) ×4                         @25Hz
  ② AA Rate ×2          BigVGAN-v2 anti-aliased upsample, 768→512      @50Hz
  ③ Prior Net           ConvNeXt-v2(causal) ×2                         @50Hz
  ④ Speaker Decoder ⭐  dilated causal TCN ×8 + AdaLN-Zero(speaker)    @50Hz
  ⑤ Post Net            ConvNeXt-v2(causal) ×2                         @50Hz
  ⑥ AA Upsampler ×9     BigVGAN-v2 anti-aliased [×3,×3]                @450Hz
  ⑦ ISTFT Head          Linear→mag/phase→iSTFT(392/98)                 → 44.1kHz

All modules are STRICTLY CAUSAL (0 look-ahead, ring-buffer streamable). Only latency is
the iSTFT overlap (3.3ms). Anti-aliasing (windowed-sinc low-pass around every upsample &
SnakeBeta) is the fix for the periodic upsampler "지지직" artifact.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import CausalConv1d

_mio = Path(__file__).resolve().parent.parent / "external" / "MioCodec" / "src"
if str(_mio) not in sys.path:
    sys.path.insert(0, str(_mio))
from miocodec.module.istft_head import ISTFTHead, SnakeBeta   # noqa: E402
from miocodec.module.adaln_zero import AdaLNZero              # noqa: E402


# ── anti-aliasing: fixed windowed-sinc low-pass, applied CAUSALLY (depthwise) ──
def _lowpass_kernel(cutoff: float, ksize: int) -> torch.Tensor:
    n = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    h = torch.where(n == 0, torch.tensor(2 * cutoff), torch.sin(2 * math.pi * cutoff * n) / (math.pi * n))
    h = h * torch.hamming_window(ksize, periodic=False)
    return h / h.sum()


class CausalLowPass(nn.Module):
    """Fixed linear-phase sinc low-pass, depthwise, causal (left-pad → 0 look-ahead)."""
    def __init__(self, channels: int, cutoff: float, ksize: int):
        super().__init__()
        if ksize % 2 == 0:
            ksize += 1
        k = _lowpass_kernel(cutoff, ksize).view(1, 1, -1).repeat(channels, 1, 1)
        self.register_buffer("k", k, persistent=False)
        self.ksize, self.ch = ksize, channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B, C, L)
        return F.conv1d(F.pad(x, (self.ksize - 1, 0)), self.k, groups=self.ch)


class AAUpsample(nn.Module):
    """Anti-aliased ×factor upsample: nearest-repeat (causal) → low-pass (kills the
    staircase images = the artifact). cutoff at the OLD Nyquist."""
    def __init__(self, channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.lp = CausalLowPass(channels, cutoff=0.5 / factor, ksize=2 * factor * 4 + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lp(F.interpolate(x, scale_factor=self.factor, mode="nearest"))


# ── causal Global Response Normalization (ConvNeXt-v2), cumulative over time ──
class CausalGRN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B, C, L)
        cnt = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)
        Gx = torch.sqrt(torch.cumsum(x.pow(2), dim=-1) / cnt + 1e-6)   # causal global response/channel
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)               # cross-channel normalize
        return self.gamma * (x * Nx) + self.beta + x


class _ChannelLN(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.ln = nn.LayerNorm(dim)
    def forward(self, x):                                          # (B, C, L)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class CausalConvNeXtBlock(nn.Module):
    """ConvNeXt-v2: depthwise CausalConv → LN → PW expand → GELU → GRN → PW contract → +res."""
    def __init__(self, dim: int, kernel: int = 7, expand: int = 4):
        super().__init__()
        self.dw = CausalConv1d(dim, dim, kernel, groups=dim)
        self.norm = _ChannelLN(dim)
        self.pw1 = nn.Conv1d(dim, expand * dim, 1)
        self.grn = CausalGRN(expand * dim)
        self.pw2 = nn.Conv1d(expand * dim, dim, 1)
        self.scale = nn.Parameter(1e-6 * torch.ones(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B, C, L)
        h = self.dw(x); h = self.norm(h)
        h = F.gelu(self.pw1(h)); h = self.grn(h); h = self.pw2(h)
        return x + self.scale * h


class AAUpStage(nn.Module):
    """BigVGAN-v2-style anti-aliased upsample stage: AA-upsample → CausalConv → SnakeBeta →
    low-pass (anti-alias the Snake's harmonics)."""
    def __init__(self, c_in: int, c_out: int, factor: int, conv_k: int = 15):
        super().__init__()
        self.up = AAUpsample(c_in, factor)
        self.conv = CausalConv1d(c_in, c_out, conv_k)
        self.snake = SnakeBeta(c_out, alpha_logscale=True)
        self.lp = CausalLowPass(c_out, cutoff=0.45, ksize=2 * 4 + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lp(self.snake(self.conv(self.up(x))))


class DilatedSpeakerBlock(nn.Module):
    """Dilated causal TCN block with per-block AdaLN-Zero(speaker) + GRN (stage ④)."""
    def __init__(self, dim: int, cond_dim: int, kernel: int = 5, dilation: int = 1):
        super().__init__()
        self.adaln = AdaLNZero(dim, cond_dim, return_gate=True)
        self.dw = CausalConv1d(dim, dim, kernel, dilation=dilation, groups=dim)
        self.snake = SnakeBeta(dim, alpha_logscale=True)
        self.grn = CausalGRN(dim)
        self.pw = nn.Conv1d(dim, dim, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:   # x (B,C,L), cond (B,1,cond)
        normed, gate = self.adaln(x.transpose(1, 2), condition=cond)          # (B,L,C),(B,L,1)
        h = normed.transpose(1, 2)
        h = self.pw(self.grn(self.snake(self.dw(h))))
        return x + gate.transpose(1, 2) * h


@dataclass
class MCSDecoderConfig:
    content_dim: int = 768
    speaker_dim: int = 128
    smooth_dim: int = 768
    wave_dim: int = 512
    smooth_blocks: int = 4
    prior_blocks: int = 2
    post_blocks: int = 2
    speaker_dilations: tuple[int, ...] = (1, 2, 4, 8, 12, 16, 24, 32)
    speaker_kernel: int = 5
    upsampler_factors: tuple[int, ...] = (3, 3)   # 50→150→450Hz
    n_fft: int = 392
    hop_length: int = 98


class MCSDecoderV3(nn.Module):
    def __init__(self, c: MCSDecoderConfig = MCSDecoderConfig()):
        super().__init__()
        self.config = c
        D, W = c.smooth_dim, c.wave_dim
        # ① content smoothing @25Hz
        self.smooth = nn.ModuleList([CausalConvNeXtBlock(D, kernel=7) for _ in range(c.smooth_blocks)])
        self.smooth_proj = nn.Conv1d(D, D, 1)
        # ② anti-aliased rate ×2, 768→512  → 50Hz
        self.up2 = AAUpStage(D, W, factor=2, conv_k=15)
        # ③ prior net @50Hz
        self.prior = nn.ModuleList([CausalConvNeXtBlock(W, kernel=5) for _ in range(c.prior_blocks)])
        # ④ speaker decoder @50Hz (the bulk)
        self.speaker = nn.ModuleList([
            DilatedSpeakerBlock(W, c.speaker_dim, c.speaker_kernel, d) for d in c.speaker_dilations])
        # ⑤ post net @50Hz
        self.post = nn.ModuleList([CausalConvNeXtBlock(W, kernel=5) for _ in range(c.post_blocks)])
        # ⑥ anti-aliased upsampler 50→450Hz
        self.upsampler = nn.ModuleList([AAUpStage(W, W, factor=f, conv_k=7) for f in c.upsampler_factors])
        # ⑦ ISTFT head
        self.istft_head = ISTFTHead(dim=W, n_fft=c.n_fft, hop_length=c.hop_length, padding="same")

    def forward(self, content: torch.Tensor, speaker: torch.Tensor, return_spec: bool = False):
        cond = speaker.unsqueeze(1)                         # (B, 1, cond)
        h = content.transpose(1, 2)                         # (B, 768, T) @25Hz
        for b in self.smooth:
            h = b(h)
        h = self.smooth_proj(h)
        h = self.up2(h)                                     # (B, 512, 2T) @50Hz
        for b in self.prior:
            h = b(h)
        for b in self.speaker:
            h = b(h, cond)
        for b in self.post:
            h = b(h)
        for b in self.upsampler:
            h = b(h)                                        # (B, 512, 18T) @450Hz
        h = h.transpose(1, 2)                               # (B, 18T, 512)
        xo = self.istft_head.out(h).transpose(1, 2)         # (B, n_fft+2, 18T)
        mag_log, phase = xo.chunk(2, dim=1)
        mag = torch.exp(mag_log).clamp(max=1e2)
        wav = self.istft_head.istft(torch.complex(mag * torch.cos(phase), mag * torch.sin(phase)))
        if return_spec:
            return wav, mag, phase
        return wav


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    c = MCSDecoderConfig()
    m = MCSDecoderV3(c).eval()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    cont, spk = torch.randn(2, 50, 768), torch.randn(2, 128)
    with torch.no_grad():
        wav, mag, phase = m(cont, spk, return_spec=True)
    algo = (c.n_fft - c.hop_length) / 2 / 44100 * 1000
    print(f"MCSDecoderV3: {n:.2f}M  out={list(wav.shape)} ({wav.shape[1]/44100:.2f}s)  "
          f"mag={list(mag.shape)}  algo-latency={algo:.1f}ms")
    c2 = cont.clone(); c2[:, 40:] += 5.0
    with torch.no_grad():
        a = m(cont, spk); b = m(c2, spk)
    edge = a.shape[1] * 40 // 50; look = (c.n_fft - c.hop_length) // 2
    print(f"strict-causal: pre-boundary max={(a[0,:edge-look-200]-b[0,:edge-look-200]).abs().max():.2e}")

"""CausalDecoder MCS — transformer prenet + dilated TCN + linear ISTFT.

Clean, minimal design inspired by MioCodec teacher's ISTFT head:
  - Prenet: causal transformer 7L (content interpretation, no speaker)
  - Speaker: FiLM (simple, per-frame scale+shift)
  - Synthesis: dilated TCN + CausalConv
  - Upsampling: ConvTranspose + Snake (mirrors teacher's UpSamplerBlock)
  - ISTFT head: single Linear (mirrors teacher — all work done by transformer)

STRICTLY CAUSAL. Only latency: iSTFT group delay 3.3ms.
Params: ~16M. Fast training, easy to debug.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import CausalConv1d

_mio = Path(__file__).resolve().parent.parent / "external" / "MioCodec" / "src"
if str(_mio) not in sys.path:
    sys.path.insert(0, str(_mio))


# ═══════════════════════════════════════════════════════════════════
# Causal ResNet block (per-position LayerNorm, SiLU)
# ═══════════════════════════════════════════════════════════════════

class CausalResBlock(nn.Module):
    def __init__(self, dim: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel, dilation=dilation)
        self.norm2 = nn.LayerNorm(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim, T)
        r = x
        h = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        h = self.conv1(F.silu(h))
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = self.conv2(F.silu(h))
        return r + h


# ═══════════════════════════════════════════════════════════════════
# Upsampler block (mirrors MioCodec UpSamplerBlock — ConvTranspose + Snake + ResNet)
# ═══════════════════════════════════════════════════════════════════

class UpsampleStage(nn.Module):
    """ConvTranspose1d(k=s*2, s=s) + SnakeBeta + CausalResBlock. Causal: trim output."""
    def __init__(self, c_in: int, c_out: int, factor: int):
        super().__init__()
        from miocodec.module.istft_head import SnakeBeta
        self.tr = nn.ConvTranspose1d(c_in, c_out, kernel_size=factor * 2, stride=factor)
        self.snake = SnakeBeta(c_out, alpha_logscale=True)
        self.res = CausalResBlock(c_out)

    def forward(self, x: torch.Tensor, out_len: int | None = None) -> torch.Tensor:
        h = self.tr(x)                           # (B, c_out, (T+1)*factor)
        if out_len is not None:
            h = h[:, :, :out_len]                # causal trim
        return self.res(self.snake(h))


# ═══════════════════════════════════════════════════════════════════
# Multi-band partition
# ═══════════════════════════════════════════════════════════════════

def _make_bands(n_fft: int):
    """Partition frequency bins into bands. Fewer bins per band = easier phase pred."""
    n_freq = n_fft // 2 + 1  # 197 for n_fft=392
    # Low freqs: finer bands (more perceptual importance, harmonic structure)
    # High freqs: wider bands (less critical, more noise-like)
    if n_freq >= 197:
        cuts = [0, 32, 64, 128, n_freq]  # 4 bands: 32, 32, 64, 69 bins
    else:
        cuts = [0, n_freq // 3, 2 * n_freq // 3, n_freq]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CausalDecoderMCSConfig:
    content_dim: int = 768
    speaker_dim: int = 128
    sample_rate: int = 44100
    content_rate: int = 25
    input_std_scale: float = 0.46 / 0.38

    # ① Prenet transformer @25Hz (content interpretation, no speaker)
    prenet_dim: int = 384
    prenet_layers: int = 7
    prenet_heads: int = 8
    prenet_window: int = 65  # odd, ~2.5s @25Hz
    prenet_dropout: float = 0.0

    # ② Rate upsampling
    up2_dim: int = 512       # 384→512 at rate ×2

    # ④ Dilated TCN @50Hz
    tcn_dim: int = 512
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8)
    tcn_kernel: int = 5

    # ⑤ Upsampler 50Hz → 450Hz (3×3)
    upsampler_factors: tuple[int, ...] = (3, 3)

    # ⑥ ISTFT head
    n_fft: int = 392
    hop_length: int = 98
    istft_padding: str = "same"


# ═══════════════════════════════════════════════════════════════════
# Decoder
# ═══════════════════════════════════════════════════════════════════

class CausalDecoderMCS(nn.Module):
    """MCS-philosophy decoder: transformer prenet + dilated TCN + linear ISTFT.

    content(768) @25Hz + speaker(128)
      ① Prenet:           causal transformer 7L @25Hz (no speaker)
      ② Rate ×2:          ConvTranspose(k=4,s=2) + Snake + ResNet → 50Hz, 512d
      ③ Speaker:           FiLM (scale, shift) per frame → 512d
      ④ Dilated TCN:      4 dilated CausalResBlocks @50Hz
      ⑤ Rate ×9:          ConvTranspose ×3→×3 + Snake ×2 → 450Hz, 512d
      ⑥ ISTFT Head:       single Linear(512→n_fft+2) → mag+phase → iSTFT
    """

    def __init__(self, c: CausalDecoderMCSConfig = CausalDecoderMCSConfig()):
        super().__init__()
        self.config = c
        D, W = c.prenet_dim, c.up2_dim
        self.register_buffer("input_scale", torch.tensor(c.input_std_scale))

        # ① Prenet transformer: content interpretation, no speaker
        from miocodec.module.transformer import Transformer
        self.prenet = Transformer(
            dim=D, n_layers=c.prenet_layers, n_heads=c.prenet_heads,
            input_dim=c.content_dim, output_dim=D,
            window_size=c.prenet_window, causal=True,
            use_rope=True, rope_theta=10000.0, dropout=c.prenet_dropout,
            use_flash_attention=False)

        # ② Rate ×2: ConvTranspose(k=4,s=2) + Snake + ResNet → 50Hz, 384→512
        self.up2 = UpsampleStage(D, W, factor=2)

        # ③ FiLM speaker injection
        self.film = nn.Linear(c.speaker_dim, W * 2)  # → (scale, shift)

        # ④ Dilated TCN @50Hz
        self.tcn = nn.ModuleList([
            CausalResBlock(W, c.tcn_kernel, d) for d in c.tcn_dilations
        ])

        # ⑤ Upsampler: ×3 → ×3, 50→450Hz
        self.upsampler = nn.ModuleList()
        cur = W
        for f in c.upsampler_factors:
            self.upsampler.append(UpsampleStage(cur, cur, factor=f))

        # ⑥ ISTFT head: multi-band Linear → mag+phase → iSTFT
        self.bands = _make_bands(c.n_fft)  # [(start, end), ...] band partitions
        self.band_heads = nn.ModuleList([
            nn.Linear(W, (end - start) * 2)  # *2 for mag_log+phase
            for start, end in self.bands
        ])
        # Frequency-axis depthwise conv: smooths band boundaries, shares phase info
        n_freq = c.n_fft // 2 + 1
        self.freq_smooth = CausalConv1d(2, 2, 5, groups=2)  # 2 channels = mag_log+phase
        from miocodec.module.istft_head import ISTFT
        self.istft = ISTFT(n_fft=c.n_fft, hop_length=c.hop_length,
                           win_length=c.n_fft, padding=c.istft_padding)

    def _compute_stft_length(self, content_frames: int) -> int:
        return int(content_frames * self.config.sample_rate
                   / self.config.hop_length / self.config.content_rate)

    def forward(self, content: torch.Tensor, speaker: torch.Tensor,
                stft_length: int | None = None, return_spec: bool = False):
        B, T, _ = content.shape
        if stft_length is None:
            stft_length = self._compute_stft_length(T)

        # Scale content
        h = content * self.input_scale.to(dtype=content.dtype)

        # ① Prenet: transformer 7L @25Hz (content interpretation)
        h = self.prenet(h)                               # (B, T, 384)

        # ② Rate ×2: ConvTranspose → Snake → ResNet → 50Hz (causal trim)
        h = self.up2(h.transpose(1, 2), out_len=T * 2)  # (B, 512, 2T)

        # ③ FiLM speaker injection
        scale, shift = self.film(speaker).chunk(2, dim=-1)  # (B, 1024) → (B,512), (B,512)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)  # broadcast over time

        # ④ Dilated TCN @50Hz
        for block in self.tcn:
            h = block(h)

        # ⑤ Upsampler 50→450Hz
        # Each ConvTranspose expands time by its factor. Track output length.
        cur_len = h.shape[-1]
        for block in self.upsampler:
            cur_len *= block.tr.stride[0]
            h = block(h, out_len=cur_len)

        # ⑥ ISTFT head: multi-band Linear → mag+phase → iSTFT
        h = h.transpose(1, 2)                            # (B, T_stft, 512)
        # Predict each band independently, then concatenate along freq axis
        mag_logs, phases = [], []
        for head, (start, end) in zip(self.band_heads, self.bands):
            xo = head(h).transpose(1, 2)                 # (B, (end-start)*2, T_stft)
            ml, ph = xo.chunk(2, dim=1)                  # (B, end-start, T_stft) each
            mag_logs.append(ml); phases.append(ph)
        mag_log = torch.cat(mag_logs, dim=1)              # (B, 197, T_stft)
        phase = torch.cat(phases, dim=1)
        # Frequency-axis smooth: stack mag+phase → conv → split
        mp = torch.stack([mag_log, phase], dim=1)          # (B, 2, 197, T_stft)
        B_s, _, nf_s, T_s = mp.shape
        mp = mp.reshape(B_s * T_s, 2, nf_s)                # (B*T, 2, 197)
        mp = self.freq_smooth(mp)                           # depthwise along freq
        mp = mp.reshape(B_s, 2, nf_s, T_s)
        mag_log, phase = mp[:, 0], mp[:, 1]
        mag = torch.exp(mag_log).clamp(max=1e2)
        wav = self.istft(torch.complex(mag * torch.cos(phase), mag * torch.sin(phase)))
        if return_spec:
            return wav, mag, phase
        return wav


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    c = CausalDecoderMCSConfig(prenet_layers=2)  # small for quick test
    m = CausalDecoderMCS(c).eval()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    cont, spk = torch.randn(2, 30, 768), torch.randn(2, 128)
    with torch.no_grad():
        wav = m(cont, spk)
    algo = (c.n_fft - c.hop_length) / 2 / c.sample_rate * 1000
    print(f"CausalDecoderMCS: {n:.2f}M  out={list(wav.shape)} "
          f"({wav.shape[1]/c.sample_rate:.2f}s)  algo-latency={algo:.1f}ms")

    # Strict-causal check
    cont2 = cont.clone(); cont2[:, 20:] += 5.0
    with torch.no_grad():
        a = m(cont, spk); b = m(cont2, spk)
    edge = a.shape[1] * 20 // 30
    look = (c.n_fft - c.hop_length) // 2
    diff = (a[0, :edge-look-200] - b[0, :edge-look-200]).abs().max().item()
    print(f"strict-causal: max_diff={diff:.2e} {'✓' if diff < 1e-3 else '✗'}")

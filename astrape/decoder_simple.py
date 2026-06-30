"""Dead-simple GRU causal decoder — the MCS-philosophy restart.

content(768) + speaker(128) → FiLM → GRU(causal) → upsample ×7 → causal convs
  → iSTFT head → wav 44.1 kHz.

The encoder went from a 215-line MCS (conv + GRU + FSQ) at val-cos 0.84 to 0.93 by
*incremental* changes. The decoder jumped straight to complex from-scratch designs
(42M transformer → v5) and stalled. So restart minimal and evolve the same way.

Streaming-native: GRU hidden state + conv ring state + iSTFT overlap-add — NO KV-cache.
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


@dataclass
class SimpleDecoderConfig:
    content_dim: int = 768
    speaker_dim: int = 128
    hidden: int = 384            # MCS gru_dim
    gru_layers: int = 2          # MCS used 2
    upsample: int = 7            # 25 → 175 Hz (matches hop=252)
    smooth_layers: int = 2
    bridge_dim: int = 512
    n_fft: int = 1512            # same iSTFT head as v5 (14.3 ms group delay)
    hop_length: int = 252
    sample_rate: int = 44100
    content_rate: int = 25


class SimpleGRUDecoder(nn.Module):
    def __init__(self, config: SimpleDecoderConfig = SimpleDecoderConfig()):
        super().__init__()
        self.config = c = config
        self.in_proj = nn.Linear(c.content_dim, c.hidden)
        self.film = nn.Linear(c.speaker_dim, c.hidden * 2)            # speaker → (scale, shift)
        self.gru = nn.GRU(c.hidden, c.hidden, c.gru_layers, batch_first=True)
        self.smooth = nn.ModuleList(
            [CausalConv1d(c.hidden, c.hidden, 3) for _ in range(c.smooth_layers)])
        self.bridge = nn.Conv1d(c.hidden, c.bridge_dim, 1)
        from miocodec.module.istft_head import ISTFTHead
        self.istft = ISTFTHead(dim=c.bridge_dim, n_fft=c.n_fft, hop_length=c.hop_length, padding="same")

    def _compute_stft_length(self, content_frames: int) -> int:
        return int(content_frames * self.config.sample_rate
                   / self.config.hop_length / self.config.content_rate)

    def forward(self, content: torch.Tensor, speaker: torch.Tensor,
                stft_length: int | None = None) -> torch.Tensor:
        B, T, _ = content.shape
        if stft_length is None:
            stft_length = self._compute_stft_length(T)
        h = self.in_proj(content)                                    # (B,T,H)
        scale, shift = self.film(speaker).chunk(2, dim=-1)
        h = h * (1 + scale).unsqueeze(1) + shift.unsqueeze(1)        # FiLM speaker conditioning
        h, _ = self.gru(h)                                           # causal recurrence (B,T,H)
        h = h.repeat_interleave(self.config.upsample, dim=1)         # 25 → 175 Hz (zero-order hold)
        h = h[:, :stft_length].transpose(1, 2)                       # (B,H,L)
        for conv in self.smooth:
            h = F.silu(conv(h))                                      # smooth the ZOH (causal)
        h = self.bridge(h).transpose(1, 2)                          # (B,L,bridge)
        return self.istft(h)                                        # (B, samples)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    m = SimpleGRUDecoder().eval()
    n = sum(p.numel() for p in m.parameters())
    B, T = 2, 50
    with torch.no_grad():
        wav = m(torch.randn(B, T, 768), torch.randn(B, 128))
    # strict-causal check: perturbing a future content frame must not change earlier output
    c = torch.randn(1, 50, 768); s = torch.randn(1, 128)
    with torch.no_grad():
        a = m(c, s); c2 = c.clone(); c2[:, 40:] += 5.0; b = m(c2, s)
    edge = a.shape[1] * 40 // 50
    fut = (a[:, :edge] - b[:, :edge]).abs().max().item()
    print(f"SimpleGRUDecoder: {n/1e6:.2f}M params  out={list(wav.shape)} ({wav.shape[1]/44100:.2f}s)")
    print(f"strict-causal (future→0): {fut:.2e}")

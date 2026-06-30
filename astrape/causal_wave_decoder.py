"""Causal-native replica of the MioCodec wave decoder.

The structure (module layout, dims, factors) mirrors MioCodec's wave decoder EXACTLY —
not its weights. Causalizing the teacher and fine-tuning was shown (prior work) not to
resemble it; instead we replicate the architecture/capacity and train causal-native from
scratch via distillation. Because the modules align 1:1 with the teacher, intermediate
features can be distilled directly.

Pipeline (rates divide evenly 25→50→450Hz, so NO interpolate → strictly causal):
  content 768d @25Hz
   → ① prenet            Transformer 6L d768 h12 (causal, windowed, RoPE) → 512d
   → ② conv_upsample     ConvTranspose1d k2 s2 (causal: no overlap) → 50Hz
   → ③ prior_net         CausalResNetStack 2 blocks
   → ④ speaker decoder   Transformer 8L d512 h8 (causal, windowed, AdaLN-Zero(speaker 128d))
   → ⑤ post_net          CausalResNetStack 2 blocks
   → ⑥ upsampler         CausalUpSampler (3,3)=9× → 450Hz
   → ⑦ ISTFTHead         n_fft=392 hop=98 (already causal; 3.3ms algorithmic latency)

Causalizations vs the teacher: Transformer `causal=True` + window (KV-cache ready);
GroupNorm→per-position LayerNorm (GroupNorm pools over time = non-causal); symmetric
conv padding → left-pad (CausalConv1d) / causal-trim transpose-conv.
"""
from __future__ import annotations

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
from miocodec.module.transformer import Transformer            # noqa: E402  (causal-capable)
from miocodec.module.istft_head import ISTFTHead, SnakeBeta    # noqa: E402


# ── ③⑤ Causal ResNet (GroupNorm → per-position channel LayerNorm; Conv1d → CausalConv1d) ──
class _ChannelNorm(nn.Module):
    """LayerNorm over channels at EACH time step (B,C,L) — causal (no time pooling)."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, C, L)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalResNetBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _ChannelNorm(channels)
        self.conv1 = CausalConv1d(channels, channels, kernel_size)
        self.norm2 = _ChannelNorm(channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, C, L)
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class CausalResNetStack(nn.Module):
    def __init__(self, channels: int, num_blocks: int = 2, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CausalResNetBlock(channels, kernel_size, dropout) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


# ── ⑥ Causal upsampler (causal-trim ConvTranspose + SnakeBeta + causal ResNet) ──
class CausalUpSamplerBlock(nn.Module):
    """Causal upsampler via RESIZE-CONV (nearest upsample + CausalConv1d) instead of
    ConvTranspose. ConvTranspose(k=2u,s=u) has overlapping kernels whose uneven sub-frame
    weights produce a periodic broadband "지지직" modulation (~100Hz) — confirmed in the
    spectrogram. Nearest-repeat (strictly causal) + a causal smoothing conv avoids it."""
    def __init__(self, in_channels: int, upsample_factors: list[int]):
        super().__init__()
        self.factors = list(upsample_factors)
        self.ups = nn.ModuleList()
        self.snakes = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, u in enumerate(self.factors):
            c_in = in_channels // (2 ** i)
            c_out = in_channels // (2 ** (i + 1))
            self.ups.append(CausalConv1d(c_in, c_out, kernel_size=2 * u + 1))   # smooth the repeat
            self.snakes.append(SnakeBeta(c_out, alpha_logscale=True))
            self.resblocks.append(CausalResNetBlock(c_out))
        final = in_channels // (2 ** len(self.factors))
        self.out_proj = nn.Linear(final, in_channels)
        self.out_snake = SnakeBeta(in_channels, alpha_logscale=True)

    @property
    def total_upsample_factor(self) -> int:
        r = 1
        for f in self.factors:
            r *= f
        return r

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, C, L) → (B, L', C)
        for up, snake, res, u in zip(self.ups, self.snakes, self.resblocks, self.factors):
            x = F.interpolate(x, scale_factor=u, mode="nearest")   # causal repeat → (B, C, L*u)
            x = res(snake(up(x)))
        x = self.out_proj(x.transpose(1, 2))                   # (B, L', C)
        return self.out_snake(x.transpose(1, 2)).transpose(1, 2)


@dataclass
class CausalWaveDecoderConfig:
    content_dim: int = 768
    speaker_dim: int = 128
    # ① prenet (teacher: 6L d768 h12, no speaker)
    prenet_dim: int = 768
    prenet_layers: int = 6
    prenet_heads: int = 12
    prenet_window: int = 129          # odd; ~64 past frames @25Hz ≈ 2.5s, KV-cache bounded
    wave_dim: int = 512
    conv_upsample_factor: int = 2     # 25→50Hz
    resnet_blocks: int = 2
    resnet_kernel: int = 3
    # ④ speaker decoder (teacher: 8L d512 h8, AdaLN-Zero)
    decoder_layers: int = 8
    decoder_heads: int = 8
    decoder_window: int = 257         # odd; ~128 past frames @50Hz ≈ 2.5s
    upsampler_factors: tuple[int, ...] = (3, 3)   # 50→450Hz
    n_fft: int = 392
    hop_length: int = 98
    rope_theta: float = 10000.0
    dropout: float = 0.0


class CausalWaveDecoder(nn.Module):
    def __init__(self, c: CausalWaveDecoderConfig = CausalWaveDecoderConfig()):
        super().__init__()
        self.config = c
        D, W = c.prenet_dim, c.wave_dim
        # ① content prenet — causal windowed transformer, no speaker
        self.prenet = Transformer(
            dim=D, n_layers=c.prenet_layers, n_heads=c.prenet_heads, output_dim=W,
            window_size=c.prenet_window, causal=True, use_rope=True, rope_theta=c.rope_theta,
            dropout=c.dropout, use_flash_attention=False)
        # ② causal conv upsample 25→50Hz  (k=s=2 → each input maps to its own 2 outputs, no future)
        self.conv_upsample = nn.ConvTranspose1d(W, W, kernel_size=c.conv_upsample_factor,
                                                stride=c.conv_upsample_factor)
        # ③ prior net
        self.prior_net = CausalResNetStack(W, c.resnet_blocks, c.resnet_kernel, c.dropout)
        # ④ speaker decoder — causal windowed transformer + AdaLN-Zero(speaker)
        self.speaker_decoder = Transformer(
            dim=W, n_layers=c.decoder_layers, n_heads=c.decoder_heads,
            window_size=c.decoder_window, causal=True, use_rope=True, rope_theta=c.rope_theta,
            dropout=c.dropout, use_adaln_zero=True, adanorm_condition_dim=c.speaker_dim,
            use_flash_attention=False)
        # ⑤ post net
        self.post_net = CausalResNetStack(W, c.resnet_blocks, c.resnet_kernel, c.dropout)
        # ⑥ upsampler 50→450Hz
        self.upsampler = CausalUpSamplerBlock(W, list(c.upsampler_factors))
        # ⑦ ISTFT head (already causal)
        self.istft_head = ISTFTHead(dim=W, n_fft=c.n_fft, hop_length=c.hop_length, padding="same")

    def forward(self, content: torch.Tensor, speaker: torch.Tensor,
                return_spec: bool = False, return_feats: bool = False):
        """content (B, T, 768) @25Hz, speaker (B, 128). Returns wav (B, samples).
        return_spec → (wav, mag, phase); return_feats → also a dict of per-module activations
        (for intermediate-feature distillation against the teacher)."""
        feats = {}
        h = self.prenet(content)                               # (B, T, 512)
        feats["prenet"] = h
        h = self.conv_upsample(h.transpose(1, 2))              # (B, 512, 2T)
        feats["upsample"] = h
        h = self.prior_net(h)                                  # (B, 512, 2T)
        feats["prior"] = h
        h = self.speaker_decoder(h.transpose(1, 2), condition=speaker.unsqueeze(1))  # (B, 2T, 512)
        feats["decoder"] = h
        h = self.post_net(h.transpose(1, 2))                   # (B, 512, 2T)
        feats["post"] = h
        h = self.upsampler(h)                                  # (B, 18T, 512)
        feats["upsampler"] = h
        # inline ISTFT head to expose mag/phase
        xo = self.istft_head.out(h).transpose(1, 2)            # (B, n_fft+2, 18T)
        mag_log, phase = xo.chunk(2, dim=1)
        mag = torch.exp(mag_log).clamp(max=1e2)
        wav = self.istft_head.istft(torch.complex(mag * torch.cos(phase), mag * torch.sin(phase)))
        if return_feats:
            return wav, mag, phase, feats
        if return_spec:
            return wav, mag, phase
        return wav


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    c = CausalWaveDecoderConfig()
    m = CausalWaveDecoder(c).eval()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    cont, spk = torch.randn(2, 50, 768), torch.randn(2, 128)
    with torch.no_grad():
        wav, mag, phase = m(cont, spk, return_spec=True)
    algo = (c.n_fft - c.hop_length) / 2 / 44100 * 1000
    print(f"CausalWaveDecoder: {n:.2f}M  out={list(wav.shape)} ({wav.shape[1]/44100:.2f}s)  "
          f"mag={list(mag.shape)}  algo-latency={algo:.1f}ms")
    # strict-causal check (perturb future content → earlier output unchanged outside iSTFT lookahead)
    c2 = cont.clone(); c2[:, 40:] += 5.0
    with torch.no_grad():
        a = m(cont, spk); b = m(c2, spk)
    edge = a.shape[1] * 40 // 50
    look = (c.n_fft - c.hop_length) // 2
    print(f"strict-causal: pre-boundary[:edge-{look}-50] max={ (a[0,:edge-look-50]-b[0,:edge-look-50]).abs().max():.2e}")

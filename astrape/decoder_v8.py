"""Decoder v8 — factorized cascade (see DECODER_V8_DESIGN.md).

Stage A  AcousticModel      content 768@25Hz + speaker 128 → mel80 + logF0 + voicing +
                            energy @150Hz.  Reuses the teacher's ①–⑤ structure (the causal
                            replica trunk) then branches to acoustic heads.  Pure regression.
Stage B  ConditionedVocoder  acoustic frames @150Hz → wav 44.1kHz.  Causal Vocos-class
                            conv vocoder (no speaker input — timbre rides in the mel), the
                            densely-conditioned regime where GAN vocoders are proven.

Both strictly causal (0 look-ahead).  Only algorithmic latency = the iSTFT overlap-add
tail (392/98 = 3.3 ms), identical to every prior decoder.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_MAG_MAX = math.log(1e2)   # clamp log-magnitude BEFORE exp (see forward) → no overflow

from .nn import CausalConv1d

_mio = Path(__file__).resolve().parent.parent / "external" / "MioCodec" / "src"
if str(_mio) not in sys.path:
    sys.path.insert(0, str(_mio))
from miocodec.module.transformer import Transformer            # noqa: E402
from miocodec.module.istft_head import ISTFTHead, SnakeBeta    # noqa: E402

# Reuse the causal ResNet already validated in the distill replica.
from .causal_wave_decoder import CausalResNetStack  # noqa: E402

N_MELS = 80
COND_DIM = N_MELS + 3          # mel | logf0 | voiced | energy


# ══════════════════════════════════════════════════════════════════════
# Shared: causal ConvNeXt block (per-position channel LayerNorm, depthwise k7)
# ══════════════════════════════════════════════════════════════════════

class CausalConvNeXt(nn.Module):
    """Depthwise causal k7 → channel-LN → pointwise ↑4 → GELU → pointwise ↓, residual."""
    def __init__(self, dim: int, kernel: int = 7, mult: int = 3):
        super().__init__()
        self.dw = CausalConv1d(dim, dim, kernel, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * mult)
        self.pw2 = nn.Linear(dim * mult, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # (B, C, T)
        r = x
        h = self.dw(x).transpose(1, 2)                        # (B, T, C)
        h = self.pw2(F.gelu(self.pw1(self.norm(h))))
        return r + h.transpose(1, 2)


# ══════════════════════════════════════════════════════════════════════
# Stage A — Acoustic model
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AcousticModelConfig:
    content_dim: int = 768
    speaker_dim: int = 128
    input_std_scale: float = 0.46 / 0.38     # Q2D2 std → teacher std (as in decoder_mcs)
    # ①–⑤ trunk (teacher-replica dims)
    prenet_dim: int = 768
    prenet_layers: int = 6
    prenet_heads: int = 12
    prenet_window: int = 129
    wave_dim: int = 512
    resnet_blocks: int = 2
    decoder_layers: int = 8
    decoder_heads: int = 8
    decoder_window: int = 257
    rope_theta: float = 10000.0
    # acoustic head
    head_dim: int = 256
    head_convnext: int = 2
    prosody_embed: int = 32
    dropout: float = 0.0


class AcousticModel(nn.Module):
    """content(B,T,768)@25Hz + speaker(B,128) → acoustic frames @150Hz.

    prosody-FIRST: predict logF0/voicing/energy, then condition the mel head on them
    (resolves the residual multimodality that makes plain mel-regression over-smooth).
    """
    def __init__(self, c: AcousticModelConfig = AcousticModelConfig()):
        super().__init__()
        self.config = c
        D, W = c.prenet_dim, c.wave_dim
        self.register_buffer("input_scale", torch.tensor(c.input_std_scale))

        # ① content prenet — causal windowed transformer, no speaker
        self.prenet = Transformer(
            dim=D, n_layers=c.prenet_layers, n_heads=c.prenet_heads, output_dim=W,
            window_size=c.prenet_window, causal=True, use_rope=True, rope_theta=c.rope_theta,
            dropout=c.dropout, use_flash_attention=False)
        # ② causal conv upsample 25→50Hz
        self.conv_upsample = nn.ConvTranspose1d(W, W, kernel_size=2, stride=2)
        # ③ prior net
        self.prior_net = CausalResNetStack(W, c.resnet_blocks, 3, c.dropout)
        # ④ speaker decoder — causal windowed transformer + AdaLN-Zero(speaker)
        self.speaker_decoder = Transformer(
            dim=W, n_layers=c.decoder_layers, n_heads=c.decoder_heads,
            window_size=c.decoder_window, causal=True, use_rope=True, rope_theta=c.rope_theta,
            dropout=c.dropout, use_adaln_zero=True, adanorm_condition_dim=c.speaker_dim,
            use_flash_attention=False)
        # ⑤ post net
        self.post_net = CausalResNetStack(W, c.resnet_blocks, 3, c.dropout)

        # ⑥ₐ upsample 50→150Hz (×3, nearest-repeat + causal smoothing conv)
        H = c.head_dim
        self.up3 = CausalConv1d(W, H, kernel_size=7)
        self.up_snake = SnakeBeta(H, alpha_logscale=True)
        # ⑦ₐ prosody head: logF0, voicing-logit, energy
        self.prosody_head = nn.Linear(H, 3)
        # ⑧ₐ mel head: conditioned on prosody
        self.prosody_embed = nn.Linear(3, c.prosody_embed)
        self.mel_in = nn.Conv1d(H + c.prosody_embed, H, kernel_size=1)
        self.mel_convnext = nn.ModuleList([CausalConvNeXt(H) for _ in range(c.head_convnext)])
        self.mel_head = nn.Linear(H, N_MELS)

    def forward(self, content: torch.Tensor, speaker: torch.Tensor,
                gt_prosody: torch.Tensor | None = None) -> dict:
        """Returns dict: mel (B,T,80), logf0 (B,T), voiced_logit (B,T), energy (B,T) @150Hz.
        If ``gt_prosody`` (B,T,3: logf0, voiced, energy) is given, the mel head is
        teacher-forced on it (training); else it uses its own prediction."""
        h = content * self.input_scale.to(content.dtype)
        h = self.prenet(h)                                    # (B,T,512)
        h = self.conv_upsample(h.transpose(1, 2))             # (B,512,2T) @50Hz
        h = self.prior_net(h)
        h = self.speaker_decoder(h.transpose(1, 2), condition=speaker.unsqueeze(1))  # (B,2T,512)
        h = self.post_net(h.transpose(1, 2))                  # (B,512,2T)

        # ⑥ₐ 50→150Hz
        h = F.interpolate(h, scale_factor=3, mode="nearest")  # causal repeat (B,512,6T)
        h = self.up_snake(self.up3(h)).transpose(1, 2)        # (B,6T,256)

        # ⑦ₐ prosody
        pros = self.prosody_head(h)                           # (B,6T,3)
        logf0, voiced_logit, energy = pros[..., 0], pros[..., 1], pros[..., 2]

        # ⑧ₐ mel, conditioned on prosody (teacher-forced or predicted)
        if gt_prosody is not None:
            pe = self.prosody_embed(gt_prosody)
        else:
            pe = self.prosody_embed(torch.stack(
                [logf0, torch.sigmoid(voiced_logit), energy], dim=-1))
        m = torch.cat([h, pe], dim=-1).transpose(1, 2)        # (B,256+32,6T)
        m = self.mel_in(m)
        for blk in self.mel_convnext:
            m = blk(m)
        mel = self.mel_head(m.transpose(1, 2))                # (B,6T,80)
        return {"mel": mel, "logf0": logf0, "voiced_logit": voiced_logit, "energy": energy}

    @torch.no_grad()
    def infer_cond(self, content: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        """Predicted acoustic conditioning (B,T,83) = [mel | logf0 | voiced_prob | energy]
        for feeding Stage B at inference."""
        o = self.forward(content, speaker)
        return torch.cat([o["mel"], o["logf0"][..., None],
                          torch.sigmoid(o["voiced_logit"])[..., None],
                          o["energy"][..., None]], dim=-1)


# ══════════════════════════════════════════════════════════════════════
# Stage B — Conditioned causal vocoder
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VocoderConfig:
    cond_dim: int = COND_DIM     # 83
    dim: int = 512
    trunk_blocks: int = 8
    # Vocos-pure: iSTFT synthesizes DIRECTLY at the 150Hz frame rate (hop = 44100/150 = 294),
    # NO time-domain upsampling (which, as ZOH, stamped a 150Hz frame-rate comb on the output).
    # OVERLAP MATTERS: with only 50% overlap (n_fft=588=2·hop) the overlap-add of 2 frames still
    # re-creates 150Hz amplitude modulation from imperfect inter-frame phase (measured: random-
    # phase frame_buzz 10.5dB vs GT 6.6). n_fft=1176 = 4·hop → 75% overlap (Vocos ratio, 4 frames
    # per sample, exact Hann COLA at hop=n_fft/4) drops it to 6.8dB ≈ GT. Latency (1176-294)/2 =
    # 441smp = 10ms (E2E ≈ 25ms, well inside the 50ms budget).
    n_fft: int = 882
    hop_length: int = 294


class ConditionedVocoder(nn.Module):
    """acoustic frames (B,T,83)@150Hz → wav (B,samples)@44.1kHz.  Conv-only, causal, no
    speaker input (timbre is carried by the mel — as in Soft-VC / StreamVC)."""
    def __init__(self, c: VocoderConfig = VocoderConfig()):
        super().__init__()
        self.config = c
        self.proj = nn.Conv1d(c.cond_dim, c.dim, kernel_size=1)
        self.trunk = nn.ModuleList([CausalConvNeXt(c.dim) for _ in range(c.trunk_blocks)])
        self.istft_head = ISTFTHead(dim=c.dim, n_fft=c.n_fft, hop_length=c.hop_length, padding="same")

    def forward(self, cond: torch.Tensor, return_spec: bool = False):
        """cond (B, T, 83) @150Hz → wav (B, T·294).  No time-domain upsampling: the iSTFT
        runs at the 150Hz frame rate, so its overlap-add (not a ZOH stair-step) interpolates."""
        h = self.proj(cond.transpose(1, 2))                   # (B,512,T) @150Hz
        for blk in self.trunk:
            h = blk(h)                                        # stays @150Hz
        xo = self.istft_head.out(h.transpose(1, 2)).transpose(1, 2)   # (B,n_fft+2,T)
        mag_log, phase = xo.chunk(2, dim=1)
        # Clamp BEFORE exp: exp(mag_log>88) overflows to inf in fp32; the old
        # `exp(mag_log).clamp(max=1e2)` then hides it in the forward, but the backward does
        # clamp'0-grad × inf = NaN → the finite-grad guard skipped ~99.7% of steps. Clamping
        # mag_log to log(100) first is forward-identical (both cap magnitude at 100) and keeps
        # the gradient finite.
        mag = torch.exp(mag_log.clamp(max=_LOG_MAG_MAX))
        wav = self.istft_head.istft(torch.complex(mag * torch.cos(phase), mag * torch.sin(phase)))
        if return_spec:
            return wav, mag, phase
        return wav


def acoustic_frames(content_frames: int) -> int:
    """150Hz acoustic/iSTFT frame count = content*6.  Vocoder output = this * hop(294)."""
    return content_frames * 6


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    Tc = 50
    content, speaker = torch.randn(2, Tc, 768), torch.randn(2, 128)
    A = AcousticModel().eval()
    B = ConditionedVocoder().eval()
    nA = sum(p.numel() for p in A.parameters()) / 1e6
    nB = sum(p.numel() for p in B.parameters()) / 1e6
    with torch.no_grad():
        o = A(content, speaker)
        cond = A.infer_cond(content, speaker)
        wav = B(cond)
    print(f"AcousticModel: {nA:.1f}M  mel={list(o['mel'].shape)} (expect T={6*Tc})")
    print(f"ConditionedVocoder: {nB:.1f}M  wav={list(wav.shape)} ({wav.shape[1]/44100:.2f}s, "
          f"expect {6*Tc*294} smp)")
    algo = (B.config.n_fft - B.config.hop_length) / 2 / 44100 * 1000
    print(f"iSTFT algo-latency={algo:.1f}ms (n_fft={B.config.n_fft}, hop={B.config.hop_length})")

    # ── strict causality: perturb future content, earlier acoustics unchanged ──
    c2 = content.clone(); c2[:, 40:] += 5.0
    with torch.no_grad():
        m1 = A(content, speaker)["mel"]; m2 = A(c2, speaker)["mel"]
    edge = 40 * 6            # acoustic frames before the perturbed content frame 40
    d = (m1[:, :edge] - m2[:, :edge]).abs().max().item()
    print(f"Stage A strict-causal: mel[:{edge}] max_diff={d:.2e} {'OK' if d < 1e-4 else 'LEAK'}")
    # Stage B: perturb future cond frames, earlier wav unchanged (outside iSTFT tail)
    cond2 = cond.clone(); cond2[:, 40*6:] += 5.0
    with torch.no_grad():
        w1 = B(cond); w2 = B(cond2)
    look = (B.config.n_fft - B.config.hop_length) // 2
    edge_s = 40 * 6 * B.config.hop_length - look - 200   # samples before perturbed frame 240
    d2 = (w1[:, :edge_s] - w2[:, :edge_s]).abs().max().item()
    print(f"Stage B strict-causal: wav[:{edge_s}] max_diff={d2:.2e} {'OK' if d2 < 1e-4 else 'LEAK'}")

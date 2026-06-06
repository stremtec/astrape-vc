"""
FlowVC configuration.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ── Encoder ────────────────────────────────────────────────────

@dataclass
class EncoderConfig:
    """F³-Encoder: causal ConvNeXt v2, KL-free continuous AE."""
    sample_rate: int = 44100
    latent_rate: int = 25              # Hz, hop = 44100/25 = 1764
    stages: tuple[int, ...] = field(
        default_factory=lambda: (32, 64, 128, 256, 512, 768)
    )
    strides: tuple[int, ...] = field(
        default_factory=lambda: (2, 2, 3, 3, 7, 7)
    )
    kernel_size: int = 7
    blocks_per_stage: int = 2
    mlp_expansion: int = 4
    noise_sigma: float = 0.01          # F³-Tokenizer noise regularization
    use_grn: bool = True


# ── Decoder ────────────────────────────────────────────────────

@dataclass
class DecoderConfig:
    """F³-Decoder: causal ConvNeXt v2 + MRF upsampler."""
    sample_rate: int = 44100
    latent_dim: int = 768
    stages: tuple[int, ...] = field(
        default_factory=lambda: (512, 256, 128, 64, 32, 16)
    )
    strides: tuple[int, ...] = field(
        default_factory=lambda: (7, 7, 3, 3, 2, 2)
    )
    kernel_size: int = 7
    pre_upsample_blocks: int = 4
    mrf_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    mrf_dilations: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    use_grn: bool = True


# ── Speaker Encoder ────────────────────────────────────────────

@dataclass
class SpeakerEncoderConfig:
    """Speaker encoder: causal ConvNeXt v2 + attention pooling."""
    sample_rate: int = 44100
    speaker_dim: int = 192
    stages: tuple[int, ...] = field(
        default_factory=lambda: (32, 48, 64, 96, 128, 192)
    )
    strides: tuple[int, ...] = field(
        default_factory=lambda: (2, 2, 3, 3, 7, 7)
    )
    kernel_size: int = 7
    blocks_per_stage: int = 2
    attn_pool_heads: int = 8
    prompt_tokens: int = 4         # P-Flow speaker prompt


# ── Flow Matching Converter ────────────────────────────────────

@dataclass
class FlowConverterConfig:
    """Vector Field Network for Conditional Flow Matching."""
    latent_dim: int = 768
    hidden_dim: int = 512
    time_dim: int = 256
    cond_dim: int = 256              # speaker(192) + prosody(3) → projection
    speaker_dim: int = 192
    prosody_dim: int = 3
    n_blocks: int = 12
    kernel_size: int = 7
    dilations: tuple[int, ...] = field(
        default_factory=lambda: (1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8)
    )
    mlp_expansion: int = 4
    dropout: float = 0.1            # stochastic depth rate
    use_cross_attn: bool = True
    cross_attn_layers: tuple[int, ...] = field(
        default_factory=lambda: (3, 6, 9)
    )
    cross_attn_heads: int = 4
    prompt_dim: int = 192           # matches speaker prompt dim
    # CFM
    sigma_min: float = 0.001        # min noise for stability
    ode_steps: int = 4              # inference: Euler steps


# ── Training ───────────────────────────────────────────────────

@dataclass
class TrainConfig:
    data_dir: str = ""
    cache_dir: str = ""
    sample_rate: int = 44100
    crop_seconds: float = 2.0
    # Phases
    phase: int = 0       # 0=AE pretrain, 1=CFM, 2=E2E+GAN
    steps: int = 200000
    batch_size: int = 1
    lr: float = 2e-4
    device: str = "cpu"
    # Logging
    log_interval: int = 50
    save_interval: int = 1000
    # Checkpoints
    resume: str = ""
    output_dir: str = "./runs"

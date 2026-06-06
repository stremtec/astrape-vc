"""
HybridVC configuration.

All hyperparameters in one place. Training uses HybridVCTrainConfig,
inference uses HybridVCInferConfig.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ── Converter (extends btrv3lite ConverterConfig) ──────────────

@dataclass
class ConverterConfig:
    """CausalConvNeXt Converter for latent-space voice conversion."""
    content_dim: int = 768       # MioCodec latent dim @ 25Hz
    speaker_dim: int = 128       # speaker condition dim
    prosody_dim: int = 3         # [log_f0, voicing, log_energy]
    hidden_dim: int = 192
    cond_dim: int = 128
    n_blocks: int = 10           # v2: 10 blocks
    kernel_size: int = 5
    dilations: tuple[int, ...] = field(
        default_factory=lambda: (1, 2, 4, 8, 16, 1, 2, 4, 8, 16)
    )
    mlp_expansion: int = 4
    cond_mlp_hidden: int = 128
    # Cross-attention (P-Flow style speaker prompt)
    use_cross_attn: bool = True
    n_speaker_prompt_tokens: int = 4
    prompt_dim: int = 192
    cross_attn_layers: tuple[int, ...] = field(
        default_factory=lambda: (3, 6, 9)  # 1-indexed
    )
    cross_attn_heads: int = 4
    # GRN (Global Response Normalization, ConvNeXt v2)
    use_grn: bool = True

    def __post_init__(self) -> None:
        if len(self.dilations) != self.n_blocks:
            raise ValueError(
                f"dilations length ({len(self.dilations)}) != n_blocks ({self.n_blocks})"
            )


# ── RAF Vocoder ────────────────────────────────────────────────

@dataclass
class RAFVocoderConfig:
    """RAF (Relativistic Adversarial Feature) BigVGAN Vocoder."""
    sample_rate: int = 24000          # base output SR before BWE
    latent_dim: int = 768             # MioCodec content dim
    speaker_dim: int = 128
    # Generator
    prenet_dim: int = 512
    prenet_blocks: int = 6
    prenet_kernel: int = 5
    upsample_rates: tuple[int, ...] = (3, 3)  # 25→75→225 Hz
    upsample_kernel: tuple[int, ...] = (9, 9)
    resblock_kernel: int = 5
    resblock_dilations: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
    )
    # RAF loss
    raf_teacher: str = "microsoft/wavlm-large"  # SSL teacher
    raf_layers: tuple[int, ...] = (6, 12, 18, 24)  # WavLM layers for FM
    lambda_raf: float = 1.0
    lambda_fm: float = 2.0
    lambda_mel: float = 45.0
    lambda_stft: float = 1.0
    gan_start_step: int = 5000


# ── BWE (Bandwidth Extension) ──────────────────────────────────

@dataclass
class BWEConfig:
    """ConvNeXt-based Bandwidth Extension: 24kHz → 44.1kHz."""
    input_sr: int = 24000
    output_sr: int = 44100
    hidden_dim: int = 256
    n_blocks: int = 6
    kernel_size: int = 7


# ── Training Config ────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Full training configuration for HybridVC."""
    # Data
    data_dir: str = ""
    cache_dir: str = ""
    sample_rate: int = 44100
    crop_seconds: float = 2.0
    # Training phases
    phase: int = 0           # 0=weight-transfer, 1=RAF-vocoder, 2=SSL-distill, 3=E2E+BWE
    steps: int = 200000
    batch_size: int = 1      # MPS: 1, CUDA: 8
    lr: float = 2e-4
    lr_raf: float = 1e-4
    device: str = "cpu"
    # Logging
    log_interval: int = 50
    save_interval: int = 1000
    # Checkpoints
    resume: str = ""         # path to checkpoint
    btrv3lite_ckpt: str = "" # Phase 0: btrv3lite converter weights
    # Bank
    bank_path: str = ""
    # Output
    output_dir: str = "./runs"


# ── Inference Config ───────────────────────────────────────────

@dataclass
class InferConfig:
    """Inference configuration for HybridVC."""
    converter_ckpt: str = ""
    vocoder_ckpt: str = ""
    bwe_ckpt: str = ""
    bank_path: str = ""
    device: str = "cpu"
    chunk_ms: int = 80        # streaming chunk size
    left_ctx_ms: int = 320    # left context
    lookahead_ms: int = 160   # lookahead

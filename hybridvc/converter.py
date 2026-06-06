"""
CausalConvNeXt Converter for HybridVC.

Extended from btrv3lite converter.py (v2 architecture):
- 10 ConvNeXt v2 blocks (GRN + LayerScale)
- AdaLN-Zero conditioning on [speaker_cond ‖ prosody]
- P-Flow cross-attention at blocks {3, 6, 9}
- Identity-init residual: z_out = z_src + out_gate * delta
- Causal: left-only padding, no future leak

Params: ~5.4M (v2) — up from 4.7M (v1, 8 blocks, no cross-attn)

Refs:
- Liu et al., "A ConvNet for the 2020s" (ConvNeXt)
- Woo et al., "ConvNeXt V2" (GRN)
- Peebles & Xie, "Scalable Diffusion Models with Transformers" (AdaLN-Zero)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ConverterConfig


# ── Causal Conv1d ──────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """1D convolution with left-only padding (causal)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))  # left-only pad
        return self.conv(x)


# ── GRN (Global Response Normalization, ConvNeXt v2) ───────────

class GRN(nn.Module):
    """
    Global Response Normalization.
    ConvNeXt v2's replacement for LayerNorm in the MLP.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)  # (B, 1, C)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x


# ── AdaLN-Zero ─────────────────────────────────────────────────

class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with zero-initialized gating.

    cond → MLP → (γ_ln, β_ln, γ_gate, γ_res)
    All gates zero-initialized → identity at init.
    """

    def __init__(self, dim: int, cond_dim: int, mlp_hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, dim * 4),  # (scale, shift, gate, res_scale)
        )
        # Zero-init last layer
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, dim)
            cond: (B, T, cond_dim) or (B, cond_dim)
        Returns:
            (x_normed, gate, res_scale)
        """
        # Expand cond if per-utterance
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, x.size(1), -1)

        params = self.mlp(cond)  # (B, T, dim*4)
        shift, scale, gate, res_scale = params.chunk(4, dim=-1)

        x_normed = self.norm(x)
        x_modulated = x_normed * (1 + scale) + shift

        # gate: sigmoid for stability, zero-init → ~0.5 after sigmoid
        # res_scale: direct multiplier, zero-init → identity
        return x_modulated, gate, res_scale


# ── ConvNeXt v2 Block ──────────────────────────────────────────

class ConvNeXtV2Block(nn.Module):
    """
    ConvNeXt v2 block with GRN + LayerScale + AdaLN-Zero conditioning.

    Structure:
        x → AdaLN(cond) → DWConv7(k, dil, causal) → LayerNorm
          → Conv1x1(dim → 4*dim) → GELU → GRN → Conv1x1(4*dim → dim)
          → LayerScale → gate → + x
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        kernel_size: int = 5,
        dilation: int = 1,
        mlp_expansion: int = 4,
        cond_mlp_hidden: int = 128,
        use_grn: bool = True,
    ):
        super().__init__()
        self.use_grn = use_grn

        self.adaln = AdaLNZero(dim, cond_dim, cond_mlp_hidden)

        # Depthwise conv
        self.dwconv = CausalConv1d(
            dim, dim,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=dim,
        )

        self.norm = nn.LayerNorm(dim)

        # Inverted bottleneck MLP
        hidden = dim * mlp_expansion
        self.pwconv1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        if use_grn:
            self.grn = GRN(hidden)
        self.pwconv2 = nn.Linear(hidden, dim)

        # LayerScale (zero-init → identity)
        self.ls_gamma = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
            cond: (B, T, cond_dim) or (B, cond_dim)
        Returns:
            (B, T, dim)
        """
        # AdaLN-Zero
        x_normed, gate, res_scale = self.adaln(x, cond)

        # ConvNeXt path
        h = x_normed.transpose(1, 2)  # (B, T, dim) → (B, dim, T)
        h = self.dwconv(h)
        h = h.transpose(1, 2)  # → (B, T, dim)

        h = self.norm(h)
        h = self.pwconv1(h)
        h = self.act(h)
        if self.use_grn:
            h = self.grn(h)
        h = self.pwconv2(h)

        # LayerScale + residual
        h = self.ls_gamma * h
        return x + gate.sigmoid() * h * res_scale


# ── Cross-Attention (P-Flow style speaker prompt) ──────────────

class SpeakerPromptCrossAttn(nn.Module):
    """
    Cross-attention to learnable speaker prompt tokens.
    P-Flow style: 4 learnable tokens attend to speaker embedding.
    """

    def __init__(
        self,
        dim: int = 192,
        n_tokens: int = 4,
        speaker_dim: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.speaker_proj = nn.Linear(speaker_dim, dim)

        # Learnable prompt tokens
        self.prompt_tokens = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)

        self.norm_x = nn.LayerNorm(dim)
        self.norm_prompt = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, speaker_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
            speaker_emb: (B, speaker_dim)
        Returns:
            (B, T, dim) attended features
        """
        B, T, D = x.shape

        # Speaker prompt tokens
        spk_bias = self.speaker_proj(speaker_emb)  # (B, dim)
        prompt = self.prompt_tokens.unsqueeze(0) + spk_bias.unsqueeze(1)
        # prompt: (B, n_tokens, dim)

        x_normed = self.norm_x(x)
        prompt_normed = self.norm_prompt(prompt)

        attn_out, _ = self.attn(
            query=x_normed,
            key=prompt_normed,
            value=prompt_normed,
        )
        return x + attn_out


# ── Full Converter ─────────────────────────────────────────────

class CausalConvNeXtConverter(nn.Module):
    """
    HybridVC converter: 10 ConvNeXt v2 blocks + cross-attn + AdaLN-Zero.

    Architecture:
        z_src (B, T, 768) @ 25Hz
          → in_proj Linear(768 → 192)
          → 10 × ConvNeXtV2Block(dim=192) + AdaLN-Zero(cond)
               blocks {3,6,9}: + SpeakerPromptCrossAttn
          → out_proj Linear(192 → 768)
          → out_gate (zero-init tanh gate)
          → z_out = z_src + out_gate * tanh(out_proj(x))
    """

    def __init__(self, cfg: ConverterConfig):
        super().__init__()
        self.cfg = cfg

        # IO
        self.in_proj = nn.Linear(cfg.content_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.content_dim)
        self.out_gate = nn.Parameter(torch.zeros(1))  # zero-init

        # Condition assembly
        cond_in_dim = cfg.speaker_dim + cfg.prosody_dim  # 128 + 3
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in_dim, cfg.cond_mlp_hidden),
            nn.SiLU(),
            nn.Linear(cfg.cond_mlp_hidden, cfg.cond_dim),
        )

        # Blocks
        self.blocks = nn.ModuleList([
            ConvNeXtV2Block(
                dim=cfg.hidden_dim,
                cond_dim=cfg.cond_dim,
                kernel_size=cfg.kernel_size,
                dilation=d,
                mlp_expansion=cfg.mlp_expansion,
                cond_mlp_hidden=cfg.cond_mlp_hidden,
                use_grn=cfg.use_grn,
            )
            for d in cfg.dilations
        ])

        # Cross-attention at specified layers
        self.cross_attns = nn.ModuleDict()
        if cfg.use_cross_attn:
            for layer_idx in cfg.cross_attn_layers:
                self.cross_attns[str(layer_idx)] = SpeakerPromptCrossAttn(
                    dim=cfg.hidden_dim,
                    n_tokens=cfg.n_speaker_prompt_tokens,
                    speaker_dim=cfg.speaker_dim,
                    n_heads=cfg.cross_attn_heads,
                )

    def _assemble_cond(
        self,
        speaker_cond: torch.Tensor,
        prosody_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            speaker_cond: (B, speaker_dim) per-utterance
            prosody_feat: (B, T, prosody_dim) per-frame
        Returns:
            (B, T, cond_dim)
        """
        if prosody_feat is None:
            return self.cond_proj(speaker_cond)  # (B, cond_dim) — per-utterance

        # Expand speaker to match temporal frames
        T = prosody_feat.size(1)
        spk_expanded = speaker_cond.unsqueeze(1).expand(-1, T, -1)
        combined = torch.cat([spk_expanded, prosody_feat], dim=-1)  # (B, T, speaker+prosody)
        return self.cond_proj(combined)  # (B, T, cond_dim)

    def forward(
        self,
        z_src: torch.Tensor,
        speaker_cond: torch.Tensor,
        prosody_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z_src: (B, T, content_dim) source latent @ 25Hz
            speaker_cond: (B, speaker_dim) target speaker condition
            prosody_feat: (B, T, prosody_dim) or None, source prosody
        Returns:
            z_out: (B, T, content_dim) converted latent
        """
        cond = self._assemble_cond(speaker_cond, prosody_feat)

        x = self.in_proj(z_src)  # (B, T, hidden_dim)

        for i, block in enumerate(self.blocks):
            # Per-frame or per-utterance condition
            block_cond = cond if cond.dim() == 3 else cond

            # Cross-attention before block (if configured)
            layer_idx = i + 1  # 1-indexed
            if str(layer_idx) in self.cross_attns:
                x = self.cross_attns[str(layer_idx)](x, speaker_cond)

            x = block(x, block_cond)

        # Output projection with gated residual
        delta = self.out_proj(x)       # (B, T, content_dim)
        delta = torch.tanh(delta)       # bound deltas
        delta = self.out_gate * delta   # zero-init → identity at step 0

        return z_src + delta


# ── Factory ─────────────────────────────────────────────────────

def make_converter(
    content_dim: int = 768,
    speaker_dim: int = 128,
    prosody_dim: int = 3,
    n_blocks: int = 10,
    use_cross_attn: bool = True,
    **kwargs,
) -> CausalConvNeXtConverter:
    """Create a HybridVC converter with sensible defaults."""
    cfg = ConverterConfig(
        content_dim=content_dim,
        speaker_dim=speaker_dim,
        prosody_dim=prosody_dim,
        n_blocks=n_blocks,
        use_cross_attn=use_cross_attn,
        **kwargs,
    )
    return CausalConvNeXtConverter(cfg)

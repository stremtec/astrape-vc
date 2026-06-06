"""
FlowVC Converter — Conditional Flow Matching ODE.

Vector Field Network v_θ(z_t, t, c) predicts velocity for source→target flow.

Architecture:
  12 ConvNeXt v2 blocks (dim=512) + AdaLN-Zero(time, condition)
  + cross-attention to speaker prompt tokens at layers [3,6,9]
  + zero-init output gate → identity at t=0

Inference: Euler or RK4 ODE solver (4-8 steps).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AdaLNZero, ConvNeXtV2Block, GRN
from .config import FlowConverterConfig


# ── Sinusoidal Time Embedding ──────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """Transformer-style sinusoidal position embedding for continuous time."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (B, 1) — continuous time in [0, 1]
        Returns:
            (B, dim) embedding
        """
        t = t.view(-1, 1).float()
        device = t.device

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


import math


# ── Time MLP ───────────────────────────────────────────────────

class TimeMLP(nn.Module):
    """Sinusoidal embedding → MLP → time conditioning."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # Zero-init last layer
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


# ── Cross-Attention to Speaker Prompt ──────────────────────────

class SpeakerCrossAttn(nn.Module):
    """Cross-attention from converter hidden states to speaker prompt tokens."""

    def __init__(self, dim: int = 512, prompt_dim: int = 192, n_heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(prompt_dim)
        self.proj_kv = nn.Linear(prompt_dim, dim * 2)  # K, V
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        # Zero-init output projection
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, x: torch.Tensor, prompt: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
            prompt: (B, n_tokens, prompt_dim)
        Returns:
            (B, T, dim)
        """
        q = self.norm_q(x)
        kv = self.norm_kv(prompt)
        k, v = self.proj_kv(kv).chunk(2, dim=-1)  # each (B, n_tokens, dim)

        attn_out, _ = self.attn(q, k, v)
        return x + self.out_proj(attn_out)


# ── Flow Block (ConvNeXt v2 + AdaLN-Zero + Cross-Attn) ────────

class FlowBlock(nn.Module):
    """Single block of the vector field network."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        kernel_size: int = 7,
        dilation: int = 1,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel_size, dilation=dilation, groups=dim)
        self.adaln = AdaLNZero(dim, cond_dim)

        # MLP (inverted bottleneck + GRN)
        hidden = dim * mlp_expansion
        self.pwconv1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.grn = GRN(hidden)
        self.pwconv2 = nn.Linear(hidden, dim)

        # LayerScale
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)  — hidden state
            cond: (B, T, cond_dim) — time + speaker + prosody
        Returns:
            (B, T, dim)
        """
        # DWConv (channel-first)
        h = x.transpose(1, 2)  # (B, dim, T)
        h = self.dwconv(h)
        h = h.transpose(1, 2)  # (B, T, dim)

        # AdaLN-Zero
        h, gate = self.adaln(h, cond)

        # MLP
        h = self.pwconv1(h)
        h = self.act(h)
        h = h.transpose(1, 2)  # (B, hidden, T) for GRN
        h = self.grn(h)
        h = h.transpose(1, 2)  # (B, T, hidden)
        h = self.pwconv2(h)

        # LayerScale + gate
        h = self.gamma * h * gate.sigmoid()

        return x + h


# Need this import for FlowBlock
from .blocks import CausalConv1d


# ── Vector Field Network ───────────────────────────────────────

class VectorFieldNet(nn.Module):
    """
    v_θ(z_t, t, c) — predicts velocity field for CFM.
    
    12 FlowBlocks with cyclic dilations + cross-attn at layers [3,6,9].
    """

    def __init__(self, cfg: FlowConverterConfig):
        super().__init__()
        self.cfg = cfg

        # Input projection
        self.in_proj = nn.Linear(cfg.latent_dim, cfg.hidden_dim)

        # Time embedding
        self.time_mlp = TimeMLP(cfg.time_dim)

        # Condition projection: speaker(192) + prosody(3) → cond_dim
        cond_in = cfg.speaker_dim + cfg.prosody_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )

        # AdaLN condition = cond_proj(frame) + time_emb(broadcast)
        # Concatenated: (B, T, cond_dim + time_dim)
        adaln_cond_dim = cfg.cond_dim + cfg.time_dim

        # Flow blocks
        self.blocks = nn.ModuleList([
            FlowBlock(
                dim=cfg.hidden_dim,
                cond_dim=adaln_cond_dim,
                kernel_size=cfg.kernel_size,
                dilation=d,
                mlp_expansion=cfg.mlp_expansion,
            )
            for d in cfg.dilations
        ])

        # Cross-attention modules
        self.cross_attns = nn.ModuleDict()
        if cfg.use_cross_attn:
            for layer_idx in cfg.cross_attn_layers:
                self.cross_attns[str(layer_idx)] = SpeakerCrossAttn(
                    dim=cfg.hidden_dim,
                    prompt_dim=cfg.prompt_dim,
                    n_heads=cfg.cross_attn_heads,
                )

        # Output projection
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.out_gate = nn.Parameter(torch.zeros(1))  # zero-init → identity

    def _assemble_cond(
        self,
        t: torch.Tensor,
        speaker_emb: torch.Tensor,
        prosody: torch.Tensor | None,
        T: int,
    ) -> torch.Tensor:
        """
        Build per-frame condition from time, speaker, prosody.
        Returns: (B, T, cond_dim + time_dim)
        """
        B = speaker_emb.size(0)

        # Time embedding → broadcast to all frames
        t_emb = self.time_mlp(t)  # (B, time_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, time_dim)

        # Speaker + prosody per-frame condition
        spk = speaker_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, speaker_dim)
        if prosody is not None:
            # Trim or pad prosody to match T
            if prosody.size(1) != T:
                if prosody.size(1) > T:
                    prosody = prosody[:, :T, :]
                else:
                    prosody = F.pad(prosody, (0, 0, 0, T - prosody.size(1)))
            cond_cat = torch.cat([spk, prosody], dim=-1)  # (B, T, speaker+prosody)
        else:
            cond_cat = spk

        cond = self.cond_proj(cond_cat)  # (B, T, cond_dim)

        return torch.cat([cond, t_emb], dim=-1)  # (B, T, cond_dim + time_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        speaker_emb: torch.Tensor,
        prompt_tokens: torch.Tensor | None = None,
        prosody: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z_t: (B, T, latent_dim) current latent state
            t: (B,) or (B, 1) time in [0, 1]
            speaker_emb: (B, speaker_dim) target speaker
            prompt_tokens: (B, n_tokens, prompt_dim) for cross-attn
            prosody: (B, T_prosody, 3) source prosody
        Returns:
            v: (B, T, latent_dim) velocity field
        """
        B, T_lat, _ = z_t.shape

        # Input projection
        x = self.in_proj(z_t)  # (B, T, 512)

        # Assemble condition
        cond = self._assemble_cond(t, speaker_emb, prosody, T_lat)

        # Flow blocks
        for i, block in enumerate(self.blocks):
            # Cross-attention before block (if configured)
            layer_idx = i + 1
            if str(layer_idx) in self.cross_attns and prompt_tokens is not None:
                x = self.cross_attns[str(layer_idx)](x, prompt_tokens)

            x = block(x, cond)

        # Output
        v = self.out_proj(x)
        v = v * self.out_gate  # zero-init → identity

        return v


# ── ODE Solver ─────────────────────────────────────────────────

def solve_cfm_euler(
    vfn: VectorFieldNet,
    z_src: torch.Tensor,
    speaker_emb: torch.Tensor,
    prompt_tokens: torch.Tensor | None,
    prosody: torch.Tensor | None,
    n_steps: int = 4,
) -> torch.Tensor:
    """
    Solve CFM ODE using Euler method.
    
    z_tgt = z_src + Σ v_θ(z_i, t_i, c) * dt
    
    Args:
        vfn: VectorFieldNet
        z_src: (B, T, latent_dim)
        speaker_emb: (B, speaker_dim)
        prompt_tokens: (B, n_tokens, prompt_dim)
        prosody: (B, T, 3)
        n_steps: number of Euler steps (4 default)
    Returns:
        z_tgt: (B, T, latent_dim)
    """
    z = z_src
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((z_src.size(0),), i * dt, device=z.device)
        v = vfn(z, t, speaker_emb, prompt_tokens, prosody)
        z = z + v * dt

    return z


def solve_cfm_rk4(
    vfn: VectorFieldNet,
    z_src: torch.Tensor,
    speaker_emb: torch.Tensor,
    prompt_tokens: torch.Tensor | None,
    prosody: torch.Tensor | None,
    n_steps: int = 4,
) -> torch.Tensor:
    """
    Solve CFM ODE using RK4 (4th-order Runge-Kutta).
    Higher quality, 2× cost vs Euler.
    """
    z = z_src
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_i = i * dt
        t = torch.full((z_src.size(0),), t_i, device=z.device)
        t_half = torch.full_like(t, t_i + dt / 2)
        t_next = torch.full_like(t, t_i + dt)

        k1 = vfn(z, t, speaker_emb, prompt_tokens, prosody)
        k2 = vfn(z + k1 * dt / 2, t_half, speaker_emb, prompt_tokens, prosody)
        k3 = vfn(z + k2 * dt / 2, t_half, speaker_emb, prompt_tokens, prosody)
        k4 = vfn(z + k3 * dt, t_next, speaker_emb, prompt_tokens, prosody)

        z = z + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    return z


# ── Factory ────────────────────────────────────────────────────

def make_vector_field_net(**kwargs) -> VectorFieldNet:
    cfg = FlowConverterConfig(**kwargs)
    return VectorFieldNet(cfg)

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FalseFutureConfig:
    input_dim: int = 768
    hidden_dim: int = 256
    horizon: int = 16
    history: int = 64
    n_heads: int = 4
    coarse_layers: int = 2
    reverse_layers: int = 2
    summary_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.0
    initial_gate_bias: float = -4.0


class CausalSummaryBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ff_mult: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=5,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.left_context = dilation * 4
        self.channel_mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x).transpose(1, 2)
        convolved = self.depthwise(F.pad(normalized, (self.left_context, 0)))
        mixed = self.channel_mixer(convolved.transpose(1, 2))
        return x + F.dropout(mixed, self.dropout, self.training)


class FalseFutureSlotGenerator(nn.Module):
    """Create pseudo-future slots for every frame without observing its future."""

    def __init__(self, config: FalseFutureConfig):
        super().__init__()
        if config.input_dim <= 0 or config.hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if config.horizon <= 0 or config.history <= 0:
            raise ValueError("horizon and history must be positive")
        if config.hidden_dim % config.n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if config.summary_layers <= 0:
            raise ValueError("summary_layers must be positive")
        self.config = config
        self.stream_history = max(
            config.history,
            config.horizon
            + 4 * sum(2**index for index in range(config.summary_layers)),
        )
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.summary_blocks = nn.ModuleList(
            [
                CausalSummaryBlock(
                    config.hidden_dim,
                    config.ff_mult,
                    dilation=2**index,
                    dropout=config.dropout,
                )
                for index in range(config.summary_layers)
            ]
        )
        self.horizon_embedding = nn.Parameter(
            torch.empty(config.horizon, config.hidden_dim)
        )
        self.slot_mixers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.hidden_dim),
                    nn.Linear(config.hidden_dim, config.hidden_dim * config.ff_mult),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim * config.ff_mult, config.hidden_dim),
                )
                for _ in range(config.coarse_layers)
            ]
        )
        self.reverse_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.n_heads,
                    dim_feedforward=config.hidden_dim * config.ff_mult,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.reverse_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        nn.init.normal_(self.horizon_embedding, std=0.02)

    def _strict_past(self, x: torch.Tensor) -> torch.Tensor:
        horizon = self.config.horizon
        padded = F.pad(x, (0, 0, horizon, 0))
        windows = padded.unfold(1, horizon, 1)[:, : x.shape[1]]
        windows = windows.permute(0, 1, 3, 2).contiguous()
        return windows.flip(2)

    def _reverse_refine(self, slots: torch.Tensor) -> torch.Tensor:
        batch, length, horizon, hidden = slots.shape
        flattened = slots.reshape(batch * length, horizon, hidden).flip(1)
        mask = torch.triu(
            torch.ones(
                horizon,
                horizon,
                device=slots.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        for block in self.reverse_blocks:
            flattened = block(flattened, src_mask=mask)
        return flattened.flip(1).reshape(batch, length, horizon, hidden)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("history must have shape [batch, time, input_dim]")
        if history.shape[1] == 0:
            raise ValueError("history must contain at least one frame")
        if history.shape[2] != self.config.input_dim:
            raise ValueError("history feature dimension does not match input_dim")
        summary = self.input_projection(history)
        for block in self.summary_blocks:
            summary = block(summary)
        folded = self._strict_past(summary)
        slots = (
            folded
            + summary.unsqueeze(2)
            + self.horizon_embedding.view(1, 1, self.config.horizon, -1)
        )
        for mixer in self.slot_mixers:
            slots = slots + mixer(slots)
        slots = self._reverse_refine(slots)
        return self.output_norm(slots)

    def forward_stream(
        self,
        current: torch.Tensor,
        history: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history is None:
            history = current[:, :0]
        joined = torch.cat((history, current), dim=1)
        slots = self(joined)[:, -current.shape[1] :]
        next_history = joined[:, -self.stream_history :]
        return slots, next_history


class FalseFutureLayerAdapter(nn.Module):
    """Read pseudo-future slots and inject a confidence-gated layer correction."""

    def __init__(
        self,
        input_dim: int,
        slot_dim: int,
        initial_gate_bias: float = -4.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.query_norm = nn.LayerNorm(input_dim)
        self.slot_norm = nn.LayerNorm(slot_dim)
        self.query = nn.Linear(input_dim, slot_dim, bias=False)
        self.key = nn.Linear(slot_dim, slot_dim, bias=False)
        self.value = nn.Linear(slot_dim, input_dim, bias=False)
        self.effect_norm = nn.LayerNorm(input_dim)
        self.gate = nn.Linear(input_dim, 1)
        self.benefit = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, initial_gate_bias)
        nn.init.zeros_(self.benefit.weight)
        nn.init.zeros_(self.benefit.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden.shape[:2] != slots.shape[:2]:
            raise ValueError("hidden and false-future slots must share batch/time")
        query = self.query(self.query_norm(hidden)).unsqueeze(2)
        normalized_slots = self.slot_norm(slots)
        keys = self.key(normalized_slots)
        scores = torch.sum(query * keys, dim=-1) * self.slot_dim**-0.5
        weights = torch.softmax(scores, dim=-1)
        values = self.value(normalized_slots)
        effect = torch.sum(weights.unsqueeze(-1) * values, dim=2)
        effect = self.effect_norm(effect)
        gate = torch.sigmoid(self.gate(hidden))
        benefit = self.benefit(hidden).squeeze(-1)
        return effect, gate.squeeze(-1), benefit

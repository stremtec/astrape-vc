"""
Manifold-Aligned Factorized Student (MAFS)

A strictly causal 50 Hz → 25 Hz content student that predicts continuous
5-dimensional FSQ codes and recovers the 768d content embedding through the
teacher's frozen FSQ projection.  The architecture is designed around four
findings from the current plateau:

1. **Effective rank = 5** — teacher content PCA shows 99% variance in 5 dims.
   Predicting 768d directly wastes 763 dimensions on a degenerate solution
   that inflates cosine by ≈0.07 while missing the actual FSQ codes.

2. **End-of-cell decimation** — emitting token k from mel frame 2k+1 (not 2k)
   improves validation cosine and gives each content prediction the complete
   40 ms cell of evidence.  A controlled probe confirmed +0.006 gain.

3. **CTC / content gradient conflict** — placing CTC after the core encoder
   produces gradients nearly orthogonal to the content loss.  Moving CTC to
   the 50 Hz edge branch and detaching its gradient from the 25 Hz core
   eliminates the conflict without losing the CTC regularisation benefit.

4. **Next-token auxiliary training** — predicting five future content states
   (stop-gradient targets) forces the latent representation to encode
   phonetic trajectory without leaking synthetic future into the production
   path (which a controlled probe showed harms hard cases).

The frozen FSQ projection is loaded at build time and never trained.
Content from the model is always ``content = W @ codes + b``, so the
output lives on the teacher manifold by construction.

Target: full-validation 5d-code cosine ≥ 0.92, per-axis accuracy ≥ 0.70.
This corresponds to ≈0.96 projected 768d cosine because the metric
inflation is eliminated by design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CausalConv1d


# ── config ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MafsConfig:
    in_dim: int = 80
    edge_dim: int = 384
    core_dim: int = 512
    edge_layers: int = 4
    core_layers: int = 8
    n_heads: int = 8
    ff_hidden: int = 1536
    edge_kernel: int = 5
    core_kernel: int = 9
    attention_context: int = 50
    dropout: float = 0.0
    content_dim: int = 768
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5)
    text_vocab_size: int = 30
    future_tokens: int = 5


# ── outputs ─────────────────────────────────────────────────────────────────────

@dataclass
class MafsOutput:
    codes: torch.Tensor              # [B, T25, 5] continuous 5d codes
    projected: torch.Tensor          # [B, 768, T25] via frozen W·codes + b
    fsq_logits: tuple[torch.Tensor, ...]  # per-axis ordinal logits
    future_codes: torch.Tensor       # [B, T25, 5, 5]  (5 future tokens × 5 axes)
    text_logits: Optional[torch.Tensor] = None  # [B, T50, vocab]


@dataclass
class MafsStreamingState:
    edge_caches: list[Optional[torch.Tensor]]
    core_conv_caches: list[Optional[torch.Tensor]]
    core_attn_histories: list[Optional[torch.Tensor]]
    recurrent_state: Optional[torch.Tensor] = None
    pending_mel: Optional[torch.Tensor] = None
    mel_position: int = 0


# ── building blocks ─────────────────────────────────────────────────────────────

class CausalEdgeBlock(nn.Module):
    """50 Hz acoustic edge: GLU + depthwise conv + LayerScale."""

    def __init__(self, dim: int, kernel_size: int, dilation: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, dim * 2)
        self.depthwise = CausalConv1d(dim, dim, kernel_size, dilation=dilation, groups=dim)
        self.project = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.full((dim,), 0.1))

    def finish(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.project(F.silu(hidden.transpose(1, 2))) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.glu(self.expand(self.norm(x)), dim=-1)
        hidden = self.depthwise(hidden.transpose(1, 2))
        return x + self.finish(hidden)

    def forward_stream(self, x: torch.Tensor, cache: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.glu(self.expand(self.norm(x)), dim=-1)
        hidden, cache = self.depthwise.forward_stream(hidden.transpose(1, 2), cache)
        return x + self.finish(hidden), cache


class MafsCoreBlock(nn.Module):
    """25 Hz core: depthwise conv (every block) + 50-frame attention (alternating) + SwiGLU FFN."""

    def __init__(self, config: MafsConfig, *, use_attention: bool, dilation: int):
        super().__init__()
        dim = config.core_dim
        self.use_attention = use_attention
        self.attention_context = config.attention_context
        # local conv path
        self.local_norm = nn.LayerNorm(dim)
        self.local_expand = nn.Linear(dim, dim * 2)
        self.local_depthwise = CausalConv1d(dim, dim, config.core_kernel, dilation=dilation, groups=dim)
        self.local_project = nn.Linear(dim, dim)
        # attention path
        if use_attention:
            self.attention_norm = nn.LayerNorm(dim)
            self.attention = nn.MultiheadAttention(dim, config.n_heads, dropout=config.dropout, batch_first=True)
            self.merge_gate = nn.Linear(dim * 2, dim)
        else:
            self.attention_norm = self.attention = self.merge_gate = None
        self.path_scale = nn.Parameter(torch.full((dim,), 1.0))  # ★ 1.0 not 0.1
        # FFN
        self.ffn_norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, config.ff_hidden, bias=False)
        self.w2 = nn.Linear(config.ff_hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, config.ff_hidden, bias=False)
        self.ffn_scale = nn.Parameter(torch.full((dim,), 1.0))  # ★ 1.0 not 0.1
        self.dropout = config.dropout

    def _local(self, x: torch.Tensor) -> torch.Tensor:
        h = F.glu(self.local_expand(self.local_norm(x)), dim=-1)
        h = self.local_depthwise(h.transpose(1, 2)).transpose(1, 2)
        return self.local_project(F.silu(h))

    def _attn_mask(self, length: int, device: torch.device) -> torch.Tensor:
        p = torch.arange(length, device=device)
        m = p.unsqueeze(0) > p.unsqueeze(1)
        m |= p.unsqueeze(0) < (p.unsqueeze(1) - self.attention_context + 1)
        return m

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        local = self._local(x)
        if self.attention is None:
            mixed = local
        else:
            normed = self.attention_norm(x)
            attn = self.attention(normed, normed, normed,
                                  attn_mask=self._attn_mask(x.shape[1], x.device),
                                  key_padding_mask=padding_mask, need_weights=False)[0]
            gate = torch.sigmoid(self.merge_gate(torch.cat((local, attn), dim=-1)))
            mixed = gate * local + (1 - gate) * attn
        x = x + F.dropout(mixed * self.path_scale, self.dropout, self.training)
        h = self.ffn_norm(x)
        return x + F.dropout(self.w2(F.silu(self.w1(h)) * self.w3(h)) * self.ffn_scale,
                             self.dropout, self.training)

    def forward_stream(self, x: torch.Tensor, conv_cache: Optional[torch.Tensor],
                       attn_history: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h = F.glu(self.local_expand(self.local_norm(x)), dim=-1)
        h, conv_cache = self.local_depthwise.forward_stream(h.transpose(1, 2), conv_cache)
        local = self.local_project(F.silu(h.transpose(1, 2)))
        next_history = attn_history
        if self.attention is None:
            mixed = local
        else:
            if attn_history is None:
                attn_history = x[:, :0]
            keys = torch.cat((attn_history, x), dim=1)
            hl = attn_history.shape[1]
            qp = hl + torch.arange(x.shape[1], device=x.device)
            kp = torch.arange(keys.shape[1], device=x.device)
            mask = kp.unsqueeze(0) > qp.unsqueeze(1)
            mask |= kp.unsqueeze(0) < (qp.unsqueeze(1) - self.attention_context + 1)
            attn = self.attention(self.attention_norm(x), self.attention_norm(keys),
                                  self.attention_norm(keys), attn_mask=mask, need_weights=False)[0]
            gate = torch.sigmoid(self.merge_gate(torch.cat((local, attn), dim=-1)))
            mixed = gate * local + (1 - gate) * attn
            next_history = keys[:, -self.attention_context:].detach()
        x = x + mixed * self.path_scale
        x = x + self.w2(F.silu(self.w1(self.ffn_norm(x))) * self.w3(self.ffn_norm(x))) * self.ffn_scale
        return x, conv_cache, next_history


# ── full model ──────────────────────────────────────────────────────────────────

class MafsModel(nn.Module):
    """Causal 50 Hz → 25 Hz content student for factorized 5d codes."""

    def __init__(self, config: MafsConfig):
        super().__init__()
        if config.edge_layers <= 0 or config.core_layers <= 0:
            raise ValueError("edge_layers and core_layers must be positive")
        if config.core_dim % config.n_heads:
            raise ValueError("core_dim must be divisible by n_heads")
        self.config = config

        # --- 50 Hz edge ---
        self.input_norm = nn.LayerNorm(config.in_dim)
        self.input_projection = nn.Linear(config.in_dim, config.edge_dim)
        edge_dilations = (1, 2, 4, 8)
        self.edge_blocks = nn.ModuleList([
            CausalEdgeBlock(config.edge_dim, config.edge_kernel,
                            edge_dilations[i % len(edge_dilations)])
            for i in range(config.edge_layers)
        ])

        # CTC head on 50 Hz edge (gradient-detached from core)
        self.text_head = nn.Linear(config.edge_dim, config.text_vocab_size)

        # --- 25 Hz core ---
        self.core_projection = nn.Linear(config.edge_dim, config.core_dim)
        self.recurrent = nn.GRU(config.core_dim, config.core_dim, num_layers=1, batch_first=True)
        core_dilations = (1, 2, 4, 8)
        self.core_blocks = nn.ModuleList([
            MafsCoreBlock(config, use_attention=(i % 2 == 1),
                          dilation=core_dilations[i % len(core_dilations)])
            for i in range(config.core_layers)
        ])
        self.output_norm = nn.LayerNorm(config.core_dim)

        # --- heads ---
        self.code_head = nn.Linear(config.core_dim, len(config.fsq_levels))           # 5d continuous
        self.ordinal_head = nn.Linear(config.core_dim, sum(config.fsq_levels))        # per-axis logits
        self.future_head = nn.Linear(config.core_dim, config.future_tokens * len(config.fsq_levels))

        # Frozen FSQ projection (loaded separately)
        self.fsq_projection = nn.Linear(len(config.fsq_levels), config.content_dim)
        self.fsq_projection.requires_grad_(False)

    def load_fsq_projection(self, state: dict[str, torch.Tensor]) -> None:
        self.fsq_projection.load_state_dict(state, strict=True)
        self.fsq_projection.requires_grad_(False)

    @staticmethod
    def output_lengths(mel_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(mel_lengths, 2, rounding_mode="floor")

    def _heads(self, hidden: torch.Tensor) -> MafsOutput:
        """hidden: [B, T25, core_dim]"""
        hidden = self.output_norm(hidden)
        codes = self.code_head(hidden)  # [B, T25, 5]
        packed = self.ordinal_head(hidden).transpose(1, 2)  # [B, 34, T25]
        fsq_logits = tuple(torch.split(packed, list(self.config.fsq_levels), dim=1))
        future = self.future_head(hidden).view(
            hidden.shape[0], hidden.shape[1],
            self.config.future_tokens, len(self.config.fsq_levels),
        )
        projected = self.fsq_projection(codes).transpose(1, 2)  # [B, 768, T25]
        return MafsOutput(codes=codes, projected=projected, fsq_logits=fsq_logits, future_codes=future)

    def forward(self, mel: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> MafsOutput:
        # --- 50 Hz edge ---
        h = self.input_projection(self.input_norm(mel.transpose(1, 2)))  # [B, T50, edge_dim]
        for block in self.edge_blocks:
            h = block(h)

        # CTC from edge (detached — no gradient to core)
        text_logits = self.text_head(h.detach() if self.training else h)

        # --- end-of-cell decimation ---
        h = h[:, 1::2].contiguous()  # [B, T25, edge_dim]
        if h.shape[1] == 0:
            empty = h.new_zeros(h.shape[0], 0, len(self.config.fsq_levels))
            return MafsOutput(codes=empty, projected=h.new_zeros(h.shape[0], self.config.content_dim, 0),
                              fsq_logits=tuple(h.new_zeros(h.shape[0], L, 0) for L in self.config.fsq_levels),
                              future_codes=h.new_zeros(h.shape[0], 0, self.config.future_tokens, len(self.config.fsq_levels)),
                              text_logits=text_logits)

        # --- 25 Hz core ---
        h = self.core_projection(h)
        h, _ = self.recurrent(h)
        padding_mask = None
        if lengths is not None:
            out_lens = self.output_lengths(lengths)
            pos = torch.arange(h.shape[1], device=h.device)
            padding_mask = pos.unsqueeze(0) >= out_lens.unsqueeze(1)
        for block in self.core_blocks:
            h = block(h, padding_mask)

        out = self._heads(h)
        out.text_logits = text_logits
        return out

    # --- streaming ---
    def initial_streaming_state(self) -> MafsStreamingState:
        return MafsStreamingState(
            edge_caches=[None] * len(self.edge_blocks),
            core_conv_caches=[None] * len(self.core_blocks),
            core_attn_histories=[None] * len(self.core_blocks),
        )

    @torch.inference_mode()
    def forward_stream(self, mel: torch.Tensor,
                       state: Optional[MafsStreamingState] = None) -> tuple[MafsOutput, MafsStreamingState]:
        if self.training:
            raise RuntimeError("forward_stream requires eval mode")
        state = state or self.initial_streaming_state()

        # Buffer odd/even alignment
        if state.pending_mel is not None:
            mel = torch.cat((state.pending_mel, mel), dim=-1)
        if state.mel_position == 0:
            process = mel.shape[-1] - (mel.shape[-1] % 2) if mel.shape[-1] >= 2 else 0
        else:
            process = mel.shape[-1] - (mel.shape[-1] % 2)
        if process == 0:
            state.pending_mel = mel
            empty = mel.new_zeros(mel.shape[0], 0, len(self.config.fsq_levels))
            return MafsOutput(codes=empty, projected=mel.new_zeros(mel.shape[0], self.config.content_dim, 0),
                              fsq_logits=tuple(mel.new_zeros(mel.shape[0], L, 0) for L in self.config.fsq_levels),
                              future_codes=mel.new_zeros(mel.shape[0], 0, self.config.future_tokens, len(self.config.fsq_levels))), state

        state.pending_mel = mel[:, :, process:]
        mel = mel[:, :, :process]

        h = self.input_projection(self.input_norm(mel.transpose(1, 2)))
        for i, block in enumerate(self.edge_blocks):
            h, state.edge_caches[i] = block.forward_stream(h, state.edge_caches[i])

        pos = state.mel_position + torch.arange(h.shape[1], device=h.device)
        selected = h[:, pos.remainder(2) == 1].contiguous()
        state.mel_position += h.shape[1]

        if selected.shape[1] == 0:
            empty = mel.new_zeros(mel.shape[0], 0, len(self.config.fsq_levels))
            return MafsOutput(codes=empty, projected=mel.new_zeros(mel.shape[0], self.config.content_dim, 0),
                              fsq_logits=tuple(mel.new_zeros(mel.shape[0], L, 0) for L in self.config.fsq_levels),
                              future_codes=mel.new_zeros(mel.shape[0], 0, self.config.future_tokens, len(self.config.fsq_levels))), state

        h = self.core_projection(selected)
        h, state.recurrent_state = self.recurrent(h, state.recurrent_state)
        for i, block in enumerate(self.core_blocks):
            h, state.core_conv_caches[i], state.core_attn_histories[i] = \
                block.forward_stream(h, state.core_conv_caches[i], state.core_attn_histories[i])
        return self._heads(h), state


# ── checkpointing ───────────────────────────────────────────────────────────────

MAFS_CHECKPOINT_VERSION = 1


def save_mafs_checkpoint(path: str | Path, model: MafsModel, *,
                         epoch: int, metrics: dict[str, float],
                         optimizer=None, scheduler=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": MAFS_CHECKPOINT_VERSION,
        "model_type": "mafs",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_mafs_checkpoint(path: str | Path, *, device="cpu") -> tuple[MafsModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    if payload.get("model_type") != "mafs":
        raise ValueError("not a MAFS checkpoint")
    config = MafsConfig(**payload["config"])
    model = MafsModel(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload

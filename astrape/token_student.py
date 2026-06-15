from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CausalConv1d


@dataclass(frozen=True)
class TokenStudentConfig:
    in_dim: int = 80
    edge_dim: int = 384
    core_dim: int = 512
    edge_layers: int = 4
    core_layers: int = 8
    n_heads: int = 8
    ff_hidden: int = 1536
    attention_context: int = 50
    edge_kernel: int = 5
    core_kernel: int = 9
    dropout: float = 0.0
    content_dim: int = 768
    text_vocab_size: int = 30
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5)


@dataclass
class TokenStudentOutput:
    content: torch.Tensor
    codes: torch.Tensor
    fsq_logits: tuple[torch.Tensor, ...]
    future_codes: torch.Tensor
    text_logits: Optional[torch.Tensor] = None


@dataclass
class TokenStreamingState:
    edge_caches: list[Optional[torch.Tensor]]
    core_caches: list[Optional[torch.Tensor]]
    attention_histories: list[Optional[torch.Tensor]]
    recurrent_state: Optional[torch.Tensor] = None
    mel_position: int = 0


class CausalEdgeBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, dim * 2)
        self.depthwise = CausalConv1d(
            dim,
            dim,
            kernel_size,
            dilation=dilation,
            groups=dim,
        )
        self.project = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.full((dim,), 0.1))

    def _finish(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(hidden.transpose(1, 2))
        return self.project(hidden) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.glu(self.expand(self.norm(x)), dim=-1)
        hidden = self.depthwise(hidden.transpose(1, 2))
        return x + self._finish(hidden)

    def forward_stream(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.glu(self.expand(self.norm(x)), dim=-1)
        hidden, cache = self.depthwise.forward_stream(
            hidden.transpose(1, 2),
            cache,
        )
        return x + self._finish(hidden), cache


class TokenDualPathBlock(nn.Module):
    def __init__(
        self,
        config: TokenStudentConfig,
        *,
        use_attention: bool,
        dilation: int,
    ):
        super().__init__()
        dim = config.core_dim
        self.use_attention = use_attention
        self.attention_context = config.attention_context
        self.local_norm = nn.LayerNorm(dim)
        self.local_expand = nn.Linear(dim, dim * 2)
        self.local_depthwise = CausalConv1d(
            dim,
            dim,
            config.core_kernel,
            dilation=dilation,
            groups=dim,
        )
        self.local_project = nn.Linear(dim, dim)
        if use_attention:
            self.attention_norm = nn.LayerNorm(dim)
            self.attention = nn.MultiheadAttention(
                dim,
                config.n_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.merge_gate = nn.Linear(dim * 2, dim)
        else:
            self.attention_norm = None
            self.attention = None
            self.merge_gate = None
        self.path_scale = nn.Parameter(torch.full((dim,), 0.1))
        self.ffn_norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, config.ff_hidden, bias=False)
        self.w2 = nn.Linear(config.ff_hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, config.ff_hidden, bias=False)
        self.ffn_scale = nn.Parameter(torch.full((dim,), 0.1))
        self.dropout = config.dropout

    def _local_full(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.glu(self.local_expand(self.local_norm(x)), dim=-1)
        hidden = self.local_depthwise(hidden.transpose(1, 2)).transpose(1, 2)
        return self.local_project(F.silu(hidden))

    def _attention_mask(self, length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(length, device=device)
        mask = positions.unsqueeze(0) > positions.unsqueeze(1)
        mask |= positions.unsqueeze(0) < (
            positions.unsqueeze(1) - self.attention_context + 1
        )
        return mask

    def _feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.ffn_norm(x)
        return self.w2(F.silu(self.w1(hidden)) * self.w3(hidden))

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        local = self._local_full(x)
        if self.attention is None:
            mixed = local
        else:
            normalized = self.attention_norm(x)
            attended = self.attention(
                normalized,
                normalized,
                normalized,
                attn_mask=self._attention_mask(x.shape[1], x.device),
                key_padding_mask=padding_mask,
                need_weights=False,
            )[0]
            gate = torch.sigmoid(self.merge_gate(torch.cat((local, attended), dim=-1)))
            mixed = gate * local + (1.0 - gate) * attended
        x = x + F.dropout(
            mixed * self.path_scale,
            self.dropout,
            self.training,
        )
        return x + F.dropout(
            self._feed_forward(x) * self.ffn_scale,
            self.dropout,
            self.training,
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        conv_cache: Optional[torch.Tensor],
        attention_history: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        hidden = F.glu(self.local_expand(self.local_norm(x)), dim=-1)
        hidden, conv_cache = self.local_depthwise.forward_stream(
            hidden.transpose(1, 2),
            conv_cache,
        )
        local = self.local_project(F.silu(hidden.transpose(1, 2)))
        next_history = attention_history
        if self.attention is None:
            mixed = local
        else:
            if attention_history is None:
                attention_history = x[:, :0]
            keys = torch.cat((attention_history, x), dim=1)
            history_length = attention_history.shape[1]
            query_positions = history_length + torch.arange(
                x.shape[1],
                device=x.device,
            )
            key_positions = torch.arange(keys.shape[1], device=x.device)
            mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            mask |= key_positions.unsqueeze(0) < (
                query_positions.unsqueeze(1) - self.attention_context + 1
            )
            attended = self.attention(
                self.attention_norm(x),
                self.attention_norm(keys),
                self.attention_norm(keys),
                attn_mask=mask,
                need_weights=False,
            )[0]
            gate = torch.sigmoid(self.merge_gate(torch.cat((local, attended), dim=-1)))
            mixed = gate * local + (1.0 - gate) * attended
            next_history = keys[:, -self.attention_context :].detach()
        x = x + mixed * self.path_scale
        x = x + self._feed_forward(x) * self.ffn_scale
        return x, conv_cache, next_history


class TokenSynchronousStudent(nn.Module):
    """Causal 50 Hz acoustic encoder with end-of-cell 25 Hz content outputs."""

    def __init__(self, config: TokenStudentConfig):
        super().__init__()
        if config.edge_layers <= 0 or config.core_layers <= 0:
            raise ValueError("edge_layers and core_layers must be positive")
        if config.core_dim % config.n_heads:
            raise ValueError("core_dim must be divisible by n_heads")
        if config.attention_context <= 0:
            raise ValueError("attention_context must be positive")
        self.config = config
        self.input_norm = nn.LayerNorm(config.in_dim)
        self.input_projection = nn.Linear(config.in_dim, config.edge_dim)
        edge_dilations = (1, 2, 4, 8)
        self.edge_blocks = nn.ModuleList(
            [
                CausalEdgeBlock(
                    config.edge_dim,
                    config.edge_kernel,
                    edge_dilations[index % len(edge_dilations)],
                )
                for index in range(config.edge_layers)
            ]
        )
        self.text_head = nn.Linear(config.edge_dim, config.text_vocab_size)
        self.core_projection = nn.Linear(config.edge_dim, config.core_dim)
        self.recurrent = nn.GRU(
            config.core_dim,
            config.core_dim,
            num_layers=1,
            batch_first=True,
        )
        core_dilations = (1, 2, 4, 8)
        self.core_blocks = nn.ModuleList(
            [
                TokenDualPathBlock(
                    config,
                    use_attention=index % 2 == 1,
                    dilation=core_dilations[index % len(core_dilations)],
                )
                for index in range(config.core_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.core_dim)
        self.code_head = nn.Linear(config.core_dim, len(config.fsq_levels))
        self.ordinal_head = nn.Linear(config.core_dim, sum(config.fsq_levels))
        self.future_head = nn.Linear(
            config.core_dim,
            5 * len(config.fsq_levels),
        )
        self.fsq_projection = nn.Linear(
            len(config.fsq_levels),
            config.content_dim,
        )
        self.fsq_projection.requires_grad_(False)

    @staticmethod
    def output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(input_lengths, 2, rounding_mode="floor")

    def load_fsq_projection(self, state: dict[str, torch.Tensor]) -> None:
        self.fsq_projection.load_state_dict(state, strict=True)
        self.fsq_projection.requires_grad_(False)

    def _heads(
        self,
        hidden: torch.Tensor,
        text_logits: Optional[torch.Tensor],
    ) -> TokenStudentOutput:
        hidden = self.output_norm(hidden)
        codes = self.code_head(hidden)
        packed = self.ordinal_head(hidden).transpose(1, 2)
        fsq_logits = torch.split(packed, self.config.fsq_levels, dim=1)
        future_codes = self.future_head(hidden).view(
            hidden.shape[0],
            hidden.shape[1],
            5,
            len(self.config.fsq_levels),
        )
        content = self.fsq_projection(codes).transpose(1, 2)
        return TokenStudentOutput(
            content=content,
            codes=codes,
            fsq_logits=fsq_logits,
            future_codes=future_codes,
            text_logits=text_logits,
        )

    def forward(
        self,
        mel: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> TokenStudentOutput:
        hidden = self.input_projection(
            self.input_norm(mel.transpose(1, 2))
        )
        for block in self.edge_blocks:
            hidden = block(hidden)
        text_logits = self.text_head(hidden)
        hidden = hidden[:, 1::2].contiguous()
        hidden = self.core_projection(hidden)
        if hidden.shape[1] == 0:
            return self._heads(hidden, text_logits)
        hidden, _ = self.recurrent(hidden)
        padding_mask = None
        if lengths is not None:
            output_lengths = self.output_lengths(lengths)
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            padding_mask = positions.unsqueeze(0) >= output_lengths.unsqueeze(1)
        for block in self.core_blocks:
            hidden = block(hidden, padding_mask)
        return self._heads(hidden, text_logits)

    def initial_streaming_state(self) -> TokenStreamingState:
        return TokenStreamingState(
            edge_caches=[None] * len(self.edge_blocks),
            core_caches=[None] * len(self.core_blocks),
            attention_histories=[None] * len(self.core_blocks),
        )

    @torch.inference_mode()
    def forward_stream(
        self,
        mel: torch.Tensor,
        state: Optional[TokenStreamingState] = None,
    ) -> tuple[TokenStudentOutput, TokenStreamingState]:
        if self.training:
            raise RuntimeError("forward_stream requires model.eval()")
        state = state or self.initial_streaming_state()
        hidden = self.input_projection(
            self.input_norm(mel.transpose(1, 2))
        )
        for index, block in enumerate(self.edge_blocks):
            hidden, state.edge_caches[index] = block.forward_stream(
                hidden,
                state.edge_caches[index],
            )
        positions = state.mel_position + torch.arange(
            hidden.shape[1],
            device=hidden.device,
        )
        selected = hidden[:, positions.remainder(2) == 1].contiguous()
        state.mel_position += hidden.shape[1]
        if selected.shape[1] == 0:
            empty_codes = hidden.new_empty(
                hidden.shape[0],
                0,
                len(self.config.fsq_levels),
            )
            empty_content = hidden.new_empty(
                hidden.shape[0],
                self.config.content_dim,
                0,
            )
            empty_logits = tuple(
                hidden.new_empty(hidden.shape[0], level, 0)
                for level in self.config.fsq_levels
            )
            empty_future = hidden.new_empty(
                hidden.shape[0],
                0,
                5,
                len(self.config.fsq_levels),
            )
            return (
                TokenStudentOutput(
                    content=empty_content,
                    codes=empty_codes,
                    fsq_logits=empty_logits,
                    future_codes=empty_future,
                    text_logits=None,
                ),
                state,
            )
        hidden = self.core_projection(selected)
        hidden, state.recurrent_state = self.recurrent(
            hidden,
            state.recurrent_state,
        )
        for index, block in enumerate(self.core_blocks):
            (
                hidden,
                state.core_caches[index],
                state.attention_histories[index],
            ) = block.forward_stream(
                hidden,
                state.core_caches[index],
                state.attention_histories[index],
            )
        return self._heads(hidden, None), state


TOKEN_CHECKPOINT_VERSION = 1


def save_token_checkpoint(
    path: str | Path,
    model: TokenSynchronousStudent,
    *,
    epoch: int,
    metrics: dict[str, float],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": TOKEN_CHECKPOINT_VERSION,
        "model_type": "token_synchronous_student",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_token_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[TokenSynchronousStudent, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    if payload.get("model_type") != "token_synchronous_student":
        raise ValueError("Not a token-synchronous student checkpoint")
    config = TokenStudentConfig(**payload["config"])
    model = TokenSynchronousStudent(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload

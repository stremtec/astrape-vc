"""
FlowVC 話者エンコーダ + P-Flow プロンプトトークン。

参照音声(1~3秒)から話者identityを抽出。
ConvNeXt v2 + アテンションプーリング + 学習可能プロンプトトークン。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CausalConv1d, ConvNeXtV2Block
from .config import SpeakerEncoderConfig


class AttentionPooling(nn.Module):
    """学習可能クエリによるマルチヘッドアテンションプーリング。"""

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim) 特徴量
        Returns:
            (B, dim) プール済みベクトル
        """
        B = x.size(0)
        q = self.query.expand(B, -1, -1)  # (B, 1, dim)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)  # (B, dim)


class PromptTokenGenerator(nn.Module):
    """話者埋め込みからP-Flowプロンプトトークンを生成。"""

    def __init__(self, speaker_dim: int = 192, n_tokens: int = 4, dim: int = 192):
        super().__init__()
        self.n_tokens = n_tokens
        # 学習可能ベーストークン
        self.base_tokens = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
        # 話者埋め込み → トークンごとのバイアス
        self.speaker_proj = nn.Linear(speaker_dim, n_tokens * dim)
        # 残差MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, speaker_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            speaker_emb: (B, speaker_dim)
        Returns:
            (B, n_tokens, dim) プロンプトトークン
        """
        B = speaker_emb.size(0)

        # ベーストークン + 話者バイアス
        bias = self.speaker_proj(speaker_emb).view(B, self.n_tokens, -1)
        tokens = self.base_tokens.unsqueeze(0) + bias

        # 残差MLP精緻化
        tokens = tokens + self.mlp(tokens)

        return tokens


class SpeakerEncoder(nn.Module):
    """
    話者エンコーダ: 参照音声 → 話者埋め込み + プロンプトトークン。

    アーキテクチャ:
      参照波形 (44.1kHz) → 6段 ConvNeXt v2
      → アテンションプーリング → spk_emb (192)
      → プロンプトトークン生成 → prompt (4, 192)
    """

    def __init__(self, cfg: SpeakerEncoderConfig):
        super().__init__()
        self.cfg = cfg

        in_ch = 1
        stages = []
        for out_ch, stride in zip(cfg.stages, cfg.strides):
            stages.append(
                CausalConv1d(in_ch, out_ch, kernel_size=stride * 3, stride=stride)
            )
            for _ in range(cfg.blocks_per_stage):
                stages.append(
                    ConvNeXtV2Block(out_ch, kernel_size=cfg.kernel_size)
                )
            in_ch = out_ch

        self.stages = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(cfg.stages[-1])

        # アテンションプーリング
        self.pool = AttentionPooling(cfg.stages[-1], cfg.attn_pool_heads)

        # spk_emb 射影
        self.spk_proj = nn.Linear(cfg.stages[-1], cfg.speaker_dim)

        # プロンプトトークン生成
        self.prompt_gen = PromptTokenGenerator(
            speaker_dim=cfg.speaker_dim,
            n_tokens=cfg.prompt_tokens,
            dim=cfg.speaker_dim,
        )

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wav: (B, 1, T_ref) 参照波形 @ 44.1kHz
        Returns:
            spk_emb: (B, speaker_dim) 話者埋め込み
            prompt_tokens: (B, n_tokens, dim) プロンプトトークン
        """
        x = self.stages(wav)  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)

        # プーリング
        pooled = self.pool(x)  # (B, C)
        spk_emb = self.spk_proj(pooled)  # (B, speaker_dim)

        # プロンプトトークン
        prompt = self.prompt_gen(spk_emb)  # (B, n_tokens, dim)

        return spk_emb, prompt

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """推論用: 話者埋め込みのみ返す。"""
        spk_emb, _ = self.forward(wav)
        return spk_emb


def make_speaker_encoder(**kwargs) -> SpeakerEncoder:
    cfg = SpeakerEncoderConfig(**kwargs)
    return SpeakerEncoder(cfg)

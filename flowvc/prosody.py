"""
FlowVC 韻律抽出器。

ソース音声からフレーム単位の韻律特徴を抽出:
  - log_f0: 対数基本周波数
  - voiced: 有声/無声フラグ
  - log_energy: 対数RMSエネルギー

軽量ConvNet → 適応的プーリング → 25Hzフレームレート。
btrv3lite f0.py の RMVPE/PENN 抽出器を再利用可能。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CausalConv1d


class ProsodyExtractor(nn.Module):
    """
    軽量韻律抽出器。

    アーキテクチャ:
      波形 (44.1kHz) → 4層 Causal ConvNet
      → AdaptiveAvgPool1d → (T_lat, 3)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        kernel_size: int = 15,
        output_dim: int = 3,
        hop_samples: int = 1764,  # 25Hz @ 44.1kHz
    ):
        super().__init__()
        self.hop_samples = hop_samples

        self.conv1 = CausalConv1d(1, 32, kernel_size=kernel_size)
        self.conv2 = CausalConv1d(32, 64, kernel_size=kernel_size)
        self.conv3 = CausalConv1d(64, hidden_dim, kernel_size=kernel_size)
        self.conv4 = CausalConv1d(hidden_dim, output_dim, kernel_size=kernel_size)

        self.act = nn.GELU()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: (B, 1, T_audio) 波形 @ 44.1kHz
        Returns:
            prosody: (B, T_lat, 3)
              [:, :, 0]: log_f0 (logスケール基本周波数)
              [:, :, 1]: voiced (有声確率, sigmoid)
              [:, :, 2]: log_energy (logスケールRMSエネルギー)
        """
        x = self.act(self.conv1(wav))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.conv4(x)  # (B, 3, T_audio)

        # 適応的プーリングで25Hzにダウンサンプル
        T_lat = max(1, wav.shape[2] // self.hop_samples)
        x = F.adaptive_avg_pool1d(x, T_lat)  # (B, 3, T_lat)

        # 転置して (B, T_lat, 3)
        x = x.transpose(1, 2)

        # 活性化
        f0 = x[:, :, 0:1]  # log_f0: そのまま
        voiced = x[:, :, 1:2].sigmoid()  # 有声確率
        energy = x[:, :, 2:3]  # log_energy: そのまま

        return torch.cat([f0, voiced, energy], dim=-1)


def make_prosody_extractor(**kwargs) -> ProsodyExtractor:
    return ProsodyExtractor(**kwargs)

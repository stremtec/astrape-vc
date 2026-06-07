"""
AudioDec Continuous VC: Splitter + Converter for cross-text voice conversion.

Key advantage over Mimi: continuous 64-dim latent → no discrete memorization.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentExtractor(nn.Module):
    """Speaker-neutral content from AudioDec latent (bottleneck: 64→16→64)."""
    def __init__(self, dim=64, bottleneck=16):
        super().__init__()
        self.compress = nn.Conv1d(dim, bottleneck, 1)
        self.expand = nn.Conv1d(bottleneck, dim, 1)

    def forward(self, z):
        """z: (B, T, D) or (B, D, T) → (B, D, T)"""
        if z.dim() == 3 and z.shape[-1] == 64:  # (B, T, D)
            z = z.transpose(1, 2)  # → (B, D, T)
        h = self.compress(z)
        h = F.gelu(h)
        h = self.expand(h)
        return z + h  # residual


class SpeakerExtractor(nn.Module):
    """Global speaker embedding from AudioDec latent."""
    def __init__(self, dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden, 5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z):
        """z: (B, T, D) → (B, D)"""
        if z.dim() == 3 and z.shape[-1] == 64:
            z = z.transpose(1, 2)
        return self.net(z)


class Converter(nn.Module):
    """Content + target speaker → VC latent (FiLM conditioning)."""
    def __init__(self, dim=64):
        super().__init__()
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        self.refine = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=1),
        )

    def forward(self, c_src, s_tgt):
        """
        c_src: (B, D, T) clean content
        s_tgt: (B, D) target speaker
        → z_vc: (B, D, T)
        """
        gamma = self.gamma(s_tgt).unsqueeze(-1)
        beta = self.beta(s_tgt).unsqueeze(-1)
        mean = c_src.mean(dim=2, keepdim=True)
        std = c_src.std(dim=2, keepdim=True) + 1e-5
        c_norm = (c_src - mean) / std
        c_mod = c_norm * gamma + beta
        return c_src + self.refine(c_mod)


class AudioDecVC(nn.Module):
    """Full AudioDec VC: splitter + converter + decoder."""

    def __init__(self, codec):
        super().__init__()
        self.codec = codec
        for p in codec.parameters():
            p.requires_grad_(False)

        self.content_ext = ContentExtractor(dim=64, bottleneck=16)
        self.speaker_ext = SpeakerExtractor(dim=64, hidden=128)
        self.converter = Converter(dim=64)

    def encode(self, audio):
        """AudioDec encode → (T, 64) continuous latent."""
        with torch.no_grad():
            return self.codec.encode(audio.squeeze())  # (T, 64)

    def decode(self, z):
        """(B, T, D) or (T, D) → audio."""
        if z.dim() == 3:
            z = z.squeeze(0)
        with torch.no_grad():
            return self.codec.decode(z)

    def forward(self, z_src, s_tgt):
        """Convert: src latent + target speaker → VC latent."""
        c_src = self.content_ext(z_src)  # (B, D, T)
        return self.converter(c_src, s_tgt)

    @torch.no_grad()
    def convert(self, src_audio, s_tgt):
        """Audio → Audio VC."""
        z_src = self.encode(src_audio)  # (T, 64)
        z_src_b = z_src.unsqueeze(0)  # (1, T, 64)
        z_vc = self.forward(z_src_b, s_tgt.unsqueeze(0))  # (1, D, T)
        # Convert back: (1, D, T) → (T, D)
        z_vc_2d = z_vc.squeeze(0).transpose(0, 1)  # (T, 64)
        return self.decode(z_vc_2d)

    def training_step(self, z_src, z_tgt, s_tgt):
        """Single training step: MSE on latent + content consistency.
        z_src, z_tgt: (B, T, D) or (B, D, T) — handles both.
        """
        # Normalize to (B, D, T) for internal processing
        if z_src.shape[-1] == 64:  # (B, T, D) → (B, D, T)
            z_src = z_src.transpose(1, 2)
            z_tgt = z_tgt.transpose(1, 2)

        c_src = self.content_ext(z_src)
        c_tgt = self.content_ext(z_tgt)

        # Content: same text → similar content
        T = min(c_src.shape[2], c_tgt.shape[2])
        loss_content = F.mse_loss(c_src[:, :, :T], c_tgt[:, :, :T])

        # Converter: c_src + s_tgt → should match z_tgt
        z_vc = self.converter(c_src[:, :, :T], s_tgt)
        T_z = min(z_vc.shape[2], z_tgt.shape[2])
        loss_convert = F.mse_loss(z_vc[:, :, :T_z], z_tgt[:, :, :T_z])

        # Speaker consistency
        s_vc = self.speaker_ext(z_vc)
        loss_spk = F.mse_loss(s_vc, s_tgt)

        return loss_convert + 0.3 * loss_content + 0.1 * loss_spk

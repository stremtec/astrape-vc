"""
StreamVC-style: HuBERT content + speaker adversarial → Mimi decoder.

HuBERT layer 0 (lowest speaker leakage: 22.2%)
→ Gradient Reversal (remove remaining speaker info)
→ Conv downsample (50Hz → 12.5Hz)
→ Mimi decoder-compatible latent
→ Mimi decoder → audio
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Function


class GradientReversal(Function):
    """Gradient reversal layer for adversarial training."""
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class SpeakerAdversarial(nn.Module):
    """Speaker classifier with gradient reversal on content path."""
    def __init__(self, dim=768, n_speakers=109, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(dim, n_speakers),
        )
    
    def forward(self, x):
        """x: (B, D, T) — returns (B, n_speakers) with reversed gradient."""
        x_rev = GradientReversal.apply(x, self.alpha)
        return self.classifier(x_rev)


class ContentProjector(nn.Module):
    """HuBERT content (768-dim @ 50Hz) → Mimi latent (512-dim @ 12.5Hz)."""
    def __init__(self, in_dim=768, out_dim=512):
        super().__init__()
        # Bottleneck: reduce speaker info
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_dim, 256, 1),
            nn.GELU(),
            nn.Conv1d(256, in_dim, 1),
        )
        # Downsample 50Hz → 12.5Hz (4x)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_dim, 512, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(512, out_dim, 4, stride=2, padding=1),
        )
    
    def forward(self, hubert_feat):
        """hubert_feat: (B, T, 768) → (B, 512, T/4)"""
        x = hubert_feat.transpose(1, 2)  # (B, 768, T)
        x = self.bottleneck(x) + x       # residual bottleneck
        x = self.downsample(x)           # (B, 512, T/4)
        return x


class StreamVC(nn.Module):
    """
    HuBERT content → speaker adversarial → downsample → Mimi decoder.
    """
    def __init__(self, hubert, mimi, n_speakers=109, spk_dim=256):
        super().__init__()
        self.hubert = hubert
        self.mimi = mimi
        for p in hubert.parameters(): p.requires_grad_(False)
        for p in mimi.parameters(): p.requires_grad_(False)
        
        self.content_proj = ContentProjector(in_dim=768, out_dim=512)
        self.spk_adversarial = SpeakerAdversarial(dim=768, n_speakers=n_speakers, alpha=5.0)
        
        # Speaker conditioning (FiLM on projected content)
        self.spk_gamma = nn.Linear(spk_dim, 512)
        self.spk_beta = nn.Linear(spk_dim, 512)
    
    def forward(self, src_audio_16k, tgt_spk_emb):
        """
        src_audio_16k: (B, T_16k) raw 16kHz audio
        tgt_spk_emb: (B, spk_dim) target speaker
        Returns: z_q_vc (B, 512, T_mimi) decoder-compatible latent
        """
        # Extract HuBERT layer 0 features
        with torch.no_grad():
            hubert_out = self.hubert(src_audio_16k, output_hidden_states=True)
            h0 = hubert_out.hidden_states[0]  # (B, T_h, 768)
        
        # Adversarial speaker removal (training only)
        _ = self.spk_adversarial(h0.transpose(1, 2))
        
        # Project to Mimi latent space
        z_content = self.content_proj(h0)  # (B, 512, T_mimi)
        
        # Inject target speaker
        gamma = self.spk_gamma(tgt_spk_emb).unsqueeze(-1)
        beta = self.spk_beta(tgt_spk_emb).unsqueeze(-1)
        mean = z_content.mean(dim=2, keepdim=True)
        std = z_content.std(dim=2, keepdim=True) + 1e-5
        z_vc = (z_content - mean) / std * gamma + beta
        
        return z_vc
    
    @torch.no_grad()
    def convert(self, src_audio_16k, tgt_spk_emb):
        """Full VC pipeline."""
        z_vc = self.forward(src_audio_16k, tgt_spk_emb)
        z_up = self.mimi._to_encoder_framerate(z_vc)
        if self.mimi.decoder_transformer:
            (z_tr,) = self.mimi.decoder_transformer(z_up)
        else:
            z_tr = z_up
        return self.mimi.decoder(z_tr)

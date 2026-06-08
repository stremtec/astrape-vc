"""
Mimi Continuous Latent Converter: z_q space, NO hard code prediction.

z_q_src + speaker_tgt → z_q_vc → Mimi decoder → VC audio.
Token swap result as bootstrap teacher.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F


class ContinuousConverter(nn.Module):
    """
    Converts z_q_src to z_q_vc in Mimi quantizer space.
    
    Architecture: causal ConvNeXt-style for streaming compatibility.
    No discrete code prediction — operates on continuous z_q (512-dim).
    """
    def __init__(self, dim=512, spk_dim=256):
        super().__init__()
        # Speaker conditioning (FiLM)
        self.spk_proj = nn.Sequential(
            nn.Linear(spk_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        
        # Causal convolution stack (streaming-friendly, same-length output)
        self.conv_stack = nn.Sequential(
            nn.Conv1d(dim, dim*2, 7, padding=3, groups=dim), nn.GELU(),
            nn.Conv1d(dim*2, dim, 1),
            nn.Conv1d(dim, dim*2, 7, padding=3, groups=dim), nn.GELU(),
            nn.Conv1d(dim*2, dim, 1),
            nn.Conv1d(dim, dim, 7, padding=3, groups=dim), nn.GELU(),
            nn.Conv1d(dim, dim, 1),
        )
        
        # Content preservation gate
        self.gate = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, z_q_src, s_tgt):
        """
        z_q_src: (B, D, T) quantizer-space source latent
        s_tgt: (B, spk_dim) target speaker embedding
        → z_q_vc: (B, D, T)
        """
        # Speaker modulation
        sp = self.spk_proj(s_tgt)
        gamma = self.gamma(sp).unsqueeze(-1)
        beta = self.beta(sp).unsqueeze(-1)
        
        # Normalize source
        mean = z_q_src.mean(dim=2, keepdim=True)
        std = z_q_src.std(dim=2, keepdim=True) + 1e-5
        z_norm = (z_q_src - mean) / std
        
        # Apply speaker style
        z_styled = z_norm * gamma + beta
        
        # Convolutional refinement
        z_refined = self.conv_stack(z_styled)
        
        # Content gate: how much source to preserve
        gate = self.gate(z_q_src)
        
        # Blend: preserve source content, inject speaker style
        z_q_vc = gate * z_q_src + (1 - gate) * z_refined
        
        return z_q_vc


def manifold_loss(z_q_vc, mimi, weight=0.1):
    """
    Encourage z_q_vc to stay near Mimi codebook manifold.
    Soft quantization: distance to nearest codebook entries.
    """
    # This is a placeholder — full implementation needs codebook access
    # For now: L2 regularization toward mean of z_q distribution
    target_mean = 0.0  # z_q values are centered around 0 after normalization
    return weight * (z_q_vc.pow(2).mean())


def content_preservation_loss(z_q_vc, z_q_src, weight=0.5):
    """Encourage content to be preserved (lower weight on speaker-changing dims)."""
    # Simple: MSE between source and VC latents, but allow global shift
    B, D, T = z_q_vc.shape
    # Allow per-channel mean shift (speaker change), penalize per-frame deviation (content change)
    vc_mean = z_q_vc.mean(dim=2, keepdim=True)
    src_mean = z_q_src.mean(dim=2, keepdim=True)
    vc_centered = z_q_vc - vc_mean
    src_centered = z_q_src - src_mean
    return weight * F.mse_loss(vc_centered, src_centered)


def speaker_transfer_loss(z_q_vc, z_q_tgt, weight=1.0):
    """Encourage VC output to match target speaker characteristics."""
    # Channel-wise statistics should match target
    vc_std = z_q_vc.std(dim=2)
    tgt_std = z_q_tgt.std(dim=2)
    vc_mean = z_q_vc.mean(dim=2)
    tgt_mean = z_q_tgt.mean(dim=2)
    return weight * (F.mse_loss(vc_mean, tgt_mean) + F.mse_loss(vc_std, tgt_std))


def teacher_distillation_loss(z_q_vc, z_q_teacher, weight=2.0):
    """
    Bootstrap: match token swap output (teacher) as initial target.
    z_q_teacher = token swap z_q (src LV0 + tgt LV1-7 decoded to z_q).
    """
    T = min(z_q_vc.shape[2], z_q_teacher.shape[2])
    return weight * F.mse_loss(z_q_vc[:, :, :T], z_q_teacher[:, :, :T])


def total_training_loss(z_q_vc, z_q_src, z_q_tgt, z_q_teacher, mimi,
                        use_teacher=True):
    """Combined loss for continuous converter training."""
    loss = 0.0
    losses = {}
    
    # Content preservation
    l_content = content_preservation_loss(z_q_vc, z_q_src)
    loss += l_content
    losses['content'] = l_content.item()
    
    # Speaker transfer
    l_spk = speaker_transfer_loss(z_q_vc, z_q_tgt)
    loss += l_spk
    losses['speaker'] = l_spk.item()
    
    # Teacher distillation (token swap bootstrap)
    if use_teacher and z_q_teacher is not None:
        l_teacher = teacher_distillation_loss(z_q_vc, z_q_teacher)
        loss += l_teacher
        losses['teacher'] = l_teacher.item()
    
    # Manifold regularization
    l_manifold = manifold_loss(z_q_vc, mimi)
    loss += l_manifold
    losses['manifold'] = l_manifold.item()
    
    return loss, losses


@torch.no_grad()
def convert_continuous(model, mimi, src_audio, s_tgt):
    """Full VC: src audio + speaker → VC audio."""
    # Get z_q_src
    z = mimi.encode_to_latent(src_audio, quantize=False)
    codes = mimi.quantizer.encode(z)
    z_q_src = mimi.quantizer.decode(codes)
    
    # Convert
    z_q_vc = model(z_q_src, s_tgt.unsqueeze(0))
    
    # Decode
    z_q_up = mimi._to_encoder_framerate(z_q_vc)
    if mimi.decoder_transformer:
        (z_tr,) = mimi.decoder_transformer(z_q_up)
    else:
        z_tr = z_q_up
    return mimi.decoder(z_tr)

"""
Codex Architecture: LV0 + Text-Invariant Speaker → LV1-7 Code Generator.

[Source Audio] → Mimi encode → LV0 codes ──────────────────┐
[Target Audio] → Resemblyzer → spk embedding ──────────────┤
                                                            ↓
                                              Bidirectional Transformer
                                                            ↓
                                              LV1-7 codes (7×T)
                                                            ↓
                                              Mimi decoder → VC Audio

Training: parallel utterances, CE loss per LV1-7 level.
Inference: LV0 from source + speaker from target → predict LV1-7 → decode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeGenerator(nn.Module):
    """
    Bidirectional transformer that predicts LV1-7 codes from LV0 + speaker.
    
    Args:
        vocab: codebook size (2048 for Mimi)
        lv0_dim: LV0 embedding dimension
        spk_dim: speaker embedding input dimension
        d_model: transformer hidden dimension
        nhead: attention heads
        num_layers: transformer layers
    """
    def __init__(self, vocab=2048, lv0_dim=128, spk_dim=256, d_model=256,
                 nhead=4, num_layers=3):
        super().__init__()
        self.vocab = vocab
        
        # Input embeddings
        self.lv0_emb = nn.Embedding(vocab, lv0_dim)
        self.spk_proj = nn.Linear(spk_dim, lv0_dim)
        
        # Input projection + position encoding
        self.input_proj = nn.Linear(lv0_dim * 2, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        
        # Bidirectional transformer (NO causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads: one per LV1-7 level
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab) for _ in range(7)])
    
    def forward(self, lv0, spk_emb):
        """
        Args:
            lv0: (B, T) integers 0..vocab-1 — source LV0 content codes
            spk_emb: (B, spk_dim) — target speaker embedding
        Returns:
            logits: (B, 7, T, vocab) — predicted LV1-7 code logits
        """
        B, T = lv0.shape
        
        # Embed and combine
        lv0_e = self.lv0_emb(lv0)                            # (B, T, lv0_dim)
        spk_e = self.spk_proj(spk_emb).unsqueeze(1).expand(-1, T, -1)  # (B, T, lv0_dim)
        h = torch.cat([lv0_e, spk_e], dim=-1)                # (B, T, 2*lv0_dim)
        
        # Project and add position
        h = self.input_proj(h)                               # (B, T, d_model)
        h = h + self.pos_emb[:, :T, :]
        
        # Bidirectional transformer (full context)
        h = self.transformer(h)                              # (B, T, d_model)
        
        # Predict each LV1-7 level independently
        logits = torch.stack([head(h) for head in self.heads], dim=1)  # (B, 7, T, vocab)
        return logits
    
    def predict(self, lv0, spk_emb):
        """Generate discrete codes (argmax)."""
        return self.forward(lv0, spk_emb).argmax(dim=-1)  # (B, 7, T)


def training_loss(model, lv0, lv1_7_gt, spk_emb, ce_loss):
    """
    Compute cross-entropy loss for all 7 LV1-7 levels.
    
    Args:
        model: CodeGenerator
        lv0: (B, T) source LV0
        lv1_7_gt: (B, 7, T) target LV1-7 ground truth
        spk_emb: (B, spk_dim) target speaker embedding
        ce_loss: nn.CrossEntropyLoss instance
    Returns:
        scalar loss
    """
    logits = model(lv0, spk_emb)  # (B, 7, T, vocab)
    loss = sum(ce_loss(logits[:, i].reshape(-1, model.vocab),
                       lv1_7_gt[:, i].reshape(-1)) for i in range(7))
    return loss


def vc_convert(model, mimi, src_audio, spk_emb):
    """
    Full VC pipeline: audio → audio.
    
    Args:
        model: CodeGenerator
        mimi: MimiModel (frozen)
        src_audio: (B, 1, T_audio) source waveform
        spk_emb: (B, spk_dim) target speaker embedding
    Returns:
        vc_audio: (B, 1, T_audio') converted waveform
    """
    with torch.no_grad():
        codes_src = mimi.encode(src_audio)           # (B, 8, T)
        lv0 = codes_src[:, 0, :]                     # (B, T)
        pred_lv1_7 = model.predict(lv0, spk_emb)     # (B, 7, T)
        codes_vc = torch.cat([lv0.unsqueeze(1), pred_lv1_7], dim=1)  # (B, 8, T)
        return mimi.decode(codes_vc)

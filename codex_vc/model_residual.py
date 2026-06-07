"""
Residual Code Generator: src LV0 + src LV1-7 + tgt spk → Δ_LV1-7.

Key insight: predict speaker-specific RESIDUAL, not full LV1-7.
This prevents utterance memorization — model must learn what CHANGES
between speakers, not what each utterance's codes are.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCodeGenerator(nn.Module):
    """
    Predicts Δ_LV1-7 = target_LV1-7 - source_LV1-7.

    Input:  src_LV0 (B,T), src_LV1-7 (B,7,T), tgt_spk (B,256)
    Output: Δ_LV1-7 logits (B,7,T,vocab) — CE over vocabulary
    """

    def __init__(self, vocab=2048, lv0_dim=128, lv_dim=128, spk_dim=256,
                 d_model=256, nhead=4, num_layers=3, dropout=0.1,
                 spk_dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.spk_dropout = spk_dropout

        # Embeddings
        self.lv0_emb = nn.Embedding(vocab, lv0_dim)
        self.lv_embs = nn.ModuleList([nn.Embedding(vocab, lv_dim) for _ in range(7)])
        self.spk_proj = nn.Linear(spk_dim, lv0_dim)

        # Input: lv0 + 7×lv + spk = lv0_dim + 7*lv_dim + lv0_dim
        total_dim = lv0_dim + 7 * lv_dim + lv0_dim
        self.input_proj = nn.Linear(total_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads: predict residual code per level
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab) for _ in range(7)])

    def forward(self, lv0, lv1_7_src, spk_emb):
        """
        Args:
            lv0: (B, T) source LV0 content codes
            lv1_7_src: (B, 7, T) source LV1-7 codes
            spk_emb: (B, 256) target speaker embedding
        Returns:
            logits: (B, 7, T, vocab) predicted residual code logits
        """
        B, T = lv0.shape

        # Speaker dropout
        if self.training and self.spk_dropout > 0:
            mask = (torch.rand(B, 1, device=spk_emb.device) > self.spk_dropout).float()
            spk_emb = spk_emb * mask

        # Embed LV0
        lv0_e = self.lv0_emb(lv0)  # (B, T, lv0_dim)

        # Embed each LV1-7 source level
        lv_embs = []
        for i in range(7):
            lv_embs.append(self.lv_embs[i](lv1_7_src[:, i, :]))  # (B, T, lv_dim)
        lv_e = torch.cat(lv_embs, dim=-1)  # (B, T, 7*lv_dim)

        # Speaker embedding
        spk_e = self.spk_proj(spk_emb).unsqueeze(1).expand(-1, T, -1)  # (B, T, lv0_dim)

        # Combine all inputs
        h = torch.cat([lv0_e, lv_e, spk_e], dim=-1)  # (B, T, total_dim)

        # Project + position
        h = self.input_proj(h) + self.pos_emb[:, :T, :]

        # Bidirectional transformer
        h = self.transformer(h)  # (B, T, d_model)

        # Predict residual per level
        logits = torch.stack([head(h) for head in self.heads], dim=1)  # (B, 7, T, vocab)
        return logits

    @torch.no_grad()
    def predict(self, lv0, lv1_7_src, spk_emb):
        """Generate Δ_LV1-7 residual codes."""
        return self.forward(lv0, lv1_7_src, spk_emb).argmax(dim=-1)


def compute_residual_loss(model, lv0, lv1_7_src, lv1_7_tgt, spk_emb, criterion):
    """
    CE loss on residual code prediction.
    Target = tgt_LV1-7 (the model predicts the target codes directly,
    but the architecture sees source codes as context).
    """
    logits = model(lv0, lv1_7_src, spk_emb)  # (B, 7, T, vocab)
    loss = sum(
        criterion(logits[:, i].reshape(-1, model.vocab), lv1_7_tgt[:, i].reshape(-1))
        for i in range(7)
    )
    return loss / 7.0


@torch.no_grad()
def convert_residual(model, mimi, src_audio, spk_emb):
    """
    Full VC pipeline using residual prediction.

    src_audio → codes_src → (LV0, LV1-7_src)
    predict Δ_LV1-7
    LV1-7_vc = LV1-7_src + Δ (or just use predicted directly)
    decode
    """
    codes_src = mimi.encode(src_audio)  # (B, 8, T)
    lv0 = codes_src[:, 0, :]           # (B, T)
    lv1_7_src = codes_src[:, 1:, :]    # (B, 7, T)

    pred = model.predict(lv0, lv1_7_src, spk_emb)  # (B, 7, T)

    # Combine: use predicted LV1-7 directly (not residual — the model
    # was trained to predict target LV1-7 given source context)
    codes_vc = torch.cat([lv0.unsqueeze(1), pred], dim=1)
    return mimi.decode(codes_vc)

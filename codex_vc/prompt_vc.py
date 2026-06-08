"""
Prompt-based VC: src codes + target prompt + speaker → VC codes.

src audio → Mimi codes ──┐
tgt prompt (2s audio) ────┤→ PromptConverter → VC codes → Mimi decode
tgt speaker embedding ─────┘

Key: target prompt provides rich acoustic reference (timbre, prosody).
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptEncoder(nn.Module):
    """Encode target prompt audio → acoustic style features."""
    def __init__(self, mimi, dim=512):
        super().__init__()
        self.mimi = mimi
        # Extract features from prompt codes
        self.code_emb = nn.Embedding(2048, 128)  # shared code embedding
        self.prompt_proj = nn.Sequential(
            nn.Conv1d(128 * 8, dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # global pooling
            nn.Flatten(),
            nn.Linear(dim, dim),
        )

    def forward(self, prompt_audio):
        """prompt_audio: (B, 1, T) → prompt_feat: (B, dim)"""
        with torch.no_grad():
            codes = self.mimi.encode(prompt_audio)  # (B, 8, T_p)
        # Embed each level
        embs = []
        for i in range(8):
            embs.append(self.code_emb(codes[:, i, :]))  # (B, T_p, 128)
        h = torch.cat(embs, dim=-1)  # (B, T_p, 1024)
        h = h.transpose(1, 2)  # (B, 1024, T_p)
        return self.prompt_proj(h)  # (B, dim)


class PromptConverter(nn.Module):
    """
    Causal transformer: src LV0 + prompt features → LV1-7 codes.
    Prompt already encodes speaker + acoustic style.
    """
    def __init__(self, vocab=2048, prompt_dim=512, d_model=512,
                 nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.lv0_emb = nn.Embedding(vocab, 128)
        self.prompt_proj = nn.Linear(prompt_dim, 128)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        self.input_proj = nn.Linear(128 * 2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab) for _ in range(7)])

    def forward(self, lv0, prompt_feat):
        B, T = lv0.shape
        lv0_e = self.lv0_emb(lv0)
        prompt_e = self.prompt_proj(prompt_feat).unsqueeze(1).expand(-1, T, -1)
        h = torch.cat([lv0_e, prompt_e], dim=-1)
        h = self.input_proj(h) + self.pos_emb[:, :T, :]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
        h = self.transformer(h, mask=causal_mask)
        return torch.stack([head(h) for head in self.heads], dim=1)

    @torch.no_grad()
    def predict(self, lv0, prompt_feat):
        return self.forward(lv0, prompt_feat).argmax(dim=-1)


def compute_loss(model, lv0, lv1_7_gt, prompt_feat, criterion):
    """CE loss over 7 levels."""
    logits = model(lv0, prompt_feat)
    loss = sum(criterion(logits[:, i].reshape(-1, model.vocab),
                         lv1_7_gt[:, i].reshape(-1)) for i in range(7))
    return loss / 7.0


@torch.no_grad()
def convert(model, mimi, src_audio, prompt_audio):
    """Full VC: src + prompt → audio."""
    prompt_enc = PromptEncoder(mimi)
    prompt_feat = prompt_enc(prompt_audio)

    codes_src = mimi.encode(src_audio)
    lv0 = codes_src[:, 0, :]
    pred_lv1_7 = model.predict(lv0, prompt_feat)

    codes_vc = torch.cat([lv0.unsqueeze(1), pred_lv1_7], dim=1)
    return mimi.decode(codes_vc)

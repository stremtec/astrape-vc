"""
Two-Stage VC: WavLM content + Resemblyzer speaker → Mimi codes.

Stage 1 (semantic): WavLM extracts speaker-independent content features
Stage 2 (acoustic): Transformer maps content + speaker → LV1-7 codes
Decoder: Frozen Mimi decoder → waveform
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel


class TwoStageVC(nn.Module):
    """
    WavLM content + Resemblyzer speaker → Mimi LV1-7 codes.

    Args:
        wavlm: Pretrained WavLMModel (frozen)
        spk_dim: Resemblyzer speaker embedding dim (256)
        mimi_vocab: Mimi codebook size (2048)
    """

    def __init__(self, wavlm, spk_dim=256, mimi_vocab=2048,
                 d_model=512, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.wavlm = wavlm
        for p in self.wavlm.parameters():
            p.requires_grad_(False)

        # Content projection: WavLM 768 → d_model
        self.content_proj = nn.Linear(768, d_model)

        # Speaker projection
        self.spk_proj = nn.Sequential(
            nn.Linear(spk_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Position encoding (for content)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: predict LV1-7 codes
        self.heads = nn.ModuleList([nn.Linear(d_model, mimi_vocab) for _ in range(7)])

    def forward(self, wavlm_input, spk_emb):
        """
        Args:
            wavlm_input: (B, T_audio) raw audio @ 16kHz for WavLM
            spk_emb: (B, 256) Resemblyzer speaker embedding
        Returns:
            logits: (B, 7, T_out, vocab)
        """
        # Extract WavLM features
        with torch.no_grad():
            wavlm_out = self.wavlm(wavlm_input).last_hidden_state  # (B, T_w, 768)

        B, T_w, _ = wavlm_out.shape

        # Project content
        content = self.content_proj(wavlm_out)  # (B, T_w, d_model)
        content = content + self.pos_emb[:, :T_w, :]

        # Project speaker and broadcast
        spk = self.spk_proj(spk_emb).unsqueeze(1)  # (B, 1, d_model)

        # Simple addition (can be improved with FiLM/cross-attention)
        h = content + spk

        # Transformer
        h = self.transformer(h)  # (B, T_w, d_model)

        # Predict LV1-7 codes
        logits = torch.stack([head(h) for head in self.heads], dim=1)  # (B, 7, T_w, vocab)
        return logits

    @torch.no_grad()
    def predict(self, wavlm_input, spk_emb):
        return self.forward(wavlm_input, spk_emb).argmax(dim=-1)


def resample_to_16k(audio_24k):
    """Resample (B, 1, T_24k) → (B, T_16k) for WavLM (using scipy)."""
    import numpy as np
    from scipy import signal
    audio_np = audio_24k.squeeze(1).numpy()  # (B, T_24k)
    B, T = audio_np.shape
    out_len = int(T * 16000 / 24000)
    out = np.zeros((B, out_len), dtype=np.float32)
    for b in range(B):
        out[b] = signal.resample(audio_np[b], out_len)
    return torch.from_numpy(out)


@torch.no_grad()
def convert_two_stage(model, mimi, src_audio_24k, spk_emb):
    """
    Full two-stage VC pipeline.

    Args:
        model: TwoStageVC
        mimi: MimiModel
        src_audio_24k: (B, 1, T) @ 24kHz
        spk_emb: (B, 256) target speaker
    Returns:
        vc_audio: (B, 1, T')
    """
    # Resample to 16kHz for WavLM
    src_16k = resample_to_16k(src_audio_24k)  # (B, T_16k)

    # Predict LV1-7 codes
    pred_lv1_7 = model.predict(src_16k, spk_emb)  # (B, 7, T_w)

    # Need LV0 codes — encode source through Mimi for LV0
    codes_src = mimi.encode(src_audio_24k)  # (B, 8, T_mimi)
    lv0 = codes_src[:, 0, :]  # (B, T_mimi)

    # Align lengths: WavLM output T_w may differ from Mimi T_mimi
    # Use min length for now
    T = min(lv0.shape[1], pred_lv1_7.shape[2])
    lv0 = lv0[:, :T]
    pred_lv1_7 = pred_lv1_7[:, :, :T]

    codes_vc = torch.cat([lv0.unsqueeze(1), pred_lv1_7], dim=1)
    return mimi.decode(codes_vc)

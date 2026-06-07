"""
Kanade v4: multi-layer features for split, full Mimi latent for decode.

Key fix: decoder uses Mimi's full encode_to_latent output (correct latent space).
Splitter uses intermediate transformer features for content/speaker separation.
"""

import torch, torch.nn as nn, torch.nn.functional as F

MIMI_DIM = 512
BOTTLENECK = 64


class MultiLayerExtractor(nn.Module):
    """Extract shallow and deep transformer features (Kanade-style)."""

    def __init__(self, mimi, shallow=(0,1,2), deep=(5,6,7)):
        super().__init__()
        self.mimi = mimi
        self.shallow = shallow
        self.deep = deep
        self.max_layer = max(max(shallow), max(deep))

    def forward(self, x):
        with torch.no_grad():
            enc = self.mimi.encoder(x)
            z_full = self.mimi.encode_to_latent(x, quantize=False)  # full latent for decode

        h = enc.transpose(1, 2)
        tt = self.mimi.encoder_transformer.transformer
        s_feats, d_feats = [], []

        for i, layer in enumerate(tt.layers):
            h = layer(h)
            if i in self.shallow: s_feats.append(h)
            if i in self.deep: d_feats.append(h)
            if i >= self.max_layer: break

        f_shallow = torch.stack(s_feats, dim=0).mean(dim=0).transpose(1,2)
        f_deep = torch.stack(d_feats, dim=0).mean(dim=0).transpose(1,2)
        return f_shallow, f_deep, z_full


class ContentBottleneck(nn.Module):
    def __init__(self, bottleneck=BOTTLENECK):
        super().__init__()
        self.compress = nn.Conv1d(MIMI_DIM, bottleneck, 1)
        self.expand = nn.Conv1d(bottleneck, MIMI_DIM, 1)
        self.norm = nn.LayerNorm(MIMI_DIM)

    def forward(self, x):
        h = self.compress(x); h = F.gelu(h); h = self.expand(h)
        h = h.transpose(1,2); h = self.norm(h)
        return (h + x.transpose(1,2)).transpose(1,2)


class SpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(MIMI_DIM, 256, 5, padding=2), nn.GELU(),
            nn.Conv1d(256, MIMI_DIM, 5, padding=2),
        )

    def forward(self, x):
        return self.conv(x).mean(dim=2).unsqueeze(-1)  # (B, D, 1)


class KanadeSplitterV4(nn.Module):
    """Multi-layer features for split, full latent for decode."""

    def __init__(self, mimi, bottleneck=BOTTLENECK):
        super().__init__()
        self.extractor = MultiLayerExtractor(mimi)
        self.content_bn = ContentBottleneck(bottleneck)
        self.speaker_enc = SpeakerEncoder()

    def forward(self, x):
        f_shallow, f_deep, z_full = self.extractor(x)
        z_content = self.content_bn(f_deep)
        z_spk = self.speaker_enc(f_shallow)
        return z_content, z_spk, z_full


# --- MimiSplitterV2 (simpler API, operates on latent directly) ---

class MimiSplitterV2(nn.Module):
    """Simple content/speaker splitter from Mimi latent z (12.5Hz)."""

    def __init__(self, dim=MIMI_DIM, bottleneck=BOTTLENECK):
        super().__init__()
        self.content = ContentBottleneck(bottleneck)
        self.speaker = SpeakerEncoder()

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, D, T) — Mimi latent @ 12.5Hz
        Returns:
            z_content: (B, D, T) time-varying content
            z_spk: (B, D, 1) per-utterance speaker (expand by caller)
        """
        z_content = self.content(z)
        z_spk = self.speaker(z)  # (B, D, 1)
        return z_content, z_spk


def splitter_loss(
    mimi,
    splitter: MimiSplitterV2,
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    audio_src: torch.Tensor,
    audio_tgt: torch.Tensor,
    ecapa_emb_src: torch.Tensor,
    ecapa_emb_tgt: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Combined splitter loss.

    z_src, z_tgt: same text, different speakers
    """
    B = z_src.size(0)

    # Split both sources
    c_src, s_src = splitter(z_src)
    c_tgt, s_tgt = splitter(z_tgt)

    # Expand speaker to match temporal dim
    T = z_src.size(2)
    s_src_exp = s_src.expand(-1, -1, T)
    s_tgt_exp = s_tgt.expand(-1, -1, T)

    # Reconstruction: decode from src content + src speaker (keep on gradient path!)
    z_recon = c_src + s_src_exp
    codes = mimi.quantizer.encode(z_recon.transpose(1, 2))
    audio_recon = mimi.decode(codes)
    recon_loss = F.l1_loss(audio_recon, audio_src)

    # Content invariance: src content ≈ tgt content (same text!)
    cos_content = F.cosine_similarity(
        c_src.transpose(1, 2).reshape(-1, MIMI_DIM),
        c_tgt.transpose(1, 2).reshape(-1, MIMI_DIM),
        dim=-1
    ).mean()
    content_loss = (1 - cos_content) ** 2

    # Speaker consistency: speaker vector should match ECAPA
    s_src_pooled = s_src.squeeze(-1)  # (B, D)
    spk_loss = 1 - F.cosine_similarity(s_src_pooled, ecapa_emb_src, dim=-1).mean()

    total = recon_loss + 0.5 * content_loss + 0.3 * spk_loss

    logs = {"recon": recon_loss.item(), "content": content_loss.item(), "spk": spk_loss.item()}
    return total, logs

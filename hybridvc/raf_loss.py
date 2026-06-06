"""
RAF (Relativistic Adversarial Feature) Loss for HybridVC.

Key insight from RAF (arXiv:2603.11678):
- SSL teacher (WavLM) replaces discriminator
- Relativistic pairing: "real is better than fake" instead of "real vs fake"
- Feature matching on WavLM internal layers
- 14M BigVGAN trained with RAF beats 112M BigVGAN with LSGAN+MPD+MRD

Reference: Lee & Choi, "RAF: Relativistic Adversarial Feedback For
Universal Speech Synthesis", ICASSP 2026.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RAFTeacher(nn.Module):
    """
    Frozen WavLM teacher for RAF loss.
    
    Extracts multi-layer features for relativistic pairing and
    feature matching loss.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        layers: tuple[int, ...] = (6, 12, 18, 24),
        device: str = "cpu",
    ):
        super().__init__()
        self.layers = layers
        self._device = device

        # Lazy load WavLM (heavy, 317M params)
        try:
            from transformers import WavLMModel
            self._wavlm = WavLMModel.from_pretrained(model_name)
        except ImportError:
            raise ImportError(
                "transformers required for RAF teacher. "
                "Install: pip install transformers"
            )
        except Exception:
            # Fallback: try loading from local cache or torch hub
            raise RuntimeError(
                f"Failed to load WavLM model '{model_name}'. "
                "Ensure internet access or local cache."
            )

        # Freeze
        for p in self._wavlm.parameters():
            p.requires_grad = False
        self._wavlm.eval()

        # Projection heads for each monitored layer → common dim
        self.layer_projs = nn.ModuleDict({
            str(l): nn.Linear(1024, 768) for l in layers
        })

    @property
    def device(self) -> torch.device:
        return next(self._wavlm.parameters()).device

    def forward(self, wav: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Args:
            wav: (B, T) waveform @ 16kHz (WavLM native SR)
        Returns:
            dict mapping layer_idx → (B, T_feat, 768) projected features
        """
        # WavLM expects 16kHz input
        with torch.no_grad():
            outputs = self._wavlm(
                wav,
                output_hidden_states=True,
            )
        
        feats = {}
        for l in self.layers:
            # hidden_states[l] shape: (B, T_feat, 1024)
            h = outputs.hidden_states[l]
            h = self.layer_projs[str(l)](h)  # → (B, T_feat, 768)
            feats[l] = h

        return feats

    def global_pool(self, feats: dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Mean-pool multi-layer features into single utterance vector.
        Used for relativistic pairing score.

        Returns:
            (B, 768) utterance-level embedding
        """
        pooled = []
        for l in self.layers:
            h = feats[l]  # (B, T, 768)
            pooled.append(h.mean(dim=1))  # (B, 768)
        return torch.stack(pooled, dim=1).mean(dim=1)  # (B, 768)


class RAFLoss(nn.Module):
    """
    RAF (Relativistic Adversarial Feature) Loss.

    Replaces GAN discriminator with frozen SSL teacher.

    Loss = RAF_adv + λ_fm * FM_loss

    RAF_adv:
        D_real = teacher.global_pool(real_wav)
        D_fake = teacher.global_pool(fake_wav)
        L_G = -log(σ(D_fake - D_real))   # generator wants D_fake > D_real
        L_D = -log(σ(D_real - D_fake))   # (unused — teacher is frozen)

    FM_loss:
        Σ_l ||teacher(real)[l] - teacher(fake)[l]||₁  per-layer L1
    """

    def __init__(
        self,
        teacher: RAFTeacher,
        lambda_fm: float = 2.0,
    ):
        super().__init__()
        self.teacher = teacher
        self.lambda_fm = lambda_fm

    def forward(
        self,
        fake_wav: torch.Tensor,
        real_wav: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            fake_wav: (B, T) generated waveform @ 16kHz
            real_wav: (B, T) ground truth waveform @ 16kHz
        Returns:
            (total_loss, logs) where logs has 'raf_adv', 'raf_fm'
        """
        # Extract multi-layer features
        real_feats = self.teacher(real_wav)
        fake_feats = self.teacher(fake_wav)

        # ── RAF adversarial loss (generator) ──
        D_real = self.teacher.global_pool(real_feats)  # (B, 768)
        D_fake = self.teacher.global_pool(fake_feats)  # (B, 768)

        # Pairwise relativistic: D_real vs D_fake
        # Score diff per pair
        diff_rf = (D_real - D_fake).mean(dim=-1)  # (B,)
        diff_fr = (D_fake - D_real).mean(dim=-1)  # (B,)

        # Generator: wants fake closer to real
        raf_adv = -F.logsigmoid(diff_fr).mean()

        # ── Feature matching loss ──
        fm_loss = 0.0
        for l in self.teacher.layers:
            fm_loss += F.l1_loss(fake_feats[l], real_feats[l].detach())
        fm_loss = fm_loss / len(self.teacher.layers)

        # ── Total ──
        total = raf_adv + self.lambda_fm * fm_loss

        logs = {
            "raf_adv": raf_adv.item(),
            "raf_fm": fm_loss.item(),
            "raf_D_real": D_real.mean().item(),
            "raf_D_fake": D_fake.mean().item(),
        }
        return total, logs


# ── Utility: resample to 16kHz for WavLM ──

def resample_wavlm(wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
    """
    Resample waveform to 16kHz for WavLM teacher.
    
    Args:
        wav: (B, T) waveform at orig_sr
        orig_sr: original sample rate (24kHz or 44.1kHz)
    Returns:
        (B, T_16k) waveform at 16000 Hz
    """
    if orig_sr == 16000:
        return wav
    import torchaudio.functional as F_audio
    return F_audio.resample(wav, orig_sr, 16000)


# ── Convenience: combine with spectral losses ──

def raf_vocoder_loss(
    fake_wav: torch.Tensor,
    real_wav: torch.Tensor,
    fake_mel: torch.Tensor,
    real_mel: torch.Tensor,
    raf_loss_fn: RAFLoss,
    sample_rate: int,
    lambda_mel: float = 45.0,
    lambda_stft: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Combined RAF + spectral loss for vocoder training.

    Args:
        fake_wav: (B, T) generated waveform
        real_wav: (B, T) ground truth waveform
        fake_mel: (B, n_mels, T_mel) generated mel-spectrogram
        real_mel: (B, n_mels, T_mel) ground truth mel-spectrogram
        raf_loss_fn: RAFLoss instance
        sample_rate: audio sample rate
        lambda_mel: mel loss weight
        lambda_stft: STFT loss weight
    Returns:
        (total, logs) with raf_*, mel_l1, stft_sc
    """
    # Resample for RAF teacher (needs 16kHz)
    fake_16k = resample_wavlm(fake_wav, sample_rate)
    real_16k = resample_wavlm(real_wav, sample_rate)

    raf_total, raf_logs = raf_loss_fn(fake_16k, real_16k)

    # Mel L1
    mel_l1 = F.l1_loss(fake_mel, real_mel)

    # Multi-res STFT (simplified single-res)
    # Full MR-STFT uses multiple n_fft/window/hop; here use one representative
    stft_sc = 0.0  # placeholder for full MR-STFT

    total = lambda_mel * mel_l1 + lambda_stft * stft_sc + raf_total

    logs = {
        **raf_logs,
        "mel_l1": mel_l1.item(),
        "stft_sc": stft_sc,
        "loss_total": total.item(),
    }
    return total, logs

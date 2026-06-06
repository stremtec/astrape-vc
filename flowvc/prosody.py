"""
FlowVC prosody extractor using FCPE (Fast Context-aware Pitch Extractor).

FCPE runs at 16kHz, outputs ~100Hz F0 frames → resampled to 25Hz.
Provides log_f0, voiced flag, and log_energy per frame.

Ref: CNChTu/FCPE (torchfcpe)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


FCPE_SR = 16000
TARGET_SR = 44100
TARGET_HZ = 25


class FCPEProsodyExtractor(nn.Module):
    """
    FCPE-based prosody extractor.

    Output per frame:
      [0]: log_f0     — log-scale fundamental frequency (0 if unvoiced)
      [1]: voiced     — voicing probability
      [2]: log_energy — log-scale RMS energy
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self._fcpe_model = None
        self._resampler_16k = None
        self._resampler_25hz = None

    def _ensure_fcpe(self):
        if self._fcpe_model is None:
            try:
                import torchfcpe
                # FCPE has MPS padding bug — always use CPU
                self._fcpe_model = torchfcpe.spawn_bundled_infer_model(device="cpu")
            except ImportError:
                raise ImportError(
                    "torchfcpe not installed. "
                    "Install: pip install git+https://github.com/CNChTu/FCPE.git"
                )

    def _ensure_resamplers(self):
        if self._resampler_16k is None:
            self._resampler_16k = torchaudio.transforms.Resample(
                orig_freq=TARGET_SR, new_freq=FCPE_SR,
            ).to(self.device)
        if self._resampler_25hz is None:
            self._resampler_25hz = torchaudio.transforms.Resample(
                orig_freq=100, new_freq=TARGET_HZ,  # FCPE outputs ~100Hz
            ).to(self.device)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: (B, 1, T) waveform @ 44100Hz
        Returns:
            prosody: (B, T_lat, 3) — [log_f0, voiced, log_energy]
        """
        self._ensure_fcpe()
        self._ensure_resamplers()

        B = wav.size(0)
        results = []

        for b in range(B):
            # Extract F0 via FCPE
            wav_b = wav[b:b+1]  # (1, 1, T)
            wav_16k = self._resampler_16k(wav_b)  # (1, 1, T_16k)

            # FCPE runs on CPU (MPS padding bug)
            wav_cpu = wav_16k.cpu()
            f0 = self._fcpe_model.infer(
                wav_cpu,
                sr=FCPE_SR,
                decoder_mode="local_argmax",
                threshold=0.006,
            )  # (1, T_frames)
            f0 = f0.to(self.device)

            # Resample to 25Hz
            f0_25hz = self._resampler_25hz(f0.unsqueeze(0)).squeeze(0).squeeze(0)  # (T_lat,)

            # Voiced flag: F0 > 0
            voiced = (f0_25hz > 1.0).float()  # (T_lat,)

            # Log F0 (0 for unvoiced)
            log_f0 = torch.where(f0_25hz > 1.0, torch.log(f0_25hz + 1e-8), torch.zeros_like(f0_25hz))

            # Log energy per frame
            T_lat = log_f0.shape[0]
            hop = wav_b.shape[2] // max(T_lat, 1)
            energy = wav_b.squeeze(0).squeeze(0).unfold(0, hop, hop)[:T_lat]
            rms = energy.pow(2).mean(dim=-1).sqrt()
            log_energy = torch.log(rms + 1e-8)

            feat = torch.stack([log_f0, voiced, log_energy], dim=-1)  # (T_lat, 3)
            results.append(feat)

        return torch.stack(results, dim=0)  # (B, T_lat, 3)


def make_prosody_extractor(device: str = "cpu") -> FCPEProsodyExtractor:
    return FCPEProsodyExtractor(device=device)

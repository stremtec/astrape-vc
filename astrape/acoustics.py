"""Causal acoustic-feature extraction for the v8 decoder interface (150 Hz).

The Stage-A acoustic model regresses these features; the Stage-B vocoder consumes
them.  ALL framing is LEFT-ALIGNED (causal): frame ``t`` summarises audio ending at
sample ``(t+1)*AC_HOP`` and NEVER reads a sample beyond it.  A centered STFT would pull
~10 ms of future audio into the target and silently give the (strictly-causal) model a
look-ahead it cannot honour at inference — see DECODER_V8_DESIGN.md §2.

Grid (44.1 kHz):  content 25 Hz → acoustic 150 Hz (hop 294) → render 450 Hz → wav.
1764 / 294 = 6 (acoustic frames per content frame);  294 / 98 = 3 (render per acoustic).

Everything here is a non-differentiable TARGET extractor (CPU/torch), so the analysis
window length is free — only its *causal alignment* matters.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio

S = 44100
AC_RATE = 150
AC_HOP = 294                 # 44100 / 150
CONTENT_HOP = 1764           # 44100 / 25  (= 6 * AC_HOP)
MEL_NFFT = 882               # 20 ms window
N_MELS = 80
F0_FRAME = 2048              # ~46 ms — enough for fmin=50 Hz (2+ periods)
FMIN, FMAX = 50.0, 600.0
VOICE_THRESH = 0.35


def causal_frames(wav: torch.Tensor, frame: int, hop: int) -> torch.Tensor:
    """Left-aligned framing: frame ``t`` = samples ``[t*hop - (frame-hop), t*hop + hop)``,
    i.e. its window ENDS at ``(t+1)*hop`` (never reads past it).  For ``len(wav)`` a
    multiple of ``hop`` this yields exactly ``len(wav)//hop`` frames (one per hop chunk)."""
    wav = F.pad(wav, (frame - hop, 0))            # left-pad only → causal
    return wav.unfold(0, frame, hop)              # (T, frame)


def extract_acoustics(wav: torch.Tensor, mel_fb: torch.Tensor | None = None) -> dict:
    """Ground-truth acoustic targets @150 Hz from a 1-D 44.1 kHz waveform.

    Returns dict of (T,)/(T,80) tensors with T = len(wav)//AC_HOP:
      mel     (T, 80)  log-mel magnitude
      logf0   (T,)     log-Hz on voiced frames, 0 elsewhere
      voiced  (T,)     {0,1}
      energy  (T,)     log-RMS of the causal window
    """
    if mel_fb is None:
        mel_fb = melscale_fbanks()
    # ── mel + energy: causal 882-window frames ──
    fr = causal_frames(wav, MEL_NFFT, AC_HOP)                 # (T, 882)
    win = torch.hann_window(MEL_NFFT, dtype=wav.dtype)
    mag = torch.fft.rfft(fr * win, n=MEL_NFFT).abs()          # (T, 442)
    mel = torch.log(torch.clamp(mag @ mel_fb, min=1e-5))      # (T, 80)
    energy = torch.log(fr.pow(2).mean(-1).clamp(min=1e-10)) * 0.5   # (T,) log-RMS

    # ── F0 + voicing: causal 2048-window FFT-autocorrelation ──
    ff = causal_frames(wav, F0_FRAME, AC_HOP)                 # (T, 2048)
    ff = ff * torch.hann_window(F0_FRAME, dtype=wav.dtype)
    nfft = 2 * F0_FRAME
    ac = torch.fft.irfft(torch.fft.rfft(ff, n=nfft).abs().pow(2), n=nfft)[..., :F0_FRAME]
    ac = ac / ac[..., :1].clamp(min=1e-9)                     # normalise by lag-0
    lmin, lmax = int(S / FMAX), int(S / FMIN)
    peak, k = ac[..., lmin:lmax].max(dim=-1)                  # (T,)
    f0 = S / (lmin + k).float()
    rms = ff.pow(2).mean(-1).sqrt()
    voiced = ((peak > VOICE_THRESH) & (rms > 1e-3)).float()
    logf0 = torch.log(f0.clamp(min=1.0)) * voiced
    return {"mel": mel.float(), "logf0": logf0.float(),
            "voiced": voiced.float(), "energy": energy.float()}


def melscale_fbanks() -> torch.Tensor:
    """(n_freqs, 80) mel filterbank for the 882-pt causal STFT."""
    return torchaudio.functional.melscale_fbanks(
        n_freqs=MEL_NFFT // 2 + 1, f_min=0.0, f_max=S / 2, n_mels=N_MELS,
        sample_rate=S, norm=None, mel_scale="htk")


def stack_acoustics(a: dict) -> torch.Tensor:
    """Pack the target dict into the (T, 83) conditioning tensor the vocoder consumes:
    [mel(80) | logf0(1) | voiced(1) | energy(1)]."""
    return torch.cat([a["mel"], a["logf0"][:, None], a["voiced"][:, None],
                      a["energy"][:, None]], dim=-1)

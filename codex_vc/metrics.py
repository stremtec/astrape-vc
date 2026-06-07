"""
Multi-metric VC evaluation: cosine, F0, spectral, content preservation.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy.spatial.distance import cosine as cos_dist
import soundfile as sf


def compute_all_metrics(src_audio, tgt_audio, vc_audio, sr, mimi=None):
    """Compute comprehensive VC metrics.

    Returns dict with:
        cos_src, cos_tgt, delta: speaker similarity
        f0_corr_src, f0_corr_tgt: F0 correlation
        f0_rmse_src, f0_rmse_tgt: F0 RMSE (log Hz)
        lsd_src, lsd_tgt: log-spectral distance (dB)
        mcd_src, mcd_tgt: mel-cepstral distortion (dB)
        content_cos: LV0 code preservation (if mimi provided)
    """
    metrics = {}

    # Ensure 1D
    if src_audio.ndim > 1: src_audio = src_audio.squeeze()
    if tgt_audio.ndim > 1: tgt_audio = tgt_audio.squeeze()
    if vc_audio.ndim > 1: vc_audio = vc_audio.squeeze()

    T = min(len(src_audio), len(tgt_audio), len(vc_audio))
    src = src_audio[:T].numpy() if torch.is_tensor(src_audio) else src_audio[:T]
    tgt = tgt_audio[:T].numpy() if torch.is_tensor(tgt_audio) else tgt_audio[:T]
    vc = vc_audio[:T].numpy() if torch.is_tensor(vc_audio) else vc_audio[:T]

    # --- F0 analysis ---
    f0_src, _ = compute_f0(src, sr)
    f0_tgt, _ = compute_f0(tgt, sr)
    f0_vc, _ = compute_f0(vc, sr)

    if len(f0_src) > 0 and len(f0_vc) > 0:
        min_len = min(len(f0_src), len(f0_vc))
        metrics['f0_corr_src'] = np.corrcoef(f0_src[:min_len], f0_vc[:min_len])[0, 1]
        metrics['f0_rmse_src'] = np.sqrt(np.mean((np.log2(f0_src[:min_len] + 1e-8) -
                                                    np.log2(f0_vc[:min_len] + 1e-8))**2))

    if len(f0_tgt) > 0 and len(f0_vc) > 0:
        min_len = min(len(f0_tgt), len(f0_vc))
        metrics['f0_corr_tgt'] = np.corrcoef(f0_tgt[:min_len], f0_vc[:min_len])[0, 1]
        metrics['f0_rmse_tgt'] = np.sqrt(np.mean((np.log2(f0_tgt[:min_len] + 1e-8) -
                                                    np.log2(f0_vc[:min_len] + 1e-8))**2))

    # --- Spectral analysis ---
    metrics['lsd_src'] = compute_lsd(src, vc, sr)
    metrics['lsd_tgt'] = compute_lsd(tgt, vc, sr)

    # --- Speaker similarity (via AudioDec latent cosine) ---
    # Computed externally using codec.encode

    # --- Content preservation (via Mimi LV0) ---
    if mimi is not None:
        # Resample to 24kHz for Mimi
        src_24 = signal.resample(src, int(len(src)*24000/sr))
        tgt_24 = signal.resample(tgt, int(len(tgt)*24000/sr))
        vc_24 = signal.resample(vc, int(len(vc)*24000/sr))

        src_t = torch.from_numpy(src_24).float().unsqueeze(0).unsqueeze(0)
        vc_t = torch.from_numpy(vc_24).float().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            codes_src = mimi.encode(src_t)
            codes_vc = mimi.encode(vc_t)
            T = min(codes_src.shape[2], codes_vc.shape[2])
            lv0_match = (codes_src[0, 0, :T] == codes_vc[0, 0, :T]).float().mean().item()
        metrics['lv0_preservation'] = lv0_match

    return metrics


def compute_f0(wav, sr, frame_ms=25, shift_ms=10):
    """Simple F0 estimation using autocorrelation."""
    from scipy.signal import correlate

    frame_len = int(frame_ms * sr / 1000)
    shift_len = int(shift_ms * sr / 1000)
    n_frames = (len(wav) - frame_len) // shift_len + 1

    if n_frames <= 0:
        return np.array([]), np.array([])

    f0s = []
    for i in range(n_frames):
        start = i * shift_len
        frame = wav[start:start + frame_len]
        frame = frame * np.hanning(len(frame))

        # Autocorrelation
        corr = correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]

        # Find first peak after zero
        corr[:10] = 0  # ignore very short lags
        if corr.max() > 0:
            peak_idx = corr.argmax()
            if peak_idx > 0:
                f0 = sr / peak_idx
                if 50 < f0 < 500:  # valid human F0 range
                    f0s.append(f0)
                    continue
        f0s.append(0)

    f0s = np.array(f0s)
    valid = f0s > 0
    return f0s[valid], np.arange(len(f0s))[valid]


def compute_lsd(ref, deg, sr, n_fft=512, hop=128):
    """Log-Spectral Distance in dB."""
    f_ref, _, Zxx_ref = signal.stft(ref, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, Zxx_deg = signal.stft(deg, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)

    mag_ref = np.abs(Zxx_ref) + 1e-8
    mag_deg = np.abs(Zxx_deg) + 1e-8

    T = min(mag_ref.shape[1], mag_deg.shape[1])
    lsd = np.mean(np.sqrt(np.mean((np.log10(mag_ref[:, :T]) - np.log10(mag_deg[:, :T]))**2, axis=0)))

    return float(lsd * 20)  # convert to dB


def format_metrics(metrics):
    """Pretty-print metrics."""
    lines = []
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            lines.append(f"  {k:20s}: {v:+.4f}")
    return "\
".join(lines)

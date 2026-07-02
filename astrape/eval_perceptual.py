"""Perceptual eval harness for the v8 decoder (rebuilds the lost /tmp HNR harness).

Metrics (proxies, but consistent for A/B — judge vs a baseline, per the project's protocol):
  HNR       harmonic-to-noise ratio (dB) over voiced frames — the "fizz"/buzz marker
  flat_hi   high-band spectral flatness (noise → 1, tonal → 0) — fizz in the highs
  centroid  spectral centroid (Hz) — brightness / metallic tilt

Modes:
  copy-synth   GT acoustics → Stage-B vocoder → wav.  Isolates rendering (no VC in loop) —
               the go/no-go for Stage B.  Compare metrics to the GROUND-TRUTH audio's own.
  vc           source wav → encoder → Stage A → Stage B → wav.  Full system.

  .venv/bin/python -m astrape.eval_perceptual --mode copy-synth \
      --vocoder /Volumes/UNTITLED/btrv5_checkpoints/v8_stage_b/last.pt --n 30
"""
import argparse, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore"); sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

S = 44100


def _frames(wav, frame=2048, hop=512):
    wav = F.pad(wav, (0, (hop - (wav.shape[-1] - frame) % hop) % hop))
    return wav.unfold(-1, frame, hop) * torch.hann_window(frame)


def hnr_db(wav, fmin=50, fmax=600, frame=2048, hop=512, vthresh=0.35):
    """Mean harmonic-to-noise ratio (dB) over voiced frames (autocorrelation method)."""
    fr = _frames(wav, frame, hop)
    nfft = 2 * frame
    ac = torch.fft.irfft(torch.fft.rfft(fr, n=nfft).abs().pow(2), n=nfft)[..., :frame]
    ac = ac / ac[..., :1].clamp_min(1e-9)
    lmin, lmax = int(S / fmax), int(S / fmin)
    r, _ = ac[..., lmin:lmax].max(dim=-1)                   # peak autocorr
    rms = fr.pow(2).mean(-1).sqrt()
    voiced = (r > vthresh) & (rms > 1e-3)
    if voiced.sum() == 0:
        return float("nan")
    r = r[voiced].clamp(1e-4, 1 - 1e-4)
    return (10 * torch.log10(r / (1 - r))).mean().item()


def flat_hi(wav, n_fft=2048, hop=512, hi_bin=0.5):
    spec = torch.stft(wav, n_fft, hop, window=torch.hann_window(n_fft),
                      center=True, return_complex=True).abs().clamp_min(1e-7)
    hi = spec[int(spec.shape[0] * hi_bin):]                 # upper half of the spectrum
    gm = hi.log().mean(0).exp()
    am = hi.mean(0)
    return (gm / am.clamp_min(1e-7)).mean().item()


def centroid_hz(wav, n_fft=2048, hop=512):
    spec = torch.stft(wav, n_fft, hop, window=torch.hann_window(n_fft),
                      center=True, return_complex=True).abs()
    freqs = torch.linspace(0, S / 2, spec.shape[0]).unsqueeze(1)
    return ((spec * freqs).sum(0) / spec.sum(0).clamp_min(1e-7)).mean().item()


def harmonic_clarity(wav, nfft=2048):
    """Peak-to-valley depth (dB) between harmonics on the most-voiced 0.37s window.
    High = clean tonal harmonics; LOW = valleys filled with energy = buzzy/rough. Broadband
    HNR is blind to this — a signal can be periodic yet have smeared harmonics."""
    from scipy.signal import stft as _stft
    x = wav.numpy()
    f, t, Z = _stft(x, S, nperseg=nfft, noverlap=nfft * 3 // 4, window="hann")
    lowE = np.abs(Z[(f >= 100) & (f <= 1000)]).mean(0)
    if lowE.size == 0:
        return float("nan")
    c = int(t[lowE.argmax()] * S)
    seg = x[max(0, c - 8192): max(0, c - 8192) + 16384]
    if len(seg) < 16384:
        return float("nan")
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    kmin, kmax = int(S / 400), int(S / 80)
    f0 = S / (kmin + np.argmax(ac[kmin:kmax]))
    fr = np.fft.rfftfreq(32768, 1 / S)
    X = np.abs(np.fft.rfft(seg * np.hanning(16384), 32768))
    harm = np.arange(f0, 6000, f0)
    mids = harm[:-1] + f0 / 2
    hb = X[[np.argmin(np.abs(fr - h)) for h in harm if h >= f0]]
    vb = X[[np.argmin(np.abs(fr - m)) for m in mids]]
    return float(20 * np.log10(hb.mean() / (vb.mean() + 1e-9)))


def frame_mod_db(wav, rate=150):
    """Prominence (dB) of the frame-rate amplitude-modulation comb in the 2-8kHz band —
    the zero-order-hold / frame-grid buzz.  Near 0 = clean; large = frame-rate imaging."""
    from scipy.signal import hilbert, butter, sosfiltfilt
    x = wav.numpy()
    sos = butter(4, [2000, 8000], "bandpass", fs=S, output="sos")
    e = np.abs(hilbert(sosfiltfilt(sos, x))); e = e - e.mean()
    E = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    fm = np.fft.rfftfreq(len(e), 1 / S)
    out = 0.0
    for k in (1, 2, 3):
        hz = rate * k; m = (fm >= hz - 6) & (fm <= hz + 6)
        base = np.median(E[(fm >= hz - 60) & (fm <= hz + 60)])
        out += 20 * np.log10(E[m].max() / (base + 1e-9) + 1e-9)
    return float(out / 3)


_MEL_FN = None


def _mel():
    global _MEL_FN
    if _MEL_FN is None:
        _MEL_FN = torchaudio.transforms.MelSpectrogram(
            S, n_fft=2048, hop_length=512, n_mels=80, f_min=0, f_max=S / 2, power=1)
    return _MEL_FN


def _content_lag(Mp, Mg, max_frames=60):
    """Best content offset (in mel frames) that aligns pred→ref, from the mel-ENERGY-envelope
    cross-correlation.  Phase-invariant (unlike waveform cross-correlation, which chases the
    vocoder's freely-generated phase and mis-aligns content), and bounded so it removes only a
    CONSTANT offset (varying speech onset, encoder/pipeline group delay, the encoder's trained
    time_shift) — not local timing errors, which stay penalised.  Same idea as encoding_gap.py
    reproducing the encoder's training shift before probing."""
    ep = torch.log1p(Mp.sum(0)).numpy(); eg = torch.log1p(Mg.sum(0)).numpy()
    ep = ep - ep.mean(); eg = eg - eg.mean()
    n = min(len(ep), len(eg))
    if n < 2:
        return 0
    r = np.correlate(ep[:n], eg[:n], "full"); lags = np.arange(-n + 1, n)
    m = np.abs(lags) <= max_frames
    return int(lags[m][r[m].argmax()])


def _align(a, b, lag):
    """Shift a→b by `lag` (in the axis-(-1) units of a,b) and return the overlapping region."""
    if lag > 0:
        a, b = a[..., lag:], b[..., :b.shape[-1] - lag]
    elif lag < 0:
        a, b = a[..., :a.shape[-1] + lag], b[..., -lag:]
    t = min(a.shape[-1], b.shape[-1])
    return a[..., :t], b[..., :t]


def mel_cos(pred, ref):
    """Content-ALIGNED mel-spectrogram cosine — phase-invariant magnitude similarity, robust
    to a constant temporal offset between pred and ref.  The gate metric for 'closeness to
    teacher/source'."""
    mf = _mel().to("cpu")
    Mp, Mg = mf(pred.float().cpu()), mf(ref.float().cpu())
    if Mp.dim() == 3: Mp, Mg = Mp[0], Mg[0]
    Mp, Mg = _align(Mp, Mg, _content_lag(Mp, Mg))
    return F.cosine_similarity(Mp.flatten(), Mg.flatten(), dim=0).item()


def wave_cos(pred, ref):
    """Time-domain cosine after CONTENT alignment (mel-envelope lag → samples).  Alignment
    removes the offset, but the residual is PHASE, which a vocoder generates freely — so this
    stays ~0 even when perceptually perfect (a 0.16ms shift of identical audio drops it to
    ~0.7).  Diagnostic only; judge with mel_cos."""
    mf = _mel().to("cpu")
    Mp, Mg = mf(pred.float().cpu()), mf(ref.float().cpu())
    if Mp.dim() == 3: Mp, Mg = Mp[0], Mg[0]
    lag = _content_lag(Mp, Mg) * 512   # mel hop → samples
    p, r = _align(pred.flatten().float().cpu(), ref.flatten().float().cpu(), lag)
    return F.cosine_similarity(p, r, dim=0).item()


def metrics(wav):
    wav = wav.detach().float().cpu()
    if wav.dim() > 1:
        wav = wav[0]
    return {"hnr": hnr_db(wav), "flat_hi": flat_hi(wav), "centroid": centroid_hz(wav),
            "harm_clarity": harmonic_clarity(wav), "frame_buzz": frame_mod_db(wav)}


def _fmt(m):
    return (f"HNR={m['hnr']:.2f}dB  flat_hi={m['flat_hi']:.3f}  centroid={m['centroid']:.0f}Hz  "
            f"harm_clarity={m['harm_clarity']:.1f}dB  frame_buzz={m['frame_buzz']:.1f}dB")


def copy_synth(args, device):
    """GT acoustics → vocoder. Compare to ground-truth audio's own metrics."""
    from .decoder_v8 import ConditionedVocoder, VocoderConfig
    from .data import AcousticDataset, collate_acoustic
    from torch.utils.data import DataLoader
    ck = torch.load(args.vocoder, map_location="cpu", weights_only=False)
    voc = ConditionedVocoder(VocoderConfig(**ck.get("vocoder_config", {}))).to(device).eval()
    voc.load_state_dict(ck["state_dict"])

    meta = np.load(args.data_dir / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"]); src = meta["source_files"][:n].astype(str)
    spk_names = meta["spk_names"]
    cz = np.load(args.data_dir / "spk_centroids.npz", allow_pickle=False)
    semap = {str(s): torch.from_numpy(e).float() for s, e in zip(cz["speakers"], cz["embeddings"])}
    idx = np.arange(n); np.random.default_rng(args.seed).shuffle(idx)
    ds = AcousticDataset(idx[int(n * 0.95):], args.data_dir / args.content_dir,
                         args.data_dir / args.acoustics_dir, src, spk_names, semap,
                         args.max_frames, args.seed, need_content=False, need_audio=True)
    loader = DataLoader(ds, 1, shuffle=False, collate_fn=collate_acoustic)
    pred_m, gt_m, cos = [], [], []
    for i, batch in enumerate(loader):
        if i >= args.n:
            break
        cond = torch.cat([batch["mel"], batch["logf0"][..., None],
                          batch["voiced"][..., None], batch["energy"][..., None]], dim=-1).to(device)
        with torch.no_grad():
            wav = voc(cond)
        pred_m.append(metrics(wav)); gt_m.append(metrics(batch["audio"]))
        # similarity to SOURCE (the GT audio being reconstructed); no teacher in copy-synth
        p, g = wav[0].cpu(), batch["audio"][0]
        cos.append({"mel_sor_cos": mel_cos(p, g), "wav_sor_cos": wave_cos(p, g)})
        if args.save and i < 5:
            _save(wav[0], f"{args.save}/cs_{i}_pred.wav"); _save(batch["audio"][0], f"{args.save}/cs_{i}_gt.wav")
    _report("COPY-SYNTH (vocoder)", pred_m); _report("GROUND TRUTH", gt_m)
    c = {k: float(np.mean([x[k] for x in cos])) for k in cos[0]}
    print(f"  vs SOURCE:  mel_sor_cos={c['mel_sor_cos']:.3f} (gate)   "
          f"wav_sor_cos={c['wav_sor_cos']:.3f} (phase-sensitive, ignore if low)")


def vc(args, device):
    """Full system: source → encoder → Stage A → Stage B."""
    from .decoder_v8 import AcousticModel, AcousticModelConfig, ConditionedVocoder, VocoderConfig
    from .train_decoder import load_encoder
    from .miocodec import load_wave
    from .voicebank import VoiceBank
    enc, _ = load_encoder(args.encoder_ckpt, device)
    cka = torch.load(args.acoustic, map_location="cpu", weights_only=False)
    A = AcousticModel(AcousticModelConfig(**cka.get("acoustic_config", {}))).to(device).eval()
    A.load_state_dict(cka["state_dict"])
    ckb = torch.load(args.vocoder, map_location="cpu", weights_only=False)
    B = ConditionedVocoder(VocoderConfig(**ckb.get("vocoder_config", {}))).to(device).eval()
    B.load_state_dict(ckb["state_dict"])
    spk = VoiceBank.load(Path(args.target)).global_embedding.float().to(device).unsqueeze(0)

    src = load_wave(Path(args.source), S, args.max_seconds)
    wl16 = _wavlm_l4(src, device)                            # (T,512) @200Hz
    with torch.no_grad():
        mask = torch.ones(1, wl16.shape[0] // 2, dtype=torch.bool, device=device)
        content = enc(wl16.unsqueeze(0).transpose(1, 2), padding_mask=mask)["projected"].transpose(1, 2)
        cond = A.infer_cond(content, spk)
        wav = B(cond)[0]
    print(f"source {_fmt(metrics(src))}")
    print(f"v8 VC  {_fmt(metrics(wav))}")
    # ── similarity references: TEACHER (MioCodec on the SAME content+target = the ceiling)
    #    and SOURCE (original audio; timbre-confounded when target≠source). mel_cos = gate.
    from .miocodec import load_mio
    mio = load_mio(device).eval()
    with torch.no_grad():
        teacher = mio.forward_wave(content, spk,
                                   stft_length=mio._calculate_target_stft_length(src.numel()))[0]
    w = wav.cpu()
    print(f"vs TEACHER: mel_tea_cos={mel_cos(w, teacher.cpu()):.3f} (gate)   "
          f"wav_tea_cos={wave_cos(w, teacher.cpu()):+.3f} (phase-sensitive)")
    print(f"vs SOURCE:  mel_sor_cos={mel_cos(w, src):.3f}   wav_sor_cos={wave_cos(w, src):+.3f}   "
          f"(only clean if target≈source; else timbre-confounded)")
    if args.save:
        _save(wav, f"{args.save}/vc_out.wav"); _save(src, f"{args.save}/vc_source.wav")
        _save(teacher, f"{args.save}/vc_teacher.wav")
        print(f"saved → {args.save}/vc_out.wav (+ _teacher, _source)")


def _wavlm_l4(wav, device):
    """5 causal WavLM conv layers @16kHz → (T,512) @200Hz (matches astrape.cache --what wavlm)."""
    import torchaudio
    from .miocodec import load_mio
    mio = load_mio(device).eval()
    fe = mio.ssl_feature_extractor.model.feature_extractor
    w16 = torchaudio.functional.resample(wav.unsqueeze(0), S, 16000).to(device)
    with torch.no_grad():
        x = w16
        for li in range(5):
            layer = fe.conv_layers[li]
            x = layer.conv(x)
            if getattr(layer, "layer_norm", None) is not None:
                x = layer.layer_norm(x.unsqueeze(0)).squeeze(0) if x.dim() == 2 else layer.layer_norm(x)
            x = F.gelu(x)
    return x.squeeze(0).transpose(0, 1)


def _save(wav, path):
    import soundfile as sf
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav.detach().float().cpu().numpy(), S)


def _report(name, ms):
    a = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
    print(f"  {name:22s} {_fmt(a)}  (n={len(ms)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["copy-synth", "vc"], required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--data-dir", type=Path, default=Path("data/mio_vctk_full_compact"))
    ap.add_argument("--content-dir", default="content_striding_8l_200hz")
    ap.add_argument("--acoustics-dir", default="acoustics_150hz")
    ap.add_argument("--vocoder", type=Path, help="Stage-B checkpoint")
    ap.add_argument("--acoustic", type=Path, help="Stage-A checkpoint (vc mode)")
    ap.add_argument("--encoder-ckpt", type=Path, help="frozen encoder (vc mode)")
    ap.add_argument("--source", help="source wav (vc mode)")
    ap.add_argument("--target", help="target .astrape voicebank (vc mode)")
    ap.add_argument("--max-seconds", type=float, default=6.0)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=None, help="dir to write sample wavs")
    args = ap.parse_args()
    device = torch.device(args.device)
    (copy_synth if args.mode == "copy-synth" else vc)(args, device)


if __name__ == "__main__":
    main()

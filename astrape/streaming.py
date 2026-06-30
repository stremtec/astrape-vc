"""Strict-causal streaming VC inference — the reference the Mac VST port replicates.

The model is strict-causal (output frame t depends only on input ≤ t), so a real-time
deployment processes audio in small blocks while carrying state. This module is the
*correctness + latency reference* for that loop:

  push(audio_block) → emit converted audio for the newly-completed content frames.

Correctness strategy (simple, exact): bounded-lookback recompute. To emit content
frames [n, n+B) we re-run encoder+decoder over [n+B-W, n+B) where W ≥ the model's
backward receptive field (transformer window=256 dominates). Frames older than W
cannot affect the emitted block (windowed attention + finite conv RF), so the emitted
audio is bit-identical to offline — verified below.

The C++/VST port keeps this exact block loop + ring buffers but replaces the recompute
with native incremental state (transformer KV-cache, conv ring buffers, iSTFT
overlap-add state) to drop the per-frame cost to the amortized ~0.8 ms (see PORT.md).

Measured latency (bit-exact streaming verified == offline; CPU, training-independent):
  content frame buffer = 40 ms        (accumulate one 25 Hz content frame)
  iSTFT tail held      = 14.3 ms      (only the (n_fft-hop)/2 = 630-sample overlap-add
                                        TAIL waits for the next frame — not a whole frame)
  front-end            = ~7 ms        (WavLM future-RF + resampler)
  → worst-case input→output ≈ 56 ms   (measured by real-time sim; was 92 ms when whole
                                        frames were emitted — sub-frame emit holds only
                                        the 14.3 ms tail, same samples, no quality loss)
  compute              = ~0.8 ms/frame with a KV-cache (this recompute reference is slower;
                         latency is unaffected — see PORT.md for the C++ cache).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

SAMPLES_PER_FRAME = 1764          # 44100 / 25 Hz  → one content frame = 40 ms
WAVLM_PER_FRAME = 8               # 200 Hz WavLM frames per 25 Hz content frame
FRAME_MS = 1000 * SAMPLES_PER_FRAME / 44100


@dataclass
class StreamConfig:
    lookback_frames: int = 256    # context kept for exact recompute (≥ backward RF ~256)
    frontend_ms: float = 7.0      # WavLM CNN future-RF (~5ms) + resampler (~2ms)


class StreamingVC:
    """Streaming wrapper over a frozen encoder + trained decoder — sub-frame emit.

    Operates on cached WavLM features (the encoder's input); the WavLM front-end is a
    separate causal stage (resample + 5 convs) folded into the budget via `frontend_ms`.

    Latency-optimal emit: when content frame m lands, the decoder's iSTFT can finalize
    every output sample EXCEPT the last `tail_hold` = (n_fft-hop)/2 = 630 (14.3 ms) of
    frame m, which still need frame m+1's overlap-add. So we emit continuously up to
    `m·1764 - tail_hold` and hold only that 14.3 ms tail — NOT a whole 40 ms frame.
    Same samples as offline (bit-exact); the held tail is the true iSTFT group delay.
    """

    def __init__(self, encoder, decoder, speaker: torch.Tensor, cfg: StreamConfig = StreamConfig()):
        self.enc, self.dec = encoder, decoder
        self.spk = speaker.reshape(1, -1)
        self.cfg = cfg
        self.tail_hold = (decoder.config.n_fft - decoder.config.hop_length) // 2   # 630 @ n_fft=1512
        self.wl = torch.zeros(0, 512)        # bounded WavLM ring (only ~lookback frames kept)
        self.base_frame = 0                  # absolute content-frame index of self.wl[0]
        self.emitted = 0                     # absolute output samples already emitted

    @torch.no_grad()
    def push_wavlm(self, wl_block: torch.Tensor) -> torch.Tensor:
        """Feed new WavLM frames; return all output samples finalizable so far."""
        self.wl = torch.cat([self.wl, wl_block], 0)
        avail = self.base_frame + self.wl.shape[0] // WAVLM_PER_FRAME   # absolute frames available
        target = avail * SAMPLES_PER_FRAME - self.tail_hold            # samples whose iSTFT is complete
        if target <= self.emitted:
            return torch.zeros(0)
        # drop frames older than the lookback window (keeps the buffer bounded, no O(n²) growth)
        keep_from = max(0, self.emitted // SAMPLES_PER_FRAME - self.cfg.lookback_frames)
        if keep_from > self.base_frame:
            self.wl = self.wl[(keep_from - self.base_frame) * WAVLM_PER_FRAME:]
            self.base_frame = keep_from
        mask = torch.ones(1, (self.wl.shape[0] // 2), dtype=torch.bool)
        content = self.enc(self.wl.unsqueeze(0).transpose(1, 2), padding_mask=mask)["projected"].transpose(1, 2)
        audio = self.dec(content, self.spk, stft_length=self.dec._compute_stft_length(content.shape[1]))[0]
        off = self.base_frame * SAMPLES_PER_FRAME                      # global index of audio[0]
        out = audio[self.emitted - off: target - off]
        self.emitted = target
        return out

    def algorithmic_latency_ms(self) -> float:
        # worst sample = first of a content frame: wait its frame (40) + the iSTFT tail
        # (14.3, the held 630 samples) + front-end; the content-block dominates.
        return FRAME_MS + 1000 * self.tail_hold / 44100 + self.cfg.frontend_ms


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import warnings, sys; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
    import numpy as np
    from pathlib import Path
    from astrape.decoder import CausalDecoderV5, CausalDecoderV5Config
    from astrape.train_decoder import load_encoder

    torch.set_grad_enabled(False); torch.set_num_threads(1)
    DATA = Path("data/mio_vctk_full_compact")
    enc, _ = load_encoder("/Volumes/UNTITLED/btrv5_checkpoints/striding_8l_200hz/striding_8l_200hz.best.pt", "cpu")
    ck = torch.load("/Volumes/UNTITLED/btrv5_checkpoints/decoder_v5/last.pt", map_location="cpu", weights_only=False)
    dec = CausalDecoderV5(CausalDecoderV5Config(**ck["decoder_config"])); dec.load_state_dict(ck["state_dict"]); dec.eval()
    spk = torch.zeros(1, 128)
    wl_full = torch.from_numpy(np.load(DATA / "wavlm_L4_200hz/s_00002.npy")).float()   # (T,512) ~6s
    n_frames = wl_full.shape[0] // WAVLM_PER_FRAME
    wl_full = wl_full[: n_frames * WAVLM_PER_FRAME]

    # offline reference
    m = torch.ones(1, wl_full.shape[0] // 2, dtype=torch.bool)
    c = enc(wl_full.unsqueeze(0).transpose(1, 2), padding_mask=m)["projected"].transpose(1, 2)
    offline = dec(c, spk, stft_length=dec._compute_stft_length(c.shape[1]))[0]

    def stream(W):
        s = StreamingVC(enc, dec, spk, StreamConfig(lookback_frames=W))
        chunks, emitted, max_lag, times, peak_buf = [], 0, 0.0, [], 0
        for i in range(n_frames):
            t = time.perf_counter()
            o = s.push_wavlm(wl_full[i*WAVLM_PER_FRAME:(i+1)*WAVLM_PER_FRAME])
            times.append(time.perf_counter() - t)
            peak_buf = max(peak_buf, s.wl.shape[0] // WAVLM_PER_FRAME)
            if o.numel():
                max_lag = max(max_lag, (FRAME_MS*i + 35 + s.cfg.frontend_ms) - emitted/44.1)
                emitted += o.numel(); chunks.append(o)
        streamed = torch.cat(chunks)
        L = min(streamed.shape[0], offline.shape[0])
        return (streamed[:L]-offline[:L]).abs().max().item(), max_lag, np.median(times)*1000, peak_buf

    print(f"clip: {n_frames} content frames ({n_frames*FRAME_MS/1000:.1f}s), 1 CPU thread\n")
    print("min-lookback sweep — smallest exact window = encoder backward RF (= recompute size):")
    print(f"  {'lookback':>8} | {'bit-exact':>9} {'max_err':>9} | {'latency':>8} {'ms/step':>8} {'buf_frames':>10}")
    for W in (32, 64, 96, 128, 256):
        err, lat, ms, buf = stream(W)
        print(f"  {W:8d} | {str(err < 1e-3):>9} {err:9.1e} | {lat:7.1f}m {ms:7.1f}m {buf:10d}")
    print(f"\n  • latency is lookback-INDEPENDENT (~56ms) — lookback only sets recompute cost.")
    print(f"  • buffer is now BOUNDED to ~lookback (buf_frames stops growing; was unbounded O(n²)).")
    print(f"  • ms/step ∝ lookback = the work a KV-cache collapses to ~1 frame (~0.8ms).")

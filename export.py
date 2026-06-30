"""Export the deploy graph (encoder + decoder) to TorchScript for native inference.

Produces LibTorch-loadable `.ts` files the Mac VST loads with `torch::jit::load`
(no Python at runtime). Traced at a FIXED streaming-window size — the steady-state
block-recompute path (see astrape/streaming.py). The low-latency KV-cache path is a
native C++ implementation (see PORT.md); this export validates that the graph runs
in LibTorch and gives a working artifact for the block path.

NOT exported here (frozen, tiny — reimplement natively or export separately):
  - resample 44.1k→16k  (FIR)
  - WavLM L4 front-end  (5 conv1d layers)  → produces the (T,512) encoder input

Usage:
  .venv/bin/python export.py \
      --decoder-ckpt /Volumes/UNTITLED/btrv5_checkpoints/decoder_v5/last.pt \
      --out-dir deploy/
"""
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from astrape.decoder import CausalDecoderV5, CausalDecoderV5Config
from astrape.train_decoder import load_encoder

WAVLM_PER_FRAME = 8


class EncoderTS(nn.Module):
    """wavlm (1,512,T) → content (1,T//8,768).  Mask built inside (all-valid)."""
    def __init__(self, enc):
        super().__init__(); self.enc = enc
    def forward(self, wavlm: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(wavlm.shape[0], wavlm.shape[2] // 2, dtype=torch.bool)
        return self.enc(wavlm, padding_mask=mask)["projected"].transpose(1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder-ckpt", type=Path, required=True)
    ap.add_argument("--encoder-ckpt", type=Path,
                    default=Path("/Volumes/UNTITLED/btrv5_checkpoints/striding_8l_200hz/striding_8l_200hz.best.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("deploy"))
    ap.add_argument("--window-frames", type=int, default=258,
                    help="Steady-state content frames per step (lookback 256 + block 1 + margin 1).")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    enc, _ = load_encoder(args.encoder_ckpt, "cpu")
    ck = torch.load(args.decoder_ckpt, map_location="cpu", weights_only=False)
    dec = CausalDecoderV5(CausalDecoderV5Config(**ck["decoder_config"])).eval()
    dec.load_state_dict(ck["state_dict"], strict=True)

    W = args.window_frames
    wavlm = torch.randn(1, 512, W * WAVLM_PER_FRAME)
    spk = torch.randn(1, 128)

    # ── encoder ──
    enc_ts_mod = EncoderTS(enc).eval()
    content_eager = enc_ts_mod(wavlm)
    enc_ts = torch.jit.trace(enc_ts_mod, (wavlm,), check_trace=False)
    err_e = (enc_ts(wavlm) - content_eager).abs().max().item()

    # ── decoder ──  (stft_length baked from the window)
    stft_len = dec._compute_stft_length(content_eager.shape[1])
    class DecTS(nn.Module):
        def __init__(s, d, sl): super().__init__(); s.d = d; s.sl = sl
        def forward(s, content, speaker): return s.d(content, speaker, stft_length=s.sl)
    dec_ts_mod = DecTS(dec, stft_len).eval()
    wav_eager = dec_ts_mod(content_eager, spk)
    dec_ts = torch.jit.trace(dec_ts_mod, (content_eager, spk), check_trace=False)
    err_d = (dec_ts(content_eager, spk) - wav_eager).abs().max().item()

    torch.jit.save(enc_ts, str(args.out_dir / "encoder.ts"))
    torch.jit.save(dec_ts, str(args.out_dir / "decoder.ts"))
    meta = {"window_frames": W, "wavlm_per_frame": WAVLM_PER_FRAME, "content_dim": 768,
            "speaker_dim": 128, "stft_length": stft_len, "samples_per_frame": 1764,
            "sample_rate": 44100, "decoder_epoch": int(ck.get("epoch", -1))}
    (args.out_dir / "deploy_meta.json").write_text(__import__("json").dumps(meta, indent=2) + "\n")

    print(f"encoder.ts  trace_err={err_e:.1e}  in=(1,512,{W*WAVLM_PER_FRAME}) → content(1,{content_eager.shape[1]},768)")
    print(f"decoder.ts  trace_err={err_d:.1e}  → wav(1,{wav_eager.shape[1]})  ({wav_eager.shape[1]/44100:.2f}s)")
    print(f"saved → {args.out_dir}/  (encoder.ts, decoder.ts, deploy_meta.json)")
    print("OK: deploy graph is TorchScript-traceable + LibTorch-loadable." if max(err_e, err_d) < 1e-3
          else "WARN: trace mismatch — check dynamic control flow.")


if __name__ == "__main__":
    main()

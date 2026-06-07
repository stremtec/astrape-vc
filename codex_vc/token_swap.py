"""
Token Swap VC: zero-shot voice conversion via Mimi RVQ code swapping.

src LV0 (content) + tgt LV1-7 (speaker) → Mimi decode → VC audio.

Parallel text Δ=+0.64, fully causal, streaming-capable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import soundfile as sf
import torch
from moshi.models import loaders
from pathlib import Path
from scipy import signal

SR = 24000
STRIDE = 1920


def load_audio(path: str, duration: float | None = None) -> torch.Tensor:
    data, orig_sr = sf.read(path)
    if orig_sr != SR:
        data = signal.resample(data, int(len(data) * SR / orig_sr), axis=0)
    if duration is not None:
        L = int(duration * SR) - (int(duration * SR) % STRIDE)
        data = data[:L]
    else:
        L = len(data) - (len(data) % STRIDE)
        data = data[:L]
    if data.ndim > 1:
        data = data.mean(axis=1)
    return torch.from_numpy(data.copy()).float().unsqueeze(0).unsqueeze(0)


def token_swap_convert(mimi, src_audio: torch.Tensor, tgt_audio: torch.Tensor) -> torch.Tensor:
    """Swap LV1-7 codes from target into source."""
    with torch.no_grad():
        codes_src = mimi.encode(src_audio)
        codes_tgt = mimi.encode(tgt_audio)
        T = min(codes_src.shape[2], codes_tgt.shape[2])
        codes_vc = codes_src[:, :, :T].clone()
        codes_vc[:, 1:, :] = codes_tgt[:, 1:, :T]  # src LV0 + tgt LV1-7
        return mimi.decode(codes_vc)


def main():
    parser = argparse.ArgumentParser(description="Token Swap VC (zero-shot)")
    parser.add_argument("--source", required=True, help="Source audio file")
    parser.add_argument("--target", required=True, help="Target audio file (speaker reference)")
    parser.add_argument("--output", default="vc_output.wav", help="Output audio file")
    parser.add_argument("--duration", type=float, default=3.0, help="Audio duration in seconds")
    args = parser.parse_args()

    print("Loading Mimi...")
    mimi = loaders.get_mimi(
        Path.home() / ".cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/"
        "snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/"
        "tokenizer-e351c8d8-checkpoint125.safetensors"
    )
    for p in mimi.parameters():
        p.requires_grad_(False)

    print("Loading audio...")
    src = load_audio(args.source, duration=args.duration)
    tgt = load_audio(args.target, duration=args.duration)

    print("Converting...")
    vc = token_swap_convert(mimi, src, tgt)

    T_out = min(vc.shape[2], src.shape[2])
    sf.write(args.output, vc[0, 0, :T_out].numpy(), SR)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

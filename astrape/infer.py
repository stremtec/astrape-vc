"""Zero-shot VC inference with the trained v5 decoder — the full deploy pipeline.

  source audio → WavLM L4 (200 Hz) → frozen Q2D2 encoder → content (25 Hz)
              → CausalDecoderV5 + target speaker (.astrape voicebank) → wav 44.1 kHz

Unlike the encoder-only eval (which synthesised with the MioCodec *teacher*
decoder), this runs OUR encoder + OUR decoder end-to-end. `--also-mio` additionally
decodes the same content+speaker with the teacher decoder as an A/B reference.

Usage:
  python -m astrape.infer --source src.wav --target p225.astrape \
      --decoder-ckpt /Volumes/UNTITLED/btrv5_checkpoints/decoder_v5/last.pt \
      --output vc_out.wav [--also-mio] [--device cpu]
"""
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torchaudio

from .miocodec import load_mio, load_wave, SAMPLE_RATE, extract_chunk_embeddings
from .decoder import CausalDecoderV5, CausalDecoderV5Config
from .train_decoder import load_encoder


@torch.no_grad()
def extract_wavlm_l4(mio, wav: torch.Tensor, device) -> torch.Tensor:
    """44.1 kHz mono → WavLM L4 raw features (T, 512) @200 Hz — the encoder's input.
    Same extraction as `astrape.cache --what wavlm` (5 causal conv layers @16 kHz)."""
    fe = mio.ssl_feature_extractor.model.feature_extractor
    x = torchaudio.functional.resample(wav.unsqueeze(0), SAMPLE_RATE, 16000).to(device)
    for li in range(5):
        layer = fe.conv_layers[li]
        x = layer.conv(x)
        if getattr(layer, "layer_norm", None) is not None:
            x = layer.layer_norm(x.unsqueeze(0)).squeeze(0) if x.dim() == 2 else layer.layer_norm(x)
        x = F.gelu(x)
    return x.squeeze(0).transpose(0, 1)              # (T, 512)


@torch.no_grad()
def speaker_embedding(mio, target: Path, device):
    """Target 128-d global embedding from a .astrape voicebank or reference audio."""
    if target.suffix == ".astrape":
        from .voicebank import VoiceBank
        vb = VoiceBank.load(target)
        return vb.global_embedding.float().reshape(-1).to(device), (vb.source_path or target.name)
    wav = load_wave(target, SAMPLE_RATE)
    embs = extract_chunk_embeddings(mio, wav, SAMPLE_RATE, device=str(device))
    return torch.stack(embs).mean(0).reshape(-1).to(device), f"{wav.shape[0]/SAMPLE_RATE:.1f}s ref audio"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True, help=".astrape voicebank or reference audio")
    ap.add_argument("--decoder-ckpt", type=Path, required=True)
    ap.add_argument("--encoder-ckpt", type=Path,
                    default=Path("/Volumes/UNTITLED/btrv5_checkpoints/striding_8l_200hz/striding_8l_200hz.best.pt"))
    ap.add_argument("--output", type=Path, default=Path("vc_out.wav"))
    ap.add_argument("--device", default="cpu", help="cpu keeps MPS free for a running training job")
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--also-mio", action="store_true",
                    help="Also synthesise via the MioCodec teacher decoder (A/B reference).")
    args = ap.parse_args()
    dev = torch.device(args.device)
    import soundfile as sf

    print(f"Loading MioCodec + encoder + v5 decoder on {dev} ...", flush=True)
    mio = load_mio(args.device).eval()
    encoder, _ = load_encoder(args.encoder_ckpt, args.device)
    ck = torch.load(args.decoder_ckpt, map_location="cpu", weights_only=False)
    decoder = CausalDecoderV5(CausalDecoderV5Config(**ck["decoder_config"])).to(dev).eval()
    decoder.load_state_dict(ck["state_dict"], strict=True)
    print(f"  decoder epoch={ck.get('epoch')}  "
          f"params={sum(p.numel() for p in decoder.parameters())/1e6:.2f}M", flush=True)

    # source → WavLM → content
    src_wav = load_wave(args.source, SAMPLE_RATE, args.max_seconds).to(dev)
    with torch.no_grad():
        wavlm = extract_wavlm_l4(mio, src_wav, dev).unsqueeze(0)          # (1, T, 512)
        mask = torch.ones(1, wavlm.shape[1] // 2, dtype=torch.bool, device=dev)
        content = encoder(wavlm.transpose(1, 2), padding_mask=mask)["projected"].transpose(1, 2)
    print(f"  source {src_wav.shape[0]/SAMPLE_RATE:.1f}s → content {tuple(content.shape)}", flush=True)

    spk, spk_src = speaker_embedding(mio, args.target, dev)
    print(f"  target speaker: {spk_src}", flush=True)

    # v5 decode
    with torch.no_grad():
        stft_len = decoder._compute_stft_length(content.shape[1])
        out = decoder(content, spk.unsqueeze(0), stft_length=stft_len).squeeze(0).cpu()
    sf.write(str(args.output), out.numpy(), SAMPLE_RATE)
    src_copy = args.output.with_name(args.output.stem + "_source.wav")
    sf.write(str(src_copy), src_wav.cpu().numpy(), SAMPLE_RATE)
    print(f"  ✓ v5 → {args.output}  ({out.shape[0]/SAMPLE_RATE:.1f}s)   [source: {src_copy}]", flush=True)

    # optional A/B reference: MioCodec teacher decoder on the same content+speaker
    if args.also_mio:
        with torch.no_grad():
            mio_len = mio._calculate_target_stft_length(src_wav.numel())
            ref = mio.forward_wave(content, spk.unsqueeze(0), stft_length=mio_len).squeeze(0).cpu()
        ref_path = args.output.with_name(args.output.stem + "_mio.wav")
        sf.write(str(ref_path), ref.numpy(), SAMPLE_RATE)
        print(f"  ✓ MioCodec teacher (A/B) → {ref_path}", flush=True)


if __name__ == "__main__":
    main()

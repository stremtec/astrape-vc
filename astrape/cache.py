"""Unified caching CLI (replaces cache_wavlm_*.py + cache_speaker_embeddings.py).

  --what wavlm     WavLM L4 raw 200Hz features → <data>/wavlm_L4_200hz/   (encoder frontend)
                   5 causal conv layers @16kHz, saved (T, 512) @200Hz, 10ms delay.
  --what speakers  per-speaker MioCodec global centroids → <data>/spk_centroids.npz (decoder)
                   chunked + energy-gated + averaged over several utterances.

Examples:
  .venv/bin/python -m astrape.cache --what wavlm --limit 0
  .venv/bin/python -m astrape.cache --what speakers --utts-per-speaker 8
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from .miocodec import load_mio, load_wave, SAMPLE_RATE, extract_chunk_embeddings


def cache_wavlm(args):
    data_dir = Path(args.data_dir)
    meta = np.load(data_dir / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"]); srcs = meta["source_files"][:n].astype(str)
    end = n if args.limit == 0 else min(n, args.start + args.limit)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "wavlm_L4_200hz"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Caching WavLM L4 raw (200Hz): {args.start}..{end-1} ({end-args.start} total)")

    mio = load_mio(args.device).eval()
    fe = mio.ssl_feature_extractor.model.feature_extractor
    for i in range(args.start, end):
        out_path = out_dir / f"s_{i:05d}.npy"
        if out_path.exists():
            continue
        wav = load_wave(Path(str(srcs[i])), SAMPLE_RATE, max_seconds=args.max_s)
        wav_16 = torchaudio.functional.resample(wav.unsqueeze(0), SAMPLE_RATE, 16000).squeeze(0)
        with torch.no_grad():
            x = wav_16.unsqueeze(0)
            for layer_idx in range(5):
                layer = fe.conv_layers[layer_idx]
                x = layer.conv(x)
                if hasattr(layer, "layer_norm") and layer.layer_norm is not None:
                    x = (layer.layer_norm(x.unsqueeze(0)).squeeze(0)
                         if x.dim() == 2 else layer.layer_norm(x))
                x = F.gelu(x)
        cnn = x.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)  # (T, 512)
        np.save(out_path, cnn)
        if (i - args.start) % 100 == 0:
            print(f"{i - args.start}/{end - args.start}")
    print(f"Done: {end - args.start} samples")


def cache_speakers(args):
    data_dir = Path(args.data_dir)
    meta = np.load(data_dir / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"])
    spk_names = meta["spk_names"][:n].astype(str); src = meta["source_files"][:n].astype(str)
    by_spk: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(spk_names):
        by_spk[s].append(i)
    speakers = sorted(by_spk)
    print(f"{len(speakers)} speakers, up to {args.utts_per_speaker} utterances each", flush=True)

    mio = load_mio(args.device).eval()
    rng = np.random.default_rng(0)
    embeddings = []
    for j, spk in enumerate(speakers):
        chosen = rng.choice(by_spk[spk], size=min(args.utts_per_speaker, len(by_spk[spk])), replace=False)
        chunks: list[torch.Tensor] = []
        for i in chosen:
            w = load_wave(Path(src[int(i)]), SAMPLE_RATE)
            chunks += extract_chunk_embeddings(mio, w, SAMPLE_RATE, device=args.device)
        embeddings.append(torch.stack(chunks).mean(0).numpy())
        if j % 10 == 0:
            print(f"  {j}/{len(speakers)} ({spk})", flush=True)

    out = Path(args.out_dir) if args.out_dir else data_dir / "spk_centroids.npz"
    np.savez(out, speakers=np.array(speakers), embeddings=np.stack(embeddings).astype(np.float32))
    print(f"Wrote {out}: {len(speakers)} speaker centroids (128-d)")


def _load_encoder(ckpt, device):
    """Frozen Q2D2 encoder from a checkpoint (same recipe as train_decoder.load_encoder)."""
    from .encoder import MCSTransQ2D2Config, MCSTransQ2D2
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    scfg = {k: tuple(v) if isinstance(v, list) else v
            for k, v in ck.get("config", {}).items() if not k.startswith("_")}
    scfg["use_wavlm_frontend"] = True
    known = set(MCSTransQ2D2Config.__dataclass_fields__.keys())
    model = MCSTransQ2D2(MCSTransQ2D2Config(**{k: v for k, v in scfg.items() if k in known}))
    model.load_state_dict(ck["state_dict"], strict=False)
    return model.to(device).eval()


def cache_content(args):
    """FULL-context content per clip: run the frozen encoder over each clip's WHOLE WavLM
    cache → content (T_content, 768). The decoder then trains on cropped content windows
    that carry their real preceding context (matches streaming inference) instead of
    re-encoding short 50-frame WavLM crops (cold-start) every step."""
    data_dir = Path(args.data_dir)
    meta = np.load(data_dir / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"])
    end = n if args.limit == 0 else min(n, args.start + args.limit)
    wl_dir = data_dir / "wavlm_L4_200hz"
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / args.content_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.encoder_ckpt:
        raise SystemExit("--encoder-ckpt is required for --what content")
    enc = _load_encoder(args.encoder_ckpt, args.device)
    print(f"Caching full-context content → {out_dir}: {args.start}..{end-1} "
          f"({end-args.start} total)", flush=True)
    done = 0
    for i in range(args.start, end):
        out_path = out_dir / f"s_{i:05d}.npy"
        if out_path.exists():
            continue
        wl_path = wl_dir / f"s_{i:05d}.npy"
        if not wl_path.exists():
            continue
        wl = torch.from_numpy(np.load(wl_path, allow_pickle=False)).float().to(args.device)  # (T,512)
        with torch.no_grad():
            mask = torch.ones(1, wl.shape[0] // 2, dtype=torch.bool, device=args.device)
            content = enc(wl.unsqueeze(0).transpose(1, 2), padding_mask=mask)["projected"]   # (1,768,Tc)
        np.save(out_path, content.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32))  # (Tc,768)
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{end - args.start}", flush=True)
    print(f"Done: {done} content files cached in {out_dir}", flush=True)


def cache_acoustics(args):
    """Ground-truth acoustic targets @150Hz per clip (Stage-A regression target / Stage-B
    conditioning): mel80 + logF0 + voicing + energy, LEFT-ALIGNED causal framing. Aligned to
    the content grid: audio is padded/cropped to Tc*1764 so T_acoustic = Tc*6 exactly."""
    from .acoustics import extract_acoustics, melscale_fbanks, CONTENT_HOP
    data_dir = Path(args.data_dir)
    meta = np.load(data_dir / "meta.npz", allow_pickle=False)
    n = int(meta["n_samples"]); srcs = meta["source_files"][:n].astype(str)
    end = n if args.limit == 0 else min(n, args.start + args.limit)
    content_dir = data_dir / args.content_dir     # to read Tc (content length) per clip
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "acoustics_150hz"
    out_dir.mkdir(parents=True, exist_ok=True)
    fb = melscale_fbanks()
    print(f"Caching acoustics (150Hz, causal) → {out_dir}: {args.start}..{end-1}", flush=True)
    done = 0
    for i in range(args.start, end):
        out_path = out_dir / f"s_{i:05d}.npz"
        if out_path.exists():
            continue
        cpath = content_dir / f"s_{i:05d}.npy"
        if not cpath.exists():
            continue
        Tc = int(np.load(cpath, allow_pickle=False).shape[0])
        wav = load_wave(Path(str(srcs[i])), SAMPLE_RATE, max_seconds=args.max_s)
        L = Tc * CONTENT_HOP
        wav = F.pad(wav, (0, L - wav.shape[0]))[:L] if wav.shape[0] < L else wav[:L]
        a = extract_acoustics(wav, fb)
        np.savez(out_path,
                 mel=a["mel"].numpy().astype(np.float16),
                 logf0=a["logf0"].numpy().astype(np.float16),
                 voiced=a["voiced"].numpy().astype(np.float16),
                 energy=a["energy"].numpy().astype(np.float16))
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{end - args.start}", flush=True)
    print(f"Done: {done} acoustic files cached in {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["wavlm", "speakers", "content", "acoustics"], required=True)
    ap.add_argument("--data-dir", default="data/mio_vctk_full_compact")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=None)
    # wavlm / content
    ap.add_argument("--max-s", type=float, default=6.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    # content
    ap.add_argument("--encoder-ckpt", default=None, help="frozen encoder checkpoint (for --what content)")
    ap.add_argument("--content-dir", default="content_striding_8l_200hz",
                    help="output subdir under data-dir for cached content")
    # speakers
    ap.add_argument("--utts-per-speaker", type=int, default=8)
    args = ap.parse_args()
    {"wavlm": cache_wavlm, "speakers": cache_speakers, "content": cache_content,
     "acoustics": cache_acoustics}[args.what](args)


if __name__ == "__main__":
    main()

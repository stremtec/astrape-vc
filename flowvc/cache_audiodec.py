"""
Pre-cache AudioDec latents for VCTK dataset.

Eliminates real-time encoding during training (major speedup).
Stores: content latent (64-dim), speaker embedding (from ECAPA), prosody (3-dim).
"""

from __future__ import annotations
import os, sys, argparse, random, torch
from pathlib import Path

sys.path.insert(0, '/Users/asill/btrvrc0')
from flowvc.audiodec import AudioDecEncoder, AUDIODEC_SR, AUDIODEC_DIM
from flowvc.dataset import VCTKDataset
import torchaudio


def build_cache(data_dir: str, cache_dir: str, device: str = "mps", max_files: int = 0):
    os.makedirs(cache_dir, exist_ok=True)
    dev = torch.device(device)
    
    enc = AudioDecEncoder(device=str(dev)).to(dev).eval()
    ds = VCTKDataset(data_dir, crop_seconds=2.0, sample_rate=AUDIODEC_SR)
    
    total = min(len(ds), max_files) if max_files > 0 else len(ds)
    print(f"Caching {total} files from {len(ds)} total...")
    
    for i in range(total):
        item = ds[i]
        src = item["src_wav"].unsqueeze(0).to(dev)
        
        # Extract speaker from file path (VCTK format: pXXX_...)
        src_path = ds.files[i]
        spk_id = Path(src_path).parent.name
        
        with torch.no_grad():
            z = enc.encode(src)  # (1, T, 64)
        
        cache = {"z": z.cpu(), "speaker_id": spk_id}
        torch.save(cache, os.path.join(cache_dir, f"{i:06d}.pt"))
        
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{total}")
    
    print(f"Done: {cache_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--cache-dir", default="./cache_audiodec")
    p.add_argument("--device", default="mps")
    p.add_argument("--max-files", type=int, default=0)
    args = p.parse_args()
    build_cache(args.data_dir, args.cache_dir, args.device, args.max_files)

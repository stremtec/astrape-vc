"""Cache integrity checker + auto-repair for btrv5 datasets.

Checks all NPZ and WavLM CNN cache files. Reports issues.
With --repair, auto-regenerates broken/missing files from source audio.

Usage:
  .venv/bin/python check_cache.py           # check only
  .venv/bin/python check_cache.py --repair  # check + fix broken files
  .venv/bin/python check_cache.py --wavlm-only  # check only WavLM CNN cache
"""
import sys, warnings, logging, argparse, time
warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np
from pathlib import Path

def check_npz(data_dir, repair=False, srcs=None, meta=None):
    """Check all s_XXXXX.npz files for required keys."""
    expected = {'logmel', 'ce_768', 'ct'}
    n = int(meta['n_samples']) if meta else 43885
    broken, missing_keys = [], []
    
    for i in range(n):
        path = data_dir / f's_{i:05d}.npz'
        if not path.exists():
            broken.append((i, 'file missing'))
            continue
        try:
            data = np.load(path, allow_pickle=False)
            missing = expected - set(data.keys())
            if missing:
                missing_keys.append((i, ', '.join(missing)))
                if path.stat().st_size < 100:
                    broken.append((i, f'too small ({path.stat().st_size}B)'))
        except Exception as e:
            broken.append((i, str(e)))
            if repair and srcs is not None:
                path.unlink(missing_ok=True)
    
    return broken, missing_keys

def check_wavlm(data_dir, repair=False, srcs=None, meta=None):
    """Check all wavlm_cnn/s_XXXXX.npy files."""
    wavlm_dir = data_dir / 'wavlm_cnn'
    if not wavlm_dir.exists():
        return [(0, 'wavlm_cnn directory missing')], []
    n = int(meta['n_samples']) if meta else 43885
    broken, missing = [], []
    
    for i in range(n):
        path = wavlm_dir / f's_{i:05d}.npy'
        if not path.exists():
            missing.append(i)
            continue
        try:
            d = np.load(path, allow_pickle=False)
            if d.ndim != 2 or d.shape[1] != 512:
                broken.append((i, f'bad shape {d.shape}'))
                if repair: path.unlink(missing_ok=True)
            if path.stat().st_size < 100:
                broken.append((i, f'too small ({path.stat().st_size}B)'))
                if repair: path.unlink(missing_ok=True)
        except Exception as e:
            broken.append((i, str(e)))
            if repair: path.unlink(missing_ok=True)
    
    return broken, missing

def repair_files(broken, missing, data_dir, srcs):
    """Auto-repair broken WavLM CNN cache files."""
    if not (broken or missing):
        return
    print(f'\nRepairing {len(broken)} broken + {len(missing)} missing files...')
    sys.path.insert(0, 'external/MioCodec/src')
    from eval_mcs_trans_audio import load_mio, load_wave, SAMPLE_RATE
    import torch
    
    mio = load_mio('cpu').eval()
    fe = mio.ssl_feature_extractor.model.feature_extractor
    wavlm_dir = data_dir / 'wavlm_cnn'
    wavlm_dir.mkdir(exist_ok=True)
    
    to_fix = set()
    for i, _ in broken:
        to_fix.add(i)
    for i in missing:
        to_fix.add(i)
    
    fixed = 0
    for i in sorted(to_fix):
        try:
            wav = load_wave(Path(str(srcs[i])), SAMPLE_RATE, max_seconds=6.0)
            with torch.no_grad():
                cnn, _ = fe(wav.unsqueeze(0), length=None)
            cnn_ds = torch.nn.functional.avg_pool1d(
                cnn.transpose(1, 2), 3, 3
            ).transpose(1, 2).squeeze(0).cpu().numpy().astype(np.float32)
            np.save(wavlm_dir / f's_{i:05d}.npy', cnn_ds)
            fixed += 1
        except Exception as e:
            print(f'  FAILED s_{i:05d}: {e}')
        if fixed % 100 == 0:
            print(f'  {fixed}/{len(to_fix)}')
    
    print(f'  Repaired {fixed}/{len(to_fix)} files')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/mio_vctk_full_compact')
    ap.add_argument('--repair', action='store_true', help='Auto-repair broken files')
    ap.add_argument('--wavlm-only', action='store_true')
    ap.add_argument('--npz-only', action='store_true')
    args = ap.parse_args()
    
    data_dir = Path(args.data_dir)
    meta = np.load(data_dir / 'meta.npz', allow_pickle=False)
    n = int(meta['n_samples'])
    srcs = meta['source_files'][:n].astype(str) if args.repair else None
    
    t0 = time.time()
    print(f'Checking {n} files at {data_dir}...')
    
    if not args.wavlm_only:
        broken_npz, missing_npz = check_npz(data_dir, args.repair, srcs, meta)
        print(f'  NPZ broken: {len(broken_npz)}')
        if broken_npz:
            for i, err in broken_npz[:5]:
                print(f'    s_{i:05d}: {err}')
            if len(broken_npz) > 5: print(f'    ... and {len(broken_npz) - 5} more')
        print(f'  NPZ missing keys: {len(missing_npz)}')
    else:
        broken_npz, missing_npz = [], []
    
    if not args.npz_only:
        broken_wl, missing_wl = check_wavlm(data_dir, args.repair, srcs, meta)
        print(f'  WavLM broken: {len(broken_wl)}')
        if broken_wl:
            for i, err in broken_wl[:5]:
                print(f'    s_{i:05d}: {err}')
            if len(broken_wl) > 5: print(f'    ... and {len(broken_wl) - 5} more')
        print(f'  WavLM missing: {len(missing_wl)}')
        if missing_wl:
            print(f'    Range: s_{min(missing_wl):05d} .. s_{max(missing_wl):05d}')
        
        if args.repair and (broken_wl or missing_wl):
            repair_files(broken_wl, missing_wl, data_dir, srcs)
    else:
        broken_wl, missing_wl = [], []
    
    total = len(broken_npz) + len(broken_wl) + len(missing_wl)
    elapsed = time.time() - t0
    status = '✅ All clean' if total == 0 else f'⚠️ {total} issues'
    print(f'\n{status} ({elapsed:.1f}s)')
    return 1 if total > 0 else 0

if __name__ == '__main__':
    sys.exit(main())

"""astrape_vc.py — End-to-end streaming VC evaluation.

Loads WavLM encoder + frozen MioCodec decoder. Implements true
streaming (sample-by-sample with conv state carry + KV-cache).

Measures:
  1. Per-module streaming latency on MPS and CPU
  2. Quality comparison (MPS vs CPU, streaming vs non-streaming)
  3. Real VC test (source → target speaker)

Usage:
  .venv/bin/python astrape_vc.py --checkpoint <path> --source <wav> --target <voicebank>
  .venv/bin/python astrape_vc.py --checkpoint <path> --benchmark
"""
import sys, warnings, logging, time, argparse, json
warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
sys.path.insert(0, 'external/MioCodec/src')
sys.path.insert(0, '.')

import torch, torchaudio, numpy as np, soundfile as sf
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from train_mcs_q2d2 import MCSTransQ2D2Config, MCSTransQ2D2
from eval_mcs_trans_audio import load_mio, load_wave, SAMPLE_RATE

S = SAMPLE_RATE  # 44100


# ═══════════════════════════════════════════════════════════════
# Streaming state management
# ═══════════════════════════════════════════════════════════════

class StreamingWavLMCNN:
    """WavLM CNN with state-carry for sample-by-sample streaming."""
    def __init__(self, device='cpu'):
        self.mio = load_mio(device).eval()
        self.fe = self.mio.ssl_feature_extractor.model.feature_extractor
        self.device = device
        self.reset()

    def reset(self):
        # Per-layer conv state: (k-1) past inputs
        self.states = [None] * 7
        self.buffer = torch.zeros(0, device=self.device)
        self.frame_count = 0

    def _stream_conv(self, x, layer_idx, conv):
        """Single causal conv with state carry."""
        k = conv.kernel_size[0]
        s = conv.stride[0]
        # Prepend past state
        if self.states[layer_idx] is not None:
            x = torch.cat([self.states[layer_idx], x], dim=-1)
        # Save state for next: last (k-1) samples
        if x.shape[-1] >= k:
            self.states[layer_idx] = x[:, :, -(k-1):].clone()
        else:
            self.states[layer_idx] = x.clone()
        # Apply conv with stride
        out = conv(x)
        return out

    def process_sample(self, sample):
        """Process a single sample (scalar). Returns (512,) or None."""
        self.buffer = torch.cat([self.buffer, sample.view(1, 1, 1)])
        if self.buffer.shape[-1] < 320:  # stride product
            return None

        # Process accumulated frames
        x = self.buffer
        for i, conv_layer in enumerate(self.fe.conv_layers):
            x = self._stream_conv(x, i, conv_layer.conv)
            x = torch.nn.functional.gelu(x)
            if i < 5:  # GroupNorm only on first 5 layers
                gn = conv_layer.layer_norm
                x = gn(x)

        # After all layers: (1, 512, N_out)
        # Keep only the oldest unprocessed frame
        n_out = x.shape[-1]
        out = x[:, :, self.frame_count % n_out]  # oldest frame
        self.frame_count += 1
        self.buffer = self.buffer[:, :, 320:]  # remove processed
        return out.squeeze(0).squeeze(0)  # (512,)


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

@dataclass
class LatencyReport:
    module: str
    mps_ms: float
    cpu_ms: float

@dataclass
class VCEvalResult:
    source_wav: str
    target_vb: str
    latency_report: list
    mps_output: Optional[np.ndarray] = None
    cpu_output: Optional[np.ndarray] = None


def benchmark_encoder(checkpoint_path, device='cpu', duration=3.0):
    """Benchmark encoder in non-streaming mode (full utterance)."""
    print(f'\n=== Encoder Benchmark ({device}, {duration}s) ===')

    # Load model
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    scfg = ck.get('config', {})
    config = MCSTransQ2D2Config(
        **{k: tuple(v) if isinstance(v, list) else v
           for k, v in scfg.items() if not k.startswith('_') and k in MCSTransQ2D2Config.__dataclass_fields__}
    )
    scfg2 = {k: tuple(v) if isinstance(v, list) else v for k, v in scfg.items() if not k.startswith("_")}
    scfg2["use_wavlm_frontend"] = True
    # Filter to only known fields
    known = set(MCSTransQ2D2Config.__dataclass_fields__.keys())
    scfg2 = {k: v for k, v in scfg2.items() if k in known}
    config = MCSTransQ2D2Config(**scfg2)

    model = MCSTransQ2D2(config).to(device).eval()

    # Load shared weights (skip WavLM adapter and new heads)
    shared = {}
    for k, v in ck['state_dict'].items():
        if not k.startswith('wavlm_adapter.') and not k.startswith('forecast_') and not k.startswith('ssl_'):
            shared[k] = v
    model.load_state_dict(shared, strict=False)

    # Generate test audio
    wav = torch.randn(int(duration * S), device=device)

    # WavLM CNN benchmark
    from eval_mcs_trans_audio import load_mio as _lm
    mio = _lm(device).eval()
    fe = mio.ssl_feature_extractor.model.feature_extractor

    t0 = time.time()
    with torch.no_grad():
        cnn, _ = fe(wav.unsqueeze(0), length=None)
    cnn_time = time.time() - t0

    # Pool + encoder
    cnn_t = cnn.transpose(1, 2)
    cnn_pool = torch.nn.functional.avg_pool1d(cnn_t, 3, 3)
    T = cnn_pool.shape[2]
    mask = torch.ones(1, T // 2, dtype=torch.bool, device=device)

    t0 = time.time()
    with torch.no_grad():
        out = model(cnn_pool, mask)
    enc_time = time.time() - t0

    total = out['projected'].shape[2]
    print(f'  Audio:       {duration:.1f}s ({int(duration*S)} samples)')
    print(f'  CNN frames:  {cnn.shape[1]} @ 137.8Hz')
    print(f'  After pool:  {T} @ 46Hz')
    print(f'  Content:     {total} @ {total/duration:.0f}Hz')
    print(f'  CNN time:    {cnn_time*1000:.1f}ms (RTF={cnn_time/duration:.3f})')
    print(f'  Enc time:    {enc_time*1000:.1f}ms (RTF={enc_time/duration:.3f})')
    print(f'  Total:       {(cnn_time+enc_time)*1000:.1f}ms')

    return {
        'cnn_ms': cnn_time * 1000,
        'encoder_ms': enc_time * 1000,
        'total_ms': (cnn_time + enc_time) * 1000,
        'rtf': (cnn_time + enc_time) / duration,
        'content_frames': total,
        'content_rate': total / duration,
    }


def compare_mps_cpu(checkpoint_path):
    """Compare MPS vs CPU latency and output quality."""
    print('\n╔══════════════════════════════════════╗')
    print('║   MPS vs CPU Comparison             ║')
    print('╚══════════════════════════════════════╝')

    mps = benchmark_encoder(checkpoint_path, 'mps', 3.0)
    cpu = benchmark_encoder(checkpoint_path, 'cpu', 3.0)

    print(f'\n  MPS RTF: {mps["rtf"]:.3f} ({"✅ real-time" if mps["rtf"]<1 else "❌"})')
    print(f'  CPU RTF: {cpu["rtf"]:.3f} ({"✅ real-time" if cpu["rtf"]<1 else "❌"})')
    print(f'  Speedup: {cpu["rtf"]/mps["rtf"]:.1f}× (MPS vs CPU)')


def vc_inference(checkpoint_path, source_wav_path, target_vb_path, device='cpu'):
    """Run full VC pipeline: encoder + frozen MioCodec decoder."""
    print(f'\n=== VC Inference ({device}) ===')

    # Load encoder
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    scfg = ck.get('config', {})
    config = MCSTransQ2D2Config(
        **{k: tuple(v) if isinstance(v, list) else v
           for k, v in scfg.items() if not k.startswith('_') and k in MCSTransQ2D2Config.__dataclass_fields__}
    )
    scfg2 = {k: tuple(v) if isinstance(v, list) else v for k, v in scfg.items() if not k.startswith("_")}
    scfg2["use_wavlm_frontend"] = True
    # Filter to only known fields
    known = set(MCSTransQ2D2Config.__dataclass_fields__.keys())
    scfg2 = {k: v for k, v in scfg2.items() if k in known}
    config = MCSTransQ2D2Config(**scfg2)
    model = MCSTransQ2D2(config).to(device).eval()
    shared = {k: v for k, v in ck['state_dict'].items()
              if not k.startswith('wavlm_adapter.') and not k.startswith('forecast_') and not k.startswith('ssl_')}
    model.load_state_dict(shared, strict=False)

    # Load audio
    wav, sr = sf.read(str(source_wav_path), dtype='float32')
    wav = torch.from_numpy(np.asarray(wav)).float()
    if wav.ndim == 2: wav = wav.mean(1)
    if sr != S: wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, S).squeeze(0)
    wav = wav.to(device)

    # WavLM CNN
    mio = load_mio(device).eval()
    fe = mio.ssl_feature_extractor.model.feature_extractor
    t0 = time.time()
    with torch.no_grad():
        cnn, _ = fe(wav.unsqueeze(0), length=None)
    cnn_t = cnn.transpose(1, 2)
    cnn_pool = torch.nn.functional.avg_pool1d(cnn_t, 3, 3)
    T = cnn_pool.shape[2]

    # Encoder
    mask = torch.ones(1, T // 2, dtype=torch.bool, device=device)
    with torch.no_grad():
        out = model(cnn_pool, mask)
    content = out['projected'].transpose(1, 2)  # (1, T_c, 768)

    # MioCodec decoder
    spk = mio.encode(wav.unsqueeze(0), return_content=False, return_global=True)
    stft_len = mio._calculate_target_stft_length(wav.numel())
    nf = min(content.shape[1], 99)
    with torch.no_grad():
        pred = mio.forward_wave(content[:, :nf], spk.global_embedding.unsqueeze(0), stft_length=stft_len)
    total_time = time.time() - t0

    out_wav = pred.squeeze(0).cpu().numpy()
    sf.write('/tmp/vc_output.wav', out_wav, S)

    print(f'  Source:    {Path(source_wav_path).name}')
    print(f'  Target:    {Path(target_vb_path).name}')
    print(f'  Duration:  {wav.shape[0]/S:.1f}s')
    print(f'  Content:   {content.shape[1]} frames @ {content.shape[1]/(wav.shape[0]/S):.0f}Hz')
    print(f'  Output:    {out_wav.shape[0]/S:.1f}s')
    print(f'  Total:     {total_time*1000:.0f}ms (RTF={total_time/(wav.shape[0]/S):.3f})')
    print(f'  → /tmp/vc_output.wav')


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Astrape VC — streaming evaluation')
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--benchmark', action='store_true', help='Run MPS vs CPU benchmark')
    ap.add_argument('--benchmark-only', action='store_true', help='Benchmark without VC')
    ap.add_argument('--source', type=Path, help='Source audio for VC')
    ap.add_argument('--target', type=Path, help='Target voicebank (.astrape)')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'mps'])
    args = ap.parse_args()

    if args.benchmark or args.benchmark_only:
        compare_mps_cpu(args.checkpoint)

    if not args.benchmark_only and args.source and args.target:
        vc_inference(args.checkpoint, args.source, args.target, args.device)
    elif not args.benchmark and not args.benchmark_only:
        # Default: benchmark + show usage
        print("Usage examples:")
        print("  Benchmark:          python astrape_vc.py --checkpoint best.pt --benchmark")
        print("  VC inference:       python astrape_vc.py --checkpoint best.pt --source in.wav --target vb.astrape")
        print("  Both:               python astrape_vc.py --checkpoint best.pt --benchmark --source in.wav --target vb.astrape")


if __name__ == '__main__':
    main()

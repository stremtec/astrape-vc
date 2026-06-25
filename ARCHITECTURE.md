# Astrape VC — 16kHz Streaming Architecture

## Full Pipeline (Training)

```
VCTK wav 44.1kHz
  │
  ▼ resample(44.1k→16k)  ← MioCodec ssl.resampler와 동일
  │
  ▼ WavLM CNN (94M, frozen, stride=320)
  │   16,000 / 320 = 50.0 Hz (정확)
  │   → (T, 512) float32 @ 50Hz
  │
  │  [cached to wavlm_16k/s_XXXXX.npy, 14GB]
  │
  ▼ WavLMFrontendAdapter (764K, learned)
  │   Linear(512→256)→GELU→Linear(256→80)
  │   → (T, 80) @ 50Hz
  │
  ▼ Causal Depthwise Stem (1.4M)
  │   8 depthwise blocks, dilations 1-16
  │   → (T, 320) @ 50Hz
  │
  ▼ CellDownsample(2×)
  │   → (T/2, 320) @ 25Hz
  │
  ▼ ProjIn(320→512) → Transformer 7L (13.8M)
  │   RoPE + SwiGLU, window=256
  │   → (T/2, 512) @ 25Hz
  │
  ▼ Q2D2 (8.5K, 3M codes)
  │   → content (T/2, 768) @ 25Hz
  │
  ▼ MioCodec Decoder (228M, frozen)
  │   → wav 44.1kHz
```

## Full Pipeline (Streaming Inference)

```
Mic input 44.1kHz
  │
  ▼ Polyphase resampler (44.1k→16k)
  │   delay: ~2ms
  │
  ▼ WavLM CNN — state-carry per conv layer
  │   7 causal convs, padding=0
  │   output: 1 frame @ 50Hz per 320 samples
  │   algorithmic delay: 400 samples @ 16kHz = 25ms
  │   compute: ~0.4ms/frame (CPU)
  │
  ▼ Adapter — per-frame, 0ms delay
  │
  ▼ Stem + Downsample — state-carry, ~0.3ms
  │   output: 1 frame @ 25Hz per 640 samples
  │
  ▼ Transformer — KV-cache per layer
  │   causal windowed attention (backlog only)
  │   compute: ~0.5ms/frame (CPU)
  │
  ▼ Q2D2 — per-frame, ~0.01ms
  │
  ▼ Decoder → audio output 44.1kHz
```

## Latency Budget

| Component | Algorithmic | Compute (CPU) |
|-----------|-------------|---------------|
| Resampler 44.1k→16k | ~2ms | ~0.1ms |
| WavLM CNN RF | 25ms | 0.4ms |
| Adapter | 0ms | 0.01ms |
| Stem + Downsample | 0ms | 0.3ms |
| Transformer 7L | 0ms | 0.5ms |
| Q2D2 | 0ms | 0.01ms |
| **Encoder Total** | **~27ms** | **~1.3ms** |

## State Carry (per component)

| Module | State | Size |
|--------|-------|------|
| Resampler | polyphase filter state | ~1KB |
| WavLM CNN L0 | last 9 input samples | 9×float32 |
| WavLM CNN L1-L6 | last k-1 frames per layer | ~4KB |
| Stem convs | last k-1 frames per block | ~10KB |
| Transformer | KV-cache (window=256, 7L) | ~7MB |
| **Total state** | | **~7MB** |

## Training vs 44.1kHz Pipeline

| | 44.1kHz (old) | 16kHz (new) |
|---|---------------|-------------|
| CNN input rate | 44.1kHz | **16kHz** |
| CNN output rate | 137.8Hz → pool → 46Hz | **50Hz (native)** |
| avg_pool | 필요 | **불필요** |
| interpolation | 46→50Hz 보간 | **불필요** |
| 8% temporal warp | 있음 | **없음** |
| CNN compute | 44.1k samples/s | **16k samples/s (2.7× faster)** |
| WavLM kernel alignment | misaligned (44.1k→16k mismatch) | **perfectly aligned** |

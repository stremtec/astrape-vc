# astrape-vst

Strict-causal, zero-shot voice conversion at 44.1 kHz — built to run as a
real-time **VST** on Apple Silicon (MPS, no CUDA).
**Encoder best: probe cos768 ≈ 0.935** (StridingAdapter, 8-layer causal Transformer).

## Architecture

```
Mic 44.1kHz
  → resample 44.1k→16k → WavLM CNN L0–L4 (frozen teacher frontend, ~10ms) → 200 Hz
  → StridingAdapter 200→50 Hz (learned, k=4) → Causal Depthwise Stem (8 blocks)
  → Downsample 2× → 25 Hz → Causal Transformer 8L (RoPE + SwiGLU, window 256)
  → Q2D2 quantizer (3.0M-code rhombic grid) → content 768d @ 25 Hz      ── encoder · 24.9M
  → Decoder v5: causal AdaLN transformer → learned upsampler → SnakeBeta
                conv stack → iSTFT (n_fft 1512) → wav 44.1kHz            ── decoder · ~16M
     ▲ target speaker = MioCodec global embedding (.astrape voicebank), via AdaLN-Zero
```

| Stage | Params | In deploy graph? |
|-------|--------|------------------|
| Encoder (StridingAdapter 8L) | 24.9M | ✅ |
| Decoder v5 | ~16M | ✅ |
| MioCodec teacher (frozen, bidirectional) | 228M | ✗ training only |
| MPD + MSD discriminators | 70.7M | ✗ training only |

Deploy graph ≈ **41M**; E2E algorithmic latency ≈ **49 ms**; **0 look-ahead**.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the latency taxonomy and the
encoder / decoder / iSTFT breakdown.

**Encoder frontend ablations** (content cos768 vs frozen MioCodec teacher):

| Frontend | Causality | cos768 | Notes |
|----------|-----------|--------|-------|
| **8L StridingAdapter 200 Hz** | **strict-causal** | **0.935** | ★ current — WavLM L4 raw + learned k=4 decimation |
| 7L WavLM 16 kHz | strict-causal | 0.934 | native 50 Hz, aligned kernels |
| 6L Mel + time-shift | strict-causal | 0.917 | mel fallback |
| 4L Mel center=True | 23 ms future | 0.911 | non-causal reference |

## Design philosophy

- **Strict causality (0 look-ahead).** Output at frame *t* depends only on input
  ≤ *t* — enforced everywhere and checked by a streaming-invariance test (adding
  future samples never changes earlier outputs). This is the non-negotiable
  constraint for real-time streaming VC.
- **Three "latencies", kept distinct.** *Look-ahead* (future context) = **0**.
  *Algorithmic latency* (group delay, e.g. iSTFT overlap-add) is the real
  streaming delay we budget (~50 ms). *Backward receptive field* (past context
  carried as recurrent state) is **free**.
- **Built for a VST.** The deploy graph is just encoder + decoder (~41M); the
  228M non-causal MioCodec teacher and the MPD/MSD discriminators live only at
  training time — **0 inference cost**.
- **Teacher–student distillation.** A frozen, bidirectional MioCodec is the
  teacher; the strict-causal student *predicts* its content embeddings. Since it
  cannot see the future, it learns to anticipate it (forecast heads, time-shift).
- **Self-reconstruction ⇒ zero-shot VC.** Disentanglement (GRL strips speaker
  from content + the Q2D2 bottleneck) makes plain reconstruction training
  generalize to cross-speaker conversion — no parallel data needed.
- **Content / speaker split.** *What* is said (content, 768d @ 25 Hz) and *who*
  says it (a MioCodec global embedding) are separate. Speaker enters only via
  AdaLN-Zero in the decoder, so one content stream drives any target voice from a
  `.astrape` voicebank.
- **Apple-Silicon native.** MPS-only, fp32 (no Tensor Cores to exploit); every op
  is MPS-safe (time-domain discriminators, on-device STFT).

## Training

Everything is in the `astrape/` package — run with `python -m astrape.<name>`.

```
astrape/
  nn  encoder  decoder  discriminators  quantizer  losses  data  miocodec  voicebank   # library
  train_encoder  train_decoder  cache  check_cache  build_voicebank  evaluate          # CLIs
tests/test_streaming_invariant.py
```

**1. Caches (one-time).**
```bash
.venv/bin/python -m astrape.cache --what wavlm    --limit 0              # WavLM L4 200Hz (encoder frontend)
.venv/bin/python -m astrape.cache --what speakers --utts-per-speaker 8   # per-speaker centroids (decoder)
.venv/bin/python -m astrape.check_cache --wavlm-only --wavlm-dir wavlm_L4_200hz   # verify integrity
```

**2. Encoder** — Q2D2 content encoder (frozen after training).
```bash
.venv/bin/python -m astrape.train_encoder \
  --device mps --epochs 60 --steps-per-epoch 2000 --batch-size 2 \
  --n-layers 8 --trans-dim 512 --n-heads 8 --ffn-dim 1024 --window 256 \
  --rope --swiglu --stem-block-type depthwise \
  --q2d2-grid rhombic --q2d2-levels 9,9,9,9,9,9 --q2d2-dim 6 \
  --wavlm-frontend --wavlm-dir wavlm_L4_200hz --wavlm-rate 200 \
  --content-cos-weight 1.0 --content-l1-weight 0.5 --delta-weight 0.04 \
  --forecast-weight 0.05 --voiced-boost 1.5 --grl-weight 0.05 \
  --lr 1e-4 --mel-frames 800 --eval-mel-frames 1200 --num-workers 6 \
  --save-every-epoch --out-dir checkpoints/striding_8l_200hz --run-name striding_8l_200hz
```

| Encoder flag | Effect |
|--------------|--------|
| `--wavlm-frontend --wavlm-rate 200` | WavLM L4 raw 200 Hz + StridingAdapter (k=4) instead of mel |
| `--rope --swiglu` | rotary positions + gated FFN in the causal Transformer |
| `--stem-block-type depthwise` | 8-block depthwise causal stem (deep RF, few params) |
| `--q2d2-levels 9,9,9,9,9,9` | 3.0M-code rhombic Q2D2 grid (ICML 2026) |
| `--grl-weight 0.05` | GRL speaker disentanglement |
| `--forecast-weight 0.05` | predict teacher[t+1], teacher[t+2] |

**3. Decoder v5** — strict-causal vocoder, 2-phase curriculum (reconstruction
warmup → MPD/MSD adversarial, recon kept as anchor). Trains on **pre-cached
full-context content**: the frozen encoder is run once over whole clips (not re-run
per step), so the decoder sees the SAME context it gets at streaming inference.
```bash
# one-time: full-context content (decoder input) + per-speaker centroids
.venv/bin/python -m astrape.cache --what content --device mps \
  --encoder-ckpt /Volumes/UNTITLED/btrv5_checkpoints/striding_8l_200hz/striding_8l_200hz.best.pt
.venv/bin/python -m astrape.cache --what speakers --utts-per-speaker 8
# train (content mode — no per-step encoder forward)
.venv/bin/python -m astrape.train_decoder \
  --device mps --epochs 60 --warmup-epochs 10 --num-workers 4 \
  --content-dir content_striding_8l_200hz
```

**Inference** — build a target voicebank, then convert:
```bash
.venv/bin/python -m astrape.build_voicebank ref1.wav [ref2.wav ...] -o p225.astrape
.venv/bin/python -m astrape.evaluate --source src.wav --target p225.astrape --output out.wav
```

## References

- **Q2D2** — Shuster & Nachmani, "Two-Dimensional Quantization for Geometry-Aware Audio Coding", ICML 2026, arXiv:2512.01537
- **MioCodec** — Aratako/MioCodec-25Hz-44.1kHz-v2 (teacher codec, HuggingFace)
- **WavLM** — Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing", 2022
- **GRL** — Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation", ICML 2015
- **Predictive Coding** — Oord et al., "Representation Learning with Contrastive Predictive Coding (CPC)", 2018
- **ConvNeXt** — Liu et al., "A ConvNet for the 2020s", CVPR 2022
- **WavTokenizer** — Ji et al., "WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer", 2024
- **Snake / BigVGAN** — Lee et al., "BigVGAN: A Universal Neural Vocoder", 2023, arXiv:2206.02944
- **iSTFT / Vocos** — Siuzdak et al., "Vocos: Closing the Gap Between Time-Domain and Fourier-Based Neural Vocoders", 2024, arXiv:2306.00819
- **APCodec** — Ai et al., "APCodec: A Neural Audio Codec", IEEE/ACM TASLP 2024, arXiv:2402.10533

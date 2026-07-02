# Astrape VC — Architecture & Experiment Log

## 1. MioCodec Teacher (Original, Bidirectional)

```
Audio (44.1kHz)
  → WavLM SSL extractor (frozen)
    ├─ local branch: L6+L9 layers → LayerNorm → Transformer 2L → Conv stride-2 → 25Hz
    │   → FSQ quantizer (5-axis, 12,800 codebook)
    │   → content embedding (768d) @ 25Hz
    └─ global branch: L1+L2 layers → ConvNeXt → attentive pool → global (128d)

Decoder (wave mode):
  content(768) @25Hz + speaker(128)
    → wave_prenet:             Transformer 6L d768 h12 (BIDIRECTIONAL, no speaker)
    → conv_upsample:           ConvTranspose1d k2 s2 → 50Hz
    → interpolate:             linear to stft_length @50Hz
    → wave_prior_net:          ResNetStack ×2 (GroupNorm)
    → wave_decoder:            Transformer 8L d512 h8 (BIDIRECTIONAL, AdaLN-Zero speaker)
    → wave_post_net:           ResNetStack ×2
    → wave_upsampler:          UpSamplerBlock (3,3)=9× → 450Hz
    → ISTFTHead:               Linear(512 → n_fft+2) → mag+phase → iSTFT(392/98)
    → waveform @44.1kHz

Key: 14 transformer layers, all bidirectional. Algorithmic latency: 3.3ms (iSTFT only,
because bidirectional = no real-time streaming possible).
```

---

## 2. Our Encoder (Q2D2, Causal)

```
WavLM L4 CNN (200Hz, 5 causal convs @16kHz, 10ms delay)  →  cached as (T,512)
  → WavLMFrontendAdapter:       CausalConv1d k4 s4 (200→50Hz) + Linear(512→80)
  → Conv stem:                  depthwise dilated CausalConv ×8 + skip connections
  → CellDownsample:             stride-2 → 25Hz
  → Transformer 8L:             512d, h8, causal windowed, RoPE, SwiGLU
  → Q2D2 projection:            Linear(512→6) → rhombic quantizer (3 pairs × L=11, 10.8M codebook)
  → Content expand:             residual MLP(768→256→768)
  → content (768d) @25Hz

cos768 vs teacher: 0.935 (probe, 8L StridingAdapter + Q2D2 L=11)
Algorithmic latency: ~12ms (WavLM CNN) + 0ms (all downstream causal)
Params: 25.4M
```

---

## 3. Decoder Experiments — All Failed to Reach Target (wave_cos < 0.25, mel_cos < 0.75)

### 3.1 v5 — CausalDecoderV5

```
content(768) + speaker(128)
  → pointwise in + 4L AdaLN transformer @25Hz (RoPE, SwiGLU, AdaLN-Zero)
  → LearnedCausalUpsampler ×18 (single giant leap 25→450Hz, ConvTranspose + SnakeBeta)
  → dil. SnakeConv stack @450Hz (d=1,2,4,1,2,4)
  → bridge + iSTFT(392/98)
```
**Result**: ~29M params. Large single-step upsample creates staircase artifacts. GAN training unstable.

### 3.2 CausalWaveDecoder — MioCodec Replica

```
content(768) + speaker(128)
  → prenet:          Transformer 6L d768 h12 (causal, windowed)
  → conv_upsample:   ConvTranspose k2 s2 → 50Hz (causal trim)
  → prior_net:       CausalResNet ×2 (LayerNorm, not GroupNorm)
  → decoder:         Transformer 8L d512 h8 (causal, AdaLN-Zero)
  → post_net:        CausalResNet ×2
  → upsampler:       CausalUpSamplerBlock (3,3)=9× (nearest+conv, not ConvTranspose)
  → ISTFT(392/98)
```
**Result**: Mirrors teacher 1:1 but causality gap (GroupNorm→LayerNorm, bidirectional→causal) causes significant quality loss. Feature distillation from teacher barely helps.

### 3.3 MCSDecoderV3 — All-Conv

```
content(768) + speaker(128)
  → ConvNeXt ×4 @25Hz (CausalGRN)
  → AAUpStage ×2 (384→512) @50Hz
  → ConvNeXt ×2 + DilatedSpeakerTCN ×8 @50Hz (AdaLN-Zero)
  → ConvNeXt ×2 + AAUpStage ×3×3 @450Hz
  → ISTFT(392/98)
```
**Result**: All-conv lacks global context. No transformer means no long-range prosody modeling. Stalled at moderate quality.

### 3.4 SimpleGRUDecoder — Minimal Baseline

```
content(768) + speaker(128)
  → FiLM + GRU 2L
  → repeat ×7 + causal convs
  → ISTFT(1512/252) → 14.3ms latency
```
**Result**: Too simple. n_fft=1512 grid wrong. Few params.

### 3.5 v6 Series — Transformer + ProsodyLSTM Iterations

**v6 (initial)**: prenet 4L + ProsodyLSTM + AdaLN 6L + AA upsample + ISTFT(392/98)
→ wave_cos=0.02, mel_cos=0.87 at E14
→ Magnitude perfect, phase completely random.

**v6 + real+imag ISTFTHead2D**: real+imag direct prediction
→ wave_cos=0.02. Network learns magnitude perfectly but ignores phase.

**v6 + mag+phase ISTFTHead2D**: two-branch shared Conv2d backbone
→ wave_cos=0.03. Same result — magnitude excellent, phase random.

**v6 + prenet 4L + no LSTM**: speaker injected directly
→ wave_cos=0.002. Worse than with LSTM.

**v6 + n_fft=1512 (757 bins)**: too many degrees of freedom
→ wave_cos=0.02. 757 phase values per frame impossible for causal system.

**Lesson**: 64M models could not learn phase. Over-engineered, too many components.

### 3.6 CausalDecoderMCS — Current Best (Still Insufficient)

```
content(768) + speaker(128)
  → prenet 10L (causal RoPE transformer) @25Hz
  → ConvTranspose ×2 + Snake → 50Hz, 512d
  → FiLM(speaker)
  → TCN ×4 dilations @50Hz
  → ConvTranspose ×3×3 + Snake → 450Hz
  → Linear(512→394) → mag+phase → iSTFT(392/98)
```

**Result**: wave_cos=0.22, mel_cos=0.73 at E14.
- 10× faster than v6 (direct ISTFT head distill, all MPS, no CPU STFT loss)
- Magnitude is good (mel_cos=0.73) but phase prediction plateaus
- Tried: multi-band head (no help), freq-axis smoothing (regression)
- 37.6M params (10L), 23.5M (7L), latency 3.3ms

**Key finding**: Phase prediction is the fundamental bottleneck for all causal decoders. The teacher uses bidirectional context at 25Hz AND 50Hz to achieve phase coherence. Our causal decoders can only see the past, making 197-bin phase prediction per frame inherently limited.

---

## 4. Summary

| Component | Teacher | Our Encoder | Our Decoders |
|-----------|---------|-------------|--------------|
| Content quantizer | FSQ (12.8K) | Q2D2 L=11 (10.8M) | — |
| cos768 vs teacher | 1.0 | 0.935 | — |
| Phase capability | Bidirectional 14L | — | Causal ≤10L |
| Best wave_cos | — | — | 0.22 (MCS) |
| Best mel_cos | — | — | 0.73 (MCS) |
| Latency | ~∞ (not streaming) | ~12ms | 3.3ms |

### Open Questions

1. Is wave_cos=0.22 actually perceptible as "reasonable VC"? Need listening tests.
2. Can we improve phase prediction by:
   - Using teacher's bidirectional prenet output as a "soft target" for our causal prenet?
   - Adding explicit F0/voicing supervision from teacher?
   - Using larger temporal context (longer window) in the fusion stage?
3. Is n_fft=392 the right tradeoff? More bins = better freq resolution but harder phase.
4. Should we revisit GAN-based training? GAN discriminators can implicitly learn phase realism.

# MioCodec Vocoder Causalization Analysis

**Status:** Teacher wave decoder (ISTFT) = non-causal. External PupuGAN vocoder = non-causal. Both need streaming replacement.

## Current Pipeline

```
Student content [25Hz]
→ Causal Mel Decoder → mel [25Hz, 80 bins]
→ ??? Vocoder ??? → waveform [44100Hz]
```

## Teacher Wave Decoder (ISTFT-based, non-causal)

The 44.1kHz checkpoint uses `use_wave_decoder=True`.
Path: `content → wave_prenet → upsampling → ResNet → wave_decoder → ISTFT → waveform`

### Non-causal components

| Component | Issue | Difficulty to Fix |
|-----------|-------|-------------------|
| wave_prenet | causal=False, window=65 (±32 frames) | Retrain with causal=True |
| ConvTranspose1d | Symmetric padding | Left-padding only |
| ResNet | Symmetric Conv1d + GroupNorm over T | Causal padding + LayerNorm |
| wave_decoder | causal=False, AdaLN-Zero | Retrain with causal=True |
| UpSamplerBlock | ConvTranspose1d symmetric | Left-padding |
| ISTFT | Full-sequence overlap-add | **Buffer-based streaming ISTFT** |

### Fixability

- Transformer layers: can be retrained with `causal=True` — same architecture, different training
- Conv1d/ConvTranspose: replace symmetric padding with left-only padding
- GroupNorm: replace with LayerNorm (channel-only, no time dependency)
- ISTFT: implement streaming overlap-add with fixed algorithmic latency

**Bottom line:** Teacher wave decoder CAN be made causal with moderate effort (retraining needed).

## External PupuGAN Vocoder (non-causal)

Architecture: `mel → conv_pre → ResampleUpsamplers → ResBlocks → conv_post → waveform`

### Non-causal components

| Component | Issue |
|-----------|-------|
| conv_pre | Conv1d(k=7, pad=3) — symmetric |
| ResampleUpsampler | **julius.lowpass_filter** — whole-signal IIR filter |
| ResampleUpsampler | **julius.highpass_filter** — whole-signal IIR filter |
| ResampleUpsampler | Zero-insertion upsampling — non-streaming |
| ResBlock | Symmetric padding (get_padding returns symmetric) |
| conv_post | Conv1d(k=7, pad=3) — symmetric |

### Key Blocker: julius filter

```python
y0 = julius.highpass_filter(y0.float(), 0.5 / scale_factor)
y = julius.lowpass_filter(y.float(), 0.5 / scale_factor)
```

These are **anti-aliasing filters** applied to the ENTIRE signal at once. They are IIR filters operating on the full temporal sequence. This is the hardest component to make causal.

### Fix options for PupuGAN

1. **Replace julius with causal FIR filter** — implement anti-aliasing as a causal FIR (finite impulse response) filter with bounded delay
2. **Remove anti-aliasing** — accept minor aliasing artifacts for streaming mode
3. **Retrain with causal replacements** — full vocoder retraining

**Bottom line:** PupuGAN can theoretically be causalized but requires significant engineering (replacing `julius` filters, causal convolutions).

## Streaming Vocoder Options

### Option A: Causalized Teacher Wave Decoder (Recommended)

**Pros:** Already integrated, high quality (44.1kHz), no external model
**Cons:** Retraining needed for causal flags

**Architecture:**
```
content [25Hz, 768d]
→ wave_prenet (causal=True, retrain)
→ causal ConvTranspose1d upsample
→ LayerNorm (replaces GroupNorm)
→ wave_decoder (causal=True, retrain)
→ Buffer-based causal ISTFT
→ waveform [44100Hz]
```

**Estimated algorithmic latency:**
- 1 content frame = 40ms
- ISTFT buffer = n_fft/2 = 196 samples ≈ 4.4ms
- Total: ~45ms

### Option B: Streaming HiFi-GAN

**Pros:** Proven streaming quality, many open-source implementations
**Cons:** Separate model, needs mel→waveform training

Architecture: standard HiFi-GAN V1/V2 with all convolutions made causal (left-padding only), transposed convolutions replaced with causal upsampling + conv.

### Option C: Lightweight Causal Conv Vocoder

**Pros:** Simple, trainable, minimal latency
**Cons:** Quality may be lower than HiFi-GAN

Architecture:
```
mel [80, T]
→ Conv1d(causal) × N blocks
→ Causal upsampling (nearest-neighbor + conv)
→ Conv1d(causal) → 1ch waveform
```

### Option D: Causal PupuGAN (High Effort)

Replace julius filters with causal FIR, retrain or fine-tune.

**Bottom line:** Option A (causal teacher wave decoder) is the most practical because the teacher already produces quality waveform and most components just need a causal flag flip + retraining.

## Action Plan

1. **Test causal wave_prenet + wave_decoder** — flip `causal=True`, check quality impact (no retraining)
2. **Implement buffer-based streaming ISTFT** — fixed-lookahead overlap-add
3. **Replace GroupNorm → LayerNorm in ResNet** — simple code change
4. **Replace symmetric Conv1d → causal Conv1d** — left-padding only
5. **Fine-tune causal decoder** — optional, depending on quality drop from step 1

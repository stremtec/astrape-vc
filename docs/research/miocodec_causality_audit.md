# MioCodec Causality Audit

Target checkpoint:

- `Aratako/MioCodec-25Hz-44.1kHz-v2`
- Local source:
  `/Users/asill/btrvrc0/.venv/lib/python3.12/site-packages/miocodec`
- HF snapshot:
  `/Users/asill/.cache/huggingface/hub/models--Aratako--MioCodec-25Hz-44.1kHz-v2/snapshots/67faba34153fe74e6665991c432a7327e23c5c1c`

## Verdict

MioCodec is not casually non-causal. It is architecturally non-causal in both
the encoder and decoder. The biggest blockers are:

1. WavLM SSL frontend with full-sequence self-attention.
2. Mio local/wave Transformers with `causal=False`.
3. Symmetric local attention windows that explicitly include future frames.
4. Global embedding branch that pools over the whole reference utterance.
5. Symmetric convolution/padding in ConvNeXt, ResNet, PostNet/upsamplers.
6. ISTFT-style synthesis that is frame/window based rather than streaming sample
   synthesis.

So the right path is not "turn MioCodec causal". It is "distill MioCodec into a
causal student that keeps the useful interfaces: content latent + native global
embedding + high-quality 44.1 kHz decoder".

## Current Checkpoint Settings

Measured from the loaded checkpoint:

```text
sample_rate = 44100
content rate = 25 Hz
n_fft = 392
hop_length = 98
use_wave_decoder = True
wave_upsampler_factors = (3, 3)

local_encoder:  6 layers, dim 768, window_size 125, causal=False
wave_prenet:    6 layers, dim 768, window_size 65,  causal=False
wave_decoder:   8 layers, dim 512, window_size 65,  causal=False, AdaLN global conditioning
global_embedding_dim = 128
```

Sources:

- `miocodec/model.py`
- `miocodec/module/transformer.py`
- `miocodec/module/global_encoder.py`
- `miocodec/module/convnext.py`
- `miocodec/module/istft_head.py`

## Encoder-Side Non-Causality

### 1. SSL frontend is WavLM, not causal

`SSLFeatureExtractor` uses torchaudio WavLM / wav2vec2-style models.

Relevant source:

- `miocodec/module/ssl_extractor.py`
- `SSLFeatureExtractor.forward()` calls `self.model.extract_features(...)`.

The loaded architecture contains:

- WavLM convolutional frontend.
- Transformer encoder layers.
- Convolutional positional embedding with `kernel_size=128, padding=64`.
- Multi-head self-attention over the sequence.

This alone breaks streaming causality. Early SSL frames can depend on future SSL
frames through self-attention and symmetric positional convolution.

### 2. Mio local encoder attends to future frames

`local_encoder` is a Mio transformer. The code supports a causal flag, but the
checkpoint has it disabled:

```text
local_encoder layer0: causal=False, local_attn=True, window_per_side=62
```

`window_size=125` means each SSL frame can attend to approximately 62 frames on
both sides. WavLM frames are around 20 ms apart, so the local encoder can see
roughly 1.2 seconds of future context before content quantization.

Relevant source:

- `miocodec/module/transformer.py`
- `Attention.create_mask()`
- `MioCodecModel.forward_content()`

### 3. SSL feature normalization uses future statistics

`MioCodecModel._normalize_ssl_features()` computes mean and std across time:

```python
mean = torch.mean(features, dim=1, keepdim=True)
std = torch.std(features, dim=1, keepdim=True)
```

This is full-utterance normalization, so even before the Mio local transformer,
features are normalized with future frames.

Relevant source:

- `miocodec/model.py`
- `_normalize_ssl_features()`

### 4. Symmetric waveform padding

`encode()` computes padding and then `forward_ssl_features()` pads both left and
right:

```python
waveform = F.pad(waveform, (padding, padding), mode="constant")
```

Padding itself is not the biggest issue, but it confirms the model is designed
for whole-clip feature extraction and length alignment, not chunked streaming.

## Global Branch Non-Causality

The target voicebank use case can tolerate offline global extraction from a few
seconds of reference audio. But Mio's native global encoder is still not causal:

- ConvNeXt blocks use symmetric `Conv1d(..., kernel_size=7, padding=3)`.
- `AttentiveStatsPool` applies `Softmax(dim=2)` over the entire time axis.
- Mean and std are computed over the whole reference sequence.

Relevant source:

- `miocodec/module/global_encoder.py`
- `miocodec/module/convnext.py`

For one-shot voicebank creation, this is acceptable because the reference audio
is known before conversion starts. It is not acceptable if we require live,
causal source-speaker embedding extraction.

## Decoder-Side Non-Causality

Even if content tokens and global embedding were already available, the Mio
decoder is still non-causal.

### 1. Wave prenet attends to future content tokens

Loaded checkpoint:

```text
wave_prenet layer0: causal=False, local_attn=True, window_per_side=32
```

At 25 Hz content rate, 32 future frames is about 1.28 seconds of lookahead.

### 2. Wave decoder attends to future acoustic frames

Loaded checkpoint:

```text
wave_decoder layer0: causal=False, local_attn=True, window_per_side=32
```

This decoder operates around the pre-ISTFT frame rate. With the 44.1 kHz
checkpoint, the pre-upsampler wave decoder is around 50 Hz, so 32 future frames
is roughly 0.64 seconds of lookahead.

Relevant source:

- `miocodec/model.py`
- `forward_wave()`
- `miocodec/module/transformer.py`

### 3. Interpolation and transposed convolutions are not streaming-native

`forward_wave()` uses:

- `ConvTranspose1d` upsample.
- `F.interpolate(..., size=stft_length, mode=wave_interpolation_mode)`.
- `UpSamplerBlock` with `ConvTranspose1d` kernel size 9, stride 3, padding 3.

Linear interpolation to a known final size is a whole-sequence operation in the
current implementation. Transposed convolutions with symmetric padding also mix
neighboring frames in a non-causal way unless rewritten as stateful causal
upsamplers.

### 4. ResNet blocks use symmetric convolution and GroupNorm

`ResNetBlock` uses:

```python
padding = (kernel_size - 1) * dilation // 2
nn.GroupNorm(...)
nn.Conv1d(..., padding=padding)
```

The symmetric padding leaks future frames. GroupNorm on `(B, C, T)` also computes
statistics across the time axis, so it is not streaming-causal.

Relevant source:

- `miocodec/module/istft_head.py`

### 5. ISTFT head is overlap-add synthesis

`ISTFTHead` predicts magnitude and phase, then reconstructs with `ISTFT`.
The checkpoint uses:

```text
n_fft = 392
hop_length = 98
istft_padding = "same"
```

This can be made streaming with buffered overlap-add and fixed algorithmic
latency, but the current implementation is a full tensor operation over all
frames. More importantly, the predicted STFT frames are already future-conditioned
by the decoder.

## Can We Patch MioCodec To Be Causal?

Not cleanly.

Possible code changes:

- Set Transformer `causal=True`.
- Replace symmetric local attention with causal local attention.
- Replace WavLM with a causal content encoder.
- Replace symmetric Conv1d/ConvTranspose/GroupNorm with causal/stateful versions.
- Replace full-sequence normalization with running/causal normalization.
- Implement streaming ISTFT overlap-add.

But the pretrained weights were trained with future context. Flipping flags would
change the distribution and likely destroy quality. This would become a full
retraining job, not a patch.

## What We Should Steal From MioCodec

Keep the interface, not the exact architecture:

- Continuous 25 Hz content embedding.
- Native global speaker/acoustic embedding.
- Global-conditioned decoder.
- High-quality 44.1 kHz waveform target.
- ISTFT/periodic-activation ideas where compatible with causal buffering.

Our student should implement these with:

- causal waveform/content encoder,
- one-shot reference/global encoder,
- causal decoder with bounded receptive field,
- optional streaming overlap-add only if we keep an ISTFT head,
- full-loss final polish against Mio recon.


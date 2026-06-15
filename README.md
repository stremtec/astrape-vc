# Astrape VC

Research code for a causal, zero-shot voice-conversion pipeline distilled from
MioCodec.

## WebUI

```bash
.venv/bin/python -m webui.server
```

Open `http://127.0.0.1:8765`. The local VC console provides VoiceBank upload,
profile diagnostics, FCPE F0 analysis, browser audio-device routing,
pitch/formant capability controls, training status, and a WebSocket streaming
path that activates when a production direct-wave checkpoint is available.

## Current Pipeline

```text
16 kHz source PCM
  -> streaming log-mel, 50 Hz
  -> strictly causal 768x10 ContentStudent, 2 s bounded history
  -> continuous 768d content, 25 Hz
     + auxiliary 5-axis FSQ prediction and frozen Mio projection
  -> DirectWaveDecoder + cached VoiceBank global embedding, 128d
  -> 44.1 kHz PCM
```

The complete model path has stateful streaming implementations and regression
tests comparing chunked output against full-sequence output. The direct causal
waveform decoder is implemented and benchmarked, but its production checkpoint
has not yet been trained. `demo_v2.py` remains available for offline comparison
through the MioCodec teacher decoder.

## Models

- `astrape.model.ContentStudent`: left-padded causal convolutions, causal
  attention, aligned 50 Hz to 25 Hz downsampling, bounded streaming state,
  a continuous 768d deployment head, and a parallel FSQ head matching
  MioCodec's `[8,8,8,5,5]` code axes.
- `astrape.mel_decoder.CausalMelDecoder`: source-restored AdaLN-Zero decoder
  matching `checkpoints/causal_mel_decoder.pt`; retained as an auxiliary
  acoustic target and diagnostic.
- `astrape.wave_decoder.DirectWaveDecoder`: stateful causal multi-dilation
  waveform generator producing exactly 1,764 samples per content frame.
- `astrape.audio.StreamingLogMel`: exact `center=False` full/chunked log-mel
  extraction for 16 kHz PCM.

Existing `checkpoints/causal_student_v3_4k.pt` was trained by the old
symmetrically padded architecture. It is therefore treated as a legacy weight
file and requires `--allow-legacy` or `--import-legacy`. Fine-tune it with the
new causal architecture before reporting causal quality.

## Training

```bash
# Standard 384d model
.venv/bin/python train_v3_4k.py

# Short run with a separate checkpoint name
.venv/bin/python train_v3_4k_mini.py

# Configured capacity tier
.venv/bin/python train_xhigh.py --tier xhigh --device mps

# Recover the teacher's exact frozen 5d-to-768d projection from cached labels
.venv/bin/python extract_fsq_projection.py

# Preferred direct continuous + auxiliary CTC training
.venv/bin/python train_content_flat_ctc.py \
  --hidden 512 --layers 10 --heads 8 \
  --steps-per-epoch 1000 --probe-samples 1024 \
  --full-validation-every 5 \
  --device mps

# Mio-shaped strict-causal backbone: 768x6, 12 heads, RoPE, SwiGLU
.venv/bin/python train_content_flat_ctc.py \
  --architecture mio_causal \
  --steps-per-epoch 1000 --probe-samples 1024 \
  --full-validation-every 5 \
  --device mps

# FFL content student: Mio backbone plus 16-slot layer-wise future effects
.venv/bin/python train_content_flat_ctc.py \
  --architecture mio_ffl \
  --steps-per-epoch 1000 --probe-samples 1024 \
  --full-validation-every 5 \
  --device mps

# Quality-first Mio causal training. Phase 1 is teacher-only; phase 2 samples
# teacher-cache and original-VCTK batches at an exact 50:50 ratio.
./run_mio_two_phase.sh

# Full 41-hour compact teacher cache, then hybrid training
./run_full_content_pipeline.sh

# Historical original-CTC curriculum, retained for comparison
.venv/bin/python train_content_curriculum.py \
  --audio-root /path/to/VCTK/wav48_silence_trimmed \
  --transcript-root /path/to/VCTK/txt

# Causal mel decoder
.venv/bin/python train_mel_decoder.py --target-mode teacher

# Direct waveform decoder, first against original waveform targets
.venv/bin/python train_wave_decoder.py --device mps

# Preferred teacher-reconstruction targets
.venv/bin/python cache_wave_targets.py --device mps
.venv/bin/python train_wave_decoder.py \
  --target-dir data/mio_4k_teacher_wave \
  --device mps
```

Training uses speaker-disjoint validation, aligned even-frame crops, causal
history warmup, masked variable-length losses, deterministic seeds, full
validation, versioned checkpoints, and separate `.best.pt`/`.last.pt` files.

The quality-first Mio causal trainer uses a direct continuous 768d content head
and a small character-CTC auxiliary head. Phase 1 trains only against cached Mio
teacher targets, including aligned transcript CTC. Phase 2 starts from the best
phase-1 probe checkpoint at a much lower learning rate and samples teacher-cache
and original full-utterance VCTK batches at an exact 50:50 ratio. The low phase-2
rate protects teacher alignment when CTC-only updates alternate with direct
distillation. Original batches apply only weighted CTC, while Mio teacher cosine
remains the checkpoint selection criterion. There is no random crop that would
invalidate transcript alignment.

Each short epoch runs 1,000 optimizer updates and evaluates a fixed,
speaker-balanced probe. Full speaker-disjoint validation runs every five
epochs and at both phase boundaries. Phase-1 best, global probe-best, and
full-validation best checkpoints are stored separately. The structured
five-axis FSQ trainer remains available as a later comparison.

`extract_content_cache.py` caches all VCTK mic1 utterances and transcripts
without storing
waveforms or global speaker embeddings. Log-mel, Mio content, and pre-FSQ
targets use FP16 storage, FSQ tokens use uint16, and character transcripts use
uint8. Teacher inference remains FP32 on MPS; only serialized arrays are
reduced in precision.

The curriculum keeps validation speakers out of both original and teacher
training. Its phases are:

1. Full original VCTK utterances with a character CTC objective.
2. A gradual mixture of original CTC and MioCodec teacher supervision.
3. 90% teacher FSQ distillation with 10% original-data retention.

The configured target remains full-context Mio teacher cosine `0.99`, measured
on the direct 768d output. Soft and hard FSQ cosine, per-axis accuracy, exact
token accuracy, sequence cosine, and frame-cosine p05 are reported separately
so FSQ fidelity cannot hide regressions in the deployment representation.

False Future Learning (FFL) is the active zero-lookahead research path for
closing the full-context gap. It synthesizes confidence-gated internal future
effects from past-only states, with no audio collection delay. The oracle
experiment, implemented `mio_ffl` content student, losses, and staged training
plan are documented in `docs/research/false_future_learning.md`. The integrated
FFL path adds 7.20M parameters across a shared slot generator and six
layer-specific gated adapters. The deterministic probe can be run with
`diagnose_false_future.py` in the Mio environment.

## VoiceBank Policy

A VoiceBank is built from one continuous target-speaker reference recording.
The minimum duration is five seconds. Longer references such as 10 seconds,
30 seconds, or one minute are accepted without changing the zero-shot
interface; the user chooses the desired quality and preparation cost.

Multiple references are not required and are not part of the core definition.
The target speaker may be unseen during training, so this remains zero-shot
voice conversion.

VoiceBank format v3 (`.astrape`) stores the Mio embedding model ID, source
hash, creation time, and non-destructive reference diagnostics for clipping,
loudness, active speech, and DC offset. The 48-byte fixed header is followed
by a raw float32 LE embedding and a JSON metadata block, so profile lists
read with a 48-byte prefix rather than decompressing a zip container.
Version 2 `.npz` files remain readable for backward compatibility; the
local `.astrape` migration is lossless (verified per file via
`migrate_voicebanks.py`).

```bash
# Build a zero-shot voice bank (default extension drives the on-disk format).
.venv/bin/python build_voicebank.py \
  --reference target_reference.wav \
  --output voicebanks/target.astrape

# Migrate every legacy .npz in voicebanks/ to .astrape, with verification.
.venv/bin/python migrate_voicebanks.py --keep-existing
.venv/bin/python migrate_voicebanks.py --move            # delete the .npz
.venv/bin/python migrate_voicebanks.py --dry-run        # preview only

# Inspect a profile (works on both .npz and .astrape).
.venv/bin/python inspect_voicebank.py voicebanks/target.astrape
```

```

To import the historical student weights:

```bash
.venv/bin/python train_v3_4k.py \
  --import-legacy checkpoints/causal_student_v3_4k.pt
```

## Extraction

```bash
.venv/bin/python extract_4k.py \
  --vctk-root /path/to/VCTK/wav48_silence_trimmed
```

Extraction randomly samples utterances per speaker with a fixed seed and stores
speaker names, utterance IDs, and source paths in `meta.npz`. MioCodec is an
optional external dependency required for extraction and teacher decoding.

## Inference And Benchmarking

```bash
# Incremental content + mel inference on cached data
.venv/bin/python stream_infer.py \
  --mel data/mio_4k_mel/m_00000.npz \
  --target data/mio_4k/s_00001.npz \
  --checkpoint checkpoints/content_student_v3_4k_causal.best.pt \
  --mel-decoder checkpoints/causal_mel_decoder.pt

# Incremental content + direct waveform inference
.venv/bin/python stream_wave_infer.py \
  --mel data/mio_4k_mel/m_00000.npz \
  --voicebank voicebanks/target.npz \
  --content-checkpoint checkpoints/content_student_768x10_fsq.best.pt \
  --wave-checkpoint checkpoints/direct_wave_decoder.best.pt \
  --device mps

# File-driven simulation of the complete PCM streaming runtime
.venv/bin/python run_streaming_e2e.py \
  --input source.wav \
  --voicebank voicebanks/target.npz \
  --content-checkpoint checkpoints/content_student_768x10_fsq.best.pt \
  --wave-checkpoint checkpoints/direct_wave_decoder.best.pt \
  --output outputs/e2e.wav \
  --device mps \
  --chunk-ms 5

# Synchronized accelerator benchmark
.venv/bin/python bench_dim.py --device mps

# Offline waveform comparison through the MioCodec teacher decoder
.venv/bin/python demo_v2.py \
  --source source.wav --reference target.wav
```

Benchmark timings synchronize MPS/CUDA before and after every measurement and
report full-sequence latency, streaming latency per 25 Hz content frame, and
real-time factor.

Frontend latency experiments and the direct causal waveform-decoder plan are
documented in `docs/latency_waveform_plan.md`.
The callback, state, buffering, compatibility, and failure contracts for the
complete runtime are documented in `docs/e2e_streaming_pipeline.md`.
The 52.17M direct + FSQ + pre-FSQ + CTC `mio_ffl` model has numerically matching
full and streaming paths and no future leakage in regression tests. A 768x6
streaming call for two 50 Hz frames measured about 10.2 ms p50 and 13.5 ms p95
on the current MPS host.

On the current Apple MPS host, the selected 768x10 model with 100 past
50 Hz frames measured 29.3 ms p50 and 30.5 ms p95 per output frame after
cache saturation.
Content plus the causal mel decoder measured 32.4 ms p50 and 33.0 ms p95.
The untrained 8.78M direct waveform decoder measured 7.86 ms p50 and 9.79 ms
p95 per 40 ms content frame. All caches are bounded and use no future
lookahead. The first output still includes the causal STFT collection delay.

## Tests

```bash
.venv/bin/python -m unittest discover -v
```

The suite covers causal prefix invariance, continuous and structured-FSQ
streaming equivalence, direct waveform full/chunk equivalence, exact waveform
length, VoiceBank migration and quality metadata, complete PCM-to-waveform
streaming equivalence, output-ring underruns, log-mel streaming,
speaker-disjoint splitting, crop alignment, CTC constraints, exact FSQ
projection recovery, checkpoint compatibility, decoder loading, and tier
construction.

# Astrape VC

Research code for a causal, zero-shot voice-conversion pipeline distilled from
MioCodec.

## Current Pipeline

```text
16 kHz source PCM
  -> streaming log-mel, 50 Hz
  -> strictly causal 768x10 ContentStudent, 2 s bounded history
  -> 5-axis FSQ prediction
  -> frozen MioCodec 5d-to-768d projection, 25 Hz
  -> CausalMelDecoder + cached target global embedding, 128d
  -> mel, 80 bins at 25 Hz
```

The source-to-mel path supports stateful streaming and has regression tests
that compare chunked output against full-sequence output. Waveform synthesis is
still a research dependency: `demo_v2.py` can use MioCodec's offline teacher
decoder, while a production causal vocoder has not yet been trained.

## Models

- `astrape.model.ContentStudent`: left-padded causal convolutions, causal
  attention, aligned 50 Hz to 25 Hz downsampling, bounded streaming state,
  and an FSQ-aware output head matching MioCodec's `[8,8,8,5,5]` code axes.
- `astrape.mel_decoder.CausalMelDecoder`: source-restored AdaLN-Zero decoder
  matching `checkpoints/causal_mel_decoder.pt`.
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

# Original VCTK CTC -> gradual blend -> MioCodec FSQ distillation
.venv/bin/python train_content_curriculum.py \
  --audio-root /path/to/VCTK/wav48_silence_trimmed \
  --transcript-root /path/to/VCTK/txt \
  --device mps

# Causal mel decoder
.venv/bin/python train_mel_decoder.py --target-mode teacher
```

Training uses speaker-disjoint validation, aligned even-frame crops, masked
variable-length losses, deterministic seeds, full validation, versioned
checkpoints, and separate `.best.pt`/`.last.pt` files.

The curriculum keeps validation speakers out of both original and teacher
training. Its phases are:

1. Full original VCTK utterances with a character CTC objective.
2. A gradual mixture of original CTC and MioCodec teacher supervision.
3. 90% teacher FSQ distillation with 10% original-data retention.

Teacher training predicts the five discrete FSQ axes directly. Exact axes
reconstruct cached `ce_768` through the frozen teacher projection, so the
deployment metric is hard-code teacher cosine rather than similarity to the
source audio. The configured target is `0.99`.

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

# Synchronized accelerator benchmark
.venv/bin/python bench_dim.py --device mps

# Offline waveform comparison through the MioCodec teacher decoder
.venv/bin/python demo_v2.py \
  --source source.wav --reference target.wav
```

Benchmark timings synchronize MPS/CUDA before and after every measurement and
report full-sequence latency, streaming latency per 25 Hz content frame, and
real-time factor.

On the current Apple MPS host, the selected 768x10 model with 100 past
50 Hz frames measured 29.3 ms p50 and 30.5 ms p95 per output frame after
cache saturation.
Content plus the causal mel decoder measured 32.4 ms p50 and 33.0 ms p95.
Both caches are bounded and use no future lookahead. The first mel output still
includes the causal STFT collection delay; waveform synthesis remains the
unresolved production stage.

## Tests

```bash
.venv/bin/python -m unittest discover -v
```

The suite covers causal prefix invariance, continuous and structured-FSQ
streaming equivalence, log-mel streaming, speaker-disjoint splitting, crop
alignment, CTC constraints, exact FSQ projection recovery, padding masks,
checkpoint compatibility, decoder loading, and tier construction.

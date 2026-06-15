# False Future Learning

## Definition

False Future Learning (FFL) is a zero-lookahead distillation method that uses
past-only evidence to synthesize internal future-context effects. The generated
state is never emitted as waveform and does not wait for incoming audio. Its
purpose is to make a causal student produce the same current representation that
a full-context teacher would produce after observing future frames.

The primary target is not exact future waveform or exact future SSL features.
It is the correction that future context causes at the current frame:

```text
teacher_future_effect = teacher_full_state - teacher_causal_state
```

This distinction matters because speech futures are multi-modal, while the
effect of those futures on Mio's current five-dimensional FSQ decision is much
lower-dimensional.

## Initial Oracle Experiment

`diagnose_false_future.py` replaces Mio's unavailable right context with several
non-learned alternatives. The experiment uses 24 utterances and three positions
per utterance.

### Mio local encoder isolated

| Internal future | No future | Oracle future | Exact FSQ |
| --- | ---: | ---: | ---: |
| 16 SSL frames / 320 ms | 0.960741 | 0.993784 | 68.1% |
| 62 SSL frames / 1240 ms | 0.960741 | 0.998669 | 94.4% |

Sixteen 50 Hz future slots already recover most of the available improvement.
This makes a compact internal future generator more attractive than predicting
the full roughly two-second teacher horizon.

### Prefix WavLM path

| Internal future | No future | Oracle future |
| --- | ---: | ---: |
| 16 SSL frames / 320 ms | 0.833461 | 0.872481 |
| 62 SSL frames / 1240 ms | 0.833461 | 0.869696 |

The smaller gain proves that the WavLM prefix representation itself has already
diverged before Mio's local encoder. FFL cannot be attached only after WavLM.
The backbone must first be causally adapted, and future-effect adapters should
be available at intermediate WavLM layers.

### Deterministic false futures

Naively replaying or reversing the past does not improve every frame. At a
16-frame horizon, zero or replay fillers improve the hardest quartile by about
0.105 and 0.119 cosine respectively, while reducing some already-correct easy
frames. At 62 frames, replay improves mean prefix-path cosine by 0.014 but wins
on only 41.7% of frames.

The conclusion is:

> False future injection must be residual, confidence-gated, and initialized as
> disabled. Unconditional fake context damages frames that do not need it.

## Implemented Content Student

```text
80d causal log-mel at 50 Hz
  -> 1x1 causal stem, 80 -> 768
  -> sequence-wide False Future Slot Generator
       768 -> 256 projection
       3 causal summary blocks, dilations [1, 2, 4]
       last 16 past states folded into 16 future positions
       current summary + learned horizon embeddings
       2 residual slot mixers
       2 reverse-causal slot refiners, far -> near
  -> repeat six times:
       Mio-style causal block, 768d / 12 heads / 2048 SwiGLU
       layer-specific attention over the 16 false-future slots
       confidence-gated residual correction
  -> layer norm
  -> character CTC head at 50 Hz
  -> causal kernel-2 stride-2 downsample
  -> direct 768d content at 25 Hz
  -> five-axis FSQ and pre-FSQ auxiliary heads when enabled
```

The architecture is selected with `architecture="mio_ffl"`. The old
`mio_causal` architecture remains unchanged as the ablation baseline.

The currently implemented model deliberately keeps the existing causal log-mel
frontend so the FFL hypothesis can be tested on the cached dataset without
mixing in a second major change. A causally adapted pretrained WavLM frontend is
the next replacement stage if the integrated FFL model establishes a useful
gain.

### Slot generator

- Input dimension: 768
- Internal dimension: 256
- Past history: 64 frames
- False-future horizon: 16 frames
- Causal summary receptive field: 29 frames
- Residual slot mixers: 2
- Reverse-causal slot refinement blocks: 2
- Shared slot generator parameters: 4.416M
- Six layer-specific adapters: 2.783M
- Total FFL parameters: 7.199M
- Direct + FSQ + pre-FSQ + CTC model total: about 52.17M

All sequence positions are generated in one causal pass during training. During
streaming, only the retained stem history and current chunk are recomputed. The
second pass orders each frame's generated slots from far to near and applies a
reverse-causal mask. This realizes reverse generation without requiring a real
far-future observation or serial waveform generation.

### Folded positions

Recent historical states are an initialization, not a fixed prediction. The
nearest past state is folded into the nearest false-future position, then a
causal summary, horizon embedding, slot mixers, and reverse refiner correct it:

```text
false_future[k] =
    folded_past[k]
    + predictor(past_summary, horizon_embedding[k])
```

The model therefore begins with the proposed "past mistaken for future"
behavior, but can learn when that analogy is wrong.

### Confidence gate

Each of the six Mio blocks has its own false-future attention and predicts a
gate in `[0, 1]`:

```text
student_state =
    causal_state
    + gate * false_future_effect
```

The current implementation predicts the gate from that layer's current hidden
state. Gate weights start at zero and biases start at `-4`, giving an initial
activation of about `0.018`. The initial model therefore stays close to the
stable causal baseline.

The model output keeps `false_future_effects`, `false_future_gates`, and
`false_future_corrections` separate. The correction is exactly the raw effect
multiplied by the gate. Effect distillation can therefore supervise the raw
prediction without forcing the gate open.

During training, the oracle benefit is available:

```text
oracle_benefit =
    loss(causal_state, teacher_full_state)
    - loss(oracle_future_state, teacher_full_state)
```

The gate receives an auxiliary target derived from positive oracle benefit.
This teaches it to activate on difficult frames rather than everywhere.

## Losses

### Stage losses

```text
L_effect
  = sum_l w_l * smooth_l1(predicted_future_effect_l,
                          teacher_full_l - teacher_causal_l)

L_hidden
  = sum_l w_l * cosine(student_ffl_l, teacher_full_l)

L_slot
  = distance_weighted cosine(predicted_future_slots,
                             teacher_future_slots)

L_gate
  = BCE(predicted_gate, oracle_benefit_target)

L_content
  = cosine_l1(student_content, teacher_content)

L_fsq
  = weighted ordinal loss over [8, 8, 8, 5, 5]

L_output
  = decoder-aware mel + multi-resolution STFT + ASR feature loss
```

`L_effect` and `L_hidden` are primary. `L_slot` is auxiliary because exact
future prediction is unnecessarily strict and may average incompatible
futures. The two five-level FSQ axes receive higher loss weights because a
single-axis error changes content cosine more severely.

## Training Schedule

### Phase 0: Baselines and cached targets

- Preserve speaker-disjoint validation.
- Keep 1,000 optimizer steps as a reporting interval, not an epoch.
- Record no-FFL, oracle-FFL, and teacher-full metrics on the same samples.
- Cache only compact final targets locally. Produce intermediate teacher states
  online or on a larger training machine to avoid exceeding local disk space.

### Phase 1: Causal WavLM adaptation

- This phase follows FFL validation with the log-mel implementation.
- Initialize from WavLM Base+.
- Replace sequence-dependent normalization and non-causal positional
  convolution.
- Distill positional convolution first.
- Distill layers 1-6 pairwise.
- Use at least LibriSpeech 960h; VCTK alone is not sufficient.
- Train until streaming/full hidden gaps stop improving, not for a fixed ten
  short epochs.

### Phase 2: FFL adapter warm-up

- Freeze causal WavLM and Mio output heads.
- Train only the generator, future-effect adapters, and gates.
- Use real teacher futures for `L_effect`, `L_hidden`, and oracle gate labels.
- Start with 16 slots.
- Learning rate: `1e-4`, warmup 2k steps, minimum 30k steps.

### Phase 3: Content and FSQ distillation

- Unfreeze the upper three WavLM layers and Mio blocks.
- Backbone learning rate: `5e-6`.
- FFL and new heads learning rate: `5e-5`.
- Keep direct 768d content as the production output.
- Keep five-axis FSQ and CTC as auxiliary losses.
- Minimum 100k supervised updates or five genuine dataset passes.

### Phase 4: Decoder-aware fine-tuning

- Compare frozen Mio decoder outputs from teacher and student content.
- Optimize output equivalence even where exact latent cosine is impossible.
- Retain a small teacher-hidden loss to prevent content drift.

### Phase 5: Ablation and compression

- Compare horizons 8, 16, and 32.
- Remove the reverse refiner, folded-past initialization, and gate separately.
- Compress WavLM-6 only after the full model establishes a quality ceiling.

## Acceptance Criteria

FFL development continues only if all of the following hold:

- FFL improves full speaker-disjoint validation, not only training crops.
- Bottom-5% and bottom-25% frame metrics improve without degrading the median.
- Gate activation is sparse and correlated with oracle benefit.
- Streaming and full-sequence student inference remain numerically equivalent.
- Added model runtime remains inside the eventual 100 ms end-to-end budget.

The full-context Mio cosine of 0.99 remains a diagnostic target. For strict
zero-lookahead production, decoder output quality, intelligibility, speaker
leakage, and streaming consistency are the release metrics.

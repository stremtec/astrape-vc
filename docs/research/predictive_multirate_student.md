# Token-Synchronous Multi-Rate Content Student

**Status:** proposed replacement for the failed Mio-shaped and FFL students  
**Primary target:** full-context Mio content cosine >= 0.92 on the complete
speaker-disjoint validation set, with zero lookahead

## Why the previous direction plateaued

- The historical `0.923` result used symmetric convolution padding, an
  utterance-random split, only 20 validation utterances, and flattened
  utterance cosine. It is not a strict-causal reference.
- The strongest current strict model reaches about `0.8996` full-validation
  frame cosine. A larger Mio-shaped backbone and 50/50 original replay did not
  break that ceiling.
- Mio's target is exactly rank five after its FSQ projection. Predicting an
  unrestricted 768-dimensional vector spends capacity outside the real target
  manifold.
- The current model's CTC gradient at the final shared block is almost
  orthogonal to content and, after the configured weight, is about three times
  larger. Removing CTC entirely also regressed validation, so CTC must move to
  an intermediate branch and be gradient-balanced.
- FFL added about 7.2M parameters but only produced a very small direct effect.
  Predicting a complete synthetic future and injecting it at every layer is a
  harder problem than the output requires.
- The current left-padded `kernel=2, stride=2` downsampler emits from frame zero
  and then groups states as `[1,2]`, `[3,4]`, and so on. Mio's padding-free
  downsampler groups `[0,1]`, `[2,3]`, and so on. The immediate-emission
  optimization changed the information boundary of every 25 Hz target.
- A controlled small probe found that selecting the end-of-cell 50 Hz state
  improved validation cosine from `0.83308` to `0.83922` and p05 from `0.57707`
  to `0.58452`. Concatenating both states reached `0.83788`; predicting the
  missing state reached only `0.82241` despite the highest training cosine.
  Generated future state therefore belongs in an auxiliary loss, not the
  production representation path.
- A second controlled probe found that a copied WavLM convolutional frontend
  with causal framewise normalization scored `0.79300`, versus `0.83386` for
  the existing log-mel frontend. Replacing WavLM's temporal GroupNorm destroys
  too much of the pretrained frontend behavior. Log-mel remains the default.

## Target geometry

The teacher FSQ axes use levels `[8, 8, 8, 5, 5]`, followed by a frozen affine
projection from five to 768 dimensions.

On 1,024 held-out validation utterances:

| Oracle correction | Resulting cosine | Gain |
| --- | ---: | ---: |
| Axis 3 only | 0.9266 | +0.0270 |
| Axis 4 only | 0.9301 | +0.0306 |
| Axes 3 and 4 | 0.9563 | +0.0567 |
| 25% correction of axes 3 and 4 errors | 0.9248 | +0.0252 |

The five axes have very low pairwise mutual information, at most about
`0.042 bit` in a 171k-frame sample. A 12,800-way joint-token head is therefore
not justified as the primary output. Independent ordinal heads plus continuous
five-axis regression are a better match.

## Proposed architecture

```text
16 kHz waveform
  -> causal 80-bin log-mel, n_fft=512, hop=320
  -> 80 -> 384 projection
  -> 4 causal acoustic-edge blocks at 50 Hz
       depthwise-separable GLU convolution
       dilation cycle [1, 2, 4, 8]
       RMSNorm and residual scaling
       no attention at this rate
  -> token-synchronous decimation
       emit h[1], h[3], h[5], ... at 25 Hz
       each output sees the complete current 40 ms token cell
       no learned stride convolution and no synthetic future injection
  -> 512d projection and one-layer GRU summary state
  -> 8 causal dual-path blocks at 25 Hz
       causal multi-scale depthwise-convolution branch in every block
       grouped-query past-only attention in alternating blocks
       two-second attention cache plus unbounded recurrent summary
       gated branch merge and SwiGLU feed-forward
  -> continuous five-axis head
  -> independent ordinal logits for [8, 8, 8, 5, 5]
  -> frozen teacher 5d-to-768d affine projection
```

This does not copy Mio's non-causal local Transformer. It only preserves the
teacher's output rate and five-axis geometry. The convolution path is biased
toward boundaries and local transitions; the recurrent and attention paths
model phonetic history.

The recurrent state and all attention/convolution caches contain past
information only. The first token is emitted when the second 20 ms log-mel
frame is available, at about 52 ms with the current 512-sample STFT. This is
not lookahead beyond the token: the output timestamp is the end of its own
40 ms cell. A 400-sample STFT can reduce this to about 45 ms after a separate
quality-preserving adaptation.

Approximate parameters:

| Component | Parameters |
| --- | ---: |
| Log-mel projection and 4 x 384d edge blocks | about 1.9M |
| 384d-to-512d projection and GRU summary | about 1.8M |
| 8 x 512d dual-path blocks | about 28.4M |
| Content, ordinal, CTC, and prediction heads | less than 1M |
| **Total** | **about 32.2M** |

This is smaller than the 44.3M Mio-shaped model. Expensive contextual
processing runs at 25 Hz, while the 50 Hz path is convolutional and local.

## Losses

The production output is continuous five-dimensional code, not hard FSQ:

```text
L_main =
    1 - cosine(frozen_projection(predicted_code), teacher_content)
    + weighted_smooth_l1(predicted_code, teacher_code)

initial axis weights ~= [1.0, 1.0, 1.0, 1.4, 1.5]
```

Auxiliary losses:

- independent ordinal cross-entropy for each FSQ axis;
- factorized five-axis prediction for each of the next five speech tokens;
- intermediate 50 Hz CTC;
- optional next-state prediction against a stop-gradient target, with the
  prediction discarded before the production path.

CTC is attached before 25 Hz decimation. Its weight is adjusted to keep its
weighted gradient norm around 15-30% of the main content gradient. If its
gradient cosine becomes negative, the conflicting component is projected out.
No fixed `0.05` CTC weight is assumed.

Per-axis regression weights start from the values above, then update slowly
from gradient norms with a bounded range of `[0.5, 2.0]`. This prevents axes 3
and 4 from being neglected without allowing them to destabilize the shared
encoder.

The current diagnostic measured final-layer CTC/content gradient cosine at
`0.224` while weighted CTC norm was `3.01x` the main loss. Delta/content
gradient cosine was `0.207` at `0.30x`. Delta loss can remain small initially,
but CTC must move to the 50 Hz intermediate branch and be balanced.

Hard FSQ cosine, continuous projected cosine, and decoder output quality are
reported separately. Hard rounding is not allowed to silently replace the
continuous validation metric.

## Training schedule

All "epochs" below are 1,000 optimizer steps, matching the short reporting
cycle used for stability checks.

### Phase 0: architecture smoke test, 3 epochs

- Treat this as a disposable run, then initialize the real Phase 1 model from
  scratch.
- Use 1,000 steps per epoch and the fixed 1,024-sample speaker-balanced probe.
- Train teacher distillation only.
- Require `>=0.885`, finite gradients, exact full/streaming equivalence, and no
  train-validation divergence larger than the current baseline.
- Compare immediate versus end-of-cell decimation once in the full model. Do
  not proceed if the full model contradicts the controlled probe.

### Phase 1: original-speech predictive pretraining, 15 epochs

- Use causal next-token prediction for five consecutive future positions,
  following the NEST-RQ pattern.
- Use a fixed random-projection quantizer over acoustic features.
- Train intermediate CTC on full utterances.
- Use mild gain, noise, and band-limit augmentation. Avoid time masking during
  later teacher distillation.

### Phase 2: teacher distillation, 30 epochs

- Main five-axis/projected-cosine loss on every teacher batch.
- Keep next-token prediction at a reduced weight.
- Use intermediate, gradient-balanced CTC.
- Use 4-8 second supervised crops with at least two seconds of past warm-up.
- Run a full-utterance CTC batch every fourth update.
- Backbone learning rate `2e-4`, heads `3e-4`.
- Train the optional next-state predictor, but never feed its output into the
  main content path.

### Phase 3: teacher-heavy refinement, 20 epochs

- Backbone learning rate `5e-5`.
- Replay one original predictive batch for every three teacher batches.
- Decay ordinal and future-prediction auxiliaries; retain the main geometric
  loss.
- Use EMA weights for validation.

### Phase 4: decoder-aware refinement, 5-10 epochs

- Begin only after full-validation cosine reaches at least `0.915`.
- Optimize decoder output equivalence while retaining the teacher-content
  anchor.
- Compare continuous, straight-through hard, and hard FSQ decoder inputs.

## Validation gates

- Speaker-disjoint 1,024-sample probe every 1,000 steps.
- Complete 6,185-utterance validation every 5,000 steps.
- Report mean frame cosine, p05, sequence cosine, five axis MAEs, ordinal
  accuracies, CTC CER, and transition-frame metrics.
- Stop a run if three complete validations improve by less than `0.0005`.
- Architecture gate: `0.905`, then `0.912`, then `0.918`, and finally `0.920`.
- The release gate is full-validation cosine `>= 0.920`, with p05 improving
  rather than trading away difficult frames.

## Latency gate

The quality path intentionally restores one 20 ms input frame that the old
immediate-emission optimization removed:

| Component | Initial budget |
| --- | ---: |
| Two log-mel frames available | 52 ms |
| Content student p95 | <=25 ms |
| Direct waveform decoder p95 | <=10 ms |
| Callback/scheduling margin | <=5 ms |
| **End-to-end target** | **<=92 ms** |

The new content p95 is a gate, not a measured claim. If it misses 25 ms, first
optimize the alternating-attention core. Test the 400-sample frontend only
after content cosine reaches `0.915`; do not trade away representation quality
prematurely.

A 400-repeat synchronized MPS benchmark brackets this budget: the existing
21.6M legacy tier measured `21.05 ms` streaming p95, while the 62.3M tier
measured `28.78 ms`. The proposed 32.2M model also moves most work to 25 Hz, so
`<=25 ms` is plausible, but it remains an acceptance test rather than an
extrapolated production claim.

## Required ablations

1. Immediate frame-zero emission versus end-of-cell emission.
2. End-of-cell selection versus learned two-frame fusion.
3. Five-future-token auxiliary on versus off.
4. Intermediate gradient-balanced CTC versus fixed final-layer CTC.
5. Adaptive five-axis loss versus uniform axes.
6. Continuous projected output versus hard FSQ at both latent and decoder
   levels.
7. 512-sample versus adapted 400-sample log-mel only after cosine reaches
   `0.915`.

The model should be enlarged only after these ablations show that the remaining
gap is capacity-limited. If the 32.2M model stalls below `0.912`, the next
action is better causal predictive pretraining data or a stronger local
transition branch, not a return to a 70M Mio-shaped encoder or another
future-injection module.

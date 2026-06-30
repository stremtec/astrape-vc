# astrape-vst

Strict-causal, zero-shot voice conversion VST — real-time on Apple Silicon (MPS).

| | |
|---|---|
| cos768 | **0.935** (probe, 8L StridingAdapter) |
| Encoder params | 24.9M |
| Algorithmic latency | ~49ms E2E (0 look-ahead) |
| Platform | macOS / Apple Silicon (MPS) |

## Design

- **0 look-ahead** — strictly causal. No future frames, KV-cache streaming.
- **Content/speaker split** — *what* is said (768d content @25Hz) vs *who* says it (global embedding).
- **Teacher–student** — trained by distilling a frozen, bidirectional MioCodec teacher into a causal student.
- **Zero-shot VC** — GRL disentanglement + Q2D2 bottleneck. No parallel data needed.

## Quick Start

```bash
# Voicebank
python -m astrape.build_voicebank ref.wav -o speaker.astrape

# Convert
python -m astrape.evaluate --source src.wav --target speaker.astrape --output out.wav
```

## References

- **Q2D2** — Shuster & Nachmani, ICML 2026, arXiv:2512.01537
- **WavLM** — Chen et al., 2022
- **MioCodec** — Aratako/MioCodec-25Hz-44.1kHz-v2 (teacher codec)

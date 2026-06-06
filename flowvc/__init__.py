"""
FlowVC — Causal-First Voice Conversion with Conditional Flow Matching.

Fully causal pipeline:
  Source(44.1k) → F³-Encoder(causal, KL-free) → z_src
    → FlowVC Converter(CFM ODE, 4-8 steps)
    → F³-Decoder(causal, MRF upsampler) → Target(44.1k)

All convolutions use left-only padding (causal).
No MioCodec dependency — own encoder/decoder trained from scratch.
"""

__version__ = "0.1.0"

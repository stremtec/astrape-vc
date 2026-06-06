"""
HybridVC — Hybrid Codec + RAF Vocoder Voice Conversion Pipeline.

btrv5 core package. Integrates:
- MioCodec continuous latent encoder (frozen)
- CausalConvNeXt Converter (10 blocks + cross-attn, ~5.4M)
- RAF BigVGAN Vocoder (14M, 24kHz output)
- ConvNeXt BWE (0.8M, 24k → 44.1k)
- RAF Loss (WavLM teacher + relativistic pairing)
"""

__version__ = "0.1.0"

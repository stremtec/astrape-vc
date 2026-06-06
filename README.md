# Astrape VC ⚡

> **Αστραπή** (astrape) — Greek for "lightning"

Real-time neural voice conversion with Conditional Flow Matching.  
Fully causal pipeline. 44.1kHz. No MioCodec dependency.

```
Source(44.1k) → F³-Encoder(causal, KL-free) → z_src
  → FlowVC Converter(CFM ODE, 4-8 steps)
  → F³-Decoder(causal, MRF upsampler) → Target(44.1k)
```

## Architecture

| Component | Params | Description |
|-----------|:------:|-------------|
| F³-Encoder | 26.5M | 6-stage causal ConvNeXt v2, KL-free, noise-reg |
| VectorFieldNet | 37.2M | 12-block CFM converter, AdaLN-Zero, cross-attn |
| F³-Decoder | 27.9M | 6-stage MRF upsampler, FiLM conditioning |
| **Total** | **91.6M** | Fully causal, streaming-ready |

## Key Features

- **Causal-first**: All convolutions use left-only padding — no future leak
- **Flow Matching**: Conditional Flow Matching with OT path, 4-step Euler solver
- **KL-free**: No VQ, no codebook collapse, continuous latent space
- **ConvNeXt v2**: GRN, LayerScale, inverted bottleneck throughout
- **44.1kHz native**: No bandwidth extension needed
- **Zero-init everywhere**: Identity at initialization for stable training

## Quick Start

```bash
# Shape test
python3 -c "
from flowvc.encoder import make_encoder
from flowvc.converter import make_vector_field_net, solve_cfm_euler
from flowvc.decoder import F3Decoder
from flowvc.config import DecoderConfig
import torch

encoder = make_encoder()
vfn = make_vector_field_net()
decoder = F3Decoder(DecoderConfig())

B, T_lat = 2, 50
wav = torch.randn(B, 1, T_lat * 1764)  # 2s @ 44.1kHz
spk = torch.randn(B, 192)
prompt = torch.randn(B, 4, 192)
prosody = torch.randn(B, T_lat, 3)

z = encoder.encode(wav)
z_tgt = solve_cfm_euler(vfn, z, spk, prompt, prosody)
out = decoder(z_tgt, spk)

assert out.shape == wav.shape
print('✅ Pipeline OK')
"
```

## Design

Based on literature review of 22 arXiv 2026 audio papers.  
Full design docs in `designs/`.  
Winning architecture selected via 5-agent parallel review + scoring.

## Status

- [x] F³-Encoder (causal ConvNeXt v2)
- [x] F³-Decoder (MRF upsampler + FiLM)
- [x] VectorFieldNet (CFM, 12 blocks)
- [x] CFM Loss (OT path + sigma regularization)
- [x] Euler/RK4 ODE Solver
- [x] Shape tests passed (91.6M params)
- [ ] Speaker Encoder + Prompt Tokens
- [ ] Prosody Extractor
- [ ] Training Pipeline
- [ ] Streaming Inference

## License

MIT

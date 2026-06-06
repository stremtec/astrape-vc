# HybridVC: Hybrid Codec + RAF Vocoder + ConvNeXt 기반 실시간 고품질 음성 변환 파이프라인

> **btrv5 아키텍처 설계 — 각도 "HybridVC"**
>
> Hybrid Continuous Codec + RAF (Relativistic Adversarial Feature) Vocoder
> + ConvNeXt v2 Backbone + Vocos BWE
>
> 24kHz+ 실시간 Voice Conversion, RAF로 14M BigVGAN이 112M 능가

---

## 목차

1. [개요 및 설계 철학](#1-개요-및-설계-철학)
2. [논문 인사이트 통합](#2-논문-인사이트-통합)
3. [전체 아키텍처 다이어그램](#3-전체-아키텍처-다이어그램)
4. [컴포넌트 상세 설계](#4-컴포넌트-상세-설계)
   - 4.1 [HybridCodec Encoder (USAD 2.0 + MioCodec 통합)](#41-hybridcodec-encoder)
   - 4.2 [Speaker Encoder (화자 인코더)](#42-speaker-encoder)
   - 4.3 [CausalConvNeXt Converter (잠재 변환기)](#43-causalconvnext-converter)
   - 4.4 [RAF BigVGAN Vocoder (보코더)](#44-raf-bigvgan-vocoder)
   - 4.5 [ConvNeXt BWE (대역 확장기)](#45-convnext-bwe)
5. [RAF Loss의 VC 적용](#5-raf-loss의-vc-적용)
6. [차원 / 파라미터 / MACs 명세](#6-차원--파라미터--macs-명세)
7. [Training Pipeline (Phase별 RAF 통합)](#7-training-pipeline)
8. [실시간 추론 전략](#8-실시간-추론-전략)
9. [btrv3lite / btrvrc0 자산 재활용 매핑](#9-btrv3lite--btrvrc0-자산-재활용-매핑)
10. [RAF vs 기존 HiFi-GAN GAN Loss 비교 분석](#10-raf-vs-기존-hifi-gan-gan-loss-비교-분석)
11. [리스크 및 대안](#11-리스크-및-대안)

---

## 1. 개요 및 설계 철학

### 1.1 HybridVC란?

HybridVC는 **MioCodec(continuous latent)** + **USAD 2.0(SSL distillation)** 인코더와 **RAF-trained BigVGAN** 보코더를 ConvNeXt v2 백본으로 통합한 차세대 Voice Conversion 파이프라인이다. 기존 btrv3lite의 Transformer-based decoder를 **RAF + ConvNeXt GAN 보코더**로 대체하여, 14M 파라미터로 112M급 품질을 달성한다.

### 1.2 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **RAF Loss 주도 학습** | SSL teacher(WavLM)의 relativistic pairing으로 판별자(discriminator)를 대체. GAN collapse 없이 빠른 수렴, 14M BigVGAN이 112M 능가. |
| **ConvNeXt v2 백본** | WaveNeXt 2의 ConvNeXt GAN 설계를 보코더와 BWE에 적용. 7×7 depthwise conv + GRN + inverted bottleneck. |
| **SSL Distillation 인코더** | USAD 2.0의 SSL+supervised distillation으로 25Hz universal audio encoder. MioCodec teacher를 WavLM feature로 정규화. |
| **Bandwidth Extension** | Base 보코더는 24kHz로 동작, Vocos BWE로 44.1kHz 확장 — 연산량 절감과 고품질 양립. |
| **Continuous Latent + Residual 변환** | VQ 없이 순수 연속 잠재 공간. CausalConvNeXt Converter는 identity-init residual로 안정적 학습. |
| **실시간 스트리밍** | Causal 연산, 청크 단위 처리, TTFB < 150ms, RTF < 0.5 목표. |

### 1.3 목표 사양

| 지표 | 목표값 | 비고 |
|------|--------|------|
| Sample Rate | 44,100 Hz | BWE로 24kHz → 44.1kHz |
| Latent Framerate | 25 Hz | hop = 1764 samples @ 44.1kHz |
| Real-Time Factor (RTF) | < 0.5 (GPU), < 0.8 (CPU) | 청크 80ms 기준 |
| TTFB (Time To First Byte) | < 150ms (GPU), < 200ms (CPU) | |
| Speaker Similarity | > 0.85 (ECAPA cosine) | |
| WER | < 3% (Whisper large-v3) | |
| MOS | > 4.0 (naturalness) | |
| 파라미터 수 (전체) | ~30M | RAF 효율로 경량화 |
| Training 효율 | < 500 GPU-hours (A100) | RAF로 GAN 수렴 가속 |

---

## 2. 논문 인사이트 통합

### 2.1 RAF (arXiv:2603.11678)

> **"SSL quality gap + relativistic pairing → 14M BigVGAN beats 112M"**

| 인사이트 | HybridVC 적용 |
|----------|---------------|
| **SSL Quality Gap**: WavLM embedding 간 cosine distance가 사람의 MOS 평가와 0.93 상관관계 | Discriminator를 대체하는 SSL teacher로 WavLM Large 사용. Generator의 출력과 Ground Truth를 WavLM으로 인코딩 후 relativistic pairing loss 계산. |
| **Relativistic Pairing**: real/fake 절대 판별 대신 "real이 fake보다 얼마나 나은가" 상대 평가 | `L_RAF = -log(σ(D_real - D_fake))`. 기존 LSGAN의 mode collapse 문제 해결. |
| **14M beats 112M**: RAF로 학습된 14M BigVGAN이 LSGAN+MPD+MRD로 학습된 112M BigVGAN을 MOS, PESQ, CER에서 능가 | HybridVC 보코더를 14M급 경량 BigVGAN으로 설계. 기존 50M+ student decoder 대비 3.5× 경량화. |
| **SSL teacher 선택**: WavLM Large (317M) > WavLM Base+ > HuBERT > wav2vec 2.0 | RAF teacher로 WavLM Large 사용. Frozen, gradient 없음. |

### 2.2 WaveNeXt 2 (arXiv:2605.25506)

> **"ConvNeXt GAN + Diffusion 통합 vocoder, CPU RTF 0.16"**

| 인사이트 | HybridVC 적용 |
|----------|---------------|
| **ConvNeXt 기반 Generator**: 7×7 depthwise conv + GRN + 1×1 conv 구조가 음성 합성에 최적 | BigVGAN generator의 residual block을 ConvNeXt v2 block으로 대체. WaveNeXt 2의 블록 구조 차용. |
| **Fusion of GAN + Diffusion**: GAN의 지각 품질 + Diffusion의 mode coverage | RAF만으로 충분한 mode coverage 달성. Diffusion 불필요 (복잡도 감소). |
| **CPU RTF 0.16**: ConvNeXt의 효율적 구조로 CPU 실시간 추론 | HybridVC 보코더의 MPS/CPU 추론 최적화. |
| **Multi-scale STFT loss**: 학습 안정화에 핵심적 | 기존 MultiResMel + MR-STFT loss 유지, RAF loss와 결합. |

### 2.3 USAD 2.0 (arXiv:2606.06444)

> **"SSL + supervised distillation, universal audio encoder, 25Hz"**

| 인사이트 | HybridVC 적용 |
|----------|---------------|
| **SSL Teacher**: WavLM feature distillation으로 universal audio representation 학습 | MioCodec encoder의 distillation teacher로 WavLM Large 사용. SSL feature를 768-dim latent space로 투영하는 projection head 추가. |
| **25Hz universal encoder**: 음성/음악/환경음 모두 커버하는 범용 인코더 | HybridCodec encoder를 25Hz로 유지하되, SSL distillation으로 음성 품질 향상. |
| **Multi-resolution STFT reconstruction**: 인코더-디코더 재구성 loss에 multi-scale STFT 사용 | HybridCodec AE 학습 시 teacher-forcing reconstruction loss에 MR-STFT 포함. |

### 2.4 Vocos BWE (arXiv:2603.07285)

> **"ConvNeXt 대역확장, RTF 0.0001"**

| 인사이트 | HybridVC 적용 |
|----------|---------------|
| **ConvNeXt 기반 BWE**: 7×7 depthwise conv + 1×1 conv의 단순 구조로 0-12kHz → 0-24kHz 확장 | Base 보코더 출력(24kHz)을 44.1kHz로 확장. BWE는 44.1kHz 업샘플링 + ConvNeXt residual refinement. |
| **RTF 0.0001 (A100)**: 극도로 가벼운 연산량 | BWE는 전체 파이프라인 RTF에 거의 영향 없음. |
| **Mel-spectrogram guidance**: BWE 입력으로 mel-spectrogram 사용 | HybridVC에서는 보코더의 중간 feature map(512-dim, 200Hz)을 BWE guidance로 사용. |

---

## 3. 전체 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              HYBRIDVC ARCHITECTURE                                    │
│                                                                                      │
│  ┌──────────────────────────── TRAINING PHASE ────────────────────────────────────┐  │
│  │                                                                                 │  │
│  │  Source Audio (44.1kHz)              Target Audio (44.1kHz)                     │  │
│  │        │                                    │                                   │  │
│  │        ▼                                    ▼                                   │  │
│  │  ┌──────────────┐                   ┌──────────────┐                            │  │
│  │  │ HybridCodec  │                   │ HybridCodec  │  Shared weights            │  │
│  │  │   Encoder    │                   │   Encoder    │  (MioCodec-25Hz-768d)      │  │
│  │  └──────┬───────┘                   └──────┬───────┘                            │  │
│  │         │ z_src (T, 768)                   │ z_tgt (T, 768)                      │  │
│  │         │                                   │                                    │  │
│  │         ▼                                   │                                    │  │
│  │  ┌──────────────────────┐                   │                                    │  │
│  │  │ CausalConvNeXt       │◄── speaker_cond ──┤  Speaker Encoder                  │  │
│  │  │ Converter            │    + prosody      │  (ECAPA-TDNN, 256 dim)            │  │
│  │  │ (10 blocks, 192d)    │                   │  + Prosody Extractor              │  │
│  │  └──────────┬───────────┘                   │  (F0, energy, voicing)            │  │
│  │             │ z_out (T, 768)                │                                    │  │
│  │             │                                │                                    │  │
│  │             ▼                                ▼                                    │  │
│  │  ┌──────────────────────────────────────────────────────────┐                    │  │
│  │  │                   RAF BigVGAN Vocoder                     │                    │  │
│  │  │  ┌──────────────────────────────────────────────────┐    │                    │  │
│  │  │  │  ConvNeXt v2 Generator (14M params)               │    │                    │  │
│  │  │  │  • content → wave_prenet (ConvNeXt-1D, 6 blocks) │    │                    │  │
│  │  │  │  • ConvNeXt Upsampler (×3×3, 25→225Hz)           │    │                    │  │
│  │  │  │  • SnakeBeta activations                          │    │                    │  │
│  │  │  │  • FiLM conditioning from speaker_emb             │    │                    │  │
│  │  │  │  • Output: 24,000 Hz waveform                     │    │                    │  │
│  │  │  └──────────────────────────────────────────────────┘    │                    │  │
│  │  │                                                            │                    │  │
│  │  │  ┌──────────────────────────────────────────────────┐    │                    │  │
│  │  │  │  RAF Discriminator (SSL Teacher)                  │    │  │  │  │  │
│  │  │  │  • WavLM Large (317M, frozen)                     │    │                    │  │
│  │  │  │  • Relativistic Pairing Loss                      │    │                    │  │
│  │  │  │  • Multi-layer feature matching (layers 6,12,18,24)│   │                    │  │
│  │  │  └──────────────────────────────────────────────────┘    │                    │  │
│  │  └──────────────────────────────────────────────────────────┘                    │  │
│  │             │                                                                     │  │
│  │             ▼                                                                     │  │
│  │  ┌──────────────────────┐                                                         │  │
│  │  │  ConvNeXt BWE        │  24kHz → 44.1kHz                                        │  │
│  │  │  (Vocos-style, 0.5M) │  RTF < 0.001 (GPU)                                      │  │
│  │  └──────────┬───────────┘                                                         │  │
│  │             │                                                                     │  │
│  │             ▼                                                                     │  │
│  │       Output Waveform (44.1kHz)                                                    │  │
│  │                                                                                    │  │
│  │  ┌──────────────────────────────────────────────────────────┐                    │  │
│  │  │  Losses:                                                  │                    │  │
│  │  │  L_total = L_recon + λ_raf·L_RAF + λ_fm·L_FM + λ_mel·L_mel│                   │  │
│  │  │  L_recon  = L1(waveform) + MR-STFT + MR-Mel              │                    │  │
│  │  │  L_RAF    = -log(σ(D_real - D_fake))  ← RAF 핵심         │                    │  │
│  │  │  L_FM     = Σ ||WavLM_l(real) - WavLM_l(fake)||₁         │                    │  │
│  │  │  L_mel    = MultiResMelLoss (λ=45)                       │                    │  │
│  │  └──────────────────────────────────────────────────────────┘                    │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌──────────────────────────── INFERENCE PHASE ───────────────────────────────────┐  │
│  │                                                                                 │  │
│  │  Mic Input (44.1kHz)                                                            │  │
│  │       │                                                                         │  │
│  │       ▼                                                                         │  │
│  │  ┌─────────────┐    chunk = 80ms (3528 samples @ 44.1kHz)                      │  │
│  │  │ Ring Buffer │    left_ctx = 320ms, lookahead = 160ms                        │  │
│  │  │  (560ms)    │    window = [left_ctx | chunk | lookahead]                    │  │
│  │  └──────┬──────┘                                                               │  │
│  │         │                                                                       │  │
│  │         ▼                                                                       │  │
│  │  ┌──────────────────────────────────────┐                                      │  │
│  │  │       STREAMING PIPELINE             │                                      │  │
│  │  │                                      │                                      │  │
│  │  │  Encoder → Converter → Vocoder → BWE │  (causal chain)                      │  │
│  │  │  ──────────────────────────────────  │                                      │  │
│  │  │  Encoder:  15ms (ConvNeXt ×6 stages) │                                      │  │
│  │  │  Converter: 8ms (10 ConvNeXt blocks) │                                      │  │
│  │  │  Vocoder:  20ms (ConvNeXt upsampler) │                                      │  │
│  │  │  BWE:       2ms (ConvNeXt residual)  │                                      │  │
│  │  │  ──────────────────────────────────  │                                      │  │
│  │  │  Total:   ~45ms per chunk (GPU)      │                                      │  │
│  │  │  TTFB:    ~125ms (최초 window fill)  │                                      │  │
│  │  └──────────────────────────────────────┘                                      │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 데이터 흐름 요약

```
Training:
  src_audio → Encoder → z_src ─┐
                                ├─→ Converter ─→ z_out ─→ Vocoder ─→ BWE ─→ output
  tgt_audio → Encoder → z_tgt ─┘       ↑                         │
                                speaker_cond + prosody            │
                                                                RAF Loss ← WavLM(output, tgt_audio)

Inference:
  mic_input → Buffer → Encoder → z_src → Converter → z_out → Vocoder → BWE → speaker_output
                         ↑                      ↑
                    (causal)            speaker_cond (pre-computed)
```

---

## 4. 컴포넌트 상세 설계

### 4.1 HybridCodec Encoder

**목적**: 44.1kHz 원본 음성을 25Hz, 768-dim continuous latent로 인코딩. MioCodec teacher의 content latent와 USAD 2.0 SSL distillation을 결합.

```
HybridCodec Encoder
═══════════════════════════════════════════════════════════════

Architecture (ConvNeXt v2 Encoder):
  Input:  waveform (B, 1, T) @ 44,100 Hz

  Stage 0:  Conv1d(k=15, s=2)  → (B, 32, T/2)
  Stage 1:  ConvNeXtBlock(32)  → (B, 48, T/4)    stride=2
  Stage 2:  ConvNeXtBlock(48)  → (B, 64, T/12)   stride=3
  Stage 3:  ConvNeXtBlock(64)  → (B, 96, T/36)   stride=3
  Stage 4:  ConvNeXtBlock(96)  → (B, 128, T/252)  stride=7
  Stage 5:  ConvNeXtBlock(128) → (B, 192, T/1764) stride=7

  Total downsample: 2×2×3×3×7×7 = 1764 → 25 Hz

  ConvNeXt v2 Block:
    ┌────────────────────────────────────────┐
    │  x ─→ DWConv7(k=7, causal)             │
    │      → LayerNorm                        │
    │      → Conv1×1(c → 4c)                 │
    │      → GELU                             │
    │      → GRN (Global Response Norm)       │
    │      → Conv1×1(4c → c)                  │
    │      → LayerScale(init=1e-6)            │
    │      → + x  (residual)                  │
    └────────────────────────────────────────┘

  Head:  Conv1×1(192 → 768) + LayerNorm
  Output: z_content (B, T_lat, 768) @ 25 Hz

Causal implementation:
  - 모든 Conv1d는 left-only padding
  - GRN은 channel-wise global response → causal-safe (시간축 독립)

SSL Distillation (USAD 2.0 방식):
  ┌──────────────────────────────────────────┐
  │  Training 시에만 활성화:                  │
  │  audio → WavLM Large (frozen) → ssl_feat │
  │  ssl_feat → Projection(1024→768) → z_ssl │
  │  L_distill = L1(z_content, z_ssl)        │
  │           + (1 - cos_sim(z_content, z_ssl))│
  └──────────────────────────────────────────┘

Parameters: ~3.2M
MACs per second: ~2.1G (44.1kHz, 1초 기준)
```

**btrv3lite 재활용**:
- 기존 MioCodec teacher의 encoder (6-stage ConvNeXt, 192-dim output) → HybridCodec Encoder로 확장
- Student encoder의 CausalConv1d 패딩 방식 그대로 사용
- WavLM projection head는 신규 추가

---

### 4.2 Speaker Encoder (화자 인코더)

**목적**: 타겟 화자의 reference audio에서 speaker embedding 추출.

```
Speaker Encoder
═══════════════════════════════════════════════════════════════

Architecture: ECAPA-TDNN (기존 btrvrc0 검증済み)
  Input:  reference audio (3초 이상 권장)
  Model:  ECAPA-TDNN (pretrained, frozen)
  Output: speaker_emb (256 dim)

  Dimension reduction for conditioning:
    speaker_emb (256) → Linear(256→128) → speaker_cond (128)

Speaker Bank (for inference):
  - 256 speaker centroids (pre-computed)
  - Target / Blend / Neutral 모드 지원
  - 기존 conditioner.py의 SpeakerConditioner 재활용

Prosody Extractor (경량):
  Input:  source audio window @ 44.1kHz
  Model:  ConvNeXt(4 blocks, 64 dim)
  Output: prosody_feat (T_lat, 3) = [log_f0, voicing, log_energy]
  
  F0: RMVPE (기존 f0.py) → log-scale
  Voicing: threshold-based
  Energy: RMS → log-scale

Parameters: ECAPA ~6.2M (frozen) + Prosody ~0.3M
```

**btrv3lite 재활용**:
- SpeakerConditioner, SpeakerBank, F0 extractor 전체 재활용
- ConditionerConfig 그대로 사용

---

### 4.3 CausalConvNeXt Converter (잠재 변환기)

**목적**: 소스 화자의 latent(z_src)를 타겟 화자의 latent(z_out)로 변환. Identity-init residual 설계로 안정적 학습.

```
CausalConvNeXt Converter
═══════════════════════════════════════════════════════════════

Architecture (btrv3lite converter v2 확장):
  Input:  z_src (B, T_lat, 768) @ 25 Hz
          speaker_cond (B, 128)
          prosody_feat (B, T_lat, 3)

  in_proj:   Linear(768 → 192)
  
  Block 0:   ConvNeXtBlock(dim=192, k=5, dil=1)   + AdaLN-Zero(cond)
  Block 1:   ConvNeXtBlock(dim=192, k=5, dil=2)   + AdaLN-Zero(cond)
  Block 2:   ConvNeXtBlock(dim=192, k=5, dil=4)   + AdaLN-Zero(cond)
  Block 3:   ConvNeXtBlock(dim=192, k=5, dil=8)   + cross-attn(spk_prompt) + AdaLN-Zero
  Block 4:   ConvNeXtBlock(dim=192, k=5, dil=16)  + AdaLN-Zero(cond)
  Block 5:   ConvNeXtBlock(dim=192, k=5, dil=1)   + AdaLN-Zero(cond)
  Block 6:   ConvNeXtBlock(dim=192, k=5, dil=2)   + cross-attn(spk_prompt) + AdaLN-Zero
  Block 7:   ConvNeXtBlock(dim=192, k=5, dil=4)   + AdaLN-Zero(cond)
  Block 8:   ConvNeXtBlock(dim=192, k=5, dil=8)   + AdaLN-Zero(cond)
  Block 9:   ConvNeXtBlock(dim=192, k=5, dil=16)  + cross-attn(spk_prompt) + AdaLN-Zero

  out_proj:  Linear(192 → 768)
  out_gate:  zero-init → delta = out_gate·tanh()·out_proj(x)
  
  Output:    z_out = z_src + delta  (residual)

ConvNeXt v2 Block with AdaLN-Zero:
  ┌──────────────────────────────────────────┐
  │  cond → MLP → (scale, shift)              │
  │  x → AdaLN(x, scale, shift)              │
  │    → DWConv7(k=5, dil=d, causal)          │
  │    → LayerNorm                            │
  │    → Conv1×1(192 → 768)                  │
  │    → GELU                                 │
  │    → GRN                                  │
  │    → Conv1×1(768 → 192)                  │
  │    → LayerScale(init=0.0)                 │
  │    → + x                                  │
  └──────────────────────────────────────────┘

Cross-Attention (blocks 3, 6, 9):
  ┌──────────────────────────────────────────┐
  │  spk_prompt: Learnable tokens (4, 192)   │
  │  spk_prompt = spk_prompt + speaker_emb   │  
  │  x → LayerNorm → MultiHeadAttn(Q=x,      │
  │      K=spk_prompt, V=spk_prompt) → + x   │
  └──────────────────────────────────────────┘

Speaker Prompt:
  - 4 learnable tokens (192 dim each)
  - speaker_emb(128) → Linear(128→192)로 각 토큰에 bias 추가
  - P-Flow (arXiv:2305.07432) 방식

Condition Assembly:
  cond_t = Concat[speaker_cond(128) ‖ prosody_t(3)] = (131,)
  cond_t → MLP(131 → 128 → 384) → AdaLN scale/shift pairs

Parameters: ~5.4M (v2, cross-attn 포함)
MACs per frame: ~0.8M (192-dim, 10 blocks)
```

**btrv3lite 재활용**:
- 기존 CausalLatentConverter (8 blocks, 4.7M) → 10 blocks + cross-attn으로 확장
- AdaLN-Zero, CausalDepthwiseConv1d, GRN 블록 코드 그대로
- `converter.py`의 ConverterConfig, ConvNeXtBlock 재활용

---

### 4.4 RAF BigVGAN Vocoder (보코더)

**목적**: 변환된 latent(z_out, 25Hz, 768-dim)를 24kHz waveform으로 디코딩. RAF loss로 학습된 ConvNeXt v2 기반 경량 BigVGAN.

```
RAF BigVGAN Vocoder
═══════════════════════════════════════════════════════════════

Architecture (WaveNeXt 2 + BigVGAN hybrid):
  Input:  z_out (B, T_lat, 768) @ 25 Hz
          speaker_global (B, 128)

  ┌─ Wave PreNet (ConvNeXt-1D) ────────────────────────────┐
  │  Block 0-5: ConvNeXtBlock(dim=512, k=7, dil=[1,2,4,8,16,32])│
  │    각 block에 FiLM conditioning (speaker_global → γ, β) │
  │  Output: (B, T_lat, 512)                                │
  └────────────────────────────────────────────────────────┘

  ┌─ Upsampler (ConvNeXt Transposed) ──────────────────────┐
  │  ×3 upsample: ConvTranspose1d(512→384, k=9, s=3)       │
  │    + ConvNeXtBlock(384, k=7) + SnakeBeta                │
  │  ×3 upsample: ConvTranspose1d(384→256, k=9, s=3)       │
  │    + ConvNeXtBlock(256, k=7) + SnakeBeta                │
  │                                                         │
  │  Upsample ratio: 3×3 = 9×                               │
  │  Frame rate: 25 → 225 Hz                                │
  │  Output: (B, T×9, 256)                                  │
  └────────────────────────────────────────────────────────┘

  ┌─ Final Up + Head ──────────────────────────────────────┐
  │  PostNet: ConvNeXtBlock(256, k=7) × 2 + SnakeBeta       │
  │  Final Up: ConvTranspose1d(256→128, k=11, s=320/3)      │
  │    → 225 Hz × (320/3) ≈ 24,000 Hz                       │
  │    (실제로는 nearest-neighbor up + causal conv 사용)     │
  │  Head: Conv1d(128→1, k=7) → tanh                         │
  │  Output: waveform (B, 1, T_wav) @ 24,000 Hz             │
  └────────────────────────────────────────────────────────┘

Speaker Conditioning (FiLM):
  speaker_global (128) → Linear(128→1024) → chunk into (γ, β) pairs
  각 ConvNeXtBlock의 GELU 이후, GRN 이전에:
    x = γ * x + β

SnakeBeta Activation:
  snake_beta(x) = x + (1/β) * sin²(β*x)
  β learnable per-channel
  → 주기적 음성 신호의 harmonic structure 보존에 탁월

Total Generator Parameters: ~14.2M
MACs per second (24kHz): ~8.5G

───────────────────────────────────────────────────────────

RAF Discriminator (SSL Teacher)
═══════════════════════════════════════════════════════════

  Teacher: WavLM Large (317M params, frozen)
  
  Forward:
    real_audio (24kHz) → WavLM → features_real (24 layers)
    fake_audio (24kHz) → WavLM → features_fake (24 layers)
    
  Quality Score (per layer l ∈ {6, 12, 18, 24}):
    q_real = Projection_l(features_real[l])  ∈ R
    q_fake = Projection_l(features_fake[l])  ∈ R
    Projection_l: LayerNorm → Linear(hidden_l → 1)
    
  Relativistic Pairing (batch 내):
    D_real = q_real - mean(q_fake)   # real이 평균 fake보다 얼마나 나은가
    D_fake = q_fake - mean(q_real)   # fake가 평균 real보다 얼마나 못한가
    
  RAF Loss:
    L_RAF_G = -log(σ(D_fake))        # Generator: fake가 real처럼 보이도록
    L_RAF_D = -log(σ(D_real))        # (Teacher는 frozen이므로 D loss 미사용)
    
  Feature Matching Loss (multi-layer):
    L_FM = Σ_l ||features_real[l] - features_fake[l]||₁ / (T_l × D_l)
    (layers 6, 12, 18, 24의 hidden feature 사용)
    
  Total Vocoder Loss:
    L_vocoder = L_recon + λ_raf·L_RAF_G + λ_fm·L_FM + λ_mel·L_mel
    L_recon = L1(wave) + MR-STFT(3 scales) + MR-Complex-STFT
    λ_raf = 2.0, λ_fm = 1.0, λ_mel = 45.0 (HiFi-GAN 멜 가중치 유지)
```

**기존 Student Decoder 대비 개선**:
| 항목 | 기존 Student v2 | RAF BigVGAN |
|------|----------------|-------------|
| 파라미터 | ~50M | ~14M |
| 출력 샘플레이트 | 44.1kHz | 24kHz (+ BWE) |
| Backbone | Transformer | ConvNeXt v2 |
| Discriminator | MPD+MRD (5.2M) | WavLM (317M, frozen) |
| GAN Loss | LSGAN | RAF (relativistic) |
| Training 안정성 | mode collapse 위험 | RAF로 안정적 |

---

### 4.5 ConvNeXt BWE (대역 확장기)

**목적**: 24kHz 보코더 출력을 44.1kHz로 확장. Vocos BWE 인사이트 기반.

```
ConvNeXt BWE (Bandwidth Extension)
═══════════════════════════════════════════════════════════════

Architecture (Vocos BWE adapted):
  Input:  waveform_24k (B, 1, T_24k) @ 24,000 Hz
  
  ┌─ Analysis ────────────────────────────────────────────┐
  │  STFT: n_fft=1024, hop=256 → spec (B, 513, T_mel)     │
  │  Mel: 128 bands → mel_spec (B, 128, T_mel)             │
  └───────────────────────────────────────────────────────┘

  ┌─ Guidance Feature (from Vocoder) ─────────────────────┐
  │  Vocoder 중간 feature (B, 256, T_voc) @ 225 Hz        │
  │  → Upsample(225→~94Hz, to match mel frames)            │
  │  → Conv1×1(256→64) + LayerNorm → guidance (B, 64, T)   │
  └───────────────────────────────────────────────────────┘

  ┌─ BWE Core (ConvNeXt) ─────────────────────────────────┐
  │  Input: Concat[mel_spec(128), guidance(64)] = (192, T) │
  │                                                         │
  │  ConvNeXtBlock(dim=256, k=7) × 4                       │
  │  ConvNeXtBlock(dim=384, k=7) × 2                       │
  │                                                         │
  │  Output: residual_spec (B, 513, T)                      │
  └───────────────────────────────────────────────────────┘

  ┌─ Synthesis ──────────────────────────────────────────┐
  │  1. Upsample 24kHz waveform:                           │
  │     Nearest-neighbor (×441/240) → 44.1kHz              │
  │     + CausalConv1d(k=31) anti-aliasing                  │
  │                                                         │
  │  2. STFT of upsampled waveform:                        │
  │     spec_naive (B, 1025, T_44k) @ 44.1kHz              │
  │                                                         │
  │  3. Apply BWE residual (high frequencies only):         │
  │     spec_naive의 0-12kHz: 그대로 사용                   │
  │     spec_naive의 12-22.05kHz: residual_spec의 해당 대역  │
  │     → soft cross-fade (gaussian window)                 │
  │                                                         │
  │  4. ISTFT → 44.1kHz waveform                           │
  └───────────────────────────────────────────────────────┘

  Output: waveform (B, 1, T_44k) @ 44,100 Hz

Parameters: ~0.8M
MACs per second: ~0.05G
RTF: < 0.001 (GPU), ~0.005 (CPU)

Losses (BWE only):
  L_BWE = L1(wave_44k, wave_gt_44k)
        + MR-STFT(wave_44k, wave_gt_44k, high_freq_weighted)
        + RAF_BWE(wave_44k, wave_gt_44k)  [경량 WavLM teacher 선택적 사용]
```

---

## 5. RAF Loss의 VC 적용

### 5.1 SSL Teacher 선택

RAF 논문의 핵심 발견: **WavLM Large (317M)의 embedding space에서의 거리가 인간 MOS 평가와 0.93 상관관계**를 보인다. 이는 기존 MPD/MRD discriminator가 포착하지 못하는 지각적 품질 차이를 SSL teacher가 포착할 수 있음을 의미한다.

```
SSL Teacher Comparison (RAF Table 1):
┌────────────────────┬──────────┬───────────┐
│ Teacher            │ MOS corr │ Params    │
├────────────────────┼──────────┼───────────┤
│ WavLM Large        │ 0.93     │ 317M      │ ← HybridVC 선택
│ WavLM Base+        │ 0.91     │ 95M       │
│ HuBERT Large       │ 0.88     │ 317M      │
│ wav2vec 2.0 Large  │ 0.85     │ 317M      │
│ MPD+MRD (baseline) │ 0.72     │ 5.2M      │
└────────────────────┴──────────┴───────────┘

선택 근거:
  - WavLM Large가 최고 MOS 상관관계
  - Frozen이므로 학습 메모리 영향 제한적 (forward only)
  - Feature matching에 multi-layer feature 사용 가능 (6,12,18,24)
```

### 5.2 Relativistic Pairing 전략

RAF의 relativistic pairing은 기존 GAN의 "진짜/가짜 절대 판별"과 달리 **"real이 fake보다 얼마나 나은가"** 를 상대 평가한다. 이는 VC에 특히 유리한 이유:

```
Relativistic Pairing for VC:
═══════════════════════════════════════════════════════════

Batch 구성 전략:
  배치 내에 다양한 화자, 다양한 발화의 real/fake pair 포함

  각 배치 (B=8 기준):
    - Sample 0-3:  화자 A의 src→tgt 변환 (fake) + tgt 실제 발화 (real)
    - Sample 4-7:  화자 B의 src→tgt 변환 (fake) + tgt 실제 발화 (real)
    
  Pairing:
    For each sample i:
      q_real[i] = Projection(WavLM(audio_real[i]))
      q_fake[i] = Projection(WavLM(audio_fake[i]))
    
    D_real[i] = q_real[i] - mean(q_fake)   # real이 fake 평균보다 얼마나 큰가
    D_fake[i] = q_fake[i] - mean(q_real)   # fake가 real 평균보다 얼마나 작은가
    
  Generator Loss (RAF-G):
    L_RAF_G = -mean(log σ(D_fake))
    = -mean(log σ(q_fake - mean(q_real)))
    → fake의 quality score가 real의 평균 이상이 되도록 유도
    
  이점:
    1. Mode collapse 방지: fake가 "모든 real보다 나아질" 필요 없이
       "real들의 평균 수준"이면 됨 → 다양한 출력 허용
    2. Gradient 안정성: LSGAN의 포화(saturation) 문제 없음
    3. 배치 내 다양성 활용: 여러 화자의 real/fake가 섞여
       더 풍부한 상대 비교 signal 제공

Multi-layer RAF:
  RAF는 단일 layer가 아닌 WavLM의 여러 transformer layer에서 계산:
  
  L_RAF = Σ_l w_l · L_RAF_l    (l ∈ {6, 12, 18, 24})
  w_l = [0.25, 0.25, 0.25, 0.25]  (uniform)
  
  - Layer 6:  낮은 수준의 음향 특징 (spectral envelope)
  - Layer 12: 중간 수준 (phonetic content)
  - Layer 18: 높은 수준 (speaker identity, prosody)
  - Layer 24: 최상위 (semantic, naturalness)
```

### 5.3 VC Training에의 RAF 통합 방식

```
Phase별 RAF 통합:
═══════════════════════════════════════════════════════════

Phase 1 (AE Pretraining): RAF 미사용
  - Encoder + Vocoder(BigVGAN) 재구성 학습
  - L = L_recon (L1 + MR-STFT + MR-Mel)
  - RAF 없는 pure reconstruction으로 기본 파형 생성 능력 확보

Phase 2 (SSL Distillation): RAF 미사용
  - Encoder + WavLM distillation
  - L = L_distill (L1 + cosine between z_content and WavLM features)
  - Encoder의 latent quality 향상

Phase 3 (Converter + Vocoder): RAF 도입
  - Converter + Vocoder (Encoder freeze, SpeakerEncoder freeze)
  - L = L_recon + λ_raf·L_RAF + λ_fm·L_FM + λ_mel·L_mel
  - RAF는 converter gradient를 vocoder로 end-to-end 전파
  - λ_raf schedule:
    step 0-10k:     λ_raf = 0.0  (warmup)
    step 10k-50k:   λ_raf = 0.5 → 2.0  (linear ramp)
    step 50k-200k:  λ_raf = 2.0  (full)

Phase 4 (End-to-End Fine-tuning): RAF full
  - Encoder + Converter + Vocoder + BWE (전체)
  - RAF 모든 컴포넌트에 gradient 전파
  - L = L_recon + 2.0·L_RAF + 1.0·L_FM + 45.0·L_mel + L_BWE

RAF Warmup이 중요한 이유:
  - 초기 vocoder 출력은 noise에 가까움 → WavLM feature가 meaningless
  - 일정 수준 재구성 품질 확보 후 RAF 도입이 효과적
  - λ_raf 점진적 증가로 안정적 transition
```

---

## 6. 차원 / 파라미터 / MACs 명세

### 6.1 차원 흐름도

```
Latent Space (Continuous, 25 Hz):
═══════════════════════════════════════════════════════════

Waveform (44.1kHz)
    │
    ▼
┌─────────────────────┐
│ HybridCodec Encoder │  in:  (B, 1, T)           @ 44,100 Hz
│   (3.2M params)     │  out: (B, T/1764, 768)    @ 25 Hz
└────────┬────────────┘
         │ z_src (B, T_lat, 768)
         ▼
┌─────────────────────┐
│ CausalConvNeXt      │  in:  z_src (B, T_lat, 768)
│ Converter           │       speaker_cond (B, 128)
│   (5.4M params)     │       prosody (B, T_lat, 3)
│                     │  out: z_out (B, T_lat, 768)
└────────┬────────────┘
         │ z_out (B, T_lat, 768)
         ▼
┌─────────────────────┐
│ RAF BigVGAN Vocoder │  in:  z_out (B, T_lat, 768)
│   (14.2M params)    │       speaker_global (B, 128)
│                     │  hid: 512 → 384 → 256 → 128
│                     │  out: (B, 1, T*960)      @ 24,000 Hz
└────────┬────────────┘
         │ waveform_24k
         ▼
┌─────────────────────┐
│ ConvNeXt BWE        │  in:  waveform_24k
│   (0.8M params)     │  out: waveform_44k (B, 1, T*1764) @ 44,100 Hz
└─────────────────────┘

Latent ↔ Waveform 관계:
  1 latent frame @ 25Hz = 1764 samples @ 44.1kHz = 960 samples @ 24kHz
  24kHz vocoder 출력 → 44.1kHz BWE 출력 (×1.8375 업샘플)
```

### 6.2 파라미터 수 상세

```
Component                    │ Params  │ Trainable │ 비고
─────────────────────────────┼─────────┼───────────┼──────────────────
HybridCodec Encoder          │  3.2M   │   3.2M    │ ConvNeXt v2 ×6 stages
  + WavLM Projection         │  0.8M   │   0.8M    │ SSL distillation
Speaker Encoder (ECAPA)      │  6.2M   │     0M    │ Frozen, pretrained
Prosody Extractor            │  0.3M   │   0.3M    │ F0+energy+voicing
CausalConvNeXt Converter     │  5.4M   │   5.4M    │ 10 blocks + cross-attn
  - in_proj                  │  0.15M  │           │
  - ConvNeXt blocks ×10      │  4.2M   │           │
  - cross-attn ×3            │  0.6M   │           │
  - speaker prompt            │  0.001M │           │ 4 tokens × 192
  - out_proj + gate          │  0.15M  │           │
RAF BigVGAN Vocoder          │ 14.2M   │  14.2M    │ ConvNeXt v2 generator
  - Wave PreNet (6 blocks)   │  5.4M   │           │
  - Upsampler (×3×3)         │  6.8M   │           │
  - PostNet + Head           │  2.0M   │           │
ConvNeXt BWE                 │  0.8M   │   0.8M    │ Vocos-style
WavLM Large (RAF Teacher)    │317.0M   │     0M    │ Frozen, training only
─────────────────────────────┼─────────┼───────────┼──────────────────
Total (inference)            │ 30.7M   │           │
Total (training, GPU memory) │ ~400M   │           │ WavLM 포함
```

### 6.3 MACs (Multiply-Accumulate) 추정

```
Component                    │ MACs/sec   │ 기준
─────────────────────────────┼────────────┼──────────────────
HybridCodec Encoder          │  2.1 G     │ 44.1kHz 1초 오디오
CausalConvNeXt Converter     │  0.02 G    │ 25Hz × 768dim
RAF BigVGAN Vocoder          │  8.5 G     │ 24kHz 1초 오디오
ConvNeXt BWE                 │  0.05 G    │ 24kHz→44.1kHz 1초
─────────────────────────────┼────────────┼──────────────────
Total per second             │ 10.67 G    │

RTF 추정:
  GPU (Apple M2 Max, ~13.6 TFLOPS FP16):
    RTF ≈ 10.67 / 13,600 ≈ 0.0008 → 실시간의 1/1250
    
  CPU (Apple M2 Max, ~3.6 TFLOPS):
    RTF ≈ 10.67 / 3,600 ≈ 0.003 → 실시간의 1/333
    
  GPU (NVIDIA A100, ~312 TFLOPS FP16):
    RTF ≈ 10.67 / 312,000 ≈ 0.00003 → 극도로 여유
    
  실제 RTF는 memory bandwidth, kernel launch overhead로 인해
  이론치의 10-50배. GPU RTF < 0.05, CPU RTF < 0.2 예상.
```

---

## 7. Training Pipeline

### 7.1 Phase 개요

```
Phase 0: Foundation (btrv3lite 자산)
Phase 1: AE Pretraining (Encoder + Vocoder)
Phase 2: SSL Distillation (Encoder + WavLM)
Phase 3: Converter + Vocoder Joint (with RAF)
Phase 4: End-to-End Fine-tuning (with RAF + BWE)
```

### 7.2 Phase 0: Foundation (기존 자산)

**목적**: btrv3lite/btrvrc0의 학습된 가중치를 HybridVC로 transfer.

```
자산 매핑:
  btrv3lite MioCodec Encoder → HybridCodec Encoder (warmstart)
    - 6-stage ConvNeXt 가중치 그대로 로드
    - 192→768 head projection만 새로 초기화
    
  btrv3lite CausalLatentConverter → CausalConvNeXt Converter (warmstart)
    - 8개 기존 블록 가중치 그대로 로드
    - 추가 2개 블록 + 3개 cross-attn은 zero-init
    - 전체가 identity로 시작하므로 수학적으로 안전
    
  btrvrc0 SpeakerConditioner → Speaker Encoder (그대로)
    - ECAPA bank, F0 stats, conditioner cfg 재활용

Warmstart 검증:
  1. 기존 converter.pt 로드 → 8개 블록 weights 복사
  2. 새 블록 2개는 gate=0으로 identity 보장
  3. 동일 입력에 대해 기존 converter와 동일 출력 확인
```

### 7.3 Phase 1: AE Pretraining

**목적**: Encoder + Vocoder(BigVGAN)의 자기재구성(self-reconstruction) 학습.

```
설정:
  모델: HybridCodec Encoder + RAF BigVGAN Vocoder
  Frozen: 없음 (전체 학습)
  데이터: VCTK + LibriTTS + 기타 multi-speaker 음성 데이터
  
손실 함수:
  L_phase1 = L1(wave) + MR-STFT + MR-Complex-STFT + MR-Mel + L_latent
  
  L_latent:
    z_pred = Encoder(audio)
    z_teacher = MioCodec_Teacher.encode(audio)  # frozen teacher
    L_latent = L1(z_pred, z_teacher) + (1 - cos_sim(z_pred, z_teacher))
    
    → Encoder가 MioCodec teacher의 latent space를 복제하도록 distillation

학습 설정:
  Optimizer:     AdamW (β₁=0.8, β₂=0.99)
  LR:            2e-4 (warmup 5k → cosine decay to 1e-6)
  Batch size:    8 (A100 40GB), 1 (MPS, grad_acc=8)
  Sequence:      3초 (audio), 75 latent frames
  Steps:         200,000
  Mixed Precision: FP16 (A100), FP32 (MPS)
  
Discriminator:
  Phase 1에서는 사용하지 않음 (pure reconstruction)
  RAF는 Phase 3에서 도입

모니터링:
  - 5,000 step마다 validation loss
  - 10,000 step마다 reconstruction sample 저장
  - MioCodec latent cosine similarity 추적 (목표: > 0.95)
```

### 7.4 Phase 2: SSL Distillation

**목적**: USAD 2.0 방식으로 WavLM feature를 Encoder latent 공간에 distillation.

```
설정:
  모델: HybridCodec Encoder (Vocoder freeze)
  Frozen: Vocoder
  학습 가능: Encoder + WavLM Projection head
  
데이터:
  - Phase 1과 동일
  - 각 batch에서 audio → WavLM feature 추출 (on-the-fly or pre-cached)
  
손실 함수:
  L_phase2 = L_phase1 + λ_distill·L_distill
  
  L_distill:
    z_content = Encoder(audio)              # (B, T_lat, 768)
    ssl_feat = WavLM(audio)                 # (B, T_ssl, 1024)
    ssl_feat → AvgPool(T_ssl→T_lat)        # temporal alignment
    ssl_feat → Proj(1024→768) → z_ssl      # (B, T_lat, 768)
    
    L_distill = L1(z_content, z_ssl.detach()) 
              + (1 - cos_sim(z_content, z_ssl.detach()))

  λ_distill = 0.5 (SSL distillation weight)

학습 설정:
  LR:            1e-4 (warmup 2k → cosine decay)
  Batch size:    4 (WavLM 메모리 부담 고려)
  Steps:         50,000
  
WavLM Caching (선택적):
  - WavLM forward는 비용이 큼 (317M params, ~25G MACs/sec)
  - 전체 데이터셋의 WavLM feature를 미리 추출하여 caching 가능
  - Cache format: (audio_hash → (ssl_feat_mean, ssl_feat_std))
  - Disk: ~500GB for 1000시간 음성
```

### 7.5 Phase 3: Converter + Vocoder Joint (RAF 도입)

**목적**: 변환기(Converter)와 보코더(Vocoder)를 RAF loss로 end-to-end 학습.

```
설정:
  모델: CausalConvNeXt Converter + RAF BigVGAN Vocoder
  Frozen: HybridCodec Encoder, Speaker Encoder, WavLM Teacher
  학습 가능: Converter + Vocoder + Speaker Prompt tokens

데이터 구성:
  - 병렬 발화 쌍 (동일 내용, 다른 화자) 또는
  - 동일 텍스트 TTS multi-speaker 데이터
  - z_src = Encoder(audio_src), z_tgt = Encoder(audio_tgt)
  - speaker_cond = SpeakerEncoder(audio_tgt_ref)
  - prosody = ProsodyExtractor(audio_src)

손실 함수:
  L_phase3 = L_vocoder + L_consistency + L_speaker

  L_vocoder (Phase 1과 동일 + RAF):
    z_out = Converter(z_src, speaker_cond, prosody)
    wave_out = Vocoder(z_out, speaker_global)
    L_vocoder = L1(wave_out, wave_tgt) 
              + MR-STFT + MR-Complex-STFT + MR-Mel(λ=45)
              + λ_raf·L_RAF_G + λ_fm·L_FM

  L_consistency (latent level):
    z_out = Converter(z_src, speaker_cond, prosody)
    L_cons = L1(z_out, z_tgt) + (1 - cos_sim(z_out, z_tgt))
    (weight: 0.3)
    
  L_speaker (speaker identity preservation):
    wave_out → Encoder → z_reencoded
    L_spk = 1 - cos_sim(SpeakerEncoder(wave_out), speaker_cond)
    (weight: 0.1)

  Total: L = L_vocoder + 0.3·L_cons + 0.1·L_spk

RAF Schedule:
  step 0-10k:     λ_raf = 0.0,    λ_fm = 0.0    (warmup)
  step 10k-30k:   λ_raf = 0→1.0,  λ_fm = 0→0.5
  step 30k-50k:   λ_raf = 1.0→2.0, λ_fm = 0.5→1.0
  step 50k-200k:  λ_raf = 2.0,    λ_fm = 1.0    (full)

학습 설정:
  Optimizer:     AdamW (β₁=0.8, β₂=0.99)
  LR Converter:  1e-4
  LR Vocoder:    5e-5
  Batch size:    6 (A100), 1 (MPS, grad_acc=6)
  Sequence:      3초 (75 latent frames → ~2.88초 @ 24kHz)
  Steps:         200,000
  
  Gradient clipping: max_norm=1.0 (안정성)
  EMA: decay=0.999 (inference용 weights)
```

### 7.6 Phase 4: End-to-End Fine-tuning

**목적**: Encoder까지 unfreeze하여 전체 파이프라인 최적화. BWE 통합.

```
설정:
  모델: Encoder + Converter + Vocoder + BWE (전체)
  Frozen: Speaker Encoder, WavLM Teacher
  학습 가능: Encoder (낮은 LR), Converter, Vocoder, BWE

손실 함수:
  L_phase4 = L_vocoder + L_BWE + 0.1·L_cons + 0.05·L_spk

  L_BWE:
    wave_24k = Vocoder(z_out, speaker_global)
    wave_44k = BWE(wave_24k)
    wave_gt_44k = Target audio @ 44.1kHz
    
    L_BWE = L1(wave_44k, wave_gt_44k)
          + MR-STFT(wave_44k, wave_gt_44k, high_freq_emphasized)
          + 0.5·L_RAF_BWE(wave_44k, wave_gt_44k)

학습 설정:
  LR Encoder:    1e-5  (very low, pretrained weights 보호)
  LR Converter:  2e-5
  LR Vocoder:    2e-5
  LR BWE:        5e-5  (새로운 컴포넌트)
  
  Batch size:    4 (A100), 1 (MPS, grad_acc=4)
  Sequence:      3초
  Steps:         50,000

  RAF: λ_raf = 1.0 (fine-tuning이므로 낮은 강도)
```

### 7.7 학습 인프라 요약

```
하드웨어 권장:
  - NVIDIA A100 40GB: Phase 1-4 전체 가능
  - Apple M2 Max (MPS): Phase 1, 2 가능 (batch=1, grad_acc)
    Phase 3-4: WavLM 메모리 부담 → feature pre-caching 필요

학습 시간 추정 (A100 기준):
  Phase 1: 200k steps × 0.3s/step ≈ 16.7 시간
  Phase 2:  50k steps × 0.5s/step ≈  7.0 시간  (WavLM overhead)
  Phase 3: 200k steps × 0.4s/step ≈ 22.2 시간
  Phase 4:  50k steps × 0.5s/step ≈  7.0 시간
  ─────────────────────────────────────────
  Total:                           ≈ 53 시간  (~2.2일)

  → 500 GPU-hours 이내 (목표 달성)

MPS 최적화:
  - WavLM feature를 CPU에서 pre-compute (MPS WavLM 불안정)
  - torch.stft → CPU fallback (기존 losses.py의 MPS safe workaround)
  - Batch size 1, gradient accumulation으로 유효 배치 구현
  - FP32 권장
```

---

## 8. 실시간 추론 전략

### 8.1 스트리밍 아키텍처

```
Streaming HybridVC Inference
═══════════════════════════════════════════════════════════

Buffer 구성:
  [ left_context | chunk | right_lookahead ]
  ├── 8 frames ──┼─ 2 frames ─┼── 4 frames ──┤
    (320ms)         (80ms)       (160ms)

  Total window: 14 latent frames = 560ms audio @ 44.1kHz
                = 24,696 samples

Pipeline per chunk (80ms = 2 latent frames):
═══════════════════════════════════════════════════════════

1. Audio Input:
   chunk_raw: 3528 samples (80ms @ 44.1kHz)
   → Ring Buffer에 추가

2. Window Assembly:
   window_44k = [history(14112) | chunk(3528) | lookahead(7056)]
              = 24,696 samples (560ms)

3. Speech Activity Detection (선택):
   energy = RMS(window_44k)
   if energy < threshold: silence insertion, skip processing

4. HybridCodec Encoder (causal, window mode):
   z_window = Encoder(window_44k)  → (14, 768) @ 25Hz
   latency: ~15ms (GPU)

5. Latent Slicing:
   z_chunk = z_window[8:10, :]  → (2, 768)
   (8 frames left context 버리고, chunk 2프레임만 추출)

6. Prosody Extraction:
   prosody = ProsodyExtractor(window_44k)  → (14, 3)
   prosody_chunk = prosody[8:10, :]        → (2, 3)

7. Speaker Condition (pre-computed, cached):
   speaker_cond: (128,) — 전체 utterance 동일
   speaker_global: (128,)
   speaker_prompt: (4, 192)

8. Converter (frame-by-frame causal):
   # 이전 chunk의 마지막 2프레임을 context로
   z_context = Converter(z_prev, speaker_cond, prosody_prev)
   z_chunk_in = Concat[z_context, z_chunk]  # (4, 768) or more
   z_out = Converter(z_chunk_in, speaker_cond, prosody_chunk)
   z_out = z_out[-2:, :]  # 마지막 2프레임만 사용
   latency: ~8ms (GPU)

9. Vocoder (causal):
   decoder_input = Concat[vocoder_history, z_out]  # 충분한 context
   waveform_24k = Vocoder(decoder_input, speaker_global)
   wave_chunk = waveform_24k[-(2*960):]  # 1920 samples @ 24kHz
   latency: ~20ms (GPU)

10. BWE:
    waveform_44k = BWE(wave_chunk)  # 1920 → 3528 samples
    latency: ~2ms (GPU)

11. Output:
    emit waveform_44k
    shift buffers:
      - left_context ← left_context[2:] + z_chunk
      - vocoder_history ← vocoder_history[2:] + z_out
      - lookahead ← new audio input

Total per-chunk latency: ~45ms (GPU)
TTFB: ~125ms (최초 window fill 560ms의 일부 + processing)
```

### 8.2 TTFB 분석

```
TTFB Breakdown (GPU, A100 기준):
─────────────────────────────────────────────────────────
Component                    │ Latency (ms) │ 비고
─────────────────────────────┼─────────────┼────────────────
Audio buffering (lookahead)  │     80      │ 4 frames minimum
Audio buffering (chunk)      │     40      │ 2 frames
Encoder (14 frames)          │     15      │ ConvNeXt ×6 stages
Prosody                      │      3      │ 경량 ConvNet
Converter (2 frames)         │      8      │ 10 ConvNeXt blocks
Vocoder (2 frames→24kHz)     │     20      │ ConvNeXt upsampler
BWE                          │      2      │ ConvNeXt residual
─────────────────────────────┼─────────────┼────────────────
Total TTFB                   │    128      │ (GPU)

CPU (M2 Max 기준):
Encoder                      │     35      │
Converter                    │     15      │
Vocoder                      │     50      │
BWE                          │      5      │
Total TTFB                   │    190      │ (CPU)
```

### 8.3 RTF 분석

```
청크 크기: 80ms (2 latent frames)

Per-chunk MACs:
  Encoder:      window(14 frames) 처리 → 2.1G/sec × 0.56sec ≈ 1.18G
  Converter:    2 frames 처리 → 0.02G/sec × 0.08sec ≈ 0.0016G
  Vocoder:      2 frames→24kHz 80ms → 8.5G/sec × 0.08sec ≈ 0.68G
  BWE:          24kHz→44.1kHz 80ms → 0.05G/sec × 0.08sec ≈ 0.004G
  ─────────────────────────────────────────────────────────
  Total per chunk: ~1.87G MACs

GPU RTF (M2 Max, ~13.6 TFLOPS):
  Processing time ≈ 1.87G / 13,600G ≈ 0.137ms
  RTF = 0.137ms / 80ms ≈ 0.0017  (실제론 memory overhead 고려해 0.02-0.05)

CPU RTF (M2 Max, ~3.6 TFLOPS):
  Processing time ≈ 1.87G / 3,600G ≈ 0.52ms
  RTF ≈ 0.0065  (실제 0.1-0.2)

→ RTF < 0.5 목표 충분히 달성 가능
```

### 8.4 메모리 사용량

```
Inference Memory (GPU/MPS):
─────────────────────────────────────────
Component              │ Memory
───────────────────────┼─────────────────
Encoder                │  13 MB
Converter              │  22 MB
Vocoder                │  57 MB
BWE                    │   4 MB
Speaker Encoder        │  25 MB
Ring Buffers           │   5 MB
───────────────────────┼─────────────────
Total                  │ ~126 MB

Training Memory (A100, batch=8):
  + WavLM (frozen)     │ +2,500 MB
  + Optimizer states   │ +1,200 MB
  + Activations        │ +3,000 MB
  ─────────────────────┼─────────────────
  Total                │ ~6,800 MB  (A100 40GB 이내)
```

---

## 9. btrv3lite / btrvrc0 자산 재활용 매핑

### 9.1 직접 재활용 컴포넌트

```
btrv3lite/btrvrc0 자산              → HybridVC 매핑                   │ 재활용 방식
─────────────────────────────────────────────────────────────────────┼──────────────
MioCodec Teacher Encoder            → HybridCodec Encoder             │ Weight transfer
  (6-stage ConvNeXt, 768-dim)         (6-stage ConvNeXt v2, 768-dim)   │ + SSL proj 신규

CausalLatentConverter               → CausalConvNeXt Converter        │ Weight transfer
  (8 blocks, 192-dim, AdaLN-Zero)     (10 blocks, 192-dim, AdaLN-Zero)│ 8개 블록 그대로
  converter.pt (12.7M)                + cross-attn                     │ +2 블록 zero-init

SpeakerConditioner                  → Speaker Encoder                 │ 코드 그대로
  (256-entry bank, ECAPA+F0 stats)    (256-entry bank)                 │ conditioner.py

F0 Extractor (f0.py)                → Prosody Extractor               │ 코드 그대로
  (RMVPE-based)                       (F0 + energy + voicing)          │

Loss functions (losses.py)          → Reconstruction losses            │ 코드 그대로
  MultiResMel, MR-STFT, MR-Complex    L1, MR-STFT, MR-Mel               │ + RAF loss 추가

Streaming inference                 → Streaming pipeline               │ 구조 재활용
  (streaming_v3.py)                   (ring buffer, causal chain)       │ 새 vocoder로 교체
  
Speaker Bank (bank.pt)              → Speaker Bank                     │ 파일 그대로
  (257 entries, sollsl)               (256 entries)                     │

Discriminator (discriminator.py)    → RAF Discriminator                │ 대체
  (MPD+MRD, 5.2M)                     (WavLM, 317M frozen)              │ RAF가 대체
```

### 9.2 Weight Transfer 상세

```
Encoder Weight Transfer:
═══════════════════════════════════════════════════════════
btrv3lite Student Encoder          HybridCodec Encoder
─────────────────────────────────────────────────────────
encoder.conv_in                    encoder.conv_in           → 직접 복사
encoder.stages.0-5                 encoder.stages.0-5        → 직접 복사
(동일 구조: CausalConv1d +        (ConvNeXt v2 block으로
 ResBlock, 192-dim)                 확장 가능)
encoder.head (192→768)             encoder.head (192→768)    → 직접 복사

Converter Weight Transfer:
═══════════════════════════════════════════════════════════
btrv3lite Converter (8 blocks)     HybridVC Converter (10 blocks)
─────────────────────────────────────────────────────────
in_proj (768→192)                  in_proj (768→192)        → 직접 복사
blocks.0-7 (ConvNeXt w/ AdaLN)     blocks.0-7               → 직접 복사
  - dwconv, pwconv1, pwconv2         - dwconv, mlp.0, mlp.2   (구조 매핑)
  - AdaLN scale/shift MLP            - AdaLN scale/shift MLP
blocks.8-9                        blocks.8-9                → zero-init
  (none)                             (ConvNeXt + AdaLN-Zero)
cross_attns.0-2                   cross_attns.0-2           → zero-init
  (none)                             (MultiHeadAttn + prompt)
out_proj (192→768)                out_proj (192→768)        → 직접 복사
out_gate                          out_gate                  → 직접 복사

Decoder → Vocoder (구조 상이, weight transfer 불가):
═══════════════════════════════════════════════════════════
Student v2 Decoder (~50M)         RAF BigVGAN Vocoder (~14M)
  - Transformer 8L                  - ConvNeXt PreNet 6 blocks
  - ResNet blocks                   - ConvNeXt Upsampler
  - ISTFT head                      - ConvNeXt Head
→ Decoder는 폐기, Vocoder는 scratch 학습
  Phase 1에서 Encoder + Vocoder 재구성으로 새로 학습
```

### 9.3 재활용 불가능한 자산 및 대체

```
자산                              │ 상태     │ 대체 방안
──────────────────────────────────┼─────────┼──────────────────────────
Student v2 Decoder                │ 폐기     │ RAF BigVGAN Vocoder로 대체
  (Transformer, 50M, 44.1kHz)     │          │ (ConvNeXt, 14M, 24kHz+BWE)
                                  
MPD/MRD Discriminator             │ 폐기     │ RAF SSL Teacher로 대체
  (5.2M, LSGAN)                   │          │ (WavLM, 317M frozen)
                                  
MioCodec Student codec 전체       │ 부분 재활용│ Encoder만 재활용
  (student.py, student_v2.py)     │          │ Decoder는 폐기
```

---

## 10. RAF vs 기존 HiFi-GAN GAN Loss 비교 분석

### 10.1 개념적 비교

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    기존 HiFi-GAN (MPD+MRD+LSGAN)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Generator G                          Discriminator D                        │
│       │                                     │                                │
│       ▼                                     ▼                                │
│  fake_audio ──────────────────────→ D(fake) → score_fake                    │
│       │                                                                      │
│  real_audio ──────────────────────→ D(real) → score_real                    │
│                                                                              │
│  L_G = (D(fake) - 1)²              ← LSGAN hinge                            │
│  L_D = (D(real) - 1)² + D(fake)²                                            │
│                                                                              │
│  특징:                                                                        │
│  • Discriminator가 학습 가능 (5.2M params, trainable)                        │
│  • MPD: 주기적 패턴 감지 (2,3,5,7,11 periods)                                │
│  • MRD: multi-resolution STFT spectrogram 판별                               │
│  • Mode collapse 위험: D가 너무 강해지면 G가 소수 mode만 생성               │
│  • 학습 불안정: G와 D의 minimax game 균형 잡기 어려움                        │
│  • Feature matching으로 안정화 (intermediate D features 비교)                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAF (Relativistic Adversarial Feature)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Generator G                          SSL Teacher T (FROZEN)                 │
│       │                                     │                                │
│       ▼                                     ▼                                │
│  fake_audio ──────────────────────→ T(fake) → features_fake                 │
│                                                                              │
│  real_audio ──────────────────────→ T(real) → features_real                 │
│                                                                              │
│  Quality projection:                                                          │
│    q_fake = Proj(features_fake)   ∈ R                                        │
│    q_real = Proj(features_real)   ∈ R                                        │
│                                                                              │
│  Relativistic pairing (batch 내):                                            │
│    D_fake = q_fake - mean(q_real)   ← fake가 전체 real보다 얼마나 못한가      │
│    D_real = q_real - mean(q_fake)   ← real이 전체 fake보다 얼마나 나은가      │
│                                                                              │
│  L_RAF_G = -log(σ(D_fake))         ← fake의 상대 점수 최대화                 │
│  (Teacher는 frozen → D loss 없음)                                            │
│                                                                              │
│  특징:                                                                        │
│  • Teacher frozen → minimax game 없음, 순수 생성 모델 최적화                 │
│  • WavLM embedding space = 인간 MOS와 0.93 상관관계                           │
│  • Relativistic → mode collapse에서 자유로움                                  │
│  • Multi-layer FM → 저수준~고수준 특징 모두 매칭                              │
│  • 14M BigVGAN이 112M BigVGAN 능가 (RAF 논문 검증)                           │
│  • 배치 크기에 민감 (상대 비교이므로)                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 정량적 비교 (RAF 논문 기반)

```
┌──────────────────────┬───────────┬───────────┬───────────┬───────────┐
│ Metric               │ HiFi-GAN  │ BigVGAN   │ BigVGAN   │ RAF       │
│                      │ (MPD+MRD) │ (112M)    │ (14M)     │ (14M)     │
├──────────────────────┼───────────┼───────────┼───────────┼───────────┤
│ MOS (naturalness)    │ 3.82      │ 4.05      │ 3.91      │ 4.12 ★    │
│ PESQ                 │ 3.21      │ 3.45      │ 3.38      │ 3.52 ★    │
│ CER (%)              │ 2.8       │ 2.1       │ 2.4       │ 1.9  ★    │
│ Speaker Similarity   │ 0.82      │ 0.87      │ 0.84      │ 0.89 ★    │
│ Training Stability   │ Low       │ Medium    │ Low       │ High ★    │
│ Params (disc/frozen) │ 5.2M/0M│ 17M/0M     │ 5.2M/0M│ 317M/317M│
│ GPU-hours (A100)     │ ~200      │ ~400      │ ~250      │ ~150 ★    │
│ Mode Collapse Risk   │ High      │ Medium    │ High      │ Low   ★    │
└──────────────────────┴───────────┴───────────┴───────────┴───────────┘

★ = HybridVC 선택 근거

핵심 인사이트:
1. RAF(14M)이 BigVGAN(112M)을 모든 지표에서 능가
2. RAF는 discriminator 학습이 필요 없어 training pipeline 단순
3. SSL teacher의 frozen weights → 학습 안정성 극대화
4. Relativistic pairing이 mode collapse 위험을 근본적으로 해결
```

### 10.3 VC 특화 비교

```
VC 관점에서의 RAF 장점:
═══════════════════════════════════════════════════════════

1. Speaker Identity 보존:
   - MPD/MRD: 주기성과 스펙트럼만 판별 → 화자 특성 간접적 포착
   - RAF: WavLM의 layer 18-24가 speaker identity에 민감
     → RAF loss가 speaker similarity를 직접적으로 개선

2. Content Preservation:
   - MPD/MRD: phonetic content 손실 가능 (GAN이 texture에 집중)
   - RAF: WavLM의 layer 6-12가 phonetic content에 민감
     → Multi-layer FM이 content 보존에 기여

3. Prosody Naturalness:
   - MPD/MRD: 국소적 파형 패턴만 평가
   - RAF: WavLM의 temporal context (self-attention)가
     장기 운율 패턴을 포착

4. Training Data Efficiency:
   - MPD/MRD: 다양한 화자/환경의 real/fake 구분 학습에 많은 데이터 필요
   - RAF: WavLM이 이미 방대한 데이터로 pre-trained
     → 적은 VC 데이터로도 효과적 학습

RAF의 VC 특화 한계:
  - WavLM이 non-speech audio에 약함 (음악, 환경음 VC에 불리)
  - 배치 크기가 작으면 relativistic pairing 효과 감소
  - SSL teacher의 language bias (영어 중심 pre-training)
```

### 10.4 RAF Loss 가중치 설계

```
Phase 3 RAF Loss 가중치:

L_RAF_total = λ_raf · Σ_l w_l · L_RAF_l  +  λ_fm · Σ_l v_l · L_FM_l

Layer weights (l ∈ {6, 12, 18, 24}):
  w_l (RAF):    [0.2, 0.3, 0.3, 0.2]  ← 중간 layer 강조
  v_l (FM):     [0.1, 0.2, 0.4, 0.3]  ← 고층 FM 강조

이유:
  - Layer 6-12: 낮은 수준 특징 (spectral, phonetic) → RAF 가중치 낮게
    (이미 MR-STFT/Mel loss가 커버)
  - Layer 18-24: 높은 수준 특징 (speaker, naturalness) → RAF 가중치 높게
    (기존 loss가 커버 못하는 지각적 품질)
  - FM은 반대로 고층 특징 매칭에 더 높은 가중치

λ schedule:
  step      λ_raf   λ_fm    설명
  ─────────────────────────────────
  0-10k     0.0     0.0     Warmup (reconstruction only)
  10k-30k   0.5     0.25    RAF 진입 (낮은 강도)
  30k-60k   1.0     0.5     중간 강도
  60k-100k  1.5     0.75    
  100k-200k 2.0     1.0     Full strength
```

---

## 11. 리스크 및 대안

### 11.1 기술적 리스크

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Risk 1: WavLM Teacher의 도메인 불일치                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 심각도: Medium                                                               │
│                                                                              │
│ 문제:                                                                         │
│   WavLM은 16kHz로 pre-trained. 24kHz/44.1kHz 음성과 mismatch 가능성.          │
│   → RAF feature가 고주파 대역(>8kHz) 품질을 제대로 반영하지 못할 수 있음.     │
│                                                                              │
│ 영향:                                                                         │
│   BWE로 생성된 12-22kHz 대역의 품질이 RAF loss에 제대로 반영되지 않아          │
│   고주파 아티팩트가 발생할 수 있음.                                              │
│                                                                              │
│ 대안:                                                                         │
│   A1) 24kHz audio를 16kHz로 downsampling하여 WavLM 입력                        │
│       → 고주파 정보 손실이지만, RAF loss는 지각적 품질 평가에는 충분            │
│   A2) WavLM 외에 48kHz 호환 SSL teacher 탐색 (예: MERT-v1-95M)                │
│   A3) BWE에 독립적인 high-freq discriminator 추가 (MPD high-period만 사용)     │
│   A4) Phase 4에서만 WavLM 사용, Phase 1-3은 기존 MPD/MRD로 학습                │
│                                                                              │
│ 권장: A1 + A3                                                               │
│   - 24kHz→16kHz downsampling (anti-alias filter 적용)                        │
│   - BWE 전용 high-frequency MPD (period 2,3) 추가                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Risk 2: RAF Relativistic Pairing의 배치 크기 의존성                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ 심각도: Medium                                                               │
│                                                                              │
│ 문제:                                                                         │
│   Relativistic pairing은 배치 내 평균을 사용 → 배치가 작으면                    │
│   mean(q_real), mean(q_fake) 추정이 불안정해짐.                               │
│   MPS 학습 시 batch=1 불가능 (grad_acc도 배치 통계에는 도움 안 됨).            │
│                                                                              │
│ 영향:                                                                         │
│   MPS에서 Phase 3, 4 학습이 어려움 (WavLM + small batch).                      │
│   RAF 효과가 감소하고 일반 LSGAN과 유사해짐.                                     │
│                                                                              │
│ 대안:                                                                         │
│   B1) RAF loss 계산 시 EMA of q_real 통계 유지 (momentum=0.99)                │
│       → D_fake = q_fake - EMA(mean(q_real))                                  │
│   B2) RAF loss를 global batch stat로 계산 (grad_acc step 간 stat 누적)         │
│   B3) MPS에서는 RAF 없이 MPD/MRD로 Phase 3 학습 → GPU에서 RAF fine-tuning     │
│   B4) Queue-based RAF: 이전 batch의 real sample을 queue에 저장하여 사용        │
│                                                                              │
│ 권장: B1 + B3                                                               │
│   - EMA of real/fake statistics로 배치 크기 의존성 완화                        │
│   - MPS 학습자는 MPD/MRD로도 충분한 품질 달성 가능 (btrv3lite 검증)            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Risk 3: BWE Cascade Error                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 심각도: Medium-Low                                                           │
│                                                                              │
│ 문제:                                                                         │
│   Vocoder(24kHz) → BWE(44.1kHz)의 cascade에서                                  │
│   Vocoder의 아티팩트가 BWE에 의해 증폭될 수 있음.                                 │
│   특히 12kHz 부근의 crossover 영역에서 spectral discontinuity.                 │
│                                                                              │
│ 영향:                                                                         │
│   고주파 대역(12-22kHz)에서 unnatural한 texture.                                │
│   Crossover 주파수(12kHz) 부근의 위상 불연속.                                    │
│                                                                              │
│ 대안:                                                                         │
│   C1) Vocoder를 바로 44.1kHz로 출력 (BWE 제거)                                │
│       → Vocoder 파라미터 14M→~25M, 연산량 8.5G→~16G                           │
│       → 여전히 실시간 가능하지만 RAF loss의 pre-trained WavLM 제약               │
│   C2) BWE 입력을 waveform 대신 Vocoder의 중간 feature map으로                    │
│       → 더 풍부한 정보로 BWE가 보간                                             │
│   C3) Crossover 영역에 learnable blending filter 적용                         │
│   C4) BWE + Vocoder를 joint training (Phase 4에서)                             │
│                                                                              │
│ 권장: C2 + C4                                                               │
│   - BWE 입력을 Vocoder의 마지막 ConvNeXtBlock 출력(256-dim, 225Hz)으로         │
│   - Phase 4에서 BWE-Vocoder joint fine-tuning                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Risk 4: MioCodec Teacher Latent Space 호환성                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 심각도: Low                                                                  │
│                                                                              │
│ 문제:                                                                         │
│   SSL distillation(USAD 2.0)으로 인해 HybridCodec Encoder의 latent space가     │
│   MioCodec teacher와 달라질 수 있음. → 기존 converter checkpoint와 호환 불가.   │
│                                                                              │
│ 영향:                                                                         │
│   Phase 0 weight transfer가 불완전할 수 있음.                                   │
│   기존 converter의 frozen condition이 깨짐.                                     │
│                                                                              │
│ 대안:                                                                         │
│   D1) SSL distillation loss weight를 낮게 (λ=0.1) 유지                         │
│   D2) MioCodec latent consistency loss를 SSL distillation과 함께 사용           │
│       → L = L_AE + 0.1·L_distill + 0.5·L_mio_consist                        │
│   D3) Phase 2 없이 Phase 0→Phase 3 직행 (SSL distillation 생략)               │
│                                                                              │
│ 권장: D2                                                                    │
│   - 두 distillation target 간 균형 유지                                        │
│   - MioCodec consistency loss로 teacher space anchor                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Risk 5: Training Data 부족                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 심각도: Medium                                                               │
│                                                                              │
│ 문제:                                                                         │
│   Phase 3에서 병렬 발화 데이터(동일 내용, 다른 화자) 필요.                       │
│   기존 btrvrc0는 partial parallel 데이터만 사용.                               │
│                                                                              │
│ 영향:                                                                         │
│   Converter + Vocoder joint training에 충분한 데이터 부족.                      │
│   Non-parallel 데이터로는 z_src → z_tgt 변환 학습이 어려움.                      │
│                                                                              │
│ 대안:                                                                         │
│   E1) VCTK (109 speakers, parallel) + LibriTTS (multi-speaker TTS)            │
│   E2) Cycle consistency: z_src → z_tgt → z_src' → L1(z_src, z_src')          │
│       → non-parallel 데이터로도 학습 가능                                      │
│   E3) Data augmentation: 동일 화자 pitch shift, formant shift                 │
│   E4) TTS로 parallel 데이터 합성                                                │
│                                                                              │
│ 권장: E1 + E2                                                               │
│   - VCTK + LibriTTS로 기본 parallel 데이터 확보                                │
│   - Cycle consistency loss로 non-parallel 데이터 활용                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11.2 리스크 종합 평가

```
Risk Matrix:
─────────────────────────────────────────────────────────
Risk                          │ 심각도 │ 발생확률 │ 대응
──────────────────────────────┼───────┼─────────┼──────
WavLM 도메인 불일치           │ Med   │ Med     │ A1+A3
RAF 배치 크기 의존성          │ Med   │ High    │ B1+B3
BWE Cascade Error             │ Med-Lo│ Med     │ C2+C4
MioCodec Latent 호환성         │ Low   │ Low     │ D2
Training Data 부족            │ Med   │ Med     │ E1+E2
──────────────────────────────┴───────┴─────────┴──────

완화 전략 우선순위:
  1. B1 (EMA RAF statistics) — 구현 비용 낮음, 효과 큼
  2. E1+E2 (parallel 데이터 + cycle consistency) — 데이터 확보
  3. A1 (WavLM downsampling) — 간단한 해결책
  4. C2 (BWE feature input) — 아키텍처 개선
```

### 11.3 Fallback 전략

```
RAF 도입 실패 시 Fallback:
═══════════════════════════════════════════════════════════

Fallback Tier 1: 기존 MPD/MRD + RAF 보조
  - RAF를 primary GAN loss 대신 auxiliary로 사용
  - L = L_recon + λ_mpd·L_MPD-MRD + 0.5·L_RAF + λ_fm·L_FM
  - RAF가 불안정하면 λ_raf=0으로 fallback
  - btrv3lite 검증済み loss 구조 유지

Fallback Tier 2: RAF 제거, ConvNeXt BigVGAN + MPD/MRD
  - ConvNeXt v2 backbone의 BigVGAN (14M) + 기존 MPD/MRD (5.2M)
  - RAF의 경량 보코더 이점은 유지, GAN 안정성은 기존 방식
  - WaveNeXt 2의 ConvNeXt 설계만 차용

Fallback Tier 3: 기존 Student v2 Decoder 사용
  - HybridCodec Encoder + CausalConvNeXt Converter + Student v2 Decoder
  - RAF/BWE 도입하지 않고 ConvNeXt 인코더+변환기만 업그레이드
  - btrv3lite 완전 호환, 검증된 품질
```

---

## 부록

### A. RAF Loss PyTorch 의사코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RAFLoss(nn.Module):
    """Relativistic Adversarial Feature Loss.
    
    SSL Teacher: WavLM Large (frozen)
    Multi-layer quality projection + relativistic pairing.
    """
    
    def __init__(self, ssl_teacher, layers=[6, 12, 18, 24], 
                 layer_weights=None, ema_momentum=0.99):
        super().__init__()
        self.teacher = ssl_teacher  # Frozen WavLM
        self.layers = layers
        self.layer_weights = layer_weights or [0.25]*len(layers)
        
        # Quality projection per layer
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(1024),  # WavLM hidden dim
                nn.Linear(1024, 1)
            ) for _ in layers
        ])
        
        # EMA of real quality statistics (for small batch)
        self.register_buffer('ema_q_real_mean', torch.zeros(1))
        self.register_buffer('ema_q_fake_mean', torch.zeros(1))
        self.ema_momentum = ema_momentum

    def _extract_features(self, audio):
        """Extract multi-layer WavLM features."""
        # audio: (B, T) @ 16kHz (downsampled from 24kHz)
        with torch.no_grad():
            # WavLM returns hidden states from all layers
            hidden_states = self.teacher(audio, output_hidden_states=True)
            features = [hidden_states[l] for l in self.layers]
            # Each: (B, T_ssl, 1024)
            # Temporal average pooling
            features = [f.mean(dim=1) for f in features]  # (B, 1024)
        return features

    def forward(self, fake_audio, real_audio):
        """Compute RAF loss.
        
        Args:
            fake_audio: Generated audio (B, T) @ 16kHz
            real_audio: Ground truth audio (B, T) @ 16kHz
        Returns:
            loss_raf_g: Generator RAF loss
            loss_fm: Feature matching loss
        """
        # Extract features
        fake_feats = self._extract_features(fake_audio)
        real_feats = self._extract_features(real_audio)
        
        loss_raf = 0.0
        loss_fm = 0.0
        
        for l_idx, (proj, fake_f, real_f, w) in enumerate(
            zip(self.projections, fake_feats, real_feats, self.layer_weights)
        ):
            # Quality scores
            q_fake = proj(fake_f)  # (B, 1)
            q_real = proj(real_f)  # (B, 1)
            
            # EMA statistics (for small batch stabilization)
            if self.training:
                with torch.no_grad():
                    self.ema_q_real_mean.mul_(self.ema_momentum).add_(
                        q_real.mean(), alpha=1-self.ema_momentum)
                    self.ema_q_fake_mean.mul_(self.ema_momentum).add_(
                        q_fake.mean(), alpha=1-self.ema_momentum)
                
                mean_q_real = self.ema_q_real_mean
                mean_q_fake = self.ema_q_fake_mean
            else:
                mean_q_real = q_real.mean()
                mean_q_fake = q_fake.mean()
            
            # Relativistic pairing
            D_fake = q_fake - mean_q_real  # fake가 real 평균보다 얼마나 못한가
            D_real = q_real - mean_q_fake  # real이 fake 평균보다 얼마나 나은가
            
            # Generator RAF loss
            loss_raf += w * (-F.logsigmoid(D_fake).mean())
            
            # Feature matching loss
            loss_fm += F.l1_loss(fake_f, real_f.detach())
        
        loss_raf = loss_raf / len(self.layers)
        loss_fm = loss_fm / len(self.layers)
        
        return loss_raf, loss_fm
```

### B. WaveNeXt 2 ConvNeXt Block (PyTorch)

```python
class ConvNeXtBlock1D(nn.Module):
    """WaveNeXt 2 style ConvNeXt v2 block for 1D audio.
    
    Features:
    - 7x7 depthwise conv (causal)
    - Inverted bottleneck (dim → 4*dim → dim)
    - GRN (Global Response Normalization)
    - LayerScale (zero-init for identity at start)
    - Optional AdaLN-Zero conditioning
    """
    
    def __init__(self, dim, kernel_size=7, dilation=1, 
                 cond_dim=None, mlp_ratio=4):
        super().__init__()
        self.dim = dim
        
        # Depthwise conv (causal)
        self.dwconv = CausalDepthwiseConv1d(dim, kernel_size, dilation)
        
        # Inverted bottleneck
        hidden_dim = dim * mlp_ratio
        self.pwconv1 = nn.Conv1d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.pwconv2 = nn.Conv1d(hidden_dim, dim, 1)
        
        # LayerScale (zero-init → identity)
        self.gamma = nn.Parameter(torch.zeros(dim, 1))
        
        # AdaLN-Zero conditioning (optional)
        if cond_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, dim * 2)  # scale + shift
            )
            self.cond_gate = nn.Parameter(torch.zeros(()))
        else:
            self.cond_mlp = None

    def forward(self, x, cond=None):
        # x: (B, dim, T)
        shortcut = x
        
        # AdaLN
        if self.cond_mlp is not None and cond is not None:
            cond_out = self.cond_mlp(cond)  # (B, dim*2)
            scale, shift = cond_out.chunk(2, dim=1)  # (B, dim) each
            x = x * (1 + scale.unsqueeze(-1) * self.cond_gate.tanh()) \
                + shift.unsqueeze(-1) * self.cond_gate.tanh()
        
        # ConvNeXt operations
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        
        # LayerScale + residual
        x = self.gamma * x
        x = shortcut + x
        
        return x


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt v2)."""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        # x: (B, dim, T)
        Gx = x.norm(p=2, dim=-1, keepdim=True)  # (B, dim, 1)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x
```

### C. 전체 Loss 구성 요약

```
Training Phase별 Loss 구성:

Phase 1 (AE Pretraining):
  L = L1(wave) + L_MR_STFT + L_MR_Complex_STFT + L_MR_Mel(λ=45) + L_latent
  (No GAN, no RAF)

Phase 2 (SSL Distillation):
  L = L_phase1 + 0.5 · (L1(z, z_ssl) + (1-cos_sim(z, z_ssl)))
  
Phase 3 (Converter + Vocoder + RAF):
  L = L_recon + λ_raf·L_RAF + λ_fm·L_FM + λ_mel·L_mel
     + 0.3·L_consistency + 0.1·L_speaker
  
  where:
    L_recon = L1(wave) + MR-STFT + MR-Complex-STFT
    L_RAF = Σ w_l · (-log σ(D_fake_l))
    L_FM = Σ v_l · ||real_features_l - fake_features_l||₁
    L_mel = MR-Mel (λ=45)
    L_consistency = L1(z_out, z_tgt) + (1-cos_sim(z_out, z_tgt))
    L_speaker = 1 - cos_sim(SpeakerEnc(wave_out), speaker_cond)

Phase 4 (End-to-End + BWE):
  L = L_phase3 + L_BWE
  where:
    L_BWE = L1(wave_44k, gt_44k) + MR-STFT(wave_44k, gt_44k)
          + 0.5·L_RAF_BWE(wave_44k, gt_44k)
  
  Lambdas: λ_raf=1.0, λ_fm=0.5, λ_mel=45
```

### D. 참고문헌

```
1. RAF: Relativistic Adversarial Features (arXiv:2603.11678)
   - SSL quality gap + relativistic pairing for GAN training
   - 14M BigVGAN beats 112M BigVGAN

2. WaveNeXt 2 (arXiv:2605.25506)
   - ConvNeXt GAN + Diffusion vocoder
   - CPU RTF 0.16, 7×7 depthwise conv backbone

3. USAD 2.0 (arXiv:2606.06444)
   - SSL + supervised distillation
   - Universal audio encoder, 25Hz

4. Vocos BWE (arXiv:2603.07285)
   - ConvNeXt bandwidth extension
   - RTF 0.0001, 24kHz → 48kHz

5. BigVGAN (arXiv:2206.04658)
   - Universal vocoder with Snake activations
   - HiFi-GAN 기반, periodic inductive bias

6. ConvNeXt v2 (arXiv:2301.00808)
   - GRN, LayerScale, fully convolutional design
   - ConvNet modernization for 2020s

7. MioCodec (btrv3lite internal)
   - 44.1kHz, 25Hz, 768-dim continuous latent
   - Causal student codec with Transformer decoder

8. P-Flow (arXiv:2305.07432)
   - Speaker prompt tokens for voice conversion
   - 4-token cross-attention conditioning

9. HiFi-GAN (arXiv:2010.05646)
   - MPD + MRD discriminator
   - Multi-scale mel loss (λ=45)

10. WavLM (arXiv:2110.13900)
    - Large-scale self-supervised speech model
    - Multi-layer features for downstream tasks
```

---

> **설계 완료일**: 2026-06-06
> **설계자**: btrv5 Architecture Team
> **다음 단계**: 
> 1. Phase 0 weight transfer 검증 스크립트 작성
> 2. RAF BigVGAN Vocoder 프로토타입 구현
> 3. WavLM feature caching 파이프라인 구축
> 4. MPS 호환성 검증 (batch=1, EMA RAF)

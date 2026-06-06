# FlowVC: Flow Matching 기반 실시간 고품질 음성 변환 파이프라인

> **btrv5 아키텍처 설계 — 각도 "FlowVC"**
> Continuous Latent Space + Conditional Flow Matching
> 44.1kHz 실시간 Voice Conversion

---

## 목차

1. [개요 및 설계 철학](#1-개요-및-설계-철학)
2. [논문 인사이트 통합](#2-논문-인사이트-통합)
3. [전체 아키텍처 다이어그램](#3-전체-아키텍처-다이어그램)
4. [컴포넌트 상세 설계](#4-컴포넌트-상세-설계)
   - 4.1 [F³-Encoder (Continuous AE 인코더)](#41-f-encoder-continuous-ae-인코더)
   - 4.2 [Speaker Encoder (화자 인코더)](#42-speaker-encoder-화자-인코더)
   - 4.3 [Prosody Extractor (운율 추출기)](#43-prosody-extractor-운율-추출기)
   - 4.4 [FlowVC Converter (조건부 Flow Matching 변환기)](#44-flowvc-converter-조건부-flow-matching-변환기)
   - 4.5 [F³-Decoder (Continuous AE 디코더)](#45-f-decoder-continuous-ae-디코더)
5. [Training Pipeline](#5-training-pipeline)
6. [실시간 추론 전략](#6-실시간-추론-전략)
7. [btrv3lite/btrvrc0 재활용 경로](#7-btrv3litebtrvrc0-재활용-경로)
8. [예상 파라미터 수 및 연산량](#8-예상-파라미터-수-및-연산량)
9. [성능 목표 및 평가 지표](#9-성능-목표-및-평가-지표)
10. [리스크 및 대안](#10-리스크-및-대안)

---

## 1. 개요 및 설계 철학

### 1.1 FlowVC란?

FlowVC는 **Continuous Latent Space**에서 **Conditional Flow Matching (CFM)** 을 통해 음성을 변환하는 차세대 Voice Conversion 파이프라인이다. 기존 btrv3lite의 결정론적(deterministic) residual 변환 방식에서 벗어나, 확률적 흐름(flow) 기반 변환을 도입하여 더 자연스럽고 다양한 음성 출력을 생성한다.

### 1.2 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **KL-free Continuous AE** | VQ(codebook collapse, commitment loss) 없이 순수 연속 잠재 공간 사용. F³-Tokenizer 방식의 noise regularization으로 정규화. |
| **Flow Matching 변환** | 소스→타겟 변환을 ODE 기반 flow로 모델링. 직선 경로(optimal transport) 가정으로 효율적 학습. |
| **ConvNeXt v2 백본** | 모든 인코더/디코더/변환기에서 ConvNeXt v2 블록 사용. 7×7 depthwise conv + GRN (Global Response Normalization) + inverted bottleneck MLP. |
| **실시간 스트리밍** | Causal 연산(좌측 패딩만 사용), 청크 단위 처리, 고정된 lookahead로 RTF < 0.3, TTFB < 150ms 목표. |
| **기존 자산 재활용** | btrv3lite의 MioCodec teacher, CausalLatentConverter, Student decoder, 스트리밍 인프라를 최대한 재활용. |

### 1.3 목표 사양

| 지표 | 목표값 | 비고 |
|------|--------|------|
| Sample Rate | 44,100 Hz | 24kHz 이상 충족 |
| Latent Framerate | 25 Hz | hop = 1764 samples |
| Real-Time Factor (RTF) | < 0.5 (GPU), < 0.8 (CPU) | 청크 40ms 기준 |
| TTFB (Time To First Byte) | < 150ms (GPU), < 200ms (CPU) | |
| Speaker Similarity | > 0.85 (ECAPA cosine) | |
| WER | < 3% (Whisper large-v3) | |
| MOS | > 4.0 (naturalness) | |
| 파라미터 수 (전체) | ~70M | 학습 가능 |

---

## 2. 논문 인사이트 통합

### 2.1 F³-Tokenizer (arXiv:2606.06357)

> **"KL-free Continuous AE + Noise Regularization + Flow Matching over Patches"**

| 인사이트 | FlowVC 적용 |
|----------|-------------|
| **KL-free**: VQ 제거, 연속 잠재 공간만 사용 | 인코더/디코더를 pure continuous AE로 설계. commitment loss, codebook collapse 문제에서 자유로움. |
| **Noise Regularization**: 인코딩된 latent에 Gaussian noise 주입 | 학습 중 `z_reg = z + σ·ε` (σ=0.01). 디코더가 latent perturbation에 강인해지고, flow matching에서 더 매끄러운 manifold 학습. |
| **Flow Matching over Patches**: latent patch 분포를 flow로 모델링 | FlowVC 변환기에서 latent의 시간 패치(4프레임) 단위로 flow matching 적용. 지역적 연속성 보장. |
| **ConvNeXt v2 백본** | 모든 컴포넌트에서 ConvNeXt v2 블록을 기본 빌딩 블록으로 사용. |

### 2.2 FMelCodec (arXiv:2605.25669)

> **"OC-VQ + Conditional Flow Matching Refiner, 250bps"**

| 인사이트 | FlowVC 적용 |
|----------|-------------|
| **CFM Refiner**: 양자화된 latent를 flow로 정제 | FlowVC 변환기가 **refiner 역할** — 소스 latent에서 시작하여 타겟 분포로 flow. 시작점이 이미 meaningful하므로 적은 스텝(4~8)으로 수렴. |
| **Optimal Transport (OT) 경로**: CFM에서 직선 경로 사용 | 학습 시 `z_t = (1-t)·z_src + t·z_tgt`, 목표 벡터장 `v = z_tgt - z_src`. 단순하고 안정적인 학습. |
| **Few-step inference** | 4-step Euler 또는 8-step RK4로 충분한 품질. RTF 최적화. |
| **경량 조건부 설계** | AdaLN-Zero + cross-attention으로 조건을 효율적으로 주입. |

### 2.3 SURF (arXiv:2606.04921)

> **"Unsupervised Source Separation via Flow Matching"**

| 인사이트 | FlowVC 적용 |
|----------|-------------|
| **Flow로 소스 분리**: 혼합 신호에서 개별 소스 복원 | 화자 정체성(speaker identity)과 언어 내용(content)을 flow의 조건부 분리로 모델링. 서로 다른 조건으로 다른 "소스"를 생성. |
| **비지도 학습 가능성** | 병렬 데이터 없이도 flow matching의 조건부 생성 능력으로 VC 가능성 열림. |
| **Score-based prior** | Speaker encoder 출력을 flow의 prior 조건으로 활용. |

---

## 3. 전체 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FLOWVC ARCHITECTURE                             │
│                     Continuous Latent + Flow Matching VC                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────┐          ┌──────────────────────┐                  │
│  │   SOURCE AUDIO        │          │  REFERENCE AUDIO     │                  │
│  │   (44.1kHz mono)      │          │  (1~3 sec, 44.1kHz) │                  │
│  └──────────┬───────────┘          └──────────┬───────────┘                  │
│             │                                 │                              │
│             ▼                                 ▼                              │
│  ┌──────────────────────┐          ┌──────────────────────┐                  │
│  │  F³-ENCODER           │          │  SPEAKER ENCODER      │                  │
│  │  (ConvNeXt-v2)        │          │  (ConvNeXt-v2)        │                  │
│  │  · 6 stages           │          │  · 6 stages           │                  │
│  │  · strides: 2,2,3,3,7,7│        │  · strides: 2,2,3,3,7,7│               │
│  │  · channels: →768     │          │  · channels: →192     │                  │
│  │  · Noise reg σ=0.01   │          │  · Global Attn Pool   │                  │
│  │  · KL-free, no VQ     │          │  · Speaker Prompt Tok │                  │
│  └──────────┬───────────┘          └──────────┬───────────┘                  │
│             │                                 │                              │
│             │  z_src (T_lat, 768)             │  spk_emb (192)               │
│             │  @ 25Hz                         │  prompt (4, 192)             │
│             │                                 │                              │
│             ├─────────────────────────────────┤                              │
│             │                                 │                              │
│  ┌──────────┴───────────┐                    │                              │
│  │  PROSODY EXTRACTOR    │                    │                              │
│  │  · Lightweight Conv   │                    │                              │
│  │  · log-F0, voiced,    │                    │                              │
│  │    log-energy         │                    │                              │
│  │  · Output: (T_lat, 3) │                    │                              │
│  └──────────┬───────────┘                    │                              │
│             │                                 │                              │
│             │  prosody (T_lat, 3)             │                              │
│             │                                 │                              │
│             ▼                                 ▼                              │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │                    FLOWVC CONVERTER                               │        │
│  │              (Conditional Flow Matching)                          │        │
│  │                                                                   │        │
│  │   Input:  z_src (T_lat, 768), t ∈ [0,1], cond = [spk+prosody]    │        │
│  │                                                                   │        │
│  │   ┌─────────────────────────────────────────────────────────┐    │        │
│  │   │  Vector Field Network v_θ(z_t, t, c)                    │    │        │
│  │   │                                                         │    │        │
│  │   │  z_t ──→ [TimeEmbed(t)] ──┐                             │    │        │
│  │   │                           ├──→ [ConvNeXt-v2 ×12] ──→ v  │    │        │
│  │   │  cond ──→ [CondProj] ────┘    (AdaLN-Zero + X-Attn)     │    │        │
│  │   │                                                         │    │        │
│  │   │  · dim=512, kernel=7, dilations cyclic [1,2,4,8]       │    │        │
│  │   │  · Cross-attn to speaker prompt @ blocks [3,6,9]       │    │        │
│  │   │  · AdaLN-Zero: zero-init → identity at t=0              │    │        │
│  │   └─────────────────────────────────────────────────────────┘    │        │
│  │                                                                   │        │
│  │   ODE Solve:  z_tgt = z_src + ∫₀¹ v_θ(z_τ, τ, c) dτ            │        │
│  │              (4~8 Euler / RK4 steps)                             │        │
│  └──────────────────────────────┬───────────────────────────────────┘        │
│                                 │                                            │
│                                 │  z_tgt (T_lat, 768)                        │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │  F³-DECODER                                                        │        │
│  │  (ConvNeXt-v2 + HiFi-GAN upsampler)                               │        │
│  │                                                                   │        │
│  │  · 6 stages (reverse strides)                                     │        │
│  │  · TransposedConv upsampling: ×7, ×7, ×3, ×3, ×2, ×2 = ×1764    │        │
│  │  · MRF (Multi-Receptive Field) blocks per stage                   │        │
│  │  · Snakeβ activation                                              │        │
│  │  · Speaker conditioning via FiLM                                   │        │
│  └──────────────────────────────┬───────────────────────────────────┘        │
│                                 │                                            │
│                                 ▼                                            │
│                     ┌──────────────────────┐                                  │
│                     │   TARGET AUDIO        │                                  │
│                     │   (44.1kHz mono)      │                                  │
│                     └──────────────────────┘                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

FIGURE 1: FlowVC 전체 파이프라인 — 소스 음성이 F³-Encoder를 통해 연속 latent로
인코딩되고, FlowVC Converter가 Conditional Flow Matching으로 타겟 화자 분포로
변환한 후, F³-Decoder가 파형으로 복원한다.
```

### 3.1 데이터 흐름 요약

```
Audio_src (44100 Hz, 16-bit)
    │
    ├──→ F³-Encoder ──→ z_src (T_lat, 768) @ 25Hz
    │
    ├──→ ProsodyExtractor ──→ prosody (T_lat, 3): [log_f0, voiced, log_energy]
    │
    └──→ SpeakerEncoder(Audio_ref) ──→ spk_emb (192), prompt_tokens (4, 192)
                │
                ▼
         Condition c = [spk_emb_per_frame ‖ prosody] → CondProj → (T_lat, 256)
                │
                ▼
         FlowVC Converter: z_src ──[4-step ODE]──→ z_tgt (T_lat, 768)
                │
                ▼
         F³-Decoder(z_tgt, spk_emb) ──→ Audio_tgt (44100 Hz)

Latent 차원: 768 (MioCodec 호환, btrv3lite weight transfer 가능)
```

---

## 4. 컴포넌트 상세 설계

### 4.1 F³-Encoder (Continuous AE 인코더)

F³-Tokenizer에서 영감을 받은 KL-free 연속 AutoEncoder의 인코더.

```
F³-Encoder Architecture
════════════════════════════════════════════════════════

Input: waveform (B, 1, T_audio) @ 44.1kHz

Stage 0: Conv1d(1→32, k=7, s=2, causal pad)  → (B, 32, T/2)
         ConvNeXt-v2 Block ×2 (dim=32, k=7)

Stage 1: Conv1d(32→64, k=7, s=2, causal pad)  → (B, 64, T/4)
         ConvNeXt-v2 Block ×2 (dim=64, k=7)

Stage 2: Conv1d(64→128, k=9, s=3, causal pad) → (B, 128, T/12)
         ConvNeXt-v2 Block ×2 (dim=128, k=7)

Stage 3: Conv1d(128→256, k=9, s=3, causal pad) → (B, 256, T/36)
         ConvNeXt-v2 Block ×2 (dim=256, k=7)

Stage 4: Conv1d(256→512, k=15, s=7, causal pad) → (B, 512, T/252)
         ConvNeXt-v2 Block ×2 (dim=512, k=7)

Stage 5: Conv1d(512→768, k=15, s=7, causal pad) → (B, 768, T/1764)
         ConvNeXt-v2 Block ×2 (dim=768, k=7)
         LayerNorm → z_raw (B, 768, T_lat)

Output: z_raw (B, T_lat, 768)  ← KL-free, no quantization

Training-only: z_reg = z_raw + σ · ε,  ε ~ N(0,I), σ=0.01
```

**ConvNeXt-v2 Block 상세:**
```
ConvNeXtV2Block(dim, kernel_size=7):
    Input: x (B, dim, T)
    
    h = DepthwiseConv1d(dim, k=kernel_size, groups=dim)  # causal (left-pad)
    h = LayerNorm(h)  # over channels
    h = Linear(dim → 4*dim)  # inverted bottleneck
    h = GELU()
    h = GRN(4*dim)  # Global Response Normalization (ConvNeXt v2 핵심)
    h = Linear(4*dim → dim)
    h = DropPath(h)  # stochastic depth
    return x + h
```

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| 입력 | (B, 1, T) waveform | 44.1kHz mono |
| 출력 | (B, T_lat, 768) | T_lat = ceil(T/1764) |
| Strides | [2, 2, 3, 3, 7, 7] | 총 downsampling = 1764 |
| Channels | [32, 64, 128, 256, 512, 768] | stage별 채널 |
| Blocks per stage | 2 | ConvNeXt-v2 × 2 |
| Kernel size | 7 (stage 내부), stride conv는 가변 | |
| Noise σ | 0.01 (고정) | F³-Tokenizer 방식 |
| VQ 여부 | 없음 (KL-free) | continuous latent |
| Causal | 예 (left-only padding) | 스트리밍 지원 |
| 예상 파라미터 | ~15.2M | |

### 4.2 Speaker Encoder (화자 인코더)

레퍼런스 음성(1~3초)에서 화자 identity를 추출.

```
Speaker Encoder Architecture
════════════════════════════════════════════════════════

Input: reference waveform (B, 1, T_ref)

Stage 0: Conv1d(1→32, k=7, s=2)  → (B, 32, T/2)
         ConvNeXt-v2 Block ×2
Stage 1: Conv1d(32→48, k=7, s=2) → (B, 48, T/4)
         ConvNeXt-v2 Block ×2
Stage 2: Conv1d(48→64, k=9, s=3) → (B, 64, T/12)
         ConvNeXt-v2 Block ×2
Stage 3: Conv1d(64→96, k=9, s=3) → (B, 96, T/36)
         ConvNeXt-v2 Block ×2
Stage 4: Conv1d(96→128, k=15, s=7) → (B, 128, T/252)
         ConvNeXt-v2 Block ×2
Stage 5: Conv1d(128→192, k=15, s=7) → (B, 192, T/1764)
         ConvNeXt-v2 Block ×2

Global Pooling: Multi-Head Attention Pooling (8 heads)
    Query: learnable [1, 192]
    Key/Value: output of Stage 5
    → spk_emb (B, 192)

Speaker Prompt Tokens (P-Flow 스타일):
    Linear(192 → 4×192) + reshape → (B, 4, 192)
    Token MLP (residual) → prompt_tokens (B, 4, 192)
```

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| 출력 spk_emb | (B, 192) | btrv3lite의 128-dim에서 확장 |
| Prompt tokens | (B, 4, 192) | P-Flow 스타일, FlowVC 변환기에서 cross-attention |
| Pooling | Multi-Head Attention Pooling | learnable query |
| 예상 파라미터 | ~8.5M | |

**btrv3lite 재활용**: 기존 128-dim → 192-dim으로 확장. MioCodec global encoder의 구조 재사용 가능.

### 4.3 Prosody Extractor (운율 추출기)

소스 음성의 prosody 정보를 프레임 단위로 추출.

```
Prosody Extractor Architecture
════════════════════════════════════════════════════════

Input: waveform (B, 1, T_audio)

Lightweight ConvNet:
    Conv1d(1→32, k=15, s=1, causal) → GELU
    Conv1d(32→64, k=15, s=1, causal) → GELU
    Conv1d(64→128, k=15, s=1, causal) → GELU
    Conv1d(128→3, k=15, s=1, causal)

AdaptiveAvgPool1d → (B, 3, T_lat)  # hop=1764

Output channels:
    [0]: log_f0   (log-scale fundamental frequency)
    [1]: voiced   (voicing probability, sigmoid)
    [2]: log_energy (log-scale RMS energy)

Output: (B, T_lat, 3)
```

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| 출력 | (B, T_lat, 3) | log_f0, voiced, log_energy |
| 예상 파라미터 | ~0.3M | 매우 경량 |
| 재활용 | btrv3lite의 F0 추출기 (PENN 기반) 대체 가능 | |

### 4.4 FlowVC Converter (조건부 Flow Matching 변환기)

FlowVC의 핵심 — Conditional Flow Matching으로 latent를 변환.

#### 4.4.1 Vector Field Network (v_θ)

```
Vector Field Network v_θ(z_t, t, c)
═══════════════════════════════════════════════════════════════

Inputs:
    z_t   : (B, T_lat, 768)  — 현재 latent 상태 (flow 중간 지점)
    t     : (B, 1)           — 연속 시간 ∈ [0,1]
    c     : (B, T_lat, 256)  — 조건 벡터 (화자+운율)

Processing:

1. Time Embedding:
    t_emb = SinusoidalEmbedding(t, dim=256)
    t_emb = Linear(256→256) → SiLU → Linear(256→256)

2. Input Projection:
    h = Linear(768→512)(z_t)                              # (B, T_lat, 512)

3. ConvNeXt-v2 Blocks ×12 (dim=512, kernel_size=7):
    
    dilations = [1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8]
    
    for i, d in enumerate(dilations):
        │
        ├── DepthwiseConv1d(512, k=7, dilation=d, causal)
        ├── LayerNorm
        ├── AdaLN-Zero(condition = c + t_emb_proj):
        │       scale, shift, gate = Linear(512+256 → 3×512)
        │       h = h * (1+scale) + shift
        ├── ConvNeXt MLP (512 → 2048 → 512) + GRN
        ├── gate 적용 (zero-init)
        │
        └── if i+1 in [3, 6, 9]:
                Cross-Attention to Speaker Prompt Tokens
                    Q = h (B, T, 512)
                    K,V = prompt_tokens (B, 4, 192) → Linear(192→512)
                    h = h + gate * MHA(Q, K, V)

4. Output Projection:
    v = Linear(512→768)(h)                               # (B, T_lat, 768)
    v = v * output_gate (zero-init)                       # identity at t=0

Output: velocity field v (B, T_lat, 768)
```

**AdaLN-Zero 상세:**
```
AdaLN-Zero(x, condition):
    # condition = concat[c_proj_frame, t_emb]
    # c_proj_frame: Linear(256→256)(c_per_frame)
    # combined: (B, T, 256+256) = (B, T, 512)
    
    params = Linear(512→3×512)(combined)  # zero-init weight, zero bias
    scale, shift, gate = params.chunk(3, dim=-1)
    
    # x: (B, T, 512)  — already LayerNorm'd
    x = x * (1.0 + scale) + shift
    # ... MLP ...
    x = x * gate  # gate zeros at init → block is identity
    return x
```

#### 4.4.2 Conditional Flow Matching 수식

**학습 (Training):**
```
주어진 것:
    z_src : 소스 latent (source speaker)
    z_tgt : 타겟 latent (target speaker, same utterance)
    c     : 조건 (타겟 speaker embedding + prosody)
    σ_min : 최소 noise level = 0.001 (안정성)

1. 시간 샘플링:
    t ~ U[0, 1]

2. 확률 경로 (Optimal Transport - 직선):
    z_t = (1 - t) · z_src + t · z_tgt + σ_min · ε
    where ε ~ N(0, I)

3. 목표 벡터장 (직선 경로의 속도):
    v_target = z_tgt - z_src

4. CFM Loss:
    L_cfm = MSE(v_θ(z_t, t, c), v_target)

    직관: 네트워크는 z_t에서 z_tgt 방향으로의 "속도"를 예측하도록 학습.
          t=0 → z_t≈z_src 근처, t=1 → z_t≈z_tgt 근처.
```

**추론 (Inference):**
```
주어진 것:
    z_src : 소스 latent
    c     : 조건 (타겟 speaker + prosody)
    N     : ODE step 수 (4~8)

1. 초기화:
    z = z_src.clone()

2. Euler Method (N steps):
    dt = 1.0 / N
    for i in range(N):
        t = i * dt
        v = v_θ(z, t, c)
        z = z + v * dt

    # 또는 RK4 (더 높은 품질, 2배 비용):
    for i in range(N):
        t = i * dt
        k1 = v_θ(z, t, c)
        k2 = v_θ(z + k1*dt/2, t+dt/2, c)
        k3 = v_θ(z + k2*dt/2, t+dt/2, c)
        k4 = v_θ(z + k3*dt, t+dt, c)
        z = z + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

3. 결과:
    z_tgt = z
```

#### 4.4.3 ConvNeXt-v2 Block (FlowVC Converter용)

```
FlowVCConvNeXtV2Block:
    Input: h (B, T, dim=512), t_emb (B, 1, 256), c_proj (B, T, 256)
    
    # Depthwise conv (causal)
    h_conv = CausalDepthwiseConv1d(h, k=7, dilation=d)
    
    # LayerNorm over channels
    h_norm = LayerNorm(h_conv)
    
    # Combine condition: c_proj per-frame + t_emb broadcast
    cond_combined = Concat[c_proj, t_emb.expand(B, T, 256)]  # (B, T, 512)
    
    # AdaLN-Zero
    adaln_params = Linear(512→3*512)(cond_combined)  # zero-init
    scale, shift, gate = adaln_params.chunk(3, dim=-1)
    h_mod = h_norm * (1.0 + scale) + shift
    
    # Inverted bottleneck MLP + GRN
    h_mlp = Linear(512→2048)(h_mod)
    h_mlp = GELU(h_mlp)
    h_mlp = GRN(2048)(h_mlp)
    h_mlp = Linear(2048→512)(h_mlp)
    
    # DropPath (stochastic depth, rate=0.1)
    h_mlp = DropPath(h_mlp)
    
    # Residual with gate
    output = h + h_mlp * gate
    
    return output
```

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| Hidden dim | 512 | ConvNeXt 블록의 working dimension |
| Num blocks | 12 | v1:8 → v2:10 → FlowVC:12 |
| Kernel size | 7 | depthwise conv |
| Dilations | [1,2,4,8] × 3 | cyclic pattern |
| Time embedding dim | 256 | sinusoidal → MLP |
| Condition dim | 256 | speaker(192)+prosody(3) → proj |
| Cross-attn heads | 4 | at blocks [3, 6, 9] |
| Cross-attn dim | 512, head_dim=128 | |
| AdaLN dim | 512 → 3×512 | zero-init |
| MLP expansion | ×4 | 512 → 2048 → 512 |
| GRN | yes | ConvNeXt v2 핵심 |
| DropPath rate | 0.1 | |
| 예상 파라미터 | ~28.3M | |

#### 4.4.4 ODE Solver 선택

| Solver | Steps | 품질 | 속도 (per frame) | 권장 |
|--------|-------|------|------------------|------|
| Euler | 4 | Good | ~3ms | 실시간 기본 |
| Euler | 8 | Better | ~6ms | 품질 우선 |
| RK4 | 4 | Very Good | ~6ms | 균형 |
| Dopri5 (adaptive) | ~12 avg | Best | ~15ms | 오프라인 |

**FlowVC 실시간 설정**: 4-step Euler 사용. z_src가 이미 meaningful한 latent이므로 소수 스텝으로 충분.

### 4.5 F³-Decoder (Continuous AE 디코더)

F³-Encoder의 역연산. HiFi-GAN 스타일의 고품질 업샘플링.

```
F³-Decoder Architecture
══════════════════════════════════════════════════════════

Input: z (B, T_lat, 768), spk_cond (B, 192)

1. Input Projection + Speaker FiLM:
    h = Linear(768→768)(z)
    FiLM(spk_cond): h = h * γ + β  (γ,β from Linear(192→2×768))
    
2. ConvNeXt-v2 Blocks (pre-upsampling):
    ConvNeXtV2Block ×4 (dim=768, k=7)  # latent refinement

3. Upsampling Stages (역방향 strides):

Stage 0: TransposedConv1d(768→512, k=15, s=7)  → (B, 512, T*7)
         MRF Block ×2 (dim=512)  # Multi-Receptive Field
         FiLM(spk_cond)

Stage 1: TransposedConv1d(512→256, k=15, s=7)  → (B, 256, T*49)
         MRF Block ×2 (dim=256)
         FiLM(spk_cond)

Stage 2: TransposedConv1d(256→128, k=9, s=3)   → (B, 128, T*147)
         MRF Block ×2 (dim=128)
         FiLM(spk_cond)

Stage 3: TransposedConv1d(128→64, k=9, s=3)    → (B, 64, T*441)
         MRF Block ×2 (dim=64)
         FiLM(spk_cond)

Stage 4: TransposedConv1d(64→32, k=7, s=2)    → (B, 32, T*882)
         MRF Block ×2 (dim=32)
         FiLM(spk_cond)

Stage 5: TransposedConv1d(32→16, k=7, s=2)    → (B, 16, T*1764)
         MRF Block ×2 (dim=16)

4. Final Projection:
    Conv1d(16→1, k=7, s=1) → tanh → waveform (B, 1, T*1764)

Output: waveform (B, 1, T_audio) @ 44.1kHz
```

**MRF (Multi-Receptive Field) Block:**
```
MRFBlock(dim, kernel_sizes=[3, 7, 11], dilations=[[1,3,5], [1,3,5], [1,3,5]]):
    Input: x (B, dim, T)
    
    residuals = []
    for ks, dils in zip(kernel_sizes, dilations):
        h = x
        for d in dils:
            h = CausalConv1d(dim, ks, dilation=d) → LeakyReLU
        residuals.append(h)
    
    h = sum(residuals) / len(residuals)
    h = LeakyReLU(h)
    return x + h  # residual connection
```

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| 입력 | (B, T_lat, 768) | F³-Encoder 출력과 동일 |
| 출력 | (B, 1, T_audio) | 44.1kHz waveform |
| Pre-refinement blocks | 4 | ConvNeXt-v2 |
| MRF blocks per stage | 2 | HiFi-GAN 스타일 |
| MRF kernel sizes | [3, 7, 11] | multi-scale |
| MRF dilations | [1,3,5] per branch | |
| Snakeβ activation | yes (MRF 내부) | periodic 신호에 유리 |
| Speaker conditioning | FiLM (each stage) | |
| Causal | yes (left-only padding) | 스트리밍 지원 |
| 예상 파라미터 | ~18.5M | |

---

## 5. Training Pipeline

### 5.1 Phase 0: 데이터 준비

```
데이터셋:
    - VCTK (109 speakers, ~44h)
    - LibriTTS (multi-speaker, ~585h)
    - 내부 한국어 데이터셋 (있는 경우)

전처리:
    - 44.1kHz resampling
    - Mono 변환
    - RMS normalization (-24dB target)
    - 3~10초 청크로 분할
    - Speaker label 태깅

Train/Val/Test split:
    - Train: 80% speakers
    - Val: 10% speakers
    - Test: 10% speakers (unseen)
```

### 5.2 Phase 1: F³-Codec 사전학습 (Encoder + Decoder)

**목적**: KL-free continuous AE 학습. 재구성 품질 확보.

```
설정:
    모델: F³-Encoder + F³-Decoder (no converter)
    입력/타겟: 동일 waveform
    손실 함수:
        L_total = L_recon + λ_adv · L_adv + λ_fm · L_fm + λ_feat · L_feat

    L_recon (재구성):
        - L1 loss: |wav - wav_recon|₁  (weight: 1.0)
        - Multi-Resolution STFT loss:
            L_stft = Σ |STFT(wav) - STFT(wav_recon)|₁ / |STFT(wav)|₁
                    + Σ |log(STFT(wav)) - log(STFT(wav_recon))|₁
          (FFT sizes: [512, 1024, 2048], hop: [128, 256, 512])

    L_adv (Adversarial):
        - LSGAN loss on Multi-Period Discriminator (MPD)
          + Multi-Resolution Discriminator (MRD)
        - Generator loss: L_adv_G = Σ (D_i(G(z)) - 1)²
        - Discriminator loss: L_adv_D = Σ (D_i(x) - 1)² + (D_i(G(z)))²

    L_fm (Feature Matching):
        - Σ_i ||D_i^l(x) - D_i^l(G(z))||₁  (discriminator 중간층)
        - weight: 2.0

    L_feat (Latent Feature):
        - L1 between intermediate encoder features of x and G(z)
        - (perceptual consistency, F³-style)
        - weight: 0.5

    Noise Regularization (F³-Tokenizer):
        - Training 시 encoder 출력에 noise 추가:
          z_reg = z_raw + σ · ε, ε ~ N(0,I), σ = 0.01
        - Decoder는 z_reg로부터 reconstruction

학습 설정:
    - Optimizer: AdamW (β₁=0.8, β₂=0.99)
    - Learning rate: G=2e-4, D=2e-4
    - LR schedule: Cosine decay to 1e-6 over 500k steps
    - Batch size: 16 (per GPU) × 1 GPU (MPS/A100)
    - Sequence length: 3초 = 132,300 samples = 75 latent frames
    - Total steps: 500,000
    - Discriminator starts at step 10,000

    Warmstart (btrv3lite 재활용):
        - MioCodec teacher의 encoder/decoder weight로 초기화
        - 또는 btrv3lite student decoder weight 활용
        - 차원 불일치 시 partial init

체크포인트:
    - Every 10,000 steps
    - Best model selection: validation L_recon + M-STFT
```

### 5.3 Phase 2: Speaker Encoder 학습

**목적**: 화자 identity 추출.

```
방식 A: ECAPA-TDNN / WavLM large pretrained 사용 (권장)
    - 사전학습된 speaker verification 모델 활용
    - Freeze, gradient 없음
    - 출력 차원: 192 (WavLM) 또는 512 (ECAPA, proj to 192)

방식 B: 자체 학습 (필요시)
    - Speaker classification head + ArcFace loss
    - AAM-Softmax (m=0.2, s=30)
    - VCTK + LibriTTS speaker labels
    - Batch size: 64
    - LR: 1e-4, 200k steps
```

### 5.4 Phase 3: FlowVC Converter 학습 (CFM)

**목적**: Conditional Flow Matching으로 latent 변환 학습.

```
설정:
    모델: FlowVC Converter (vector field network v_θ)
    Frozen: F³-Encoder, F³-Decoder, Speaker Encoder
    학습 가능: FlowVC Converter 전체

데이터 구성:
    - 동일 발화, 다른 화자 쌍 (parallel data)
    - 또는 동일 내용 다른 화자 (multi-speaker TTS 데이터)
    - z_src = F³-Encoder(audio_src)
    - z_tgt = F³-Encoder(audio_tgt)
    - c = SpeakerEncoder(audio_tgt_ref) + ProsodyExtractor(audio_src)

손실 함수:
    L_cfm = MSE(v_θ(z_t, t, c), z_tgt - z_src)

    추가 보조 손실:
    - Latent Consistency Loss (SURF-inspired):
        z_tgt_pred = z_src + v_θ(z_t, t, c)  (1-step prediction)
        L_consist = L1(z_tgt_pred, z_tgt)  (weight: 0.2)
    
    - Speaker Consistency Loss:
        spk_pred = SpeakerEncoder(F³-Decoder(z_tgt_pred))
        L_spk = 1 - cos_sim(spk_pred, spk_gt)  (weight: 0.1)

    - Prosody Preservation Loss:
        pros_pred = ProsodyExtractor(F³-Decoder(z_tgt_pred))
        L_pros = L1(pros_pred, pros_src)  (weight: 0.05)

    Total: L = L_cfm + 0.2·L_consist + 0.1·L_spk + 0.05·L_pros

학습 설정:
    - Optimizer: AdamW (β₁=0.9, β₂=0.999)
    - Learning rate: 1e-4 (warmup 5k steps → cosine decay to 1e-6)
    - Batch size: 8 × 1 GPU
    - Sequence length: 75 latent frames (3초)
    - t sampling: U[0, 1] (uniform)
    - σ_min: 0.001
    - Total steps: 200,000
    - Gradient clipping: max_norm=1.0

    Mixed precision: FP16 (AMP) 사용 가능

ODE Step Curriculum:
    - Steps 0~50k: 1-step prediction 위주 (t ~ U[0.9, 1.0] bias)
    - Steps 50k~200k: uniform t sampling
```

### 5.5 Phase 4: End-to-End Fine-tuning

**목적**: Decoder를 unfreeze하여 전체 파이프라인 최적화.

```
설정:
    모델: Encoder + FlowVC + Decoder (전체)
    Frozen: Speaker Encoder
    학습 가능: Decoder + FlowVC (Encoder는 낮은 LR)

손실 함수:
    L_total = L_recon + L_adv + L_fm + L_cfm
    (Phase 1과 Phase 3의 손실 결합)

학습 설정:
    - G Learning rate: 5e-5 (Encoder), 2e-5 (Decoder), 2e-5 (FlowVC)
    - D Learning rate: 5e-5
    - Batch size: 4
    - Sequence: 3초
    - Total steps: 50,000

Discriminator:
    - MPD + MRD (Phase 1과 동일)
    - Discriminator는 step 5,000부터 활성화
```

### 5.6 학습 인프라

```
하드웨어:
    - Apple Silicon (MPS): batch=1, ~2.0s/step (Phase 1), ~1.5s/step (Phase 3)
    - NVIDIA A100 40GB: batch=16, ~0.3s/step (Phase 1), ~0.2s/step (Phase 3)

MPS 최적화:
    - batch=1로 Phase 1 학습 가능 (btrv3lite 검증済み)
    - Gradient accumulation으로 유효 batch size 증가
    - FP32 권장 (MPS FP16 불안정)

모니터링:
    - TensorBoard / Wandb logging
    - Validation every 5,000 steps
    - Audio sample generation every 10,000 steps
```

---

## 6. 실시간 추론 전략

### 6.1 스트리밍 아키텍처

```
Streaming FlowVC Inference
══════════════════════════════════════════════════════════

Buffer 구조:
    
    [ left_context | chunk | right_lookahead ]
    ├── 8 frames ──┼─ 1 frame ─┼── 2 frames ──┤
      (320ms)        (40ms)       (80ms)

    Total window: 11 frames = 440ms = 19,404 samples

Pipeline per chunk:

    1. Audio Input:
       chunk_raw: 1764 samples (40ms, 1 latent frame)

    2. Window Assembly:
       window = [history(8 frames) | chunk(1 frame) | lookahead(2 frames)]
       (11 frames = 19,404 samples)

    3. F³-Encoder (causal window mode):
       z_window = Encoder(window)  → (11, 768)

    4. Latent Slice:
       z_frame = z_window[8:9, :]  → (1, 768)
       (left_context + chunk만큼, lookahead는 제외)

    5. Prosody Extraction:
       prosody_window = ProsodyExtractor(window)  → (11, 3)
       prosody_frame = prosody_window[8:9, :]    → (1, 3)

    6. Condition Assembly:
       spk_emb: (192,) — utterance-level, 한 번만 계산
       spk_prompt: (4, 192)
       cond = Concat[spk_emb ‖ prosody_frame]  → (1, 259)

    7. FlowVC Converter (frame-by-frame ODE):
       # 이전 프레임의 최종 latent를 context로 사용 (연속성)
       z_prev = last_output_frame  # (1, 768)
       z_t = z_frame.clone()
       
       # 4-step Euler
       for i in range(4):
           t = i / 4.0
           # 현재 프레임 + 이전 프레임 context (2 frames)
           z_input = Concat[z_prev, z_t]  along time → (2, 768)
           cond_input = Concat[cond_prev, cond]
           v = FlowVC([z_prev, z_t], t, [cond_prev, cond])
           z_t = z_t + v[-1:] * 0.25  # 마지막 프레임만 업데이트
       
       z_out = z_t  # (1, 768)

    8. F³-Decoder (causal):
       decoder_input = Concat[decoder_history, z_out]  # (9, 768)
       waveform_chunk = Decoder(decoder_input)[-1764:]  → (1764,)

    9. Output:
       emit waveform_chunk
       shift buffers: history ← history[1:] + z_out
```

### 6.2 TTFB (Time To First Byte) 분석

```
TTFB Breakdown (GPU, A100 기준):

    Component              │ Latency (ms) │ 비고
    ───────────────────────┼─────────────┼─────────────────────
    Audio input buffering  │     40      │ 1 frame = 1764 samples @ 44.1kHz
    Lookahead buffering    │     80      │ 2 frames (최초 1회)
    Encoder (11 frames)    │     15      │ ConvNeXt-v2 × 12 stages
    Prosody                │      3      │ 경량 ConvNet
    FlowVC (4 ODE steps)   │     12      │ 4 × 3ms per step
    Decoder (1 frame)      │      5      │ ConvNeXt-v2 + MRF
    ───────────────────────┼─────────────┼─────────────────────
    Total TTFB             │   ~155      │ (GPU)
    
    CPU (M2 Max 기준):
    ───────────────────────┼─────────────┼─────────────────────
    Encoder                │     40      │
    FlowVC                 │     35      │
    Decoder                │     15      │
    Total TTFB             │   ~210      │ (CPU, 최적화 전)
```

### 6.3 RTF (Real-Time Factor) 분석

```
정상 상태(per frame, 40ms audio):

    GPU (A100):
    ┌──────────────┬────────┬──────┐
    │ Component    │ ms     │ %    │
    ├──────────────┼────────┼──────┤
    │ Encoder      │  1.5   │  15% │
    │ Prosody      │  0.3   │   3% │
    │ FlowVC (4st) │  3.0   │  30% │
    │ Decoder      │  2.0   │  20% │
    │ Overhead     │  1.0   │  10% │
    ├──────────────┼────────┼──────┤
    │ Total        │  7.8   │  78% │
    └──────────────┴────────┴──────┘
    RTF = 7.8ms / 40ms = 0.195  ✓ (< 0.5 목표)

    CPU (M2 Max, CoreML 최적화):
    ┌──────────────┬────────┬──────┐
    │ Encoder      │  5.0   │  20% │
    │ Prosody      │  1.0   │   4% │
    │ FlowVC (4st) │ 10.0   │  40% │
    │ Decoder      │  6.0   │  24% │
    │ Overhead     │  3.0   │  12% │
    ├──────────────┼────────┼──────┤
    │ Total        │ 25.0   │ 100% │
    └──────────────┴────────┴──────┘
    RTF = 25ms / 40ms = 0.625  ✓ (< 0.8 목표)
```

### 6.4 최적화 전략

```
CPU/MPS 최적화:
    1. CoreML 변환 (Apple Silicon): Encoder + Decoder + FlowVC
       - 2~3× speedup 예상
    2. INT8 양자화: Encoder/Decoder weight → 30% 속도 향상
    3. 연산 융합: Conv + Norm + Activation → fused kernel
    4. Lookahead 축소: 2→1 frame (품질-속도 tradeoff)
       - RTF 20% 개선, TTFB 40ms 감소

GPU 최적화:
    1. CUDA Graph: 반복적인 추론 패턴 캡처 → overhead 제거
    2. FP16 inference: 1.5× speedup
    3. Batch inference: 여러 청크를 동시에 처리 (offline)

ODE Step 최적화:
    1. Adaptive step sizing: 초기 프레임 8 steps, 이후 2~4 steps
    2. Step distillation: 4-step student가 8-step teacher 모방
    3. Lookup table: t=0.0, 0.5, 1.0에서만 평가하고 보간
```

---

## 7. btrv3lite/btrvrc0 재활용 경로

### 7.1 직접 재활용 가능한 자산

| btrv3lite/btrvrc0 자산 | FlowVC 적용 | 재활용 방식 |
|------------------------|-------------|------------|
| **MioCodec Teacher** (`teacher.py`) | F³-Encoder/Decoder warmstart | MioCodec의 encoder 6-stage 구조를 F³-Encoder 초기화에 사용. 768-dim latent 호환 유지. |
| **CausalLatentConverter** (`converter.py`) | FlowVC Converter 구조 참조 | ConvNeXt-1D 블록, AdaLN-Zero, CausalDepthwiseConv, SpeakerPromptEncoder — 코드 구조 90% 재사용. |
| **SpeakerPromptEncoder** (`converter.py`) | FlowVC의 cross-attention prompt | P-Flow 스타일의 learnable speaker prompt token — 그대로 사용. |
| **CausalDepthwiseConv1d** (`converter.py`) | 모든 컴포넌트 | Causal conv 구현 그대로 재사용. |
| **ChannelLayerNorm** (`converter.py`) | 모든 컴포넌트 | LayerNorm over channels — 그대로 재사용. |
| **Student Decoder** (`student.py`) | F³-Decoder 구조 참조 | MRF 블록, Snake activation, FiLM conditioning, upsampling 구조 — 코드 재사용 70%. |
| **StreamingV3Inference** (`streaming_v3.py`) | FlowVC 스트리밍 | Buffer 관리, 청크 처리, lookahead 윈도우 — 인프라 코드 직접 재사용. |
| **F0/Prosody Extractor** (`f0.py`, `conditioner.py`) | Prosody Extractor | log-F0, voicing, energy 추출 — 그대로 사용. |
| **Speaker Bank** (`adcvc_bank.py`, `bank*.pt`) | Speaker conditioning | 기존 speaker bank (256 entries) 재사용 가능. |
| **Discriminator** (`discriminator.py`) | F³-Codec GAN 학습 | MPD + MRD discriminator — 그대로 사용. |
| **Dataset / Caching** (`dataset.py`, `adcvc_dataset.py`) | 학습 데이터 파이프라인 | Audio loading, teacher feature caching 인프라 — 재사용. |
| **Warmstart Checkpoint** | FlowVC warmstart | `/Users/asill/btrvrc0/models/btrv3lite_v1/converter.pt` → FlowVC Converter 초기화. |

### 7.2 Weight Transfer 매핑

```
btrv3lite CausalLatentConverter → FlowVC Converter

btrv3lite (dim=192)                 FlowVC (dim=512)
─────────────────────────           ─────────────────────
in_proj:  Linear(768→192)     →    in_proj:  Linear(768→512)    [partial: 192 cols만]
blocks.N.dwconv (192, g=192)  →    blocks.N.dwconv (512, g=512) [partial: 192 ch만]
blocks.N.pwconv1 (192→768)    →    blocks.N.mlp.0 (512→2048)    [부분 초기화]
blocks.N.pwconv2 (768→192)    →    blocks.N.mlp.2 (2048→512)    [부분 초기화]
out_proj:  Linear(192→768)    →    out_proj:  Linear(512→768)   [partial: 192 cols만]
adaln (128→3*192)             →    adaln (512→3*512)            [zero-init이므로 skip]
cross_attns                   →    cross_attns                  [dim 확장, partial init]
prompt_encoder (128→4*192)   →    prompt_encoder (192→4*512)   [partial init]

전략: 확장된 차원은 Kaiming init, 기존 차원은 btrv3lite weight로 초기화.
```

### 7.3 코드베이스 구조 제안

```
btrv5/
├── flowvc/
│   ├── __init__.py
│   ├── encoder.py           # F³-Encoder (ConvNeXt-v2 기반)
│   ├── decoder.py           # F³-Decoder (MRF + ConvNeXt-v2)
│   ├── converter.py         # FlowVC Converter (v_θ + ODE solver)
│   ├── speaker_encoder.py   # Speaker Encoder (ECAPA/WavLM wrapper)
│   ├── prosody.py           # Prosody Extractor (F0, energy)
│   ├── convnext_v2.py       # ConvNeXt-v2 block (GRN 포함)
│   ├── solver.py            # ODE solvers (Euler, RK4, Dopri5)
│   ├── discriminator.py     # MPD + MRD (btrvrc0에서 복사)
│   ├── streaming.py         # Streaming inference (btrvrc0 streaming_v3 기반)
│   ├── losses.py            # CFM loss, STFT loss, GAN loss
│   ├── dataset.py           # Data loading + caching
│   ├── train_codec.py       # Phase 1: Codec pretraining
│   ├── train_flowvc.py      # Phase 3: FlowVC CFM training
│   └── train_e2e.py         # Phase 4: End-to-end fine-tuning
├── configs/
│   ├── codec_44k.yaml       # F³-Codec 설정
│   ├── flowvc_base.yaml     # FlowVC 기본 설정
│   └── streaming.yaml       # 추론 설정
├── scripts/
│   ├── cache_features.py    # Teacher feature caching
│   ├── infer.py             # Offline inference
│   └── benchmark.py         # RTF 측정
└── designs/
    └── 01_flowvc.md         # 본 설계 문서
```

---

## 8. 예상 파라미터 수 및 연산량

### 8.1 파라미터 수

| 컴포넌트 | Sub-module | Params (M) | 비고 |
|----------|-----------|-----------|------|
| **F³-Encoder** | 6 stages, 각 2 ConvNeXt-v2 blocks | 15.2 | |
| **Speaker Encoder** | 6 stages + Attn Pool + Prompt | 8.5 | (또는 pretrained WavLM) |
| **Prosody Extractor** | 경량 ConvNet | 0.3 | |
| **FlowVC Converter** | 12 ConvNeXt-v2 blocks + cross-attn | 28.3 | 핵심 컴포넌트 |
| **F³-Decoder** | 4 refine blocks + 6 upsampling stages | 18.5 | |
| **Discriminator** | MPD + MRD (학습 전용) | 45.2 | 추론 시 미사용 |
| **Total (inference)** | | **70.8M** | |
| **Total (training)** | | **116.0M** | D 포함 |

### 8.2 MACs (Multiply-Accumulate) 추정

```
1프레임 (40ms, 1764 samples @ 44.1kHz) 기준:

┌─────────────────────┬──────────────┬──────────────────┐
│ Component           │ Input Shape   │ MACs (per frame) │
├─────────────────────┼──────────────┼──────────────────┤
│ F³-Encoder          │ (1, 11*1764)  │ ~2.8G            │
│ Prosody Extractor   │ (1, 11*1764)  │ ~0.05G           │
│ FlowVC (4 ODE steps)│ (1, 768)×4   │ ~1.2G            │
│ F³-Decoder          │ (1, 768)      │ ~1.5G            │
├─────────────────────┼──────────────┼──────────────────┤
│ Total per frame     │              │ ~5.55 GMACs      │
├─────────────────────┼──────────────┼──────────────────┤
│ Per second of audio │ 25 frames     │ ~138.75 GMACs    │
│ GPU (A100, 312 TFLOPS FP16)        │ 0.04% utilization │
│ CPU (M2 Max, ~15 TFLOPS)           │ 0.9% utilization  │
└─────────────────────┴──────────────┴──────────────────┘

실시간 충분: A100에서는 0.04%만 사용, M2 Max에서도 여유.
```

### 8.3 메모리 사용량

```
추론 메모리 (FP32):
    Model weights: 70.8M × 4 bytes = ~283 MB
    Activations (peak): 11 frames × encoder activations ≈ 120 MB
    Total: ~400 MB

MP3/INT8 최적화:
    INT8 weights: 70.8M × 1 byte = ~71 MB
    Total: ~200 MB → 모바일/엣지 디바이스 가능
```

---

## 9. 성능 목표 및 평가 지표

### 9.1 자동 평가 지표

| 지표 | 측정 방법 | 목표값 | btrv3lite 기준 |
|------|----------|--------|---------------|
| **Speaker Similarity** | ECAPA-TDNN cosine similarity (src↔tgt) | > 0.85 | 0.78 (개선 목표) |
| **WER** | Whisper large-v3 ASR | < 3% | 3.5% |
| **MCD** (Mel Cepstral Distortion) | Dynamic Time Warping | < 5.0 dB | 5.8 dB |
| **F0 RMSE** | log-F0 RMSE (cents) | < 50 cents | 65 cents |
| **PESQ** (WB) | ITU-T P.862.2 | > 3.0 | 2.7 |
| **UTMOS** | UTMOS strong model | > 4.0 | 3.8 |
| **RTF** | wall-clock / audio duration | < 0.5 (GPU) | 0.3 |
| **TTFB** | 첫 오디오 출력까지 시간 | < 150ms (GPU) | 120ms |

### 9.2 주관적 평가

| 테스트 | 방법 | 목표 |
|--------|------|------|
| **MOS (naturalness)** | 20명 청취자, 5점 척도 | > 4.0 |
| **MOS (similarity)** | 동일/다른 화자 판별 | > 3.8 |
| **ABX test** | FlowVC vs btrv3lite | > 60% 선호 |
| **Stress test** | unseen speaker, noisy input | acceptable quality |

### 9.3 벤치마크 데이터셋

```
VCTK test set (unseen speakers):
    - 10 speakers, 20 utterances each
    - 모든 가능한 src→tgt 쌍 (90 direction)

LibriTTS test-clean:
    - 10 speakers, 20 utterances

자체 한국어 테스트셋 (있는 경우):
    - 5 speakers, 10 utterances
```

---

## 10. 리스크 및 대안

### 10.1 주요 리스크

| # | 리스크 | 심각도 | 확률 | 완화 전략 |
|---|--------|--------|------|----------|
| **R1** | Flow Matching 학습 불안정 | High | Medium | OT 경로(직선) 사용으로 안정성 확보. σ_min > 0.001로 numerical stability. Gradient clipping. |
| **R2** | ODE step 수 부족 → 품질 저하 | High | Low | 4-step으로도 충분함 (FMelCodec에서 검증). 필요시 8-step으로 증가. Step distillation 기법 적용. |
| **R3** | ConvNeXt v2 GRN의 MPS 미지원 | Medium | Medium | GRN 구현은 단순하므로 pure PyTorch로 구현. 또는 LayerNorm으로 대체 가능. |
| **R4** | 768-dim latent가 flow matching에 너무 큼 | Medium | Low | FMelCodec에서 더 큰 차원에서도 성공. 필요시 latent 차원을 512로 축소. |
| **R5** | 실시간 성능 미달 (CPU) | Medium | Medium | CoreML 변환, INT8 양자화, ODE step 축소(4→2), lookahead 2→1. |
| **R6** | Unseen speaker에서 화자 유사도 저하 | Medium | Medium | Speaker encoder로 WavLM large (pretrained) 사용. FlowVC의 확률적 특성이 도움. |
| **R7** | btrv3lite warmstart 불완전 | Low | Medium | 차원 확장 시 partial init으로 충분. Full random init도 수렴 가능. |

### 10.2 설계 대안

#### 대안 A: Discrete Latent 보조 경로 (FMelCodec 스타일)

```
현재 설계(FlowVC)는 continuous latent만 사용하지만,
필요시 discrete bottleneck을 보조 경로로 추가:

    Encoder → z_cont (768-dim continuous)
            → z_disc (VQ, 1024 codebook, 256-dim)  ← 보조

    FlowVC는 z_cont에서 동작, z_disc는 저비트레이트 전송용.

    장점: FMelCodec의 OC-VQ insight 활용, 초저비트레이트(250bps) 전송 가능
    단점: 파이프라인 복잡도 증가, VQ 학습 추가 부담
```

#### 대안 B: Patch-based Flow Matching (F³-Tokenizer 스타일)

```
현재: 프레임 단위 ODE
대안: latent patch (4프레임) 단위 Flow Matching

    장점: 시간적 일관성 향상, 더 적은 ODE step으로 더 좋은 품질
    단점: TTFB 증가 (4프레임 lookahead 필요), 스트리밍 복잡도 증가
```

#### 대안 C: Knowledge Distillation (Teacher-Student)

```
2-step inference student가 8-step teacher 모방:

    Teacher: 8-step Euler/RK4
    Student: 2-step Euler
    Loss: L_consistency (학생 출력 vs 교사 출력)

    장점: 추론 속도 4배 향상 (RTF 0.1 이하)
    단점: 추가 학습 단계 필요, 품질 손실 가능성
```

### 10.3 단계적 구현 계획 (Milestone)

```
Milestone 1: F³-Codec (Week 1-3)
    - ConvNeXt-v2 Encoder + Decoder 구현
    - GAN 학습 (Phase 1)
    - 44.1kHz reconstruction 품질 검증
    - 목표: PESQ > 3.5, M-STFT < 0.8

Milestone 2: FlowVC Converter (Week 3-6)
    - Vector field network 구현
    - CFM 학습 (Phase 3)
    - Offline VC 품질 검증
    - 목표: Speaker sim > 0.80, WER < 5%

Milestone 3: Streaming Inference (Week 6-8)
    - Causal buffer management
    - 실시간 ODE solver 최적화
    - RTF/TTFB 측정
    - 목표: RTF < 0.5 (GPU), TTFB < 200ms

Milestone 4: End-to-End + 최적화 (Week 8-10)
    - E2E fine-tuning (Phase 4)
    - CoreML/ONNX 변환
    - 벤치마크 테스트
    - 목표: 전체 지표 충족
```

---

## 부록 A: 참고 문헌

| 논문 | arXiv | 핵심 기여 |
|------|-------|----------|
| F³-Tokenizer | 2606.06357 | KL-free continuous AE, noise regularization, flow matching over patches |
| FMelCodec | 2605.25669 | OC-VQ + Conditional Flow Matching refiner, 250bps |
| SURF | 2606.04921 | Unsupervised source separation via Flow Matching |
| ConvNeXt v2 | 2301.00808 | GRN, fully convolutional modern backbone |
| DiT (AdaLN-Zero) | 2212.09748 | Adaptive Layer Norm with zero-initialization |
| P-Flow | 2305.07432 | Speaker prompt tokens for VC |
| HiFi-GAN | 2010.05646 | Multi-Receptive Field blocks, multi-period discriminator |
| Flow Matching | 2210.02747 | Simulation-free training of continuous normalizing flows |
| MioCodec | - | btrv3lite teacher codec (44.1kHz, 25Hz, 768-dim) |

## 부록 B: 설정 파일 예시

```yaml
# configs/flowvc_base.yaml

sample_rate: 44100
latent_rate: 25
hop_samples: 1764

encoder:
  type: "convnext_v2"
  stages: 6
  strides: [2, 2, 3, 3, 7, 7]
  channels: [32, 64, 128, 256, 512, 768]
  blocks_per_stage: 2
  kernel_size: 7
  noise_sigma: 0.01
  use_grn: true

decoder:
  type: "convnext_v2_hifigan"
  refine_blocks: 4
  upsampling_stages: 6
  reverse_strides: [7, 7, 3, 3, 2, 2]
  channels: [768, 512, 256, 128, 64, 32]
  mrf_blocks_per_stage: 2
  mrf_kernel_sizes: [3, 7, 11]
  mrf_dilations: [[1,3,5], [1,3,5], [1,3,5]]
  speaker_conditioning: "film"

converter:
  type: "flowvc_cfm"
  hidden_dim: 512
  n_blocks: 12
  kernel_size: 7
  dilations: [1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8]
  mlp_expansion: 4
  cond_dim: 256
  time_emb_dim: 256
  cross_attn_layers: [3, 6, 9]
  cross_attn_heads: 4
  speaker_prompt_tokens: 4
  ode_solver: "euler"
  ode_steps: 4
  sigma_min: 0.001

speaker_encoder:
  type: "wavlm_large"  # or "ecapa_tdnn" or "trainable_convnext"
  output_dim: 192
  freeze: true

training:
  phase1:
    steps: 500000
    batch_size: 16
    lr_g: 0.0002
    lr_d: 0.0002
    seq_seconds: 3.0
    noise_sigma: 0.01
  phase3:
    steps: 200000
    batch_size: 8
    lr: 0.0001
    seq_seconds: 3.0
    cfm_weight: 1.0
    consist_weight: 0.2
    spk_weight: 0.1
    prosody_weight: 0.05
  phase4:
    steps: 50000
    batch_size: 4
    lr_g: 0.00005
    lr_d: 0.00005
```

---

> **문서 버전**: v1.0
> **작성일**: 2026-06-06
> **작성자**: btrv5 Architecture Team
> **상태**: Draft (리뷰 대기)

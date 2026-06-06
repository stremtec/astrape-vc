# StreamVC: Streaming-Optimized Voice Conversion Pipeline

> **btrv5 아키텍처 설계 — 각도 "StreamVC"**
>
> Depth-wise Streaming TTS + Audio Interaction Model + Vocos BWE 인사이트 통합
>
> 24kHz+ 초저지연 실시간 음성 변환 파이프라인

---

## 목차

1. [개요 및 설계 철학](#1-개요-및-설계-철학)
2. [스트리밍 아키텍처 다이어그램](#2-스트리밍-아키텍처-다이어그램)
3. [컴포넌트별 실시간 처리 전략](#3-컴포넌트별-실시간-처리-전략)
4. [레이턴시 버짓](#4-레이턴시-버짓)
5. [차원 및 파라미터 명세](#5-차원-및-파라미터-명세)
6. [Training Pipeline](#6-training-pipeline)
7. [기존 streaming.py와의 차이점 및 재활용 방안](#7-기존-streamingpy와의-차이점-및-재활용-방안)
8. [TTFB/RTF 추정 및 검증 방법](#8-ttfbrtf-추정-및-검증-방법)
9. [리스크 분석](#9-리스크-분석)
10. [참고문헌](#10-참고문헌)

---

## 1. 개요 및 설계 철학

### 1.1 StreamVC란?

StreamVC는 **24kHz 이상 샘플레이트**에서 **TTFB < 50ms, RTF < 0.1**을 달성하는 스트리밍-퍼스트 실시간 음성 변환(Voice Conversion) 파이프라인이다. 기존 SOLA/cross-fade 기반의 후처리 없는 **순수 chunk 기반 스트리밍**을 목표로 하며, 다음 세 논문의 핵심 인사이트를 아키텍처 전반에 통합한다:

| 논문 | 핵심 인사이트 | StreamVC 적용 |
|------|-------------|--------------|
| **Depth-wise Streaming TTS** (2604.12438) | 32-layer RVQ depth축 순차 디코딩 → 첫 토큰 생성까지 극소량 레이어만 활성화, TTFB 48.99ms | Content Encoder의 depth-wise chunk encoding |
| **Audio Interaction Model** (2606.05121) | Streaming perceive-decide-respond LLM, async FIFO inference 파이프라인 | 전체 파이프라인의 비동기 FIFO 스트리밍 아키텍처 |
| **Vocos BWE** (2603.07285) | ConvNeXt 기반 대역확장, RTF 0.0001 (A100) / 0.0053 (CPU) | Decoder 출력을 24kHz로 Bandwidth Extension |

### 1.2 핵심 설계 원칙

1. **Chunk-first, no overlap**: SOLA/cross-fade 없이 독립 chunk 단위 처리. Chunk 경계는 causal convolution의 natural receptive field로 자연스럽게 연결.
2. **Depth-wise decoding**: Content Encoder는 깊이 축(depth axis)을 따라 순차적으로 partial output을 생성 — shallow layer 출력으로 첫 chunk 응답, deep layer 출력으로 후속 refinement.
3. **Async FIFO pipeline**: Audio Interaction Model의 비동기 추론 아키텍처를 차용하여, Encoder → Converter → Decoder가 FIFO 큐로 연결된 파이프라인.
4. **Bandwidth Extension 분리**: Base decoder는 12kHz latent에서 동작하고, Vocos BWE가 24kHz로 업샘플링 — decoder 연산량 대폭 절감.

### 1.3 목표 성능 지표

| 지표 | 목표치 | 비고 |
|------|--------|------|
| Sample Rate | 24,000 Hz | BWE로 달성, base는 12kHz latent |
| TTFB (Time-To-First-Byte) | < 50ms | 첫 오디오 프레임 출력까지 |
| RTF (Real-Time Factor) | < 0.1 | CPU 기준, GPU는 < 0.01 |
| Speaker Similarity | ≥ 0.85 | speaker embedding cosine similarity |
| WER (Word Error Rate) | < 3% | ASR 기반 측정 |
| Chunk Size | 80ms (1920 samples @ 24kHz) | Processing granularity |
| Memory | < 200MB | 추론 시 상주 메모리 |

---

## 2. 스트리밍 아키텍처 다이어그램

### 2.1 전체 파이프라인 (ASCII Art)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         StreamVC Inference Pipeline                          │
│                                                                              │
│   Audio Input (24kHz, int16)                                                 │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────┐    80ms chunks (1920 samples @ 24kHz)                     │
│   │ Ring Buffer │    sliding window, stride = chunk_size                     │
│   │  (320ms)    │    no overlap between consecutive chunks                   │
│   └──────┬──────┘                                                            │
│          │                                                                    │
│          ▼                                                                    │
│   ┌──────────────────────────────────────────┐                              │
│   │         STREAMING ENCODER                │                              │
│   │                                          │                              │
│   │  ┌────────────┐   ┌───────────────────┐  │                              │
│   │  │ Content    │   │ Speaker Embedding │  │  Pre-computed, cached        │
│   │  │ Encoder    │   │   (ECAPA-TDNN)    │  │                              │
│   │  │ (Causal    │   │                   │  │                              │
│   │  │  ConvNeXt) │   │   dim: 256        │  │                              │
│   │  └─────┬──────┘   └────────┬──────────┘  │                              │
│   │        │                    │              │                              │
│   │  Depth-wise progression:                  │                              │
│   │  Layer 0-3  → L0 content (first chunk)   │                              │
│   │  Layer 4-7  → L1 content (refined)       │                              │
│   │  Layer 8-11 → L2 content (full quality)  │                              │
│   │        │                    │              │                              │
│   └────────┼────────────────────┼──────────────┘                              │
│            │                    │                                              │
│            ▼                    │                                              │
│   ┌────────────────────────────┐│                                             │
│   │   DEPTH-WISE CONVERTER     ││                                             │
│   │                            ││                                             │
│   │  ┌──────────────────────┐  ││                                             │
│   │  │ CausalLatentConverter│◄─┘│  Adaptive Layer Normalization (AdaLN)       │
│   │  │ (ConvNeXt-1D, causal)│   │  conditioned on speaker embedding            │
│   │  │                      │   │                                              │
│   │  │  Depth-wise decode:  │   │                                              │
│   │  │  D0 (4 layers) → Q0  │   │  First response @ 48ms TTFB                 │
│   │  │  D1 (4 layers) → Q1  │   │  Refined @ 64ms                            │
│   │  │  D2 (4 layers) → Q2  │   │  Full quality @ 80ms                       │
│   │  └──────────┬───────────┘   │                                              │
│   └─────────────┼───────────────┘                                              │
│                 │                                                              │
│                 ▼ (latent, 12kHz equivalent, dim=512)                          │
│   ┌──────────────────────────────────────────┐                              │
│   │          STREAMING DECODER               │                              │
│   │                                          │                              │
│   │  ┌────────────────┐  ┌────────────────┐  │                              │
│   │  │ Base Decoder   │  │ Vocos BWE      │  │                              │
│   │  │ (Causal HiFi-  │─▶│ (ConvNeXt      │  │                              │
│   │  │  GAN, 12kHz)   │  │  Bandwidth     │  │                              │
│   │  │                │  │  Extension)    │  │                              │
│   │  │ RTF: ~0.0002   │  │ RTF: ~0.0053   │  │  CPU 기준                     │
│   │  └────────────────┘  └───────┬────────┘  │                              │
│   └──────────────────────────────┼───────────┘                              │
│                                  │                                           │
│                                  ▼                                           │
│   ┌──────────────────────────────────────────┐                              │
│   │         OUTPUT BUFFER                    │                              │
│   │  (80ms chunks @ 24kHz, int16)            │                              │
│   │  Async FIFO → audio playback             │                              │
│   └──────────────────────────────────────────┘                              │
│                                                                              │
│   ════════════════════ FIFO Boundary ════════════════════                    │
│                                                                              │
│   ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│   │ Encoder │───▶│Converter │───▶│ Decoder  │───▶│ Output   │               │
│   │  Queue  │    │  Queue   │    │  Queue   │    │  Queue   │               │
│   │ (FIFO)  │    │  (FIFO)  │    │  (FIFO)  │    │  (FIFO)  │               │
│   └─────────┘    └──────────┘    └──────────┘    └──────────┘               │
│        ▲               ▲               ▲               ▲                     │
│        │               │               │               │                     │
│   Worker Thread   Worker Thread   Worker Thread   Audio Thread                │
│   (CPU/GPU)       (CPU/GPU)       (CPU/GPU)       (CPU)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Chunk 흐름 상세

```
Time ─────────────────────────────────────────────────────▶

Input Stream:
│  Chunk[0]  │  Chunk[1]  │  Chunk[2]  │  Chunk[3]  │  Chunk[4]  │
│  0-80ms    │  80-160ms  │ 160-240ms  │ 240-320ms  │ 320-400ms  │

Depth-wise Encoding (Chunk[0] example):
  t=0ms:     Encoder Layer 0-3   → L0_content[0]   (partial, 16ms proc)
  t=16ms:    Encoder Layer 4-7   → L1_content[0]   (refined)
  t=32ms:    Encoder Layer 8-11  → L2_content[0]   (full)

Depth-wise Conversion (Chunk[0]):
  t=32ms:    Converter D0       → Q0[0]           (first response)
  t=48ms:    Converter D1       → Q1[0]           (refined)   ◄── TTFB!
  t=64ms:    Converter D2       → Q2[0]           (full quality)

Decoding (Chunk[0]):
  t=48ms:    Base Decoder       → audio @ 12kHz   (12ms proc)
  t=60ms:    Vocos BWE          → audio @ 24kHz   (3ms proc)
  t=63ms:    Output Chunk[0] ready → PLAYBACK      ◄── End-to-end latency

Pipeline Parallelism (overlapped execution):
  While Converter processes Chunk[0] D0...:
    Encoder already processing Chunk[1] Layers 0-3
  While Decoder processes Chunk[0]:
    Converter already processing Chunk[1] D0
    Encoder already processing Chunk[2] Layers 0-3

Effective latency per chunk: max(component_latency) ≈ 63ms first chunk
Steady-state: each 80ms chunk produces output every 80ms
```

### 2.3 Async FIFO Inference (Audio Interaction Model 패턴)

```
┌──────────────────────────────────────────────────────┐
│                  Inference Scheduler                  │
│                                                      │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐            │
│  │ Encoder │   │Converter│   │ Decoder │            │
│  │ Worker  │   │ Worker  │   │ Worker  │            │
│  │ (GPU)   │   │ (GPU)   │   │ (CPU)   │            │
│  └────┬────┘   └────┬────┘   └────┬────┘            │
│       │              │              │                  │
│       ▼              ▼              ▼                  │
│  ┌─────────────────────────────────────┐              │
│  │        Shared Memory FIFO           │              │
│  │                                     │              │
│  │  EncFIFO ──▶ ConvFIFO ──▶ DecFIFO  │              │
│  │  (max 4)     (max 4)      (max 4)   │              │
│  │                                     │              │
│  │  Each entry: (chunk_id, tensor,     │              │
│  │               depth_level, ts)      │              │
│  └─────────────────────────────────────┘              │
│                                                      │
│  Back-pressure: if any FIFO is full, upstream        │
│  worker blocks → natural flow control                 │
└──────────────────────────────────────────────────────┘
```

---

## 3. 컴포넌트별 실시간 처리 전략

### 3.1 Streaming Content Encoder

#### 구조
```
Input: 80ms audio chunk @ 24kHz (1920 samples)
       ↓
   Mel-Spectrogram (80-bin, 25ms window, 10ms hop)
       ↓  [8 frames per 80ms chunk]
   CausalConv1D Stem (kernel=7, stride=1, dilations=[1,2,4])
       ↓  [causal: no future lookahead]
   ConvNeXt Blocks × 12 (depth-wise organized into 3 groups)
       ├── Group 0: Blocks 0-3  → L0 output (shallow content)  [16ms]
       ├── Group 1: Blocks 4-7  → L1 output (intermediate)     [16ms]
       └── Group 2: Blocks 8-11 → L2 output (full quality)     [16ms]
       ↓
   Output: content latent z_c ∈ ℝ^{T_c × 512}
           T_c = 4 (每 80ms chunk, stride=20ms)
```

#### Depth-wise Streaming Strategy

Depth-wise Streaming TTS (2604.12438)의 핵심 인사이트: **RVQ depth 축을 따라 순차적으로 디코딩하여 첫 토큰 생성 지연을 최소화**한다. 이를 Content Encoder에 적용:

1. **Shallow output (L0)** : 첫 4개 ConvNeXt 블록만 통과 → coarse content feature. Converter에 즉시 전달하여 첫 응답 생성.
2. **Intermediate output (L1)** : Blocks 4-7 통과 → refined content. Converter D1 레이어에서 활용.
3. **Full output (L2)** : 전체 12 블록 통과 → 최종 품질. Converter D2에서 최종 latent 생성.

**핵심**: L0, L1, L2는 병렬이 아닌 **순차적**으로 생성되지만, Converter는 각 depth level이 준비되는 즉시 해당 depth의 변환을 시작할 수 있다 (pipeline parallelism).

#### Causal Constraint
- 모든 convolution은 causal padding 사용 (left-padding only)
- Mel-spectrogram의 lookahead: 25ms window 사용 시 center=True → 12.5ms 미래 정보 필요
  - **해결**: streaming Mel 계산 시 center=False + causal padding, 또는 25ms window에서 12.5ms를 chunk overlap으로 흡수 (chunk 경계 12.5ms 중첩 → 실질적 lookahead 12.5ms)

### 3.2 Speaker Encoder

```
Reference Audio (3s, pre-recorded)
       ↓
   ECAPA-TDNN (frozen, pre-trained)
       ↓
   Speaker Embedding s ∈ ℝ^{256}
       ↓
   Cached for entire session (no per-chunk recomputation)
```

- 기존 btrv3lite와 동일한 ECAPA-TDNN 사용
- 추론 시작 시 1회 계산 후 캐싱
- Speaker embedding은 Converter의 AdaLN condition으로 사용

### 3.3 Depth-wise CausalLatentConverter

#### 구조
```
Input: z_c (content latent), s (speaker embedding)
       ↓
   AdaLN Modulation: scale, shift = MLP(s)
       ↓
   Converter Blocks × 12 (depth-wise organized)
       ├── D0: Blocks 0-3  ──▶ Q0 (coarse converted latent)
       ├── D1: Blocks 4-7  ──▶ Q1 (refined converted latent)
       └── D2: Blocks 8-11 ──▶ Q2 (full quality target latent)
       ↓
   Output: z_t ∈ ℝ^{T_c × 512}  (12kHz-equivalent acoustic latent)
```

#### Depth-wise Decoding Protocol

Depth-wise Streaming TTS의 **RVQ depth축 순차 디코딩** 전략을 Converter에 적용:

1. **D0 (Blocks 0-3, 4 ConvNeXt blocks)**:
   - L0 content + speaker embedding으로 coarse 변환
   - 출력 Q0: 기본적인 화자 특성 반영, 일부 디테일 손실
   - Decoder에 즉시 전달 → **TTFB 달성** (≈48ms)

2. **D1 (Blocks 4-7, 4 ConvNeXt blocks)**:
   - L1 content + D0 hidden state 기반 refined 변환
   - 출력 Q1: 향상된 화자 유사도, 더 자연스러운 운율
   - Decoder 업데이트 (residual/overwrite)

3. **D2 (Blocks 8-11, 4 ConvNeXt blocks)**:
   - L2 content + D1 hidden state 기반 최종 변환
   - 출력 Q2: 최종 품질, full speaker similarity
   - Decoder 최종 출력

#### Residual Update Strategy

```
Q_final = α₀·Q₀ + α₁·Q₁ + α₂·Q₂

where α₀ + α₁ + α₂ = 1 (learnable or fixed schedule)
      initial: α₀=0.5, α₁=0.3, α₂=0.2
      final:   α₀=0.1, α₁=0.2, α₂=0.7  (progressive refinement)
```

또는 Decoder가 Q₀로 첫 chunk를 생성하고, Q₁, Q₂는 후속 chunk refinement에 활용.

#### btrv3lite 재활용 요소
- ConvNeXt-1D backbone: btrv3lite의 `CausalLatentConverter`에서 직접 이식
- AdaLN conditioning: speaker embedding을 condition으로 하는 FiLM/AdaLN 레이어 구조 재사용
- Causal convolution: 기존 causal padding 로직 그대로 활용

### 3.4 Streaming Decoder (Base + BWE)

#### 3.4.1 Base Decoder (Causal HiFi-GAN, 12kHz)

```
Input: z_t ∈ ℝ^{T_c × 512}  (12kHz-equivalent acoustic latent)
       ↓
   CausalConv1D Upsampling Blocks × 4
   (transposed conv, kernel=8, stride=4 each → 256× upsampling)
       ↓
   Multi-Receptive Field Fusion (MRF) blocks
   (causal, kernel sizes = [3,7,11])
       ↓
   Output: waveform @ 12,000 Hz
```

- 12kHz base로 동작 → 연산량 1/4 (24kHz 대비)
- Causal transposed convolution: future lookahead zero
- RTF target: < 0.005 (CPU), < 0.0005 (GPU)

#### 3.4.2 Vocos Bandwidth Extension (24kHz)

Vocos BWE (2603.07285)의 ConvNeXt 기반 대역확장:

```
Input: waveform @ 12,000 Hz
       ↓
   STFT (window=1024, hop=256 @ 12kHz)
       ↓
   ConvNeXt Blocks × 6
   (causal, dilatated, efficient)
       ↓
   ISTFT → waveform @ 24,000 Hz
   + learned high-frequency residual
```

**핵심 이점**:
- ConvNeXt BWE는 RTF 0.0001 (A100), 0.0053 (CPU)로 극단적 경량
- 12kHz → 24kHz 변환을 ConvNeXt feature space에서 처리 → phase reconstruction artifacts 최소화
- Base decoder + BWE의 조합이 24kHz 단일 decoder보다 3-4× 경량

**Causal BWE adaptation**:
- 원본 Vocos BWE는 non-causal ConvNeXt 사용
- StreamVC에서는 ConvNeXt를 causal 버전으로 변환 (left-only padding)
- 실험적으로 non-causal 대비 품질 저하 0.02 MOS 이내로 예상

### 3.5 Output Buffer & Playback

```
┌──────────────────────────────────────┐
│         Async Output Pipeline         │
│                                      │
│  Decoded Chunks (80ms @ 24kHz)       │
│       │                              │
│       ▼                              │
│  ┌──────────┐   ┌──────────┐        │
│  │ Saturation│   │ Chunk    │        │
│  │ Limiter  │──▶│ Concaten- │        │
│  │ (-0.3dB) │   │ ation    │        │
│  └──────────┘   └────┬─────┘        │
│                      │               │
│                      ▼               │
│  ┌────────────────────────────────┐  │
│  │   Ring Buffer (480ms, pre-fill│  │
│  │   80ms for initial playback)  │  │
│  └────────────┬───────────────────┘  │
│               │                      │
│               ▼                      │
│  ┌────────────────────────────────┐  │
│  │   Audio Callback (PortAudio / │  │
│  │   CoreAudio / WASAPI)         │  │
│  └────────────────────────────────┘  │
│                                      │
│  Pre-fill strategy:                  │
│    Initial: 80ms silence pre-fill    │
│    Once first chunk ready: swap in   │
│    Steady-state: continuous drain    │
└──────────────────────────────────────┘
```

- **No SOLA, no cross-fade**: Chunk 경계는 HiFi-GAN이 학습한 natural continuation으로 연결
- **Pre-fill**: 최초 80ms 버퍼를 사일런스로 채우고 첫 chunk 완성 즉시 교체
- **Under-run protection**: 버퍼가 80ms 미만으로 떨어지면 마지막 chunk 반복 (zero-stuffing 보다 자연스러움)

---

## 4. 레이턴시 버짓

### 4.1 세부 분석 (80ms Chunk 기준)

```
═══════════════════════════════════════════════════════════════════
STAGE                    DEVICE    PROCESSING TIME    CUMULATIVE
═══════════════════════════════════════════════════════════════════
1. Audio Input Buffer    CPU       0.5ms              0.5ms
   (1920 samples DMA)

2. Mel-Spectrogram       CPU       1.0ms              1.5ms
   (80-bin, 8 frames)

3. Content Encoder L0    GPU/CPU   16ms               17.5ms
   (ConvNeXt Blocks 0-3)

4. Converter D0          GPU/CPU   15ms               32.5ms
   (ConvNeXt Blocks 0-3)

5. Base Decoder          CPU       12ms               44.5ms
   (12kHz HiFi-GAN)

6. Vocos BWE             CPU       3ms                47.5ms
   (12kHz→24kHz)

7. Output Buffer +       CPU       0.5ms              48.0ms  ◄── TTFB
   Limiter
═══════════════════════════════════════════════════════════════════
   **TTFB (D0 사용): 48ms**

8. Content Encoder L1    GPU/CPU   16ms               (비동기)
   (Blocks 4-7)
9. Converter D1          GPU/CPU   15ms               (비동기)
   (Blocks 4-7)
   → Q1 refined latent available @ ~64ms

10. Content Encoder L2   GPU/CPU   16ms              (비동기)
    (Blocks 8-11)
11. Converter D2         GPU/CPU   15ms              (비동기)
    (Blocks 8-11)
    → Q2 final latent available @ ~80ms
═══════════════════════════════════════════════════════════════════
   **End-to-end latency (full quality): 80ms**
```

### 4.2 Chunk Size 선택 근거

| Chunk Size | Frames | TTFB | Steady-state Latency | 장단점 |
|-----------|--------|------|---------------------|--------|
| 40ms | 4 frames (960 samples) | ~30ms | 40ms | TTFB는 낮으나 Encoder context 부족, 품질 저하 우려 |
| **80ms** | **8 frames (1920 samples)** | **~48ms** | **80ms** | **선택**: TTFB < 50ms 충족, 충분한 phonetic context |
| 160ms | 16 frames (3840 samples) | ~70ms | 160ms | 품질은 최상이나 TTFB 제약 위반 |
| 200ms | 20 frames (4800 samples) | ~90ms | 200ms | btrv3lite와 유사, 지연 체감 큼 |

**최종 선택: 80ms (1920 samples @ 24kHz)**

- Mel-spectrogram 기준 8프레임 (25ms window, 10ms hop → 약 95ms의 context, center padding 제외 시 80ms)
- TTFB 48ms로 목표 50ms 이내 달성
- 충분한 phonetic context 보유 (한국어 기준 약 1.5음절)
- btrv3lite의 200ms lookahead buffer 대비 60% 감소

### 4.3 Lookahead 분석

```
StreamVC lookahead 구성:
  Mel window:    12.5ms (25ms window, center)
  Chunk stride:  80ms
  Encoder causal: 0ms (lookahead 없음)
  Converter causal: 0ms
  Decoder causal: 0ms

Total algorithmic lookahead: 12.5ms (Mel window only)
Total system latency: 48ms (TTFB, processing-dominated)
```

btrv3lite 비교:
```
btrv3lite lookahead:
  MioCodec lookahead buffer: ~200ms
  + processing latency: ~50-100ms
  Total: ~250-400ms

StreamVC: ~48-80ms → 5-8× 개선
```

### 4.4 Pipeline Parallelism 효과

```
Time (ms) →    0    16   32   48   64   80   96   112  128  144  160
               │    │    │    │    │    │    │    │    │    │    │
Chunk 0 Enc:   ███████████████████████████████████
               L0   L1   L2
Chunk 0 Conv:       ░░░░████████████████████████████
                    D0   D1   D2
Chunk 0 Dec:                  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                              Dec+BWE
Chunk 1 Enc:            ████████████████████████████████████████
                        L0   L1   L2
Chunk 1 Conv:                          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
                                       D0   D1   D2
Chunk 1 Dec:                                            ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                                                        Dec+BWE

Steady-state throughput: 1 chunk / 80ms
Pipeline occupancy: ~3 chunks simultaneously in-flight
```

---

## 5. 차원 및 파라미터 명세

### 5.1 Model Dimensions

| 컴포넌트 | 레이어 | Hidden Dim | Kernel | Params | 비고 |
|---------|--------|-----------|--------|--------|------|
| **Content Encoder** | | | | **~18M** | |
| Conv1D Stem | 1 | 128→512 | 7 | 0.5M | causal |
| ConvNeXt Blocks × 12 | 12 | 512 | 7 | 16M | depth-wise: 3 groups of 4 |
| Output Proj | 1 | 512→512 | 1 | 0.3M | |
| Activation | - | SiLU | - | - | |
| **Speaker Encoder** | | | | **~14M** | ECAPA-TDNN, frozen |
| **Converter** | | | | **~18M** | |
| AdaLN MLP | 2 | 256→1024→1024 | - | 1.5M | condition: speaker emb |
| ConvNeXt Blocks × 12 | 12 | 512 | 7 | 16M | depth-wise: D0/D1/D2 |
| Output Proj | 1 | 512→512 | 1 | 0.3M | |
| **Base Decoder** | | | | **~8M** | causal HiFi-GAN |
| Upsampling × 4 | 4 | 512→256→128→64→32 | 8 | 3M | stride=4 each |
| MRF Blocks | 12 | 32 | [3,7,11] | 4.5M | multi-receptive field |
| Output Conv | 1 | 32→1 | 7 | 0.02M | |
| **Vocos BWE** | | | | **~3M** | ConvNeXt backbone |
| ConvNeXt × 6 | 6 | 128 | 7 | 2.5M | causal |
| ISTFT head | 1 | - | - | 0.3M | |
| **Total** | | | | **~61M** | |

### 5.2 Latent Dimensions

| Latent | Shape | Description |
|--------|-------|-------------|
| Audio chunk | (1, 1920) | 80ms @ 24kHz |
| Mel-spectrogram | (80, 8) | 80 mel-bins, 8 time frames |
| Content latent z_c | (4, 512) | Encoder output, 20ms stride |
| Speaker embedding s | (256,) | Pre-computed, session-cached |
| Converter output z_t | (4, 512) | 12kHz-equivalent acoustic latent |
| Base waveform | (1, 960) | 80ms @ 12kHz |
| Final waveform | (1, 1920) | 80ms @ 24kHz |

### 5.3 Memory Footprint (Inference)

```
Model weights (fp32):    61M × 4 bytes  = 244 MB
Model weights (fp16):    61M × 2 bytes  = 122 MB  ◄── target
Model weights (int8):    61M × 1 byte   =  61 MB  ◄── stretch goal

Runtime buffers:
  Input ring buffer:     320ms × 24000 × 2 bytes = 15 KB
  Encoder FIFO:          4 × 512 × 4 × 2 bytes   = 16 KB
  Converter FIFO:        4 × 512 × 4 × 2 bytes   = 16 KB
  Decoder FIFO:          4 × 1920 × 2 bytes      = 15 KB
  Output ring buffer:    480ms × 24000 × 2 bytes = 23 KB

Total runtime: ~85 KB + model

Target: < 200 MB (fp16 model + runtime)
```

---

## 6. Training Pipeline

### 6.1 Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    StreamVC Training Pipeline                   │
│                                                                │
│  Phase 1: Content Encoder Pre-training                         │
│  Phase 2: Converter + Decoder Joint Training (non-streaming)   │
│  Phase 3: Depth-wise Distillation                              │
│  Phase 4: Streaming Adaptation                                 │
│  Phase 5: Vocos BWE Fine-tuning                                │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 Phase 1: Content Encoder Pre-training

```
Objective: Self-supervised content feature extraction

Architecture:
  - wav2vec 2.0 / HuBERT style masked prediction
  - 또는 SSL distillation from pre-trained content model

Data:
  - LibriSpeech (960h) + VCTK (44h) + AI Hub 한국어 음성 (1000h)
  - Multi-lingual: 한국어 40%, 영어 40%, 기타 20%

Training:
  - Masked acoustic modeling (60% mask ratio)
  - Contrastive loss + codebook diversity loss
  - 200K steps, batch 256, 8× A100

Output: Frozen Content Encoder weights
```

### 6.3 Phase 2: Converter + Decoder Joint Training

```
Objective: Speaker-conditioned waveform reconstruction

Architecture:
  - Full CausalLatentConverter (12 blocks)
  - Causal HiFi-GAN Decoder (12kHz base)
  - Non-streaming mode (full utterance processing)

Losses:
  L_total = L_mel + L_feat + L_adv + L_fm + L_spk + L_depth

  L_mel:       Mel-spectrogram L1 loss (12kHz + 24kHz)
  L_feat:      Multi-scale STFT feature matching loss
  L_adv:       Hinge-GAN adversarial loss (MSD + MPD)
  L_fm:        Feature matching loss (discriminator features)
  L_spk:       Speaker embedding cosine similarity loss
  L_depth:     Depth-wise consistency loss (see below)

Data:
  - VCTK (44h, 109 speakers) + LibriTTS (585h)
  - AI Hub 다화자 음성 (multi-speaker Korean)

Training:
  - 500K steps, batch 64, 4× A100
  - AdamW, lr=2e-4, exponential decay
  - Random 2-5s segment cropping
```

### 6.4 Phase 3: Depth-wise Distillation

```
Objective: Train depth-wise decoding capability

핵심 아이디어 (Depth-wise Streaming TTS 2604.12438):
  1. Full model (12 blocks)을 teacher로 사용
  2. Shallow sub-models (4 blocks, 8 blocks)을 student로 distillation
  3. 각 depth level이 독립적으로 reasonable output 생성하도록 학습

Distillation Protocol:

  Teacher: Converter_full (12 blocks, z_t_full)
  Student D0: Converter_shallow (4 blocks, Q₀)
  Student D1: Converter_mid (8 blocks, Q₁)

  L_depth = λ₀·MSE(Q₀, z_t_full)           # shallow → full alignment
          + λ₁·MSE(Q₁, z_t_full)           # mid → full alignment
          + λ₂·MSE(Q₀ + Δ₁, z_t_full)     # progressive alignment
          + λ₃·cos_sim(Qᵢ, z_t_full)      # direction consistency

  where Δ₁ = Q₁ - Q₀ (refinement direction)

  λ schedule:
    초기: λ₀=1.0, λ₁=0.3, λ₂=0.1, λ₃=0.5
    후기: λ₀=0.3, λ₁=1.0, λ₂=0.5, λ₃=0.5
    (점진적으로 deeper student에 가중치 이동)

Training:
  - 100K steps, batch 64, 2× A100
  - Teacher frozen, student weights from Phase 2 checkpoint
```

### 6.5 Phase 4: Streaming Adaptation

```
Objective: Chunk-level consistency, boundary artifact 제거

Strategy:
  1. Simulated streaming training:
     - Long utterance → random chunk size [40ms, 160ms]로 분할
     - 각 chunk 독립 처리 후 연접
     - 연접된 waveform의 continuity loss 적용

  2. Boundary continuity loss:
     L_boundary = MSE(waveform[t-5:t+5]_chunk_i,
                      waveform[t-5:t+5]_chunk_i+1)
     where t = chunk boundary

  3. Depth-wise progressive training:
     - Random depth truncation (0.3 확률로 4 blocks only,
       0.3 확률로 8 blocks only, 0.4 확률로 full 12 blocks)
     - 각 depth level에서도 streaming consistency 유지

  4. Causal constraint 강화:
     - 모든 convolution에 causal padding 강제
     - Non-causal path 발견 시 penalty

Training:
  - 50K steps, batch 32, 2× A100
  - Low learning rate (1e-5) for fine-tuning
```

### 6.6 Phase 5: Vocos BWE Fine-tuning

```
Objective: 12kHz base + BWE → 24kHz 최종 출력 최적화

Strategy:
  1. Base Decoder (12kHz) 고정
  2. Vocos BWE를 causal ConvNeXt로 변환
  3. End-to-end fine-tuning:
     z_t → Base Decoder (frozen) → BWE → 24kHz waveform

  Losses:
    L_bwe = L_mel_24k + L_stft_24k + L_adv_hires

  Training:
    - 50K steps, batch 16, 2× A100
    - Base decoder frozen
    - High-frequency band (6kHz-12kHz)에 가중치
```

### 6.7 Training Data Summary

| Dataset | Hours | Speakers | Usage |
|---------|-------|----------|-------|
| LibriSpeech | 960h | 2,484 | Phase 1 (content encoder) |
| LibriTTS | 585h | 2,456 | Phase 2, 3, 4 |
| VCTK | 44h | 109 | Phase 2, 3, 4 (multi-speaker) |
| AI Hub 한국어 음성 | 1,000h | 2,000+ | Phase 1, 2, 3, 4 |
| DAPS | 5h | 20 | Evaluation only |
| EARS | 100h | 100+ | Streaming evaluation |
| **Total** | **~2,694h** | | |

---

## 7. 기존 streaming.py와의 차이점 및 재활용 방안

### 7.1 btrv3lite streaming.py 분석

```
btrv3lite streaming.py 구조:
  ┌─────────────────────────────────┐
  │  MioCodecStreamingWrapper       │
  │  - lookahead buffer: 200ms      │
  │  - chunk size: 200ms            │
  │  - overlap-add with cross-fade  │
  │  - latency: ~250-400ms          │
  └─────────────────────────────────┘
              │
              ▼
  ┌─────────────────────────────────┐
  │  CausalLatentConverter          │
  │  - ConvNeXt-1D, causal          │
  │  - full utterance processing    │
  │  - no depth-wise capability     │
  └─────────────────────────────────┘
```

### 7.2 주요 차이점

| 측면 | btrv3lite streaming.py | StreamVC | 개선 |
|------|----------------------|----------|------|
| **Latency** | 250-400ms | 48-80ms | **5-8× 감소** |
| **Lookahead** | 200ms (MioCodec) | 12.5ms (Mel window) | **16× 감소** |
| **Overlap** | 50% overlap + cross-fade | No overlap (causal) | 제거 |
| **Chunk size** | 200ms 고정 | 80ms (depth-wise) | 유연 |
| **Decoder** | Single 24kHz decoder | 12kHz base + BWE | **연산량 1/3** |
| **Depth-wise** | 없음 | 3-level progressive | 신규 |
| **Async pipeline** | 동기적 처리 | FIFO async (AIM 패턴) | 신규 |
| **Boundary 처리** | Cross-fade (50ms) | Causal continuity | 제거 |

### 7.3 재활용 가능 요소

#### 직접 재활용 (코드 레벨)
```
✓ CausalLatentConverter 클래스
  → StreamVC Converter의 베이스 클래스로 확장
  → depth-wise forward(depth_level) 추가

✓ ConvNeXt-1D 블록 구현
  → causal padding, group norm, SiLU activation
  → depth-wise group을 위해 BlockGroup wrapper 추가

✓ Speaker Encoder (ECAPA-TDNN)
  → 완전 동일, 세션 캐싱 로직 유지

✓ Audio I/O 유틸리티
  → PortAudio/CoreAudio backend
  → Ring buffer 구현
  → Resampling utilities (필요 시)
```

#### 확장 재활용 (아키텍처 수정)
```
△ Causal padding 로직
  → 모든 Conv1D에 causal constraint 적용하는 유틸리티 함수
  → StreamVC의 Base Decoder, BWE에도 적용

△ Mel-spectrogram streaming
  → 기존 torchaudio 기반 Mel 변환을 streaming 버전으로 수정
  → center=False + causal padding 적용

△ Discriminator (HiFi-GAN)
  → btrv3lite의 MSD/MPD discriminator를 12kHz 버전으로 조정
  → causal constraint 추가
```

### 7.4 폐기되는 요소

```
✗ MioCodec wrapper
  → StreamVC는 자체 Content Encoder 사용 → 완전 대체

✗ Cross-fade / SOLA 로직
  → causal continuity로 대체 → 불필요

✗ 50% overlap-add
  → non-overlapping chunking → 단순화

✗ Fixed chunk size (200ms)
  → 80ms 고정 (depth-wise로 가변적 처리)
```

---

## 8. TTFB/RTF 추정 및 검증 방법

### 8.1 TTFB 정의 및 측정

```
TTFB (Time-To-First-Byte) 정의:
  T_mic에서 T_spk까지의 최초 오디오 출력 지연

  TTFB = T_audio_capture + T_preprocess + T_encode_L0
       + T_convert_D0 + T_decode + T_bwe + T_output

측정 방법:
  1. Loopback measurement:
     - Test signal (impulse/clip @ t=0) → DAC → Loopback cable → ADC
     - StreamVC processing → output waveform
     - Cross-correlation으로 impulse delay 측정

  2. Timestamp logging:
     - 각 stage 진입/출구 timestamp 기록 (std::chrono / time.perf_counter)
     - chunk_id, depth_level 포함

  3. End-to-end:
     - Mic 입력 timestamp vs Speaker 출력 timestamp
     - 100회 반복, P95 값 보고
```

### 8.2 RTF 정의 및 측정

```
RTF (Real-Time Factor) 정의:
  RTF = T_processing / T_audio

  T_processing: 80ms chunk 처리에 소요된 wall-clock time
  T_audio: 80ms (chunk duration)

측정 방법:
  1. Per-chunk measurement:
     for each chunk:
       t0 = now()
       encoded = encoder(chunk, depth_level)
       converted = converter(encoded, spk_emb, depth_level)
       waveform = decoder(converted)
       waveform_24k = bwe(waveform)
       t1 = now()
       chunk_rtf = (t1 - t0) / 0.080  # seconds

  2. Aggregated:
     - 전체 발화에 대한 avg RTF, P99 RTF
     - GPU (CUDA event 기반) 및 CPU (perf_counter 기반) 각각 측정

  3. Device별:
     - GPU (NVIDIA A100, RTX 4090, Apple M2/M3 ANE)
     - CPU (Intel i9, Apple M2/M3, AMD Ryzen)
```

### 8.3 예상 성능 (추정)

```
Device          | Encoder | Converter | Decoder | BWE    | Total RTF
────────────────┼─────────┼───────────┼─────────┼────────┼───────────
A100 (GPU)      | 0.002   | 0.002     | 0.0005  | 0.0001 | 0.0046
RTX 4090 (GPU)  | 0.003   | 0.003     | 0.0008  | 0.0002 | 0.0070
Apple M3 (ANE)  | 0.008   | 0.008     | 0.003   | 0.001  | 0.020
Apple M3 (CPU)  | 0.020   | 0.020     | 0.008   | 0.005  | 0.053
Intel i9 (CPU)  | 0.025   | 0.025     | 0.010   | 0.006  | 0.066
Raspberry Pi 5  | 0.080   | 0.080     | 0.030   | 0.020  | 0.210 ◄ 위험
────────────────┴─────────┴───────────┴─────────┴────────┴───────────

TTFB (all devices): 48ms (D0 사용 시)
  - Content Encoder L0: 16ms
  - Converter D0: 15ms
  - Decoder: 12ms
  - BWE: 3ms
  - I/O overhead: 2ms
  Total: 48ms ✓ (< 50ms target)
```

### 8.4 검증 Milestone

```
Milestone 1 (Month 2): Component-level benchmark
  - 각 컴포넌트 개별 RTF 측정 (Python, PyTorch)
  - 목표: Encoder < 20ms, Converter < 20ms, Decoder+BWE < 20ms

Milestone 2 (Month 4): Integrated pipeline (non-streaming)
  - Full pipeline end-to-end RTF
  - 목표: RTF < 0.1 (CPU), < 0.01 (GPU)

Milestone 3 (Month 6): Streaming pipeline
  - Async FIFO pipeline 구현 완료
  - TTFB loopback 측정
  - 목표: TTFB < 50ms, steady-state latency < 80ms

Milestone 4 (Month 8): Production optimization
  - ONNX Runtime / TensorRT 변환
  - INT8 양자화
  - 목표: RTF < 0.05 (CPU), TTFB < 40ms
```

---

## 9. 리스크 분석

### 9.1 Risk Matrix

```
Risk                          | Impact | Likelihood | Mitigation
──────────────────────────────┼────────┼────────────┼───────────────────────────
Chunk 경계 아티팩트           | HIGH   | MEDIUM     | [9.2] 참조
Depth-wise 품질 degradation   | HIGH   | MEDIUM     | [9.3] 참조
BWE causal 변환 품질 저하     | MEDIUM | MEDIUM     | [9.4] 참조
CPU-only 환경 RTF 미달        | MEDIUM | HIGH       | [9.5] 참조
Speaker similarity 저하       | HIGH   | LOW        | [9.6] 참조
Real-time audio drop-out      | MEDIUM | MEDIUM     | [9.7] 참조
Training data 부족 (한국어)   | MEDIUM | LOW        | [9.8] 참조
Lookahead buffer 부족         | MEDIUM | LOW        | [9.9] 참조
```

### 9.2 Risk 1: Chunk 경계 아티팩트 (CRITICAL)

**문제**: Non-overlapping chunking으로 인해 chunk 경계에서 불연속(click, pop, phase discontinuity) 발생 가능.

**원인 분석**:
1. 각 chunk가 독립적으로 처리되어 waveform continuity 보장 불가
2. Decoder의 causal receptive field가 chunk 경계에서 truncated
3. Mel-spectrogram의 frame-level discontinuity가 latent로 전파

**해결 전략**:

```
Strategy A: Causal Receptive Field Bridge
  ┌─────────────────────────────────────────┐
  │  Chunk[i-1] 마지막 N개 sample을         │
  │  Chunk[i]의 causal padding으로 활용     │
  │                                         │
  │  Decoder가 Chunk[i] 시작부를 생성할 때  │
  │  Chunk[i-1]의 마지막 hidden state로     │
  │  초기화 → natural continuation          │
  └─────────────────────────────────────────┘

Strategy B: Hidden State Carry-over
  ┌─────────────────────────────────────────┐
  │  Decoder의 마지막 layer hidden state를  │
  │  다음 chunk의 초기 state로 전달         │
  │                                         │
  │  state[i] = Decoder(chunk[i], state[i-1])│
  │                                         │
  │  → Recurrent-like continuity without    │
  │    explicit overlap                     │
  └─────────────────────────────────────────┘

Strategy C: Boundary-aware Training
  - Phase 4에서 chunk 경계 5-sample window에
    continuity loss 가중치 10배 적용
  - 경계 artifact가 발생하면 큰 패널티

Strategy D: Soft Cross-fade (Fallback)
  - 극단적 artifact 발생 시 5-sample linear cross-fade만 적용
  - 80ms chunk에서 5-sample (0.2ms)은 실질적 SOLA 아님
```

**권장**: Strategy B (Hidden State Carry-over) + Strategy C (Boundary-aware Training) 우선 적용. 실패 시 Strategy D를 minimal fallback으로.

### 9.3 Risk 2: Depth-wise 품질 Degradation

**문제**: Shallow depth (D0, 4 blocks) 출력이 현저히 낮은 품질을 보일 경우, 초기 TTFB 달성은 가능하나 사용자 경험이 저하됨.

**영향**: TTFB 48ms에 출력되는 첫 chunk (D0)의 speaker similarity가 0.85 미만일 수 있음.

**해결**:
```
1. Aggressive distillation:
   - D0 student가 teacher(12 blocks)의 출력을 최대한 모방하도록
     distillation temperature 조정 (τ=2.0 → 부드러운 target)
   - D0에 speaker embedding을 더 강하게 injection (AdaLN scale 확대)

2. Progressive transition:
   - D0 → D1 → D2 전환 시 cross-fade가 아닌
     Decoder hidden state의 smooth transition으로 자연스럽게 연결
   - D0 첫 chunk만 coarse하고, 80ms 이내에 D2 품질 도달

3. User perception masking:
   - 발화 초반 80ms는 주로 무성음/묵음인 경우가 많음
   - 초기 chunk 품질 저하가 실제 체감에 미치는 영향 제한적
   - AB 테스트로 체감 품질 검증 필수
```

### 9.4 Risk 3: Vocos BWE Causal 변환

**문제**: 원본 Vocos BWE는 non-causal ConvNeXt를 사용. Causal 변환 시 고주파 대역 복원 품질 저하 가능.

**해결**:
```
1. Non-causal BWE as baseline:
   - Causal BWE와 non-causal BWE의 PESQ/LSD 비교
   - 차이가 MOS 0.1 이내면 causal 채택
   - 차이가 크면 non-causal BWE 유지 + 추가 12.5ms lookahead 감수

2. Hybrid approach:
   - BWE ConvNeXt의 초기 2개 layer는 causal
   - 마지막 4개 layer는 5ms lookahead 허용 (center padding)
   - 전체 lookahead는 5ms 증가하나 품질 보존

3. Distillation from non-causal to causal:
   - Non-causal BWE teacher → causal BWE student distillation
   - High-frequency band (6-12kHz)에 loss 가중치
```

### 9.5 Risk 4: CPU-only 환경 RTF 미달

**문제**: 모바일/엣지 디바이스에서 CPU-only 추론 시 RTF < 0.1 달성 어려움.

**예상**: Raspberry Pi 5에서 RTF ~0.21로 목표 초과.

**해결**:
```
1. Model compression:
   - INT8 양자화 (1.2-1.5× speedup)
   - Depth-wise pruning: D2 사용률 낮은 layer 제거
   - Block sharing: Encoder/Converter 유사 블록 weight sharing

2. Adaptive depth:
   - 런타임 RTF 모니터링
   - RTF > 0.08 진입 시 depth level 자동 하향
   - 항상 D0+D1만 사용, D2 skip → RTF 40% 감소

3. Frame skipping:
   - 연속 chunk 처리 시 2번째 chunk마다 Encoder skip
   - 이전 chunk의 latent를 linear interpolation으로 추정
   - RTF 50% 감소, 품질 저하 0.05 MOS

4. Minimal configuration:
   - "StreamVC-Mini": Encoder 6 blocks, Converter 6 blocks,
     Decoder 2× upsampling, BWE 3 blocks
   - Total ~15M params, RTF < 0.05 (CPU)
```

### 9.6 Risk 5: Speaker Similarity 저하

**문제**: Content Encoder가 content-disentanglement에 실패하여 speaker 정보가 converter에 의해 완전히 변환되지 않음.

**해결**:
```
1. Information bottleneck:
   - Content Encoder 출력에 Gaussian noise injection (σ=0.1)
   - Content Encoder에 speaker classification adversary 추가
     (gradient reversal layer)

2. Multi-scale speaker conditioning:
   - Converter 매 block마다 speaker embedding 재주입
   - AdaLN + cross-attention 조합

3. Evaluation protocol:
   - Speaker verification (ECAPA-TDNN cosine similarity)
   - Same-different speaker discrimination test
   - 목표: cosine similarity ≥ 0.85
```

### 9.7 Risk 6: Real-time Audio Drop-out

**문제**: Async FIFO 파이프라인에서 under-run 발생 시 audio drop-out (무음 구간).

**해결**:
```
1. Buffer sizing:
   - Pre-fill buffer: 80ms (초기 지연 최소화)
   - Steady-state buffer: 160ms (2 chunks) 안전 마진
   - Max buffer: 480ms (6 chunks) → overflow 시 oldest drop

2. Predictive scheduling:
   - 최근 10개 chunk의 RTF 추세로 다음 chunk RTF 예측
   - RTF 상승 추세 감지 → depth level 사전 하향

3. Graceful degradation:
   - Under-run → 마지막 chunk 반복 (pitch-preserved)
   - Over-run → oldest chunk drop, 로그 기록
   - 3회 연속 under-run → depth level 강제 하향
```

### 9.8 Risk 7: Training Data

**문제**: 고품질 multi-speaker 한국어 음성 데이터 부족.

**해결**:
```
1. AI Hub 데이터셋 적극 활용:
   - 한국어 자유대화 음성 (2000시간+)
   - 감정 표현 음성 합성 데이터

2. Self-supervised pre-training:
   - 대량의 unlabeled 한국어 음성으로 SSL pre-training
   - 소량의 labeled multi-speaker 데이터로 fine-tuning

3. Cross-lingual transfer:
   - 영어 VCTK/LibriTTS로 학습된 speaker conversion 능력이
     한국어에도 전이되는지 검증
   - Content encoder를 multilingual로 학습
```

### 9.9 Risk 8: Lookahead Buffer 부족

**문제**: Mel-spectrogram의 12.5ms lookahead만으로 충분한 phonetic context 확보가 어려울 수 있음.

**영향**: Content Encoder가 onset detection, coarticulation 처리에 실패할 가능성.

**해결**:
```
1. Content Encoder에 dilated causal convolution 적용:
   - dilation rates = [1, 2, 4, 8, 16, 32]
   - 32 × 7 kernel width = 224 samples ≈ 9.3ms effective receptive field
   - 추가 lookahead 없이 context 확장

2. Streaming-friendly context buffer:
   - 과거 2개 chunk의 Mel frame을 context로 캐싱
   - 현재 chunk + past context로 encoding
   - Lookahead 없이 240ms context 확보 (160ms past + 80ms current)

3. Parallel context encoding:
   - 과거 context는 pre-computed key-value cache로 관리
   - 현재 chunk만 forward pass → 연산량 증가 최소화
```

### 9.10 추가 Risk: Depth-wise Encoder-Converter 불일치

**문제**: Encoder L0 출력 시점과 Converter D0 실행 시점 사이에 latency가 발생하여, 실제 D0가 L0가 아닌 L1/L2 입력을 기다리게 될 수 있음.

**해결**: Async FIFO에서 depth level tag를 확인하여, 각 Converter depth는 해당 Encoder depth의 출력이 준비된 경우에만 consume. 더 깊은 Encoder 출력이 먼저 도착한 경우에도 D0가 L0를 기다리도록 protocol 정의.

---

## 10. 참고문헌

1. **Depth-wise Streaming TTS** (arXiv:2604.12438)
   - 32-layer RVQ depth축 순차 디코딩
   - TTFB 48.99ms, RTF 0.0033
   - StreamVC: Content Encoder & Converter depth-wise 적용

2. **Audio Interaction Model** (arXiv:2606.05121)
   - Streaming perceive-decide-respond LLM
   - Async FIFO inference 파이프라인
   - StreamVC: Async FIFO pipeline architecture

3. **Vocos BWE** (arXiv:2603.07285)
   - ConvNeXt 기반 bandwidth extension
   - RTF 0.0001 (A100), 0.0053 (CPU)
   - StreamVC: 12kHz→24kHz BWE module

4. **btrv3lite**
   - CausalLatentConverter (ConvNeXt-1D, causal)
   - MioCodec streaming wrapper
   - StreamVC: 직접 재활용 및 개선 베이스라인

5. **ConvNeXt** (Liu et al., 2022)
   - Modernized CNN backbone
   - Depth-wise conv, inverted bottleneck
   - StreamVC: 모든 encoder/converter 블록 백본

6. **HiFi-GAN** (Kong et al., 2020)
   - Multi-scale discriminator, multi-period discriminator
   - StreamVC: Base decoder 구조 기반

7. **ECAPA-TDNN** (Desplanques et al., 2020)
   - Speaker embedding extractor
   - StreamVC: Speaker encoder (frozen)

---

## 부록 A: 구현 로드맵

```
Month 1-2: Phase 1 - Content Encoder pre-training
Month 2-3: Phase 2 - Converter + Decoder joint training
Month 3-4: Phase 3 - Depth-wise distillation
Month 4-5: Phase 4 - Streaming adaptation
Month 5-6: Phase 5 - Vocos BWE fine-tuning
Month 6-7: Integration & Async FIFO pipeline
Month 7-8: Optimization (ONNX, INT8, mobile port)
Month 8-9: Production deployment & monitoring
```

## 부록 B: 평가 지표 상세

| 지표 | 측정 방법 | 목표 | 허용 범위 |
|------|---------|------|----------|
| TTFB | Loopback impulse measurement | < 50ms | < 60ms (P95) |
| RTF (GPU) | CUDA event timing / 80ms | < 0.01 | < 0.02 |
| RTF (CPU) | perf_counter / 80ms | < 0.1 | < 0.15 |
| Speaker Similarity | ECAPA-TDNN cosine sim | ≥ 0.85 | ≥ 0.80 |
| WER | Whisper/Conformer ASR | < 3% | < 5% |
| MOS | Crowd-sourced listening test | ≥ 4.0 | ≥ 3.5 |
| MCD (Mel Cepstral Distortion) | Dynamic Time Warping | < 5.0 | < 6.0 |
| PESQ | ITU-T P.862.2 | ≥ 3.0 | ≥ 2.5 |
| Chunk Boundary SNR | Boundary ±5 samples vs center | > 40dB | > 35dB |

---

## 부록 C: 용어 정의

| 약어 | Full Name | 설명 |
|------|-----------|------|
| TTFB | Time-To-First-Byte | 첫 오디오 프레임 출력까지의 지연 시간 |
| RTF | Real-Time Factor | 처리 시간 / 오디오 길이 비율 |
| BWE | Bandwidth Extension | 협대역 → 광대역 주파수 확장 |
| FIFO | First-In-First-Out | 선입선출 큐 |
| AdaLN | Adaptive Layer Normalization | 조건부 레이어 정규화 |
| SSL | Self-Supervised Learning | 자기지도학습 |
| SOLA | Synchronized Overlap-and-Add | 크로스페이드 기반 시간축 조정 |
| MOS | Mean Opinion Score | 주관적 음질 평가 점수 |
| MCD | Mel Cepstral Distortion | 스펙트럼 왜곡 측정 지표 |
| WER | Word Error Rate | 음성 인식 오류율 |

---

> **문서 버전**: v1.0  
> **작성일**: 2026-06-06  
> **아키텍처 각도**: StreamVC — Streaming-Optimized + Depth-wise Decoding  
> **목표**: 24kHz+ 초저지연 실시간 음성 변환 파이프라인

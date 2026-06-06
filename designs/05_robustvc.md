# RobustVC: Denoising-First + Style Control + FiLM 기반 실시간 음성 변환 파이프라인

> **btrv5 아키텍처 설계 — 각도 "RobustVC"**
>
> SB-RF 1-step Denoising → WavLM+FiLM Speaker Conditioning → DUET/GLASS Style Steering → Causal VC
> 24kHz+ Real-time Voice Conversion, Noise-robust & Style-controllable

---

## 목차

1. [개요 및 설계 철학](#1-개요-및-설계-철학)
2. [논문 인사이트 통합](#2-논문-인사이트-통합)
3. [전체 아키텍처 다이어그램](#3-전체-아키텍처-다이어그램)
4. [컴포넌트 상세 설계](#4-컴포넌트-상세-설계)
   - 4.1 [SB-RF Denoising Preprocessor](#41-sb-rf-denoising-preprocessor)
   - 4.2 [WavLM Speaker Encoder + FiLM Gating](#42-wavlm-speaker-encoder--film-gating)
   - 4.3 [DUET-Style Hidden Steering](#43-duet-style-hidden-steering)
   - 4.4 [GLASS LoRA Style Arithmetic](#44-glass-lora-style-arithmetic)
   - 4.5 [CausalLatentConverter (FiLM 확장)](#45-causallatentconverter-film-확장)
   - 4.6 [MioCodec Decoder + UniPASE Enhancement](#46-miocodec-decoder--unipase-enhancement)
5. [차원 / 파라미터 명세](#5-차원--파라미터-명세)
6. [Training Pipeline](#6-training-pipeline)
7. [실시간 추론 전략](#7-실시간-추론-전략)
8. [평가 메트릭](#8-평가-메트릭)
9. [btrvrc0 FiLM 확장 매핑](#9-btrvrc0-film-확장-매핑)
10. [리스크 및 대안](#10-리스크-및-대안)

---

## 1. 개요 및 설계 철학

### 1.1 RobustVC란?

RobustVC는 **"Denoising-First"** 원칙 아래 SB-RF로 입력 노이즈를 제거한 후, **WavLM+FiLM**으로 강인한 화자 조건화를 수행하고, **DUET hidden steering + GLASS LoRA**로 감정/스타일을 제어하는 차세대 Voice Conversion 파이프라인이다.

### 1.2 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **Denoising-First** | VC 이전에 SB-RF 1-step으로 입력 클린업. 노이즈에 강인한 VC 보장 |
| **Human-Aligned Speaker Conditioning** | WavLM 기반 speaker encoder + FiLM gating으로 인간 지각과 정렬된 화자 유사도 |
| **Frozen Hidden Steering** | DUET 방식: VC converter의 중간 hidden state를 frozen style vector로 조정 |
| **Composable LoRA Style** | GLASS 방식: LoRA weight arithmetic으로 다중 스타일 조합 (감정+강도+운율) |
| **Identity-Init Everywhere** | SB-RF, FiLM, LoRA, Converter 모두 zero/identity init으로 안정적 학습 |
| **실시간 스트리밍** | Causal 연산, SB-RF 1-step → VC → style 순차 파이프라인, RTF < 0.8 |

### 1.3 목표 사양

| 지표 | 목표값 | 비고 |
|------|--------|------|
| Sample Rate | 44,100 Hz | MioCodec native, 24kHz+ 충족 |
| Latent Framerate | 25 Hz | hop = 1764 samples |
| Real-Time Factor (RTF) | < 0.6 (GPU), < 0.9 (CPU) | SB-RF 1-step 포함 |
| Speaker Similarity | > 0.85 (WavLM cosine) | Speaker Perception 기준 |
| WER | < 3% (Whisper large-v3) | noisy input 포함 |
| MOS-Naturalness | > 4.0 | |
| Denoising PESQ gain | > 0.3 (vs. no denoising) | SB-RF on/off 비교 |
| Style Control Fidelity | > 0.80 (emotion classifier acc) | DUET+GLASS |

---

## 2. 논문 인사이트 통합

### 2.1 SB-RF (arXiv:2606.05575) — 1-step Schrödinger Bridge Rectified Flow

> **"1-step rectified flow enhancement, Schrödinger Bridge prior"**

| 인사이트 | RobustVC 적용 |
|----------|---------------|
| **1-step denoising**: ReFlow로 학습된 single-step enhancement | 입력단 SB-RF preprocessor. Raw waveform → denoised waveform, 1 NFE |
| **Schrödinger Bridge prior**: noisy↔clean 사이의 최적 transport | 학습 시 SB prior로 noisy-clean pair 매핑. 적은 데이터로 일반화 |
| **Rectified Flow**: 직선 경로로 단순화된 ODE | `x_t = (1-t)·x_noisy + t·x_clean`, velocity = x_clean - x_noisy |

### 2.2 DUET (arXiv:2606.00066) — Frozen Hidden State Emotion Steering

> **"Frozen hidden state steering + dual-space emotion control"**

| 인사이트 | RobustVC 적용 |
|----------|---------------|
| **Frozen steering**: 사전학습된 모델의 hidden state를 freeze하고 steering vector만 학습 | CausalLatentConverter frozen, 중간 hidden에 style steering vector 주입 |
| **Dual-space control**: valence-arousal space + discrete emotion label | 2D V-A 연속 제어 + 8-class discrete emotion |
| **Hidden injection point**: converter block 4~6 사이 hidden state에 주입 | n_blocks=10 기준 block 4, 7에 steering vector add |

### 2.3 GLASS (arXiv:2606.05889) — LoRA + GRPO Composable Style Control

> **"LoRA arithmetic + GRPO optimization for composable acoustic style"**

| 인사이트 | RobustVC 적용 |
|----------|---------------|
| **LoRA weight arithmetic**: `W_style = W_base + α·ΔW_emotion + β·ΔW_intensity` | Converter의 linear layer들에 LoRA adapter. 추론 시 weight merge로 zero-latency |
| **GRPO training**: Group Relative Policy Optimization으로 style reward 최적화 | Style adapter를 GRPO로 학습. Discriminator-free style fidelity |
| **Composable**: 다중 LoRA를 가중합으로 조합 | `α·ΔW_happy + β·ΔW_loud` → blended style |

### 2.4 FiLM ASR (arXiv:2606.06211) — Gating + Identity Init

> **"FiLM gating mechanism + identity initialization, 1.6% param overhead"**

| 인사이트 | RobustVC 적용 |
|----------|---------------|
| **Gating mechanism**: FiLM layer를 gate(gamma=1, beta=0) init으로 identity 보존 | Speaker conditioner에 FiLM gating 적용. `γ=1+Δγ, β=Δβ` with zero init |
| **1.6% param**: 전체 파라미터의 1.6%만으로 효과적 conditioning | RobustVC conditioner: 전체 60M 중 ~1M (1.7%) |
| **Feature-wise modulation**: 채널별 scale/shift | WavLM embedding → FiLM scale/shift → converter hidden modulation |

### 2.5 UniPASE (arXiv:2604.14606) — WavLM Prior Universal Enhancement

> **"WavLM prior for universal speech enhancement"**

| 인사이트 | RobustVC 적용 |
|----------|---------------|
| **WavLM prior**: frozen WavLM을 feature extractor로 사용하여 universal enhancement | SB-RF denoiser의 condition으로 WavLM feature 사용 |
| **Multi-resolution**: 다양한 time resolution에서 feature 추출 | WavLM layer 6, 12, 18, 24의 hidden state를 multi-scale condition으로 |
| **Universal**: speaker, noise, channel 불문 일관된 enhancement | SB-RF가 unseen noise에도 강인하도록 UniPASE-style multi-condition 학습 |

### 2.6 Speaker Perception (arXiv:2606.05739) — WavLM Human-Aligned Similarity

> **"WavLM encoder for human-aligned speaker similarity"**

| 인사이트 | RobustVC 적용 |
|----------|---------------|
| **WavLM > ECAPA**: 인간 지각과의 상관관계에서 WavLM이 ECAPA 능가 | Speaker encoder를 기존 ECAPA에서 WavLM+FiLM으로 교체 |
| **Layer-weighted pooling**: WavLM layer별 가중치를 두고 weighted-sum pooling | Learnable layer weights + attentive pooling → speaker embedding |
| **Human-aligned metric**: 평가 지표도 WavLM cosine similarity 사용 | SECS → WavLM-SECS (Speaker Perception 기준) |

---

## 3. 전체 아키텍처 다이어그램

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           RobustVC Inference Pipeline                         │
│                        (Denoising-First + Style Control)                      │
└──────────────────────────────────────────────────────────────────────────────┘

   Noisy Source Audio (44.1kHz, mono)
          │
          ▼
   ┌──────────────────────────────────────┐
   │  ① SB-RF Denoising Preprocessor     │  ← UniPASE-style WavLM condition
   │     (1-step Rectified Flow)          │     Frozen WavLM Large (layer 6,12,18,24)
   │     x_clean = x_noisy + v_θ(x,t=0)   │     ~8M params, 1 NFE, ~15ms latency
   └──────────────────────────────────────┘
          │
          ▼  Clean Audio
   ┌──────────────────────────────────────┐
   │  ② MioCodec Encoder (Frozen)        │
   │     z_src ∈ ℝ^(T×768), @25Hz         │
   │     g_src ∈ ℝ^128 (utterance global)  │
   └──────────────────────────────────────┘
          │  z_src               g_src (for reference)
          ▼                       │
   ┌──────────────────────────────────────┐
   │  ③ WavLM Speaker Encoder + FiLM     │  ← Speaker Perception (arXiv:2606.05739)
   │     Reference audio → WavLM Large    │     Frozen WavLM backbone
   │     Layer-weighted attentive pool    │     Learnable: layer weights + FiLM head
   │     → spk_emb ∈ ℝ^256                │     ~1M trainable params
   │     → FiLM (γ, β) per converter block │
   └──────────────────────────────────────┘
          │  spk_emb, FiLM params
          ▼
   ┌──────────────────────────────────────┐
   │  ④ CausalLatentConverter (btrvrc0)  │  ← FiLM ASR gating 확장
   │     10× ConvNeXt-1D blocks          │     Base: btrvrc0 CausalLatentConverter
   │     FiLM conditioning per block      │     + FiLM gating (γ=1+Δγ, β=Δβ)
   │     ├─ Block 0..3: FiLM only         │
   │     ├─ Block 4: + DUET hidden steer  │  ← DUET (arXiv:2606.00066)
   │     ├─ Block 5..6: FiLM only         │
   │     ├─ Block 7: + DUET hidden steer  │
   │     └─ Block 8..9: FiLM only         │
   │     Identity-init output gate        │
   │     z_out = z_src + δ                 │
   └──────────────────────────────────────┘
          │  z_out
          ▼
   ┌──────────────────────────────────────┐
   │  ⑤ GLASS LoRA Style Adapter         │  ← GLASS (arXiv:2606.05889)
   │     LoRA on converter linear layers  │     Rank=8, α=16
   │     W' = W + Σ α_i · ΔW_i            │     Composable: emotion + intensity
   │     Zero-init → identity at t=0      │
   └──────────────────────────────────────┘
          │  z_styled
          ▼
   ┌──────────────────────────────────────┐
   │  ⑥ MioCodec Decoder (Frozen)        │  + UniPASE post-enhancement (opt)
   │     z_styled + g_tgt → waveform      │     Post-decoder: light 1-step enhancement
   │     44.1kHz output                   │
   └──────────────────────────────────────┘
          │
          ▼
   Clean, Style-controlled Output Audio


   ┌──────────────────────────────────────────────────────────────┐
   │                    Training-Only Paths                        │
   │                                                              │
   │  SB-RF Learner:  x_noisy → v_θ → x_clean (CFM loss)          │
   │  DUET Trainer:   frozen converter + learned steer vectors    │
   │  GLASS Trainer:  GRPO on LoRA ΔW with style reward           │
   │  FiLM Trainer:   WavLM layer weights + FiLM γ,β projections  │
   │  Converter:      Phase 2 (identity) → Phase 3 (all-spk)       │
   └──────────────────────────────────────────────────────────────┘
```

---

## 4. 컴포넌트 상세 설계

### 4.1 SB-RF Denoising Preprocessor

**목적**: VC 진입 전 입력 오디오의 노이즈 제거. 1-step inference로 실시간성 보장.

```
Input:  x_noisy ∈ ℝ^T        (raw waveform, 44.1kHz mono, arbitrary length)
Output: x_clean ∈ ℝ^T        (denoised waveform, same length)

Architecture:
  ┌─ WavLM Large (frozen, 317M) ─┐
  │  layer 6,12,18,24 hidden     │  → multi-scale condition c ∈ ℝ^(T'×4096)
  └──────────────────────────────┘
              │
              ▼  c (downsampled to match time resolution)
  ┌──────────────────────────────────────┐
  │  Denoiser Network (ConvNeXt-1D, ~8M) │
  │                                      │
  │  x_noisy → STFT(mag+phase)           │
  │         → ConvNeXt Encoder (4 blocks) │
  │         → FiLM(c) modulation          │
  │         → ConvNeXt Decoder (4 blocks) │
  │         → ISTFT → x_clean             │
  └──────────────────────────────────────┘

Training:
  - CFM loss: ℒ = 𝔼_t‖v_θ(x_t, t, c) - (x_clean - x_noisy)‖²
  - x_t = (1-t)·x_noisy + t·x_clean
  - Rectified Flow: 직선 경로 학습 후 1-step inference 가능
  - Noise types: Gaussian, environmental, room reverb, codec artifacts
  - ReFlow distillation: teacher (100-step) → student (1-step)

Latency: ~15ms (1 NFE, causal ConvNeXt on 80ms chunks)
Params: ~8M (denoiser only, WavLM shared with speaker encoder)
```

**통합 방식**:
- SB-RF는 VC pipeline의 **첫 단계**로 고정 배치
- WavLM은 SB-RF conditioner와 speaker encoder가 **공유** (메모리 절약)
- Denoising on/off 선택 가능 — clean input에서는 bypass

### 4.2 WavLM Speaker Encoder + FiLM Gating

**목적**: 인간 지각과 정렬된 화자 임베딩 생성 + FiLM 기반 converter conditioning

```
Input:  x_ref ∈ ℝ^T_ref     (reference audio, 3~10초)
Output: spk_emb ∈ ℝ^256     (speaker embedding)
        FiLM_params = {(γ_i, β_i)} for i=0..9 (per converter block)

Architecture:
  ┌─────────────────────────────────────────────┐
  │  WavLM Large (frozen, shared with SB-RF)    │
  │                                              │
  │  x_ref → WavLM → {h_6, h_12, h_18, h_24}     │
  │           (layer-wise hidden states)          │
  └─────────────────────────────────────────────┘
              │
              ▼  Learnable layer weights α ∈ ℝ^4 (softmax)
  ┌──────────────────────────────────────────────┐
  │  Weighted Sum: h = Σ α_k · h_k                │
  │  Attentive Pooling (over time)                │
  │  → spk_emb ∈ ℝ^256                            │
  └──────────────────────────────────────────────┘
              │
              ├──→ SpeakerSimilarity loss target (WavLM cosine)
              │
              ▼
  ┌──────────────────────────────────────────────┐
  │  FiLM Head (per converter block i=0..9)       │  ← FiLM ASR gating
  │                                              │
  │  γ_i = 1 + proj_γ_i(spk_emb)  ← zero init   │
  │  β_i = 0 + proj_β_i(spk_emb)  ← zero init   │
  │                                              │
  │  proj: Linear(256→192) for each block         │
  └──────────────────────────────────────────────┘

Total trainable: ~1M params (layer weights 4 + pool attn + 10×2×Linear(256→192))
```

**FiLM ASR 인사이트 적용**:
- Gating mechanism: `γ = 1 + Δγ, β = 0 + Δβ` — identity init으로 기존 converter 출력 보존
- per-block modulation: 각 ConvNeXt block마다 독립적인 FiLM 파라미터
- 전체 파라미터의 1.6%만 추가 (60M 중 ~1M)

### 4.3 DUET-Style Hidden Steering

**목적**: Frozen converter 중간 hidden state에 style vector를 주입하여 감정/스타일 제어

```
Style Space (Dual-Space):
  ┌─────────────────────────────────────┐
  │  Continuous:  valence, arousal ∈ ℝ² │  ← 2D V-A space
  │  Discrete:    emotion ∈ {neutral,    │
  │               happy, sad, angry,     │
  │               surprised, fearful,    │
  │               disgusted, calm} 8-class│
  └─────────────────────────────────────┘
              │
              ▼  Learnable embedding lookup
  ┌──────────────────────────────────────┐
  │  Style Embedding: s ∈ ℝ^192          │
  │  V-A → MLP(2→192) → s_va            │
  │  Discrete → Embedding(8, 192) → s_d  │
  │  s = s_va + s_d  (combined)           │
  └──────────────────────────────────────┘
              │
              ▼  Steering vector injection
  ┌──────────────────────────────────────┐
  │  Converter Hidden State:              │
  │  h_block4 ← h_block4 + α·W_steer·s   │  ← DUET injection point 1
  │  h_block7 ← h_block7 + β·W_steer·s   │  ← DUET injection point 2
  │                                      │
  │  W_steer ∈ ℝ^(192×192), zero-init    │
  └──────────────────────────────────────┘

Training:
  - Converter frozen (btrvrc0 checkpoint)
  - Only s_va embedding, W_steer, and MLPs trained
  - Emotion classification loss on z_out
  - V-A regression loss on z_out
  - Cycle consistency: style(s_out) ≈ style(s_in)

Inference:
  - V-A slider (continuous) + emotion select (discrete)
  - Zero steering (α=β=0) → original voice
  - Positive steering → target style
```

### 4.4 GLASS LoRA Style Arithmetic

**목적**: LoRA weight arithmetic으로 다중 스타일을 composable하게 조합

```
Base Model: CausalLatentConverter (frozen)

LoRA Adapters (applied to converter's Linear layers):
  ┌──────────────────────────────────────────────────┐
  │  For each Linear(dim_in, dim_out):                │
  │    W' = W + Σ_i α_i · (B_i @ A_i)                │
  │                                                   │
  │  A_i ∈ ℝ^(r × dim_in)   (rank r=8)                │
  │  B_i ∈ ℝ^(dim_out × r)  (zero-init)               │
  │                                                   │
  │  Style library:                                    │
  │    ΔW_happy, ΔW_sad, ΔW_angry, ΔW_calm            │
  │    ΔW_loud, ΔW_soft, ΔW_fast, ΔW_slow             │
  │    ΔW_breathy, ΔW_whisper, ΔW_chesty              │
  └──────────────────────────────────────────────────┘

Inference:
  - Style interpolation: α·ΔW_happy + β·ΔW_loud + γ·ΔW_breathy
  - Weight merge: inference 직전에 한 번 merge → zero runtime overhead
  - Style slider: α continuous, -1.0 ~ +1.0

Training (GRPO):
  - Group Relative Policy Optimization
  - Reward model: WavLM-based style classifier + MOS predictor
  - 각 style adapter를 독립적 GRPO로 학습
  - Negative style (α=-1)도 지원: anti-style

Composable Arithmetic:
  happy voice:    W' = W + 0.7·ΔW_happy
  sad+loud:       W' = W + 0.5·ΔW_sad + 0.8·ΔW_loud
  whisper+calm:   W' = W + 0.9·ΔW_whisper + 0.3·ΔW_calm
```

**GRPO 세부사항**:
- Group size: 4 outputs per prompt → 상대 비교로 reward shaping
- KL penalty: base model 대비 divergence 제한 (β=1e-3)
- Reward signal: WavLM style classifier confidence + speaker similarity penalty
- 각 LoRA ~0.3M params, 총 10개 adapter → ~3M 추가

### 4.5 CausalLatentConverter (FiLM 확장)

**btrvrc0 converter 기반, FiLM 확장**

```
Config:
  content_dim: 768      (MioCodec latent)
  hidden_dim: 192       (ConvNeXt internal)
  speaker_dim: 256      (WavLM speaker emb, 확장)
  cond_dim: 128         (FiLM conditioning projection)
  n_blocks: 10
  kernel_size: 5
  dilations: (1, 2, 4, 8, 16, 1, 2, 4, 8, 16)
  mlp_expansion: 4

Block 구조 (ConvNeXt-1D + FiLM):
  ┌───────────────────────────────────────────┐
  │  CausalDepthwiseConv1d(k=5, dil=d)        │
  │  ChannelLayerNorm                          │
  │  FiLM(γ_i, β_i)  ← WavLM speaker condition│
  │    γ_i = 1 + proj_γ_i(spk_emb)  ← FiLM ASR│
  │    β_i = 0 + proj_β_i(spk_emb)  ←         │
  │  GELU                                      │
  │  Linear(hidden → hidden*4)                 │
  │  Linear(hidden*4 → hidden)                 │
  │  Residual connection                       │
  └───────────────────────────────────────────┘

  Block 4, 7: + DUET hidden steering injection
  All Linear layers: GLASS LoRA 적용 가능

Identity at init:
  - FiLM: γ=1, β=0 (zero Δ)
  - DUET: W_steer=0
  - GLASS LoRA: B=0
  - Output gate: 0 → z_out = z_src

btrvrc0 diff:
  - speaker_dim: 128 → 256 (ECAPA → WavLM)
  - AdaLN-Zero → FiLM gating (FiLM ASR 방식)
  - Cross-attention 제거 (simpler, faster)
  - n_blocks: 10 (v2 spec 유지)
  - DUET injection points: block 4, 7
  - GLASS LoRA ready: 모든 Linear layer에 adapter slot
```

### 4.6 MioCodec Decoder + UniPASE Enhancement

**MioCodec Decoder** (frozen, from btrv3lite):
- Input: z_styled (B, T, 768) + g_tgt (B, 128)
- Output: 44.1kHz waveform
- 기존 btrv3lite decoder 그대로 사용

**UniPASE Post-Enhancement** (optional, 경량):
```
  Decoder output → Light Enhancer (2 ConvNeXt blocks, ~1M)
                  → WavLM prior condition
                  → Final output
```
- Decoder 출력의 잔여 아티팩트 제거
- WavLM multi-layer feature를 condition으로
- 추론 시 bypass 가능 (quality vs speed trade-off)

---

## 5. 차원 / 파라미터 명세

### 5.1 Tensor Flow

| Stage | Input | Output | Shape | Rate |
|-------|-------|--------|-------|------|
| SB-RF | raw waveform | clean waveform | (B, T_audio) | 44.1kHz |
| MioCodec Enc | clean waveform | z_src, g_src | (B, T_lat, 768), (B, 128) | 25Hz |
| WavLM Speaker | reference audio | spk_emb, {γ_i,β_i} | (B, 256), 10×(B, 192) | per-utt |
| DUET Style | style params | steer vectors | 2× (B, 192) | per-utt |
| GLASS LoRA | style weights | merged ΔW | — (merged) | — |
| Converter | z_src + conds | z_out | (B, T_lat, 768) | 25Hz |
| MioCodec Dec | z_out + g_tgt | waveform | (B, T_audio) | 44.1kHz |

### 5.2 Parameter Count

| Module | Trainable | Frozen | Total |
|--------|-----------|--------|-------|
| SB-RF Denoiser | 8M | — | 8M |
| WavLM Large (shared) | — | 317M | 317M (shared) |
| WavLM FiLM Head | 1M | — | 1M |
| MioCodec Encoder | — | 25M | 25M |
| CausalLatentConverter | 5.4M | — | 5.4M |
| DUET Steering | 0.2M | — | 0.2M |
| GLASS LoRA (×10) | 3M | — | 3M |
| MioCodec Decoder | — | 40M | 40M |
| UniPASE Enhancer | 1M | — | 1M |
| **Total Trainable** | **18.6M** | | |
| **Total (w/o shared)** | **18.6M** | **382M** | **~59M** |

> WavLM은 SB-RF와 Speaker Encoder에서 공유. 전체 런타임 메모리는 ~59M (frozen WavLM 제외 시).

### 5.3 MACs / Latency (추론)

| Stage | MACs | Latency (GPU, 80ms chunk) | Latency (CPU) |
|-------|------|---------------------------|---------------|
| SB-RF Denoiser (1-step) | 2.1G | 12ms | 45ms |
| MioCodec Encoder | 0.8G | 5ms | 18ms |
| WavLM Speaker (precomputed) | — | 0ms | 0ms |
| Converter | 1.5G | 8ms | 30ms |
| GLASS LoRA (merged) | 0G | 0ms | 0ms |
| MioCodec Decoder | 2.5G | 14ms | 50ms |
| UniPASE Enhancer | 0.3G | 3ms | 10ms |
| **Total** | **7.2G** | **42ms** | **153ms** |

> RTF @ 80ms chunk: **0.53 (GPU), 1.9 (CPU)** — CPU에서는 SB-RF 또는 UniPASE bypass 필요.
> SB-RF bypass 시 CPU RTF: 1.35. UniPASE bypass + SB-RF bypass: CPU RTF 1.23.
> 실시간 CPU 타겟 달성을 위해 양자화(INT8) 또는 SB-RF distillation(0.5-step) 고려.

---

## 6. Training Pipeline

### Phase 1: SB-RF Denoiser Pretraining (독립)

```
Data: Noisy-clean pair dataset (DNS Challenge, VCTK-DEMAND, 자체 합성)
      x_noisy = clean + noise @ random SNR (0~20dB)

Stage 1.1: Rectified Flow 학습 (teacher, 100-step)
  - CFM loss: 𝔼‖v_θ(x_t,t,c) - (x_clean-x_noisy)‖²
  - WavLM condition: frozen WavLM multi-layer features
  - 200K steps, batch 32, A100 × 4

Stage 1.2: ReFlow Distillation (student, 1-step)
  - Teacher(100-step) → Student(1-step) distillation
  - MSE + multi-resolution STFT loss
  - 100K steps

Output: sb_rf_denoiser.pt (~8M)
```

### Phase 2: WavLM Speaker Encoder + FiLM Head Training

```
Data: VoxCeleb2, LibriTTS, VoiceBank (multi-speaker)

Stage 2.1: Speaker Embedding 학습
  - WavLM layer weight + attentive pooling 학습
  - Speaker classification loss (AAM-Softmax)
  - WavLM cosine similarity loss (Speaker Perception metric)
  - 50K steps

Stage 2.2: FiLM Head 학습 (with frozen converter)
  - btrvrc0 converter checkpoint freeze
  - FiLM γ,β projections 학습
  - Speaker similarity loss + reconstruction loss
  - 30K steps

Output: wavlm_speaker.pt, film_head.pt
```

### Phase 3: CausalLatentConverter Fine-tuning (FiLM integration)

```
Stage 3.1: Identity Warmup (converter freeze → FiLM only)
  - Same-speaker reconstruction
  - FiLM head가 identity를 벗어나지 않도록 regularization
  - 10K steps

Stage 3.2: Full Converter + FiLM Joint Training
  - Multi-speaker VC with WavLM speaker conditioning
  - Loss: L1 content + multi-resolution STFT + WavLM cosine + F0 consistency
  - Converter unfrozen, FiLM head unfrozen
  - 100K steps, batch 16, A100 × 4

Stage 3.3: Privacy/Anonymization (optional)
  - Speaker adversary classifier + gradient reversal (btrvrc0 방식 유지)
  - 30K steps

Output: converter_film.pt (~5.4M)
```

### Phase 4: DUET Style Steering Training

```
Data: Emotional speech (ESD, CREMA-D, RAVDESS)

Stage 4.1: Discrete Emotion Steering
  - Converter frozen (Phase 3 checkpoint)
  - Style embedding s_d + W_steer 학습
  - Emotion classification loss on z_out
  - 20K steps

Stage 4.2: Continuous V-A Steering
  - V-A embedding s_va + shared W_steer fine-tuning
  - V-A regression loss
  - 10K steps

Output: duet_steering.pt (~0.2M)
```

### Phase 5: GLASS LoRA GRPO Training

```
Data: Emotional speech + style-labeled data

For each style adapter (10 styles × 30K steps):
  - GRPO: 4 outputs per input
  - Reward: style classifier confidence + MOS predictor + speaker sim penalty
  - KL penalty to base model
  - LoRA rank=8, α=16

Output: lora_{style}.pt × 10 (~0.3M each, 3M total)
```

### Phase 6: UniPASE Enhancer (optional)

```
Data: VC output ↔ ground truth pairs
  
Stage 6.1: Light enhancer training
  - MSE + multi-resolution STFT loss
  - WavLM condition (frozen)
  - 30K steps

Output: unipase_enhancer.pt (~1M)
```

---

## 7. 실시간 추론 전략

### 7.1 Pipeline Latency 분석

```
Audio Input (streaming chunks, 80ms = 3528 samples @ 44.1kHz)
  │
  ├─[12ms]── SB-RF Denoiser (1-step, causal ConvNeXt)
  │          80ms chunk → 80ms clean chunk
  │
  ├─[5ms]─── MioCodec Encoder
  │          80ms audio → 2 latent frames (25Hz, 40ms/frame)
  │
  ├─[0ms]─── Speaker Conditioning (precomputed per voicebank switch)
  │          WavLM emb + FiLM params cached
  │
  ├─[8ms]─── CausalLatentConverter + DUET steering
  │          2 frames × 10 blocks → 2 output frames
  │          DUET steer vector add: negligible
  │
  ├─[0ms]─── GLASS LoRA (pre-merged weights per style config)
  │
  ├─[14ms]── MioCodec Decoder
  │          2 latent frames → 80ms audio
  │
  └─[3ms]─── UniPASE Enhancer (optional)
             
Total GPU Latency: 42ms per 80ms chunk → RTF = 0.53
TTFB: ~30ms (first chunk encoding + conversion only, decoder buffered)
```

### 7.2 Optimization Strategies

| Strategy | Latency Reduction | Quality Impact |
|----------|-------------------|----------------|
| SB-RF bypass (clean input) | -12ms | None (clean input) |
| UniPASE bypass | -3ms | Minor artifact increase |
| INT8 Quantization (converter) | -40% MACs | <0.05 MOS drop |
| SB-RF distillation to 0.5-step | -6ms | <0.1 PESQ drop |
| Chunk size 40ms (vs 80ms) | -50% buffer | TTFB 감소, RTF 증가 |
| Weight pre-merge (LoRA) | -0ms (runtime) | None |

### 7.3 Streaming State Machine

```
State: IDLE → WARMING → STREAMING → DRAINING → IDLE

WARMING:  First chunk → SB-RF → Encoder → buffer N frames for causal RF
          (N = receptive_field_frames = ~121 frames ≈ 4.8s context)
          실제로는 1초 warmup으로 충분 (dilated conv의 실효 receptive field)

STREAMING: Chunk-by-chunk processing
           SB-RF(lookahead=0, causal) → Encoder → Converter → Decoder
           Output: audio chunk with <42ms latency

DRAINING:  Last chunk → flush decoder buffer → final audio
```

---

## 8. 평가 메트릭

### 8.1 Primary Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Speaker Similarity | > 0.85 (WavLM cosine) | Speaker Perception metric |
| WER | < 3% | Whisper large-v3 on VC output |
| MOS-Naturalness | > 4.0 | Crowd-sourced (P.808) |
| RTF | < 0.6 GPU | wall-clock / audio duration |
| PESQ (noisy input) | > 3.5 | SB-RF on vs off |

### 8.2 Denoising Evaluation

| Scenario | PESQ (no denoising) | PESQ (SB-RF) | Δ |
|----------|---------------------|--------------|---|
| Clean input | 4.2 | 4.2 | 0.0 |
| SNR 10dB (Gaussian) | 2.8 | 3.6 | +0.8 |
| SNR 5dB (environmental) | 2.1 | 3.2 | +1.1 |
| Room reverb (RT60=0.6s) | 2.5 | 3.4 | +0.9 |

### 8.3 Style Control Evaluation

| Style Dimension | Control Accuracy | Interpolation Smoothness |
|-----------------|------------------|--------------------------|
| Emotion (8-class) | > 80% classifier acc | — |
| Valence (continuous) | > 0.75 correlation | monotonic |
| Arousal (continuous) | > 0.75 correlation | monotonic |
| LoRA blend (α sweep) | > 0.70 correlation | smooth transition |
| Multi-style compose | > 70% dual-classification acc | — |

### 8.4 Ablation Targets

| Ablation | Expected Impact |
|----------|-----------------|
| SB-RF off (noisy input) | WER +15%, Speaker Sim -0.10 |
| FiLM ASR gating → no gating | Speaker Sim -0.05, training instability |
| DUET steering off | No style control (baseline) |
| GLASS LoRA off | Single-style only (DUET works) |
| WavLM → ECAPA | Speaker Sim -0.08 (human alignment) |

---

## 9. btrvrc0 FiLM 확장 매핑

### 9.1 변경 사항 요약

| Component | btrvrc0 (current) | RobustVC (proposed) |
|-----------|-------------------|---------------------|
| Speaker Encoder | ECAPA-TDNN (frozen, 192-dim) | WavLM Large + learnable pool (256-dim) |
| Conditioning | AdaLN-Zero (scale/shift/gate per block) | FiLM Gating (γ=1+Δγ, β=Δβ, per block) |
| Conditioning Dim | 128-dim (bank centroid) | 256-dim (WavLM speaker embedding) |
| Speaker Bank | 256×128 centroid buffer | 유지 + WavLM 256-dim 대응 |
| Converter Backbone | ConvNeXt-1D + Cross-Attention | ConvNeXt-1D only (simpler) |
| Converter Blocks | 10 (dilations 1,2,4,8,16 repeat) | 10 (same) |
| Output Gate | zero-init scalar gate | zero-init per-channel gate vector |
| Additional Modules | — | SB-RF, DUET, GLASS LoRA, UniPASE |

### 9.2 체크포인트 마이그레이션

```
기존 btrvrc0 converter.pt:
  in_proj:  Linear(768→192)         → 유지
  blocks:   10× ConvNeXt + AdaLN    → ConvNeXt 유지, AdaLN → FiLM (재학습)
  out_proj: Linear(192→768)         → 유지
  out_gate: scalar                  → per-channel vector (192-dim)

마이그레이션:
  1. btrvrc0 converter.pt 로드
  2. AdaLN weights → FiLM projection 초기화 (차원 변경)
  3. FiLM ASR identity init: γ=1+proj(spk), β=proj(spk) with proj zero-init
  4. Phase 3.1 identity warmup으로 안정화
  5. Phase 3.2 full training
```

### 9.3 보호된 체크포인트

```
/Users/asill/btrvrc0/models/btrv3lite_v1/
  ├── converter_final.pt           → RobustVC base weights
  ├── conditioner.pt               → speaker bank (마이그레이션)
  └── MioCodec encoder/decoder     → Frozen, 그대로 사용
```

---

## 10. 리스크 및 대안

### 10.1 기술적 리스크

| Risk | Severity | Mitigation |
|------|----------|------------|
| **SB-RF 1-step denoising artifacts** | Medium | ReFlow distillation 단계에서 multi-resolution STFT loss + adversarial loss 추가. Fallback: 2-step으로 증가 |
| **FiLM ASR gating instability** | Low | Zero-init으로 identity 보장. Phase 3.1 warmup에서 gate magnitude 모니터링 |
| **DUET + GLASS 간섭** | Medium | DUET steering → GLASS LoRA 순서로 적용 (hidden → weight). 두 효과가 additive하게 동작하도록 orthogonal initialization |
| **CPU RTF > 1.0** | High | INT8 양자화 필수. SB-RF INT8 distillation (0.5-step). UniPASE bypass. CPU-only 모드에서 SB-RF off 옵션 |
| **WavLM 메모리 (317M)** | Medium | SB-RF + Speaker Encoder가 WavLM 공유. INT8 또는 pruning. Long audio에서는 chunked processing |
| **btrvrc0 converter 호환성** | Low | ConvNeXt backbone 유지. AdaLN→FiLM 변경은 projection layer 재학습으로 대응. Identity init으로 안전 |

### 10.2 대안 아키텍처

| Scenario | Alternative |
|----------|-------------|
| SB-RF 대신 | Conv-TasNet, Demucs 기반 denoiser. But 1-step 보장 어려움 |
| FiLM 대신 | AdaLN-Zero 유지 (btrvrc0 호환성 최대). WavLM condition만 추가 |
| DUET 대신 | StyleToken (GST), GlobalStyleToken. But continuous V-A control 약함 |
| GLASS 대신 | Prompt-based style control. But composability 부족 |
| WavLM 대신 | ECAPA 유지 + HuBERT (경량). But human-alignment 저하 |

### 10.3 단계적 구현 우선순위

```
Priority 1 (Core):  SB-RF Denoiser + WavLM FiLM Converter
  → Denoising-First + Robust Speaker Conditioning 실현
  → btrvrc0 converter 마이그레이션
  → 목표: noisy input에서도 speaker sim > 0.85

Priority 2 (Style): DUET Steering + GLASS LoRA
  → Style controllability 추가
  → Phase 4, 5 training

Priority 3 (Polish): UniPASE Enhancer + INT8 Optimization
  → 최종 품질 향상 + CPU 실시간 최적화
  → Phase 6 training + quantization
```

---

## 부록 A: 핵심 논문 요약

| Paper | arXiv | Key Idea |
|-------|-------|----------|
| SB-RF | 2606.05575 | 1-step Schrödinger Bridge rectified flow for enhancement |
| DUET | 2606.00066 | Frozen hidden state emotion steering, dual V-A + discrete |
| GLASS | 2606.05889 | LoRA arithmetic + GRPO for composable acoustic style |
| FiLM ASR | 2606.06211 | Gating mechanism with identity init, 1.6% param overhead |
| UniPASE | 2604.14606 | WavLM prior for universal speech enhancement |
| Speaker Perception | 2606.05739 | WavLM encoder for human-aligned speaker similarity |

## 부록 B: FiLM ASR vs AdaLN-Zero 비교

| 특성 | AdaLN-Zero (btrvrc0) | FiLM Gating (RobustVC) |
|------|----------------------|------------------------|
| Modulation | scale, shift, gate (3 params) | scale, shift (2 params) |
| Identity condition | scale=0, shift=0, gate=0 | γ=1, β=0 |
| Parameter count | 3×dim per block | 2×dim per block |
| Gate semantic | Controls residual contribution | Built into γ (1+Δγ) |
| Known stability | DiT, StreamVC 검증됨 | FiLM ASR 논문 검증 |
| Choice rationale | — | Simpler, 33% fewer params, identity semantics clearer |

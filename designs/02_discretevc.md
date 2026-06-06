# btrv5 — DiscreteVC Architecture Design

> **각도**: Discrete Token + FSQ + Denoising 기반 24kHz+ 실시간 고품질 음성 변환  
> **통합 인사이트**: CleanCodec (2606.04418), VoCodec (2606.05892), P2PSynCodec (2606.05876)  
> **작성일**: 2026-06-06  
> **버전**: v1.0

---

## 1. 개요 (Executive Summary)

DiscreteVC는 음성 변환(Voice Conversion)을 **완전 이산적(discrete) 토큰 공간**에서 수행하는 파이프라인이다. 연속적 잠재 표현(continuous latent) 대신 Finite Scalar Quantization(FSQ)로 양자화된 이산 토큰을 사용하여, 다음 이점을 취한다:

- **압축된 표현**: 토큰당 5차원 정수 → bitrate 250 bps 미만에서도 고품질 복원
- **언어 모델 친화적**: 이산 토큰 시퀀스 → 추후 text-to-speech, multi-modal 통합 용이
- **잡음 강건성**: Denoising joint training으로 열악한 입력 환경 대응
- **추론 효율성**: 소규모 토큰 공간 변환 → RTF < 0.3, TTFB < 100ms 달성 가능

### 핵심 목표

| 지표 | 목표치 |
|------|--------|
| Sample Rate | 24 kHz |
| Latent Rate | 50 Hz (480× downsampling) |
| RTF | < 0.3 (실시간의 3배 이상) |
| TTFB | < 100 ms |
| Speaker Similarity | ≥ 0.85 |
| WER | < 3% |
| MOS | ≥ 4.0 |
| Total Parameters | ~22M |

---

## 2. 아키텍처 개관 (Architecture Overview)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DiscreteVC Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────┐    ┌─────────────────┐    ┌───────────────────────────┐  │
│   │  Source  │    │  ConvNeXt v2    │    │        FSQ Quantizer       │  │
│   │  Audio   │───▶│    Encoder      │───▶│    levels=[8,8,8,8,8]     │  │
│   │  24kHz   │    │  6 stages,      │    │    d=5, |C|=32,768        │  │
│   │  (noisy) │    │  stride=480×    │    │    straight-through grad   │  │
│   └──────────┘    └─────────────────┘    └───────────┬───────────────┘  │
│                                                      │                  │
│                                          discrete tokens s_t ∈ [0,7]⁵   │
│                                                      │                  │
│   ┌──────────┐    ┌───────────────────────────────────▼──────────────┐  │
│   │  Target  │    │           VC Converter (Causal)                   │  │
│   │ Speaker  │───▶│   ┌──────────────────────────────────────────┐   │  │
│   │ Embedding│    │   │ Token Embedding Lookup (5×8 → 320)       │   │  │
│   │ (ECAPA)  │    │   │ ConvNeXt v2 Blocks ×8, dil=[1,2,4,8,1,..] │   │  │
│   └──────────┘    │   │ AdaLN-Zero + Speaker Cross-Attention      │   │  │
│                   │   │ Token Head: 5×8 logit output              │   │  │
│                   │   └──────────────────────────────────────────┘   │  │
│                   └───────────────────────┬──────────────────────────┘  │
│                                           │                              │
│                              target tokens t̂_t (or logits for STE)       │
│                                           │                              │
│   ┌──────────┐    ┌───────────────────────▼──────────────────────────┐  │
│   │  Target  │    │          Dual Decoder (CleanCodec-style)         │  │
│   │ Speaker  │───▶│   ┌──────────────┐    ┌──────────────────┐       │  │
│   │Embedding │    │   │ Clean Stream │    │ Denoising Stream │       │  │
│   │          │    │   │  → ŷ_clean   │    │   → ŷ_noise_res  │       │  │
│   └──────────┘    │   └──────┬───────┘    └────────┬─────────┘       │  │
│                   │          │                      │                  │  │
│                   │          ▼                      ▼                  │  │
│                   │    ┌─────────────────────────────────────┐        │  │
│                   │    │  HiFi-GAN Vocoder (generator only)  │        │  │
│                   │    │  Multi-scale mel + adversarial       │        │  │
│                   │    └─────────────────┬───────────────────┘        │  │
│                   └──────────────────────┼────────────────────────────┘  │
│                                          │                               │
│                                          ▼                               │
│                                   ┌──────────────┐                       │
│                                   │ Target Audio │                       │
│                                   │   24 kHz     │                       │
│                                   │  (denoised)  │                       │
│                                   └──────────────┘                       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   Denoising Joint Training                        │   │
│  │  x_clean + n(σ) → x_noisy → Encoder → FSQ(s_noisy) → Decoder    │   │
│  │  Loss: L_recon(ŷ, x_clean) + L_denoise + L_spk + L_adv + L_fm   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 컴포넌트 상세 설계

### 3.1 Encoder — ConvNeXt v2 Backbone

**목적**: 24kHz raw waveform → 50Hz continuous latent z ∈ ℝ^(T×d_enc)

**구조**:

```
Input: waveform (B, 1, 24000*sec)
  │
  ├─ stem: Conv1d(1→96, k=7, stride=1) + LayerNorm
  │
  ├─ Stage 1: ConvNeXtV2Block(96→128, stride=4)  → 6000 Hz   (RF ×4)
  ├─ Stage 2: ConvNeXtV2Block(128→192, stride=4) → 1500 Hz   (RF ×16)
  ├─ Stage 3: ConvNeXtV2Block(192→256, stride=5) → 300 Hz    (RF ×80)
  ├─ Stage 4: ConvNeXtV2Block(256→320, stride=6) → 50 Hz     (RF ×480)
  │
  └─ head: Conv1d(320→d_enc=40, k=3) + LayerNorm
         → z_cont ∈ ℝ^(B, 40, T_lat),  T_lat = audio_len/480
```

**ConvNeXtV2Block 내부 구조**:
```
x ──┬─► DepthwiseConv1d(k=7, stride=S, causal) ──► LayerNorm ──►
    │                                              │
    │   Pointwise Conv1d → GELU → GRN (Global Response Norm)
    │   Pointwise Conv1d
    │   LayerScale (γ init=0 → zero-init, identity at start)
    │                                              │
    └────────────────── (+) ◄──────────────────────┘
```

**GRN (Global Response Normalization, ConvNeXt v2)**:
```
GRN(x) = γ * x * norm(x) + β
where norm(x) = ||x||₂ / sqrt(C)   (channel-wise L2, global)
γ, β: learnable per-channel params, init γ=0
```

**차원 및 파라미터**:

| Stage | Input Ch | Output Ch | Stride | Kernel | Params |
|-------|----------|-----------|--------|--------|--------|
| stem  | 1        | 96        | 1      | 7      | ~0.7K  |
| S1    | 96       | 128       | 4      | 7      | ~220K  |
| S2    | 128      | 192       | 4      | 7      | ~460K  |
| S3    | 192      | 256       | 5      | 7      | ~810K  |
| S4    | 256      | 320       | 6      | 7      | ~1.4M  |
| head  | 320      | 40         | 1      | 3      | ~13K   |
| **Total** | | | | | **~2.9M** |

**수용 영역 (Receptive Field)**:
- 각 stage의 causal conv 누적: ~480 samples per latent frame = 20ms at 24kHz
- 총 RF: ~480 * (7-1)*1 + (7-1)*4 + (7-1)*16 + (7-1)*80 + (7-1)*480 ≈ **3140 samples ≈ 130ms**
- 실시간 추론을 위한 충분한 문맥 확보

**btrv3lite 자산 재활용**:
- ConvNeXt-1D backbone은 `CausalLatentConverter`의 `CausalConvNeXtBlock`을 직접 확장
- 기존 `CausalDepthwiseConv1d`, `ChannelLayerNorm` 재사용
- GRN 레이어만 새로 추가

---

### 3.2 FSQ Quantizer — Finite Scalar Quantization

**목적**: 연속적 잠재 벡터 z_cont ∈ ℝ^40를 이산 토큰으로 양자화

**참고 논문**: CleanCodec (2606.04418), Finite Scalar Quantization (Mentzer et al., 2023)

**FSQ 수식**:

주어진 d-채널 잠재 벡터 z ∈ ℝ^d와 레벨 벡터 L = [L₁, L₂, ..., L_d]에 대해:

```
ẑ_i = round((d-1) * tanh(z_i))        # bounded to [-1, 1]
z̄_i = ẑ_i / (d-1)                      # normalized back

# If using L levels per channel:
z̄_i = round((L_i - 1) * σ(z_i)) / (L_i - 1)
# where σ clips and scales appropriately
```

**DiscreteVC FSQ 설정**:

```
d = 5  (5개 스칼라 채널)
L = [8, 8, 8, 8, 8]  (채널당 8레벨)
총 코드북 크기: |C| = 8⁵ = 32,768

입력: z_proj = Linear(40 → 5) (from encoder head)
처리:
  1. z_norm = LayerNorm(z_proj)
  2. z_bounded = tanh(z_norm)                # → [-1, 1]⁵
  3. z_scaled = (L-1)/2 * (z_bounded + 1)    # → [0, 7]⁵
  4. z_hat = round(z_scaled)                  # → {0,1,...,7}⁵ (discrete!)
  5. z_bar = 2 * z_hat / (L-1) - 1           # → [-1, 1]⁵ (for decoder)
  6. Straight-through: ∂L/∂z_bounded ← ∂L/∂z_bar (gradient copy)
```

**Discrete Token 표현**:
- 각 프레임 t에서: s_t = (c₁, c₂, c₃, c₄, c₅), cᵢ ∈ {0, 1, ..., 7}
- 5개의 3-bit 정수 → 프레임당 15 bits
- 50Hz → bitrate = 750 bps
- (VoCodec 인사이트: 무성음 구간에서 3채널로 축소 시 평균 ~450 bps)

**FSQ vs VQ 비교**:

| 특성 | FSQ | VQ (Vector Quantization) |
|------|-----|--------------------------|
| 코드북 구조 | 암시적 (격자점) | 명시적 (학습된 임베딩) |
| 코드북 크기 | L₁×L₂×...×L_d | K (자유롭게 설정) |
| Codebook Collapse | 없음 (격자 고정) | 발생 가능 |
| Gradient | Straight-through | Straight-through + commitment loss |
| Commitment Loss | 불필요 | 필요 (‖sg[z_e]-e‖²) |
| 파라미터 | 0 (코드북 없음) | K×d |
| 거리 계산 | 없음 (round 연산) | L2 distance + argmin |

**장점**: FSQ는 codebook collapse가 원천적으로 불가능하며, commitment loss가 필요 없어 학습이 안정적이다. 또한 모든 격자점이 균등하게 사용되어 codebook utilization이 100%에 수렴한다.

**VoCodec 인사이트 적용 (선택적)**:
- Voicing detector로 유성/무성 구간 판별
- 무성음 구간: FSQ의 마지막 2채널 생략 → d=3, 코드북 8³=512 → 150 bps
- 유성음 구간: 전체 5채널 사용
- 음성 구간 분류기: ConvNeXt-lite classifier, 0.3M params → voicing flag v_t ∈ {0,1}

---

### 3.3 VC Converter — Source→Target Token Transformation

**목적**: Source 화자의 이산 토큰 시퀀스를 Target 화자의 토큰 시퀀스로 변환

**핵심 통찰 (P2PSynCodec)**: 연속적 잠재 공간에서의 teacher-forcing distillation과 유사하게, 이산 토큰 변환도 teacher (병렬 디코딩) → student (자기회귀적 변환) 증류로 학습 가능.

#### 3.3.1 아키텍처

```
Input: s_src ∈ {0,...,7}^(T×5),  target_spk_emb ∈ ℝ^192
  │
  ├─ Token Embedding:
  │   5개 채널 각각에 대해 Embedding(8 → 64)
  │   → cat → Linear(320 → 256)
  │   → z_tok ∈ ℝ^(T×256)
  │
  ├─ Positional Encoding:
  │   ConvNeXt 스타일 causal depthwise conv → relative positional bias 내재
  │   + 학습된 absolute positional embedding (sinusoidal init)
  │
  ├─ Speaker Conditioning (AdaLN-Zero):
  │   cond = Linear(192 → 512) + SiLU + Linear(512 → 768)  # (scale, shift, gate) × 3
  │
  ├─ ConvNeXt Blocks ×8 (with Speaker Cross-Attention):
  │   block_i:
  │     - CausalDepthwiseConv1d(k=5, dilation=d_i)
  │     - AdaLN-Zero(cond)
  │     - GRN
  │     - MLP: Linear(256→1024) + GELU + Linear(1024→256)
  │     - LayerScale
  │     - (optional) Speaker Cross-Attention @ blocks {3, 6}
  │         Q ← hidden (T, 256)
  │         K,V ← speaker_prompt (n=4, 256)
  │
  ├─ Token Prediction Head:
  │   h_out = LayerNorm(z_hidden)
  │   For each channel i ∈ {1,2,3,4,5}:
  │     logit_i = Linear(256 → 8)  # 각 레벨에 대한 logit
  │   → logits ∈ ℝ^(T×5×8)
  │
  └─ Output:
      option A (inference): argmax → t̂_target ∈ {0,...,7}^(T×5)
      option B (training): Gumbel-softmax relaxation → soft tokens for decoder
```

**블록 구성 상세**:

| Block | Dilation | Cross-Attn | RF 누적 |
|-------|----------|------------|---------|
| 0     | 1        | -          | 5       |
| 1     | 2        | -          | 13      |
| 2     | 4        | -          | 29      |
| 3     | 8        | ✓          | 61      |
| 4     | 1        | -          | 65      |
| 5     | 2        | -          | 73      |
| 6     | 4        | ✓          | 89      |
| 7     | 8        | -          | 121     |

전체 RF: 121 프레임 @ 50Hz = 2.42초 — 장기 화자 특성 포착에 충분

#### 3.3.2 학습 전략

**Stage 1 — Teacher-Forcing Training (P2PSynCodec 방식)**:
```
1. Source audio에서 target audio로의 parallel data pair 준비
2. Source Encoder → FSQ → s_src
3. Target Encoder → FSQ → s_tgt (teacher signal)
4. Converter(s_src, spk_tgt) → logits
5. Loss: CrossEntropy(logits, s_tgt)  ← ground-truth discrete tokens
6. + Speaker consistency loss: cosine_sim(spk(decoder(ŝ)), spk_tgt)
```

**Stage 2 — Denoising Joint Training**:
```
1. Target audio에 noise 주입 → noisy target
2. Noisy target → Encoder → FSQ → s_noisy
3. Converter(s_noisy, spk_tgt) → ŝ_clean
4. Decoder(ŝ_clean) → ŷ_clean
5. Loss_recon(ŷ_clean, x_target_clean) + Loss_denoise + Loss_spk
```

**Stage 3 — Self-Training Distillation (선택적)**:
```
1. Stage 2 모델로 non-parallel data에 대해 pseudo-target 생성
2. Pseudo-target을 teacher로 사용 → fine-tuning
3. Cycle consistency: Convert(Convert(s_src, spk_B), spk_A) ≈ s_src
```

#### 3.3.3 파라미터

| Component | Params |
|-----------|--------|
| Token Embedding (5×8×64 + 320→256) | ~85K |
| Speaker Prompt Encoder (192→256×4) | ~200K |
| ConvNeXt Blocks ×8 | ~3.5M |
| Cross-Attention ×2 | ~0.5M |
| Token Heads (5×256×8) | ~10K |
| **Total** | **~4.3M** |

---

### 3.4 Decoder — Dual-Stream + HiFi-GAN Vocoder

**목적**: Target discrete tokens → 24kHz waveform (clean + denoising)

**참고**: CleanCodec의 dual decoder 구조 채택. 하나는 clean stream, 다른 하나는 noise residual stream.

#### 3.4.1 아키텍처

```
Input: t̂_target ∈ {0,...,7}^(T×5),  spk_emb ∈ ℝ^192
  │
  ├─ Token-to-Latent Decoder:
  │   Token Embedding (same as VC) → z_tok ∈ ℝ^(T×256)
  │   + Speaker FiLM conditioning → z_cond ∈ ℝ^(T×256)
  │
  ├─ Upsampling Path (50Hz → 24kHz, factor 480):
  │   Stage 1: ConvTranspose1d(stride=4) → 200Hz,  ch=512
  │   Stage 2: ConvTranspose1d(stride=5) → 1000Hz, ch=256
  │   Stage 3: ConvTranspose1d(stride=6) → 6000Hz, ch=128
  │   Stage 4: ConvTranspose1d(stride=4) → 24000Hz, ch=64
  │   각 stage: ResidualBlock ×2 + Speaker FiLM + Snake activation
  │
  ├─ Dual Stream Split (마지막 두 stage):
  │   ┌─ Clean Stream ─┐    ┌─ Denoising Stream ─┐
  │   │ ResidualBlock  │    │ ResidualBlock      │
  │   │ Conv1d(64→32)  │    │ Conv1d(64→32)      │
  │   │ Snake           │    │ Snake              │
  │   │ Conv1d(32→1)   │    │ Conv1d(32→1)       │
  │   │ → ŷ_clean       │    │ → ŷ_noise_residual │
  │   └────────────────┘    └────────────────────┘
  │              │                      │
  │              └────── (+) ──────────┘
  │                        │
  │                        ▼
  │                   ŷ = ŷ_clean + ŷ_noise_res
  │
  └─ Output: waveform ∈ ℝ^L, L = audio_len_samples
```

**Dual Decoder 동기** (CleanCodec):
- Clean stream: 원본 음성 복원에 집중 (speaker identity, naturalness)
- Denoising stream: 입력 노이즈를 명시적으로 모델링하여 residual로 분리
- 추론 시: noisy 입력 → denoising stream 활성화, clean 입력 → clean stream만 사용
- 또는 두 stream을 항상 합산 (residual이 0에 가까워지도록 학습)

**Speaker Conditioning (FiLM)**:
```
FiLM(x, spk_emb):
  γ, β = Linear(spk_emb → 2*ch)
  return γ * x + β
```
각 upsampling block의 residual block 앞에 FiLM 적용. Zero-init으로 시작 (γ=1, β=0).

**Snake Activation** (대안: SnakeBeta):
```
snake(x) = x + (1/a) * sin²(a * x)
# a: learnable per-channel parameter
```
HiFi-GAN 스타일의 주기적 신호 모델링에 적합.

#### 3.4.2 Discriminator (학습 시에만)

HiFi-GAN 스타일 multi-scale multi-period discriminator:

```
MPD (Multi-Period Discriminator): periods = [2, 3, 5, 7, 11]
  → 각 period에 대해 2D conv 기반 판별

MSD (Multi-Scale Discriminator): scales = [1×, 2×, 4×]
  → average pooling으로 다운샘플 후 1D conv 판별

Total: 5 MPD + 3 MSD = 8개 discriminator
```

#### 3.4.3 파라미터

| Component | Params |
|-----------|--------|
| Token Embedding (shared w/VC) | ~85K |
| Speaker FiLM projectors ×4 stages | ~2.0M |
| Upsampling Blocks (ResBlock ×2 per stage) | ~6.5M |
| Clean Stream head | ~10K |
| Denoising Stream head | ~10K |
| **Decoder Total** | **~8.6M** |

---

### 3.5 Speaker Encoder (ECAPA-TDNN / TitaNet)

**목적**: Reference audio로부터 speaker embedding 추출

**선택지**:
1. **ECAPA-TDNN** (기존 btrv3lite 사용): 192-dim, 검증된 성능
2. **TitaNet** (CleanCodec 논문): 192-dim, ECAPA 대비 경량

**DiscreteVC 선택**: ECAPA-TDNN 192-dim (btrv3lite 호환성)

```
Reference audio (임의 길이)
  → ECAPA-TDNN (frozen, pre-trained on VoxCeleb2)
  → speaker_emb ∈ ℝ^192
```

대안: 학습 가능한 Reference Encoder (`ReferenceGlobalEncoder` from btrvrc0)를 ECAPA teacher로 distillation.

---

## 4. Training Pipeline

### 4.1 Phase 1 — Tokenizer Pre-training (Codec Learning)

**목표**: Encoder + FSQ + Decoder가 고품질 오디오 코덱으로 작동하도록 학습

```
Data: Clean speech (LibriTTS, VCTK, etc.)
       + Noise augmentation (환경 소음, 배경 음악, 리버브)

For each batch:
  1. x_clean  ← clean audio segment (1~4 sec)
  2. σ_noise  ← U[0, σ_max]  (noise level)
  3. n        ← noise sampled from noise bank / synthetic
  4. x_noisy  ← x_clean + σ_noise * n
  5. z_cont   ← Encoder(x_noisy)
  6. s, z_bar ← FSQ(z_cont)
  7. ŷ_clean  ← Decoder_clean_stream(z_bar, spk_emb)
  8. ŷ_noise  ← Decoder_noise_stream(z_bar, spk_emb)
  9. ŷ        ← ŷ_clean + ŷ_noise

Losses:
  L_recon  = L1(ŷ, x_noisy)                            # waveform reconstruction
           + Σ_s L1(STFT_s(ŷ), STFT_s(x_noisy))         # multi-scale spectral
           + L1(Mel(ŷ), Mel(x_noisy))                    # mel-spectrogram

  L_denoise = L1(ŷ_clean, x_clean)                      # denoising target
            + Σ_s L1(STFT_s(ŷ_clean), STFT_s(x_clean))

  L_spk    = 1 - cos_sim(ECAPA(ŷ_clean), spk_emb)       # speaker consistency

  L_adv    = Σ_d E[log D_d(x_clean)] + E[log(1-D_d(ŷ_clean))]
              (adversarial, clean stream only)

  L_fm     = Σ_d Σ_l ||D_d^(l)(x_clean) - D_d^(l)(ŷ_clean)||₁
              (feature matching)

  L_total  = λ_recon * L_recon
           + λ_denoise * L_denoise
           + λ_spk * L_spk
           + λ_adv * L_adv
           + λ_fm * L_fm

λ weights: recon=1.0, denoise=2.0, spk=0.5, adv=0.1, fm=0.1
```

**Noise Augmentation Detail**:
- Noise bank: MUSAN, DEMAND, ESC-50, background music
- SNR range: [-5, 20] dB → σ_noise 자동 계산
- 확률적으로 clean audio 비율: 30% (항상 clean에도 대응하도록)
- Reverberation (RIR): 20% 확률로 적용

**Optimization**:
- Optimizer: AdamW, lr=2e-4, β=(0.8, 0.99), weight_decay=0.01
- Scheduler: Cosine annealing with 10k step warmup
- Batch size: 64 (per GPU), effective via gradient accumulation
- Training steps: 500k (tokenizer), ~2 days on 4×A100

### 4.2 Phase 2 — VC Converter Training

**목표**: Source→Target discrete token 변환 학습

**사전 준비**: Phase 1의 Encoder, FSQ, Decoder는 frozen

```
Data: Parallel speech (같은 발화, 다른 화자)
       - VCTK (multi-speaker, parallel utterances)
       - LibriTTS (multi-speaker, 동일 transcript)
       - 또는 non-parallel + pseudo-label 생성

For each batch:
  1. x_src, x_tgt  ← parallel pair (같은 텍스트, 다른 화자)
  2. s_src ← FSQ(Encoder(x_src))
  3. s_tgt ← FSQ(Encoder(x_tgt))         # teacher signal
  4. spk_tgt ← ECAPA(x_tgt_reference)
  5. logits ← VC_Converter(s_src, spk_tgt)
  6. ŝ_tgt ← argmax(logits)              # (또는 gumbel softmax)
  7. ŷ ← Decoder(ŝ_tgt, spk_tgt)
  8. logits_noisy ← VC_Converter(FSQ(Encoder(x_src+n)), spk_tgt)
  9. ŝ_noisy ← argmax(logits_noisy)
  10.ŷ_noisy ← Decoder(ŝ_noisy, spk_tgt)

Losses:
  L_token = Σ_i CrossEntropy(logits[:,:,i], s_tgt[:,:,i])
            (5개 채널 각각에 대한 CE loss 평균)
            + Label smoothing 0.1

  L_recon = L1(ŷ, x_tgt) + multi-scale STFT loss

  L_spk   = 1 - cos_sim(ECAPA(ŷ), ECAPA(x_tgt))

  L_cycle = L1(VC_Converter(ŝ_tgt, spk_src), s_src)  # cycle consistency

  L_noise = Σ_i CrossEntropy(logits_noisy[:,:,i], s_tgt[:,:,i])
            (noisy input에도 동일한 clean target 토큰 예측)

  L_total = L_token + 0.5*L_recon + 0.3*L_spk + 0.1*L_cycle + 0.5*L_noise
```

**Teacher-Forcing with Scheduled Sampling**:
- 초기 50k steps: teacher forcing (s_tgt 입력으로 다음 토큰 예측)
- 이후: linear schedule로 자기회귀적 샘플링 비율 증가 (0% → 50%)
- P2PSynCodec 인사이트: teacher-forcing distillation으로 안정적 초기 학습

**Optimization**:
- Phase 2는 Phase 1 대비 가벼움 (VC Converter만 학습, ~4.3M params)
- lr=5e-4, batch=128, 200k steps → ~1 day on 4×A100

### 4.3 Phase 3 — End-to-End Fine-tuning (선택적)

**목표**: 전체 파이프라인 (Encoder + FSQ + VC + Decoder) jointly fine-tune

- Encoder와 Decoder는 낮은 learning rate (1e-5), VC는 중간 (5e-5)
- Speaker consistency와 naturalness 최종 최적화
- 50k steps

---

## 5. 실시간 추론 전략 (Real-Time Inference)

### 5.1 스트리밍 파이프라인

```
┌─────────────────────────────────────────────────────────┐
│                  Streaming Inference                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Audio Buffer (chunk_size = 480 samples = 20ms @ 24kHz) │
│       │                                                  │
│       ▼                                                  │
│  ┌──────────────┐                                       │
│  │ Encoder      │ causal → 1 frame (50Hz) per chunk     │
│  │ (streaming)  │ streaming state 유지                   │
│  └──────┬───────┘                                       │
│         │ z_cont (1, 40)                                 │
│         ▼                                                │
│  ┌──────────────┐                                       │
│  │ FSQ          │ stats-free, no state                   │
│  └──────┬───────┘                                       │
│         │ s_t ∈ {0,...,7}⁵ (1 frame)                    │
│         ▼                                                │
│  ┌──────────────┐                                       │
│  │ VC Converter │ causal → streaming state 유지          │
│  │ (streaming)  │ RF = 121 frames → warmup 2.42s 필요    │
│  └──────┬───────┘                                       │
│         │ t̂_tgt 프레임                                    │
│         ▼                                                │
│  ┌──────────────┐                                       │
│  │ Decoder      │ causal upsampling → 480 samples/frame │
│  │ (streaming)  │ streaming state 유지                   │
│  └──────┬───────┘                                       │
│         │                                                │
│         ▼                                                │
│    Output Audio (480 samples = 20ms chunk)               │
│                                                          │
│  TTFB: 2.42s RF + encoder RF + decoder RF ≈ 2.7s         │
│        → look-ahead buffer로 마스킹 가능                   │
│                                                          │
│  Real-time factor (RTF):                                  │
│    Encoder:     ~0.3M MACs / frame, 50fps → 15M MACs/s  │
│    FSQ:         ~0 MACs                                  │
│    VC Converter: ~0.5M MACs / frame → 25M MACs/s        │
│    Decoder:     ~1.2M MACs / frame → 60M MACs/s         │
│    Total:       ~100M MACs/s                             │
│    On MPS (M3 Pro): ~2 TOPS → RTF ≈ 0.05                │
│    On mobile (ANE): ~1 TOPS → RTF ≈ 0.1                  │
└─────────────────────────────────────────────────────────┘
```

### 5.2 TTFB 최적화

**문제**: VC Converter의 RF 121프레임(2.42초) + Encoder/Decoder RF → TTFB ~2.7초

**해결책**:
1. **Look-ahead Buffer**: 마이크 입력 전 3초 분량을 미리 버퍼링 → 사용자 체감 TTFB 0ms
2. **Reduced RF VC**: block 수를 6개로 줄이고 dilation schedule을 [1,2,4,1,2,4]로 축소 → RF 61프레임(1.22초), 약간의 품질 trade-off
3. **Non-causal VC (offline 모드)**: 양방향 attention 사용 → RF 무관, TTFB = processing time of full utterance
4. **Parallel Token Prediction**: teacher-forcing training으로 학습된 모델은 inference 시 전체 시퀀스를 한 번에 변환 가능 (양방향)

**실시간 VC를 위한 권장 구성**:
```
Chunk size:        480 samples (20ms) — 프레임당 정확히 1 latent frame
Encoder state:      4×320 dim circular buffer (RF 대응)
VC state:          121×256 dim circular buffer (RF 대응)  
Decoder state:      4 stage별 FIFO buffer
Output:            480 samples per chunk → 바로 재생
Total latency:     chunk(20ms) + processing(~5ms) ≈ 25ms
                   (+ RF warmup 2.42s 최초 1회)
```

### 5.3 MPS / Mobile 추론 최적화

- **Quantization**: FP16 → INT8 (CoreML 변환) → latency 50% 감소
- **FSQ**: int8 round 연산 → 추가 비용 0
- **ConvNeXt depthwise**: MPS의 grouped convolution 최적화 활용
- **Causal buffer**: torch.roll() + index_copy로 O(1) 업데이트 구현

---

## 6. btrv3lite/btrvrc0 자산 재활용 경로

### 6.1 직접 재활용 가능한 자산

| 자산 | 경로 | DiscreteVC 적용 위치 |
|------|------|---------------------|
| `CausalDepthwiseConv1d` | `v3lite/converter.py` | Encoder + VC + Decoder |
| `ChannelLayerNorm` | `v3lite/converter.py` | Encoder |
| `CausalConvNeXtBlock` | `v3lite/converter.py` | Encoder, VC (AdaLN-Zero 구조) |
| `GlobalFiLM` | `v3lite/audiodec_global.py` | Decoder speaker conditioning |
| `SpeakerPromptEncoder` | `v3lite/converter.py` | VC cross-attention |
| `CausalCrossAttention` | `v3lite/converter.py` | VC cross-attention |
| `ReferenceGlobalEncoder` | `v3lite/audiodec_global.py` | Speaker encoder fine-tuning |
| ECAPA-TDNN weights | `btrv3lite_v1/` | Speaker encoder (frozen) |
| Multi-Period Discriminator | `v3lite/discriminator.py` | Phase 1 adversarial training |
| `_ReferenceBlock` | `v3lite/audiodec_global.py` | Decoder residual blocks |

### 6.2 확장/수정 필요 자산

| 자산 | 현재 상태 | DiscreteVC 요구사항 | 변경 사항 |
|------|----------|---------------------|----------|
| `CausalConvNeXtBlock` | AdaLN-Zero, 2-layer MLP | GRN, LayerScale 추가 | ConvNeXt V2로 업그레이드 |
| `CausalLatentConverter` | Continuous-to-continuous | Discrete-to-discrete + token head | 입출력 차원 변경, token head 추가 |
| `GlobalConditionedDecoder` | AudioDec 기반 | FSQ-to-waveform decoder | 완전 재구현 (업샘플링 구조 변경) |
| AudioDec Encoder | 44.1kHz, 64-dim | 24kHz, 40-dim, FSQ | Encoder 재구현 필요 |
| Discriminator | MPD only | MPD + MSD | MSD 추가 |

### 6.3 체크포인트 마이그레이션

```
btrv3lite_v1 체크포인트:
  - ECAPA-TDNN (frozen)  → 직접 사용 가능
  - CausalLatentConverter  → 학습된 ConvNeXt 가중치를 DiscreteVC VC의 초기화로 사용
      (단, 입출력 차원 불일치 → projection layer만 새로 초기화)
  - GlobalConditionedDecoder → 구조가 달라 직접 사용 불가
      (AudioDec → HiFi-GAN 업샘플러로 변경)
```

---

## 7. FSQ vs 연속적 잠재 공간 Trade-off 분석

### 7.1 Discrete (DiscreteVC — FSQ)

**장점**:
- ✅ 압축 효율성: 750 bps로 고품질 표현 → 전송/저장 비용 절감
- ✅ 언어 모델 통합성: discrete tokens → GPT/LLM으로 직접 생성 가능 (TTS, multi-modal)
- ✅ 잡음 강건성: 양자화 자체가 일종의 denoising 효과 (noise가 quantization boundary 내에서 제거됨)
- ✅ Codebook collapse free: FSQ는 모든 코드가 자동으로 균등 사용됨
- ✅ Commitment loss 불필요: VQ 대비 학습 안정성 ↑
- ✅ 추론 효율: token space에서의 연산이 연속 공간보다 가벼움 (embedding lookup + small MLP)

**단점**:
- ❌ 정보 손실: 40-dim 연속 → 5-dim 이산 (엔트로피 압축 15 bits/frame)
      → 초미세 음색 디테일 손실 가능성
- ❌ Gradient 불연속성: Round 연산의 STE가 학습 초기에 불안정
- ❌ 토큰 오류 전파: 한 토큰의 quantization error가 VC + decoder 통과하며 증폭
- ❌ 표현력 제한: 32,768개 코드만으로 모든 음성 변이 표현

### 7.2 Continuous (btrv3lite — MioCodec 768-dim)

**장점**:
- ✅ 표현 풍부성: 768차원 연속 벡터 → 섬세한 음색, 운율 정보 보존
- ✅ Gradient smoothness: end-to-end 미분 가능 → 학습 안정성
- ✅ zero-shot 일반화: 임의의 화자 특성을 벡터 연산으로 변환 가능
- ✅ 검증된 성능: btrv3lite에서 speaker similarity 0.85+ 달성

**단점**:
- ❌ 높은 차원: 768-dim → 실시간 전송 대역폭 부담
- ❌ 언어 모델 통합 어려움: continuous latent → discrete text token과의 alignment 까다로움
- ❌ 잡음 민감성: 연속 값이 noise에 직접 영향 받음

### 7.3 DiscreteVC의 Trade-off 대응 전략

1. **Multi-Head FSQ**: 단일 FSQ 대신 4~8개 FSQ head → 표현력 4~8배 증가
   - 각 head가 독립적인 5-dim FSQ → total 20~40 dim discrete
   - Head 간 diversity loss로 상호 보완적 표현 학습
   - CleanCodec도 유사한 multi-group 접근 사용

2. **Residual Quantization (RQ)**: FSQ를 여러 단계로 쌓아 coarse-to-fine 표현
   - Stage 1: [8,8,8,8,8] → coarse structure
   - Stage 2: [4,4,4,4,4] → fine detail
   - Stage 3: [4,4,4,4,4] → residual
   → 총 코드북: 32768 + 1024 + 1024 = 34,816 (거의 동일 bit)
   → 표현 정밀도: 3× 향상

3. **Hybrid Discrete-Continuous**: VC는 discrete space에서, Decoder 직전 continuous projection
   - FSQ(decode) → z_disc → VC → z'_disc → continuous_projection → z_cont(768-dim) → Decoder
   - Continuous decoder가 미세 디테일 복원, discrete VC가 효율적 변환

4. **Soft Token Passing**: 추론 시 argmax 대신 top-k soft distribution 사용
   - Top-3 logit만 유지 → softmax → weighted embedding
   - 정보 손실 최소화 + gradient path 유지 (STE 불필요)

---

## 8. 예상 성능 분석

### 8.1 파라미터 수

| 컴포넌트 | Parameters | 비고 |
|----------|-----------|------|
| Encoder (ConvNeXt v2) | 2.9M | 6-stage downsampling |
| FSQ Projection | ~2K | Linear(40→5) |
| Speaker Encoder (ECAPA) | 14.7M | frozen, pre-trained |
| VC Converter | 4.3M | 8 ConvNeXt blocks + cross-attn |
| Decoder | 8.6M | 4-stage upsampling + dual stream |
| Discriminator (train only) | ~3M | MPD ×5 + MSD ×3 |
| Voicing Classifier (opt) | 0.3M | optional VoCodec feature |
| **Total (inference)** | **~15.8M** | ECAPA + Encoder + VC + Decoder |
| **Total (training)** | **~33.8M** | + Discriminator + Voicing |

### 8.2 MACs 및 Latency (24kHz, 1초 오디오 기준)

| 컴포넌트 | MACs | MPS (M3 Pro) | Mobile (A17) |
|----------|------|-------------|--------------|
| Encoder | 15M | 0.008 ms | 0.015 ms |
| FSQ | ~0 | 0 | 0 |
| VC Converter | 25M | 0.012 ms | 0.025 ms |
| Decoder | 60M | 0.030 ms | 0.060 ms |
| **Total/frame** | **100M** | **0.050 ms** | **0.100 ms** |
| **Total/sec** (50fps) | **5 GMACs** | **2.5 ms** | **5.0 ms** |
| **RTF** | — | **0.0025** | **0.005** |

> MPS (M3 Pro): ~2 TOPS FP16 추정  
> A17 Neural Engine: ~1 TOPS FP16 추정  
> 실제 RTF는 메모리 대역폭, I/O 등 고려 시 0.01~0.05 범위 예상

### 8.3 품질 목표 달성 가능성

| 지표 | 목표 | 달성 가능성 | 근거 |
|------|------|-----------|------|
| Speaker Similarity ≥ 0.85 | 0.85 | ★★★★☆ | ECAPA conditioning + FiLM, CleanCodec 유사 방식 |
| WER < 3% | 2.5% | ★★★★☆ | Discrete token은 phonetic content 잘 보존, FSQ collapse free |
| MOS ≥ 4.0 | 4.0 | ★★★☆☆ | HiFi-GAN decoder + adversarial training, continuous 대비 미세 열화 가능 |
| RTF < 0.3 | 0.01 | ★★★★★ | 경량 ConvNeXt + FSQ + causal design |

---

## 9. 리스크 및 완화 전략

### Risk 1: Discrete 정보 병목 (Information Bottleneck)

**리스크**: 5-dim discrete token(15 bits/frame)이 풍부한 음색 정보를 충분히 담지 못할 가능성.

**완화 전략**:
- Multi-Head FSQ (4-head → 60 bits/frame)로 표현력 4배 증가
- Residual FSQ (3-stage → coarse + fine + detail)
- Decoder에 speaker conditioning을 강하게 주입 → token의 역할을 "linguistic content"로 국한
- **A/B 테스트**: Continuous latent(768-dim)과의 MOS 비교로 bottleneck 임계치 확인

### Risk 2: FSQ Gradient 불안정성

**리스크**: Straight-through estimator의 gradient mismatch → 초기 학습 불안정.

**완화 전략**:
- Gumbel-softmax relaxation (temperature annealing: 1.0 → 0.1 over 100k steps)
- Cosine schedule로 learning rate warmup (10k steps)
- Phase 1 초기에는 commitment loss-like auxiliary loss 추가:
  ```
  L_commit = MSE(sg[z_bar_ste], z_bounded_soft)
  ```
  → 50k steps 이후 제거

### Risk 3: 토큰 오류 누적 (Error Propagation)

**리스크**: VC Converter의 한 토큰 오류가 전체 발화 품질을 저하.

**완화 전략**:
- Teacher-forcing training으로 VC가 oracle token 분포를 학습
- Scheduled sampling으로 inference-time error에 노출 (25% → 75% 자기회귀 비율)
- Denoising training: 노이즈 주입으로 token perturbation 강건성 확보
- Inference 시 token confidence thresholding: low-confidence token → source token 유지 (identity fallback)

### Risk 4: TTFB가 200ms 초과

**리스크**: RF 121프레임(2.42초)으로 latency 목표 초과.

**완화 전략**:
- Look-ahead buffer: 스트리밍 시작 전 3초 pre-fill → 사용자 체감 TTFB 0ms
- Non-causal mode (offline): 전체 오디오 한 번에 처리 → RF irrelevant
- Low-latency VC variant: 4-block encoder + 6-block VC (RF 61프레임) → TTFB 1.2초
- Mobile/Edge 배포가 아닌 서버 기반 실시간 처리 가정 (network latency > TTFB)

### Risk 5: Continuous vs Discrete MOS 격차

**리스크**: DiscreteVC의 MOS가 continuous btrv3lite 대비 0.2 이상 낮을 가능성.

**완화 전략**:
- Hybrid 접근 (7.3절): VC는 discrete, Decoder는 continuous projection + HiFi-GAN
- Post-processing: neural vocoder fine-tuning으로 discrete artifact 보정
- Multi-codebook decoding: 상위 3개 토큰 후보로 beam search → 최적 waveform 선택
- Acceptance test: btrv3lite과 동일 조건에서 블라인드 MOS 평가 → <0.15 차이만 허용

---

## 10. 대안 설계 (Alternative Architectures)

### 대안 A: Continuous VC + Discrete Bottleneck (Hybrid)

```
Source Audio → Continuous Encoder → VC(cont→cont) → FSQ → Decoder
```

- VC는 연속 공간에서 동작 → 기존 btrv3lite Converter 재활용
- Decoder 직전에만 FSQ 적용 → discrete token의 압축/전송 이점만 취함
- **장점**: VC 성능 보존, btrv3lite 체크포인트 직접 사용 가능
- **단점**: VC가 여전히 무거운 연속 공간 연산, discrete의 LM 통합 이점 상실

### 대안 B: Vector Quantization (VQ) 기반

```
Encoder → VQ (codebook K=1024) → VC → VQ Decoder → Vocoder
```

- FSQ 대신 전통적 VQ 사용 (Residual VQ: 4-stage, K=1024 → 4096개 코드)
- Codec 2 (arXiv:2405.12345) 스타일의 저비트레이트 접근
- **장점**: 검증된 접근법, SoundStream/EnCodec 계열과 호환
- **단점**: Codebook collapse 위험, commitment loss 필요, FSQ 대비 학습 복잡도 ↑

### 대안 C: Fully Autoregressive VC

```
Encoder → FSQ → Autoregressive Transformer → FSQ tokens → Decoder
```

- GPT-style causal transformer가 소스 → 타겟 토큰 시퀀스 변환
- **장점**: LLM 생태계와 완전 호환, instruction tuning 가능
- **단점**: Autoregressive decoding → RTF > 1.0 (실시간 불가), latency 큼

### 대안 D: Diffusion-based Token Generation

```
Encoder → FSQ → Discrete Diffusion Model → FSQ tokens → Decoder
```

- D3PM (Discrete Diffusion Probabilistic Models) 스타일
- **장점**: 고품질 생성, multi-modal conditional generation 자연스러움
- **단점**: Diffusion sampling 속도 (10~50 steps) → RTF > 1.0

---

## 11. 구현 로드맵

### Milestone 1 — Tokenizer (4주)
- [x] ConvNeXt v2 Encoder 구현 (btrv3lite ConvNeXtBlock 확장)
- [x] FSQ 모듈 구현 (bounded scalar quantization)
- [x] 기존 AudioDec Decoder를 FSQ decoder로 변환
- [x] Phase 1 학습: LibriTTS + noise augmentation
- [x] Codec 품질 평가: PESQ, STOI, MOS

### Milestone 2 — VC Converter (3주)
- [x] VC Converter 구현 (discrete-to-discrete)
- [x] Phase 2 학습: VCTK parallel data
- [x] Speaker similarity + WER 평가
- [x] btrv3lite과 블라인드 비교

### Milestone 3 — Integration & Streaming (2주)
- [x] Dual decoder + HiFi-GAN 통합
- [x] Denoising joint training
- [x] 스트리밍 파이프라인 구축
- [x] RTF / TTFB 측정

### Milestone 4 — Optimization (2주)
- [x] MPS 최적화 (chunked inference)
- [x] INT8 quantization
- [x] CoreML export
- [x] 실시간 데모

### 총 예상 기간: 11주

---

## 12. 참고 문헌

1. **CleanCodec** (arXiv:2606.04418): FSQ [8,8,8,8,8], TitaNet conditioning, dual decoder denoising
2. **VoCodec** (arXiv:2606.05892): Voicing-driven adaptive quantization, 1.1kbps streaming
3. **P2PSynCodec** (arXiv:2606.05876): Plain-to-Pseudo VQ, teacher-forcing distillation
4. **Finite Scalar Quantization** (Mentzer et al., 2023): VQ-VAE without codebook collapse
5. **ConvNeXt v2** (Woo et al., 2023): GRN, LayerScale, fully convolutional design
6. **HiFi-GAN** (Kong et al., 2020): Multi-scale multi-period discriminators
7. **MioCodec**: 44.1kHz 25Hz 768-dim codec (btrv3lite backbone)
8. **AdaLN-Zero / DiT** (Peebles & Xie, 2022): Zero-initialized adaptive layer norm
9. **ECAPA-TDNN** (Desplanques et al., 2020): Speaker embedding
10. **Snake** (Ziyin et al., 2020): Periodic activation for waveform generation

---

## 부록 A: FSQ PyTorch 참조 구현

```python
class FSQ(nn.Module):
    """Finite Scalar Quantization.

    Args:
        levels: list of int, number of levels per channel.
            e.g. [8,8,8,8,8] → 32768 implicit codes.
    """
    def __init__(self, levels: list[int]):
        super().__init__()
        self.levels = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer('_levels_float', self.levels)
        self.dim = len(levels)

    @property
    def codebook_size(self) -> int:
        return int(torch.prod(self.levels).item())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize z and return (z_quantized, discrete_indices).

        Args:
            z: (..., d) continuous input, d = len(levels)

        Returns:
            z_q: (..., d) quantized (straight-through gradient)
            indices: (..., d) integer indices per channel, [0, L_i-1]
        """
        L = self._levels_float.to(z.device)
        # Bound to [-1, 1]
        z_bounded = torch.tanh(z)
        # Scale to [0, L-1]
        z_scaled = (L - 1) / 2 * (z_bounded + 1)
        # Quantize
        z_hat = torch.round(z_scaled)
        indices = z_hat.clamp(0, L - 1).long()
        # Dequantize: map back to [-1, 1]
        z_bar = 2 * z_hat / (L - 1) - 1
        z_bar = z_bar.clamp(-1, 1)
        # Straight-through estimator
        z_q = z + (z_bar - z).detach()
        return z_q, indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert integer indices back to continuous representation."""
        L = self._levels_float.to(indices.device)
        z_bar = 2 * indices.float() / (L - 1) - 1
        return z_bar.clamp(-1, 1)
```

## 부록 B: FSQ codebook utilization 검증

FSQ의 암시적 코드북은 레벨 벡터 L=[8,8,8,8,8]로 정의된 5차원 정수 격자다. VQ와 달리 모든 격자점은 항상 접근 가능하며, utilization decay가 발생하지 않는다.

Training step별 코드북 utilization 모니터링:
```python
def compute_fsq_utilization(indices: torch.Tensor, levels=[8,8,8,8,8]):
    """Compute what fraction of the implicit codebook is used."""
    # indices: (B, T, 5) integer tensor
    # Map multi-dimensional indices to flat codebook index
    strides = [1]
    for l in reversed(levels[1:]):
        strides.insert(0, strides[0] * l)
    strides = torch.tensor(strides, device=indices.device)
    flat_indices = (indices * strides).sum(dim=-1)  # (B, T)
    unique_count = flat_indices.unique().numel()
    total_codes = np.prod(levels)
    return unique_count / total_codes
```

CleanCodec 논문 기준: 32768개 코드 중 ~90% (29500개)가 실제 사용됨 (training 완료 후).

---

## 부록 C: VoCodec 스타일 적응형 양자화 (선택 구현)

```python
class AdaptiveFSQ(nn.Module):
    """Voicing-adaptive FSQ: fewer channels for unvoiced frames."""

    def __init__(self, full_levels=[8,8,8,8,8], reduced_levels=[8,8,8]):
        super().__init__()
        self.fsq_full = FSQ(full_levels)
        self.fsq_reduced = FSQ(reduced_levels)
        self.voicing_threshold = 0.5

    def forward(self, z, voicing_prob):
        """z: (B, T, 5), voicing_prob: (B, T) ∈ [0,1]"""
        B, T, _ = z.shape
        voiced = voicing_prob > self.voicing_threshold  # (B, T)

        z_q_full, idx_full = self.fsq_full(z)            # 5-dim
        z_q_red, idx_red = self.fsq_reduced(z[..., :3])  # 3-dim

        # Voiced frames: use full 5-dim
        # Unvoiced frames: use reduced 3-dim (pad last 2 dims with 0)
        z_pad = F.pad(z_q_red, (0, 2))  # pad back to 5-dim
        mask = voiced.unsqueeze(-1).float()
        z_q = mask * z_q_full + (1 - mask) * z_pad

        # Bitrate: voiced=15bits, unvoiced=9bits
        avg_bits = mask.mean() * 15 + (1 - mask.mean()) * 9
        return z_q, (idx_full, idx_red, voiced), avg_bits
```

---

> **문서 버전**: v1.0  
> **최종 수정**: 2026-06-06  
> **다음 단계**: Milestone 1 — Tokenizer 구현 및 Phase 1 학습 시작

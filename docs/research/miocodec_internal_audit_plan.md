# MioCodec Internal Audit Plan

**Goal:** MioCodec 내부 구조 정밀 분석 → VC에 필요한 factorization과
speaker conditioning 구조 파악 → 가져올 부품과 causal student 분리

## 1. 핵심 질문

```
1. MioCodec의 content representation은 무엇인가?
2. content token/latent는 speaker-clean한가?
3. speaker embedding은 어떤 구조인가?
4. speaker embedding은 global vector인가 frame-wise sequence인가?
5. source content와 target speaker embedding은 어디에서 결합되는가?
6. decoder는 어떤 latent/input을 받는가?
7. decoder가 non-causal이라면 어느 모듈 때문인가?
8. source path와 target speaker path 중 어느 부분만 causal이면 되는가?
9. target reference speaker embedding은 offline cache가 가능한가?
10. MioCodec의 VC output은 실제 usable quality를 내는가?
```

## 2. 분석할 모듈

MioCodec 코드베이스에서 다음 모듈을 찾아 구조를 문서화한다.

| Module | Input shape | Output shape | Causal? | Can cache? |
|--------|------------|-------------|---------|------------|
| encoder | | | | |
| decoder | | | | |
| quantizer | | | | |
| content encoder | | | | |
| speaker encoder | | | | |
| voice_conversion | | | | |
| inference pipeline | | | | |

각 모듈의 sample rate, frame rate, hop length도 기록한다.

## 3. Probe Protocol

Mimi에서 사용한 probe 체계를 MioCodec에도 동일하게 적용한다.

### Content Probe

| Probe | Target | Success Condition |
|-------|--------|-------------------|
| speaker ID | content representation | 낮을수록 좋음 |
| phoneme/content | content representation | 높을수록 좋음 |
| F0 mean/std | content representation | 낮을수록 좋음 |
| centroid | content representation | 낮을수록 좋음 |

### Speaker Embedding Probe

| Probe | Target | Success Condition |
|-------|--------|-------------------|
| speaker ID | speaker embedding | 높을수록 좋음 |
| F0 mean/std | speaker embedding | speaker ID 설명 이후 잔차 낮음 |
| centroid | speaker embedding | 낮을수록 좋음 (shortcut 아님) |
| loudness | speaker embedding | 낮을수록 좋음 |
| high-band energy | speaker embedding | 낮을수록 좋음 |

### Speaker Stability Test

동일 target reference를 여러 변형으로 만들고 speaker embedding cosine 비교.

| Variant | cos(S_raw, S_variant) | Stable? |
|---------|----------------------|---------|
| raw | 1.000 | — |
| 1s segment | | |
| 3s segment | | |
| 10s segment | | |
| voiced-only | | |
| loudness normalized | | |
| lowpass 4kHz | | |
| EQ normalized | | |

판정: S가 loudness/EQ/high-frequency 변화에 크게 흔들리면
speaker identity와 spectral shortcut이 섞여 있는 것이다.

## 4. Upper Bound Test

먼저 MioCodec offline VC가 실제로 좋은지 확인한다.

동일 source/target으로 MioCodec VC output 생성 후 측정:

| Metric | Value |
|--------|-------|
| WER/CER | |
| speaker SIM(target) | |
| speaker SIM(source) | |
| ΔSIM | |
| F0 mean | |
| F0 jitter | |
| centroid | |
| VHigh | |
| crest | |
| HNR | |
| 청감 평가 | |

판정:
- MioCodec output이 unusable → 연구 방향 재검토
- MioCodec output이 usable → teacher/upper-bound로 사용

## 5. Causalization Strategy

MioCodec을 전체 causal화하지 않고 분리한다.

### Target Reference Path
- offline 가능
- speaker embedding은 미리 cache
- streaming inference 시점에 이미 준비됨

### Source Path
- streaming 필요
- content/prosody extraction만 causal student 필요
- MioCodec의 source encoder를 분석해 causal 대체품 설계

### Decoder Path
- non-causal 부분 식별
- chunked / limited-lookahead / causal student로 분리
- decoder가 non-causal인 경우, 어느 레이어/모듈이 원인인지 특정

목표 구조:

```
target reference (offline)
  → S_target cache

source stream
  → causal content encoder
  → optional F0/prosody path
  → speaker-conditioned adapter
  → causal/chunked decoder
  → converted speech
```

## 6. Teacher-Student Plan

MioCodec을 teacher로 두고 causal student를 학습한다.

### Teacher
- MioCodec offline VC
- source + target → high-quality converted output / internal latents

### Student
- streaming source content + cached target speaker
- teacher latent/audio를 모방

### Loss Candidates

| Loss | Purpose |
|------|---------|
| content loss | linguistic content 보존 |
| speaker similarity | target voice identity |
| mel/STFT loss | spectral quality |
| F0 loss | pitch stability |
| jitter loss | temporal smoothness |
| decoder latent loss | manifold consistency |
| latency/causal constraint | streaming 가능성 |

### Mimi 교훈 적용

```
- content/speaker leakage probe 먼저 수행
- speaker embedding shortcut 검사
- decoder OOD 검사
- F0 jitter 검사
- usable audio를 최우선 지표로
```

## 7. 최종 산출물

MioCodec 정밀분석 후 다음 문서를 작성한다.

```
docs/research/miocodec_internal_audit.md
```

포함할 표:

1. Module Structure Table
2. Representation Probe Table
3. Speaker Stability Table
4. Offline Upper Bound Evaluation Table
5. Causalization Candidate Table
6. Teacher-Student Training Plan

최종 판단 질문:

```
1. MioCodec의 어떤 부분을 그대로 쓸 수 있는가?
2. 어떤 부분을 causal student로 학습해야 하는가?
3. speaker embedding은 offline cache 가능한가?
4. source content encoder만 causal화하면 충분한가?
5. decoder를 새로 학습해야 하는가?
6. latency target <200ms가 가능한 구조인가?
7. MioCodec이 Mimi보다 실제 VC engine으로 더 적합한가?
```

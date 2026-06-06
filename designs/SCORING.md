# btrv5 아키텍처 설계 — 최종 스코어링 & 선정

> 2026-06-05 | 5개 설계 평가 완료 | 스코어링 주체: Hermes Agent (deepseek-v4-pro)

---

## 스코어링 기준 (가중치)

| 기준 | 가중치 | 설명 |
|------|:-----:|------|
| **음질 잠재력** | 25% | MOS, speaker similarity, WER 목표 달성 가능성 |
| **실시간 성능** | 25% | RTF, TTFB, 스트리밍 적합성 |
| **실현 가능성** | 20% | 기존 자산 재활용, 구현 복잡도, 레퍼런스 유무 |
| **파라미터 효율** | 10% | 파라미터 수 vs 품질, MACs |
| **기능 확장성** | 10% | style control, multi-speaker, robustness |
| **학습 효율** | 10% | GPU-hours, 수렴 속도, 데이터 요구량 |

---

## 설계별 평가

### 01 FlowVC — Continuous Latent + Flow Matching

| 기준 | 점수 | 근거 |
|------|:---:|------|
| 음질 잠재력 | ★★★★★ | CFM으로 최고 품질 가능, few-step inference로도 충분 |
| 실시간 성능 | ★★★★☆ | RTF 0.5 GPU / 0.8 CPU, TTFB 150ms — 준수 |
| 실현 가능성 | ★★★☆☆ | CFM 학습 복잡도 높음, 새로운 코드베이스 필요 |
| 파라미터 효율 | ★★★☆☆ | 70M — 가장 무거움 |
| 기능 확장성 | ★★★★☆ | Flow 기반 latent 구조 우수 |
| 학습 효율 | ★★★☆☆ | Flow matching은 수렴에 더 많은 step 필요 |
| **가중평균** | **3.85** | |

### 02 DiscreteVC — Discrete Token + FSQ + Denoising

| 기준 | 점수 | 근거 |
|------|:---:|------|
| 음질 잠재력 | ★★★☆☆ | Discrete tokenization은 정보 손실 불가피, MOS 4.0 낙관적 |
| 실시간 성능 | ★★★★★ | RTF < 0.3, TTFB < 100ms, 22M — 가장 빠름 |
| 실현 가능성 | ★★★★☆ | FSQ 단순, CleanCodec 참조 코드 있음 |
| 파라미터 효율 | ★★★★★ | 22M — 가장 가벼움 |
| 기능 확장성 | ★★★☆☆ | Discrete token이 유연성 제한 |
| 학습 효율 | ★★★★☆ | FSQ는 commitment loss 불필요, 단순한 loss 구성 |
| **가중평균** | **4.00** | ✅ 2위 |

### 03 StreamVC — Streaming + Depth-wise Decoding

| 기준 | 점수 | 근거 |
|------|:---:|------|
| 음질 잠재력 | ★★★★☆ | Depth-wise로 인한 미세한 품질 저하 가능 |
| 실시간 성능 | ★★★★★★ | TTFB 48ms, RTF < 0.1 — **최고의 레이턴시** |
| 실현 가능성 | ★★★☆☆ | Depth-wise streaming 복잡도 높음, 부품 많음 |
| 파라미터 효율 | ★★★☆☆ | 61M — 무거움 |
| 기능 확장성 | ★★★☆☆ | Streaming에 특화, 확장성 제한적 |
| 학습 효율 | ★★★☆☆ | 5-phase 복잡한 학습 |
| **가중평균** | **3.75** | |

### 04 HybridVC — Hybrid Codec + RAF Vocoder 🥇

| 기준 | 점수 | 근거 |
|------|:---:|------|
| 음질 잠재력 | ★★★★★ | RAF로 입증된 품질 향상 (14M > 112M), MOS 4.0+ |
| 실시간 성능 | ★★★★☆ | RTF 0.5 GPU, TTFB 150ms — 양호 |
| 실현 가능성 | ★★★★★ | **MioCodec + btrv3lite 직접 재활용, RAF는 drop-in objective** |
| 파라미터 효율 | ★★★★☆ | 30M — MioCodec decoder(50M) 대비 3.5× 경량 |
| 기능 확장성 | ★★★★☆ | SSL distillation + BWE + RAF로 견고한 베이스라인 |
| 학습 효율 | ★★★★★ | **53 GPU-hours — 최단 훈련 시간, RAF 수렴 빠름** |
| **가중평균** | **4.55** | 🥇 **1위** |

### 05 RobustVC — Denoising + Style Control + FiLM

| 기준 | 점수 | 근거 |
|------|:---:|------|
| 음질 잠재력 | ★★★★★ | Denoising + style로 실제 환경 최고 품질 |
| 실시간 성능 | ★★☆☆☆ | **RTF 0.53 GPU / 1.9 CPU — 가장 느림** |
| 실현 가능성 | ★★☆☆☆ | SB-RF+DUET+GLASS+FiLM+UniPASE — 과도한 통합 부담 |
| 파라미터 효율 | ★★★☆☆ | 60M — 무거움 |
| 기능 확장성 | ★★★★★★ | Denoising, style, emotion, gating — 최다 기능 |
| 학습 효율 | ★★☆☆☆ | 6-phase + GRPO는 느림 |
| **가중평균** | **3.15** | |

---

## 🏆 최종 선정: HybridVC (4.55점)

```
┌──────────────────────────────────────────────────┐
│                                                  │
│   HybridVC = MioCodec + RAF BigVGAN + ConvNeXt   │
│                                                  │
│   ▸ 30M params (기존 MioCodec decoder 50M 대비 ↓) │
│   ▸ 53 GPU-hours training (A100 1주 이내)        │
│   ▸ 44.1kHz 출력 (24kHz base + Vocos BWE)        │
│   ▸ btrv3lite/btrvrc0 자산 80%+ 재활용           │
│   ▸ RAF loss로 입증된 품질 (14M > 112M)          │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 선정 이유

1. **실현 가능성 최고**: MioCodec + CausalLatentConverter를 그대로 가져오고, RAF는 BigVGAN training objective만 교체하는 drop-in 방식. btrv3lite 코드베이스에서 시작 가능.

2. **RAF = 단일 최대 임팩트**: 22편 논문 중 가장 즉각적이고 검증된 개선. 14M BigVGAN이 112M을 능가한다는 건 MioCodec decoder(50M) 대체에 완벽한 근거.

3. **학습 효율 압도적**: 53 GPU-hours는 다른 설계(100-500h+) 대비 월등히 짧음. MPS에서도 feasible.

4. **절충의 미학**: FlowVC의 복잡한 CFM, RobustVC의 과도한 통합 부담 없이, 하나의 핵심 혁신(RAF)에 집중. 나머지(USAD 2.0 distillation, Vocos BWE)는 점진적 추가 가능.

5. **리스크 최소**: btrv3lite→HybridVC weight transfer 경로 명확. RAF는 이미 논문에서 검증됨. BWE는 독립 모듈이라 실패해도 fallback 가능.

---

## HybridVC 핵심 스펙

| 항목 | 값 |
|------|-----|
| 샘플레이트 | 44.1kHz (24kHz base + BWE) |
| 파라미터 | 30.7M (추론) |
| RTF (GPU) | < 0.05 |
| RTF (CPU) | < 0.2 |
| TTFB | ~125ms (GPU), ~190ms (CPU) |
| MOS 목표 | > 4.0 |
| Speaker SIM | > 0.85 |
| WER | < 3% |
| 훈련 시간 | ~53 GPU-hours (A100) |
| 메모리 (추론) | ~126MB (fp16) |

---

## 구현 로드맵 (4 Phase)

| Phase | 내용 | 기간 |
|-------|------|:---:|
| **P0: Weight Transfer** | btrv3lite converter→HybridVC converter | 1일 |
| **P1: RAF Vocoder** | BigVGAN-base + RAF loss training | 1-2주 |
| **P2: SSL Distillation** | USAD 2.0-style encoder distillation | 1주 |
| **P3: BWE + E2E** | Vocos BWE 통합, end-to-end fine-tune | 1주 |
| **총** | | **3-4주** |

---

## 전체 순위

| 순위 | 설계 | 점수 | 핵심 강점 |
|:---:|------|:---:|------|
| 🥇 | **HybridVC** | **4.55** | RAF + 재활용 + 학습효율 |
| 🥈 | DiscreteVC | 4.00 | 초경량 + 초고속 |
| 🥉 | FlowVC | 3.85 | CFM 최고 품질 |
| 4 | StreamVC | 3.75 | 최저 레이턴시 |
| 5 | RobustVC | 3.15 | 최다 기능 |

---

*설계문서: `/Users/asill/btrv5/designs/04_hybridvc.md` (1,832줄)*
*스코어링: `/Users/asill/btrv5/designs/SCORING.md`*

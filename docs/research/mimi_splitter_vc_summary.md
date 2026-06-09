# Mimi Splitter VC Research Summary

**Status: Negative Result — VC engine 실패, 연구적 발견은 유의미**

## 1. 연구 질문

초기 질문:
```
Mimi codec 내부의 semantic/acoustic split을 이용해서
저지연 voice conversion을 구현할 수 있는가?
```

중간 가설:
```
Mimi는 실패한 codec이 아니라, 단순 token swap으로는 안 되고
내부 factor를 splitter로 분리해야 하는 codec일 수 있다.
```

최종 진단:
```
Mimi 내부 구조 분석과 splitter 실험은 연구적으로 유의미했지만,
실제 usable VC 음성 생성에는 실패했다.
Mimi는 low-latency backbone으로는 매우 빠르지만,
VC용 speaker-conditioned decoder로는 부적합하거나
추가적인 강한 재합성 구조가 필요하다.
```

## 2. 핵심 발견

| Finding | Evidence | Interpretation |
|---------|----------|---------------|
| q0는 semantic codebook | `n_q_semantic=1` | Mimi는 이미 semantic/acoustic split 내장 |
| q1-q7은 acoustic codebook | `n_q_acoustic=7`, RVQ residual | Acoustic detail은 7개 codebook에 분산 |
| Decoder는 단일 latent만 받음 | `decode_latent(codes)` = Σ embed[code_i] | Per-codebook injection 불가 |
| q0는 speaker-neutral에 가까움 | Speaker probe: q0=8.3%, C=8.3% (chance=5%) | Content carrier로 검증 |
| q1-q7/A는 speaker leak 있음 | A speaker probe=33.3% | Acoustic path에 source speaker 잔존 |
| S는 speaker identity + shortcut | S speaker=81.7%, loudnorm cos=0.77, lowpass cos=0.82 | Pure identity 아님 |
| FiLM만 centroid shift 성공 | γ mean=-29.9, tanh(γ)≈-1 | Source acoustic zeroing + β injection |
| FiLM은 decoder OOD 유발 | z_vc vs z_rt L2=226, cosine=0.80 | Artifact/jitter 원인 |
| Latency는 매우 우수 | Encode 18.7ms + FiLM 0.4ms + Decode 9.7ms = 28.6ms | RTF 0.386, effective ~110ms |

### FiLM 작동 메커니즘

```text
z_mod = z_ac * (1 + tanh(γ)) + β

γ ≈ -30 → tanh(-30) ≈ -1
→ z_mod ≈ 0 * z_ac + β
→ source acoustic 완전 제거, target β만 주입

z_vc = C + β (거의)
```

즉 FiLM은 source acoustic을 target 방향으로 "변환"하는 게 아니라,
**source를 kill하고 target bias를 주입**하는 극단적 연산이다.
이게 화자전이(centroid 920→1429Hz)의 실체.

## 3. 실험 결과 요약

| Method | Centroid | Jitter | OOD L2 | Result |
|--------|----------|--------|--------|--------|
| **Source p255** | 1045Hz | 17.5% | — | — |
| **Target origin** | 1415Hz | 11.0% | — | — |
| FiLM v1 (n_c=1) | 1428Hz | 37.8% | 226 | 화자전이 성공, jitter 폭발 |
| FiLM (n_c=2) | 1001Hz | 26.2% | — | 화자전이 실패 |
| Transformer 4L | — | — | — | 과적합, 불안정 |
| Transformer 2L | — | — | — | 화자전이 약함 |
| Acoustic Generator | 1073Hz | 27.0% | — | 전이 실패 |
| Adv Acoustic | 958Hz | 23.2% | — | 전이 실패 |
| P-path + TCN v1 | 968Hz | 20.7% | — | 전이 실패 |
| Smooth adapter W=2.0 | 1056Hz | 25.0% | — | 전이 실패 |
| Smooth adapter W=0.3 | 1083Hz | 26.5% | — | 전이 실패 |
| P-path v2 + F0 norm | 968Hz | 20.7% | — | 전이 실패 |
| Post-process v5 | 1394Hz | 22.8% | — | centroid 유지 |
| Post-process v6 | 1411Hz | 25.5% | — | centroid 유지 |
| Temporal smooth post | 1275Hz | 36.9% | — | 더블링 발생 |
| **Refine α=0.10** | 1201Hz | 5.4% | 306 | 거의 깨끗, 약한 전이 |
| **Refine α=0.30** | 1229Hz | 7.1% | 260 | 최적 combo |
| **Refine α=0.80** | 1408Hz | 16.8% | 210 | 강한 전이, jitter 절반 |
| Splitter v2 | 1125Hz | 23.1% | 194 | OOD 개선, 전이 약화 |
| β-only (α=0) | 1182Hz | 5.5% | 331 | 드론, 음성 안 들림 |
| **Mimi RT** | 920Hz | 20.9% | 0 | Mimi 자체 reconstruction |

### 해석

- FiLM v1은 강한 화자/스펙트럼 shift를 만들지만 jitter와 artifact가 크다.
- Temporal adapter들은 안정성은 올릴 수 있었지만 speaker transfer를 죽였다.
- Refine alpha는 speaker transfer strength와 stability 사이의 연속 제어 노브를 제공했다.
- 그러나 **최종적으로 usable voice quality는 확보하지 못했다.**

## 4. 실패 원인

```
Mimi Splitter VC failure modes:

1. q0-only content는 speaker-clean하지만 acoustic/prosody detail이 부족하다.
   → LSD 27.9dB, VHigh 7% (source 14%)

2. q1-q7/A를 쓰면 source speaker leakage가 돌아온다.
   → A speaker probe 33.3%, centroid shift 사라짐

3. S는 speaker identity와 spectral/domain shortcut을 섞어서 사용한다.
   → loudnorm cos=0.77, lowpass cos=0.82

4. FiLM은 source acoustic을 zeroing하고 target beta를 주입해
   transfer를 만들지만, decoder manifold 밖으로 latent를 밀어
   artifact와 jitter를 만든다.
   → OOD L2=226, cos(z_vc, z_rt)=0.80

5. Post-process는 crest/VHigh는 개선했지만 F0 jitter는 해결하지 못했다.
   → Jitter 37.8% → 25.5% (여전히 높음)

6. 보코더/generative enhancer는 가능하지만 저지연 VC 목표와 충돌한다.
   → RTF 0.386 → enhancer 추가 시 0.5-1.0+
```

## 5. Latency 결과

| Component | CPU latency per 80ms chunk |
|-----------|---------------------------|
| Mimi encode | 18.7ms |
| FiLM adapter | 0.4ms |
| Mimi decode | 9.7ms |
| **Total p50** | **28.6ms** |
| **RTF** | **0.386** |
| **Effective latency** | **~110ms** |
| Target | <200ms — 달성 |

**해석:** Mimi + FiLM은 latency 측면에서는 매우 훌륭했지만,
output quality가 usable 수준에 도달하지 못했다.

## 6. 결론

```
Mimi는 연구 대상으로는 성공적이었다.
내부 codebook 구조, q0 content carrier, FiLM zeroing mechanism,
decoder OOD failure mode를 명확히 밝혔다.

그러나 실제 VC engine으로는 실패했다.
생성된 음성들은 noise, non-voice artifact, severe jitter,
trembling voice 문제를 보였고, usable voice가 확보되지 않았다.

따라서 MimiSplit VC는 negative result로 정리하고,
다음 단계는 VC 구조에 더 적합한 MioCodec을 정밀 분석하는 것이다.
```

## 7. 주요 파일

| Category | Files |
|----------|-------|
| Core splitter | `mimi_splitter_v2.py` |
| Adapter 실험 | `train_60spk.py`, `train_transformer.py`, `train_nc2.py`, `train_smooth_adapter.py`, `train_acoustic_gen.py`, `train_adv_acoustic.py`, `train_prosody.py`, `train_ppath_v2.py`, `train_splitter_v2.py` |
| Post-process | `post_process.py` ~ `post_process_v6.py` |
| Refine sweep | `sweep_refine.py`, `sweep_hardfilm.py` |
| Diagnosis | `diagnose_v2.py`, `diagnose_artifact.py`, `audit_structure.py` |
| Latency | `measure_latency.py`, `measure_mps*.py` |
| VC test | `vc_test.py`, `vc_origin.py`, `eval_compare.py`, `eval_wer.py` |
| Transformer adapter | `transformer_adapter.py` |
| Output audio | `~/Desktop/vc_*.wav`, `~/Desktop/vc_refine_*.wav` |
| Checkpoints | `checkpoints/mimi_splitter_v2_60spk.pt` 등 |

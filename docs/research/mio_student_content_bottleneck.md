# MioCodec Causal Student — Content Bottleneck Diagnosis

**Status:** 음색 이동 성공, content intelligibility 실패

## 현재 Pipeline

```
44100Hz source
→ causal logmel 50Hz [B, 80, T50]
→ Causal Content Student v1 (mel+TCN, 4-layer, 256dim)
→ FSQ 5d prediction @25Hz [B, 5, T25]
→ teacher FSQ hard quantize → proj_out
→ content_embedding 768d [B, T25, 768]
→ cached target global 128d [B, 128]
→ causal AdaLN-Zero mel decoder
→ teacher wave decoder
→ VC waveform @44100Hz
```

## 성공한 것

| Component | Status |
|-----------|--------|
| MioCodec teacher VC | usable quality (jitter 7.1%, crest 6.0) |
| Target global embedding | stable, cacheable, speaker conditioning works |
| Causal Content Student v1 | functional, generalizes to unseen speakers |
| Causal Mel Decoder | content-driven, global conditioning works |
| Global conditioning | tgt > src > other > zero ✓ |
| 음색/화자 이동 | 성공 |

## 실패한 것

| Attempt | Result |
|---------|--------|
| Hard FSQ path | token match 0%, content garbled |
| Soft FSQ path | worse than hard (OOD for teacher proj_out) |
| StudentProjOut | +0.053 cos, still unintelligible |
| Residual 768d embed | degraded FSQ base |
| Level CE | dominated MSE, worsened FSQ |
| Student-aware decoder | wash |
| VC pair decoder distill | global already works |
| Bigger TCN (v1.1) | too slow, killed |

## 핵심 수치

| Metric | Value |
|--------|-------|
| ContentStudent FSQ cos (aggregate) | ~0.899 |
| ContentStudent → teacher proj_out cos | 0.626 |
| Frame median cosine | 0.725 |
| Token match (12800-class) | 0.0% |
| StudentProjOut cos | 0.679 (+0.053) |
| StudentProjOut frame median | 0.744 |
| Causal decoder mel cos (student content) | 0.832 |
| Causal decoder mel cos (teacher content) | 0.918 |

## 병목 진단

```
문제: student output이 영어로 들리지 않음 (garbled)
원인: student content embedding이 teacher content trajectory를 충분히 따르지 못함
근본: 작은 TCN만으로 WavLM+Transformer teacher의 phonetic content를 distill 불가

해결: 
1. Teacher intermediate features (pre-FSQ, local SSL)를 추가 distillation target으로
2. Content encoder를 더 강한 구조로 업그레이드
3. Content embedding 768d를 main loss로 직접 최적화
4. FSQ 5d는 auxiliary로만 사용
```

## v2 설계 방향

```
Input: logmel [B, 80, T50]
→ Conv stem
→ causal Conformer/Transformer encoder (6-8 layers, dim 384-512)
→ causal downsample (50Hz→25Hz)
→ Heads:
    content_embed_768 (MAIN)
    pre_fsq_768 (intermediate distill)
    fsq_5d (auxiliary)
→ Loss: cosine + L1 on content_embed, cosine on pre_fsq, MSE on fsq
```

## v2 목표

| Metric | Current (v1 hard) | v2 Target |
|--------|-------------------|-----------|
| Content cos | 0.626 | ≥ 0.80 |
| Frame median cos | 0.725 | ≥ 0.85 |
| Anti-corr frames | 2.7% | < 1% |
| Mel cos (decoder) | 0.832 | ≥ 0.85 |

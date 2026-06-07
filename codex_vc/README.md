# Codex VC

Mimi 기반 음성 변환 — Codex 리뷰 아키텍처 구현.

```
source audio → Mimi encode → LV0 codes ──────────┐
target audio → Resemblyzer → spk embedding ───────┤
                                                   ↓
                                     Bidirectional Transformer
                                                   ↓
                                     LV1-7 codes (7×T)
                                                   ↓
                                     Mimi decoder → VC audio
```

## 구조

```
codex_vc/
├── model.py              # CodeGenerator (bidirectional transformer)
├── train.py              # Basic training script
├── train_eval_split.py   # 🆕 Split-based training (utterance/speaker val)
├── infer.py              # CLI inference (backward-compat checkpoint)
├── README.md
└── __init__.py
```

## 설치

```bash
pip install moshi resemblyzer soundfile scipy
```

## 사용법

### 학습 (split 기반 — 권장)

```bash
# Utterance split (20% val)
python codex_vc/train_eval_split.py \
    --cache runs/vctk_codes_full.pt \
    --spk-emb runs/vctk_full_spk.pt \
    --out runs/codex_model.pt \
    --steps 500 \
    --val-interval 50 \
    --ablation

# Speaker split (unseen speaker evaluation)
python codex_vc/train_eval_split.py --split-mode speaker --val-ratio 0.2 ...

# With random crop (prevents position memorization)
python codex_vc/train_eval_split.py --segment-frames 10 ...
```

### 추론

```bash
python codex_vc/infer.py \
    --source input.wav \
    --target-speaker p226 \
    --output vc_output.wav
```

## ⚠️ 주의: Overfitting

기존 `train_full3.py`의 **99.9% accuracy는 train set memorization**일 가능성이 높다:

- Train/val split이 없었음
- 같은 utterance가 train과 eval 모두에 등장
- Checkpoint를 train acc 기준으로 저장

**반드시 `train_eval_split.py`로 검증할 것.**

## 성능

| 조건 | Δ | 비고 |
|------|:--:|------|
| 5화자 parallel (train) | +0.70 | ⚠️ train set |
| 109화자 parallel (train) | +0.58 | ⚠️ train set |
| Cross-text (unseen spk) | -0.21 | 일반화 실패 |

> **train accuracy ≠ generalization.** utterance split val 기준으로 재평가 필요.

## Ablation 테스트

```bash
python codex_vc/train_eval_split.py --ablation ...
```

출력:
- `abl_target_acc`: 정상 target speaker embedding
- `abl_shuffled_acc`: shuffled speaker embedding
- `abl_zero_acc`: zero speaker embedding
- `abl_source_acc`: source speaker embedding

**해석:** target만 높고 나머지가 낮아야 speaker conditioning이 유효.

## 아키텍처 상세

- **LV0**: Mimi semantic/content codes (2048-dim codebook)
- **Speaker**: Resemblyzer 256-dim text-independent embedding
- **Transformer**: 3-layer bidirectional, d_model=256, 4 heads
- **Output**: 7 independent linear heads → LV1-7 acoustic codes
- **Decoder**: Frozen Mimi SEANet decoder

## References

- Kyutai Mimi: https://github.com/kyutai-labs/moshi
- Resemblyzer: https://github.com/resemble-ai/Resemblyzer
- Codex review: https://github.com/stremtec/astrape-vc/pull/7

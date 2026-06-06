# FlowVC コード品質監査レポート

エージェント #8: コード品質・デッドコード・カバレッジ
監査日: 2026-06-06
対象: flowvc/ 以下の全14ファイル (__init__.py, blocks.py, config.py, encoder.py, decoder.py, converter.py, speaker.py, prosody.py, cfm_loss.py, dataset.py, train.py, infer.py, tests.py, causality_audit.py)

---

## 1. 未使用インポート

### 1.1 speaker.py:12 — `torch.nn.functional` 未使用
```python
import torch.nn.functional as F
```
ファイル内で `F.` を一度も使用していない。
**修正案**: 削除する。

### 1.2 decoder.py:15 — `torch.nn.functional` 未使用
```python
import torch.nn.functional as F
```
ファイル内で `F.` を一度も使用していない。
**修正案**: 削除する。

### 1.3 causality_audit.py:11 — `torch.nn.functional` 未使用
```python
import torch.nn.functional as F
```
ファイル内で `F.` を一度も使用していない。
**修正案**: 削除する。

### 1.4 train.py:9 — `EncoderConfig`, `FlowConverterConfig` 未使用
```python
from .config import EncoderConfig, DecoderConfig, FlowConverterConfig
```
`DecoderConfig` のみ使用 (`F3Decoder(DecoderConfig())` at line 41)。`EncoderConfig`, `FlowConverterConfig` は使用されていない。
**修正案**: `from .config import DecoderConfig` に絞る。

### 1.5 infer.py:8 — `F3Encoder` 未使用
```python
from .encoder import F3Encoder, make_encoder
```
`make_encoder` のみ使用。`F3Encoder` は型アノテーションにも使われていない。
**修正案**: `from .encoder import make_encoder` に絞る。

### 1.6 infer.py:10 — `VectorFieldNet` 未使用
```python
from .converter import VectorFieldNet, solve_cfm_euler, make_vector_field_net
```
`solve_cfm_euler`, `make_vector_field_net` のみ使用。
**修正案**: `from .converter import solve_cfm_euler, make_vector_field_net` に絞る。

### 1.7 infer.py:11 — `SpeakerEncoder` 未使用
```python
from .speaker import SpeakerEncoder, make_speaker_encoder
```
`make_speaker_encoder` のみ使用。
**修正案**: `from .speaker import make_speaker_encoder` に絞る。

### 1.8 infer.py:12 — `FCPEProsodyExtractor` 未使用
```python
from .prosody import FCPEProsodyExtractor, make_prosody_extractor
```
`make_prosody_extractor` のみ使用。
**修正案**: `from .prosody import make_prosody_extractor` に絞る。

---

## 2. 到達不能コード / デッドコード

### 2.1 cfm_loss.py:77-136 — `FlowVCLoss` クラス (全60行) が未使用
`FlowVCLoss` は定義されているが、train.py・infer.py のどこからも import/使用されていない。
音声変換の統合損失として設計されたが、train.py では各 phase で個別に損失計算しており FlowVCLoss をバイパスしている。
**修正案**: 削除するか、train.py の Phase 2 (E2E) で FlowVCLoss を使用するように統合する。

### 2.2 dataset.py:177-198 — `CachedDataset` クラス (全22行) が未使用
`build_cache()` でキャッシュを構築するが、train.py は `CachedDataset` ではなく `VCTKDataset` を直接使用している。
**修正案**: train.py にキャッシュ利用オプション (`--cache-dir`) を追加し `CachedDataset` を活用するか、未実装なら削除する。

### 2.3 config.py:104-120 — `TrainConfig` dataclass (全17行) が未使用
`TrainConfig` は定義されているが、train.py はこれを全く使わず argparse で重複定義している。
**修正案**: train.py に `TrainConfig` を統合するか、TrainConfig を削除する。

### 2.4 speaker.py:140-143 — `SpeakerEncoder.encode()` が未使用
```python
def encode(self, wav: torch.Tensor) -> torch.Tensor:
    spk_emb, _ = self.forward(wav)
    return spk_emb
```
全ファイル中でこのメソッドを呼び出している箇所がない。train.py も infer.py も `speaker_enc(ref)` で `forward()` を直接呼んでいる。
**修正案**: 削除するか、infer.py で推論時のプロンプト不要ケースに使う。

### 2.5 converter.py:308-333 — `solve_cfm_rk4()` (全26行) が本番未使用
`solve_cfm_rk4` は tests.py の `test_cfm_ode_solver` でのみ呼ばれる。train.py/infer.py は Euler のみ使用。
**修正案**: infer.py に `--ode-method rk4` オプションを追加して活用するか、「将来の高品質オプション」としてコメントを明記。

### 2.6 config.py:98 — `FlowConverterConfig.ode_steps` がデッドフィールド
```python
ode_steps: int = 4  # 推論: Eulerステップ数
```
この値はどこからも `cfg.ode_steps` として読まれていない。train.py/infer.py/cfm_loss.py/tests.py 全てが `n_steps=4` をハードコードしている。
**修正案**: 各呼び出し箇所で `cfg.ode_steps` を参照するよう統一する。

---

## 3. 未使用変数・関数

### 3.1 __init__.py:13 — `__version__` が未使用
```python
__version__ = "0.1.0"
```
どのファイルからも `from flowvc import __version__` されていない。
**修正案**: 削除するか、CLI の `--version` で表示する。

### 3.2 train.py:137 — `elapsed` 変数が未使用
```python
elapsed = time.time()  # 代入のみで未使用
```
**修正案**: 削除するか、ログに経過時間を表示する。

---

## 4. コメントと実装の不一致

### 4.1 converter.py:7-9 — ブロック数コメント
```python
# 12 個の ConvNeXt v2 ブロック (dim=512) + AdaLN-Zero(時間, 条件)
```
実際のブロック数は `cfg.dilations` の長さで決まる（デフォルト12）。コメントは現状正しいが、dilations を変更するとコメントが嘘になる。
**修正案**: `# N 個の ConvNeXt v2 ブロック (N=len(cfg.dilations))` に変更。

### 4.2 converter.py:303 — 台形修正子コメント
```python
z = z + v_end * dt * 0.5  # trapezoidal corrector: uses velocity at both t≈1 and t=1
```
実際には `t=1` の1点のみ評価。`t≈1` は最終 Euler ステップの時刻。厳密には「最終 Euler ステップ (t≈1-dt) の速度と t=1 の速度の平均」。
**修正案**: コメントを `# trapezoidal corrector: (v(t≈1-dt) + v(t=1)) * dt/2` に修正。

### 4.3 config.py:104 — TrainConfig docstring
```python
class TrainConfig:
    data_dir: str = ""
```
docstring がない。他の Config クラスには docstring がある。
**修正案**: docstring を追加。

---

## 5. ハードコードされたマジックナンバー

### 5.1 重大: `n_steps=4` の重複ハードコード (6箇所)
| ファイル | 行番号 | コンテキスト |
|----------|--------|-------------|
| train.py | 123 | Phase 2 E2E CFM ODE |
| cfm_loss.py | 129 | FlowVCLoss spk_consistency |
| tests.py | 191, 196, 201 | CFM ODE solver test |
| causality_audit.py | 414, 418 | CFM ODE causality test (×2) |

`FlowConverterConfig.ode_steps` が定義されているが使われていない。
**修正案**: 全箇所を `cfg.ode_steps` 参照に統一する。

### 5.2 decoder.py:98 — `mlp_expansion=4` ハードコード
```python
ConvNeXtV2Block(latent_dim, kernel_size=cfg.kernel_size,
                mlp_expansion=4, use_grn=cfg.use_grn)
```
`DecoderConfig` に `mlp_expansion` フィールドがない。`EncoderConfig` には存在するのに欠落。
**修正案**: `DecoderConfig` に `mlp_expansion: int = 4` を追加し、ここで `cfg.mlp_expansion` を参照。

### 5.3 speaker.py:99 — `ConvNeXtV2Block` のパラメータ欠落
```python
ConvNeXtV2Block(out_ch, kernel_size=cfg.kernel_size)
```
`EncoderConfig` 相当の `mlp_expansion` と `use_grn` が `SpeakerEncoderConfig` にないため、ConvNeXtV2Block のデフォルトに依存している。
**修正案**: `SpeakerEncoderConfig` に `mlp_expansion: int = 4` と `use_grn: bool = True` を追加。

### 5.4 その他のマジックナンバー

| ファイル | 行 | 値 | 説明 | 推奨 |
|----------|-----|-----|------|------|
| blocks.py | 134 | `1e-4` | LayerScale 初期値 | `LAYERSCALE_INIT = 1e-4` 定数化 |
| blocks.py | 220 | `0.01` | out_gate 初期値 | 設定可能に (`FlowConverterConfig` に追加) |
| prosody.py | 49 | `0.006` | FCPE 閾値 | `FCPE_THRESHOLD = 0.006` 定数化 |
| prosody.py | 35 | `100` | FCPE 出力レート | `FCPE_OUTPUT_HZ = 100` 定数化 |
| train.py | 49,57 | `betas=(0.8, 0.9)` | AdamW betas | TrainConfig に移動 |
| train.py | 50 | `weight_decay=0.01` | Weight decay | TrainConfig に移動 |
| train.py | 61 | `0.1` | Phase 2 lr factor | TrainConfig に移動 |
| train.py | 101 | `0.1` | Latent consistency weight | TrainConfig に移動 |
| train.py | 129 | `1.0` | Grad clip max norm | TrainConfig に移動 |
| dataset.py | 117 | `0.5` | 同/異話者比 | 設定可能に |
| cfm_loss.py | 129 | `n_steps=4` (call) | ODE steps | cfg 参照に |
| decoder.py | 119 | `kernel_size=7` | 最終畳み込み | DecoderConfig に追加 |
| decoder.py | 85 | `speaker_dim=192` | デフォルト話者次元 | config と一致確認のみ |

---

## 6. エラーハンドリングの欠如

### 6.1 prosody.py:53-54 — NumPy 変換のデバイス判定が脆弱
```python
f0_np = f0.squeeze(0).cpu().numpy() if f0.is_cuda or f0.device.type == "mps" else f0.squeeze(0).numpy()
```
MPS デバイスでも `.cpu().numpy()` が動作する PyTorch 2.x+ では二重判定は不要だが、古いバージョンでは必要。また `f0.is_cuda` が False かつ MPS でもない場合（CPU 以外の未知デバイス）のフォールバックがない。
**修正案**: `f0_np = f0.detach().cpu().numpy()` で統一（CPU テンソルは no-op）。

### 6.2 dataset.py:79-82 — 広すぎる例外捕捉
```python
try:
    info = torchaudio.info(self.files[0])
    self.sr_orig = info.sample_rate
except Exception:
    self.sr_orig = sample_rate
```
`Exception` で全て捕捉しているため、ファイル破損やメモリ不足も沈黙処理される。
**修正案**: 最低限 `except (RuntimeError, OSError)` に絞り、警告を print する。

### 6.3 train.py:88-89 — エポック枯渇の未処理
```python
while step < args.steps:
    for batch in loader:
        if step >= args.steps:
            break
```
`DataLoader` が 1 epoch 分のデータを返し終わると `for batch in loader` が終了し、そのまま外側の `while` が無限ループになる（step < args.steps は真のまま）。
**修正案**: `for batch in loader` の外で `if step >= args.steps: break` するか、`loader` を無限繰り返しにする。

### 6.4 infer.py:154-156 — チェックポイント欠落時の無警告フォールバック
```python
vfn.load_state_dict(ckpt.get("vfn", vfn.state_dict()))
```
`"vfn"` キーがチェックポイントにない場合、ランダム初期重みのまま無警告で進む。
**修正案**: `strict=False` でロードし、欠落キーがある場合は警告を出力する。

### 6.5 train.py:29 — チェックポイント不一致のサイレント失敗
```python
model.load_state_dict(ckpt[name])
```
モデル構造がチェックポイントと異なる場合、`RuntimeError` で即死する。`strict=False` により不一致を警告に留めるべき。
**修正案**: `model.load_state_dict(ckpt[name], strict=False)` とし、欠落/余剰キーをログ出力。

---

## 7. 型ヒントの完全性

### 7.1 戻り値型ヒント欠如 (関数/メソッド)

| ファイル | 行 | シンボル | 欠如 |
|----------|-----|----------|------|
| blocks.py | 182 | `AdaLNZero.forward()` | 戻り値型 |
| decoder.py | 74 | `DecoderStage.forward()` | 戻り値型 |
| decoder.py | 151 | `make_decoder()` | 戻り値型 |
| encoder.py | 84 | `make_encoder()` | 引数型 + 戻り値型 |
| converter.py | 338 | `make_vector_field_net()` | 戻り値型 |
| speaker.py | 146 | `make_speaker_encoder()` | 戻り値型 |
| prosody.py | 23 | `_ensure_fcpe()` | 戻り値型 |
| prosody.py | 31 | `_ensure_resamplers()` | 戻り値型 |
| dataset.py | 203 | `build_cache()` | 戻り値型 |
| dataset.py | 246 | `main()` | 戻り値型 |
| train.py | 19 | `save_checkpoint()` | 戻り値型 |
| train.py | 35 | `train()` | 引数型 (`args` は未指定) + 戻り値型 |
| train.py | 158 | `main()` | 戻り値型 |
| infer.py | 137 | `load_models()` | 戻り値型 (tuple) |
| infer.py | 160 | `convert_file()` | 戻り値型 |
| infer.py | 181 | `profile()` | 戻り値型 |
| infer.py | 236 | `main()` | 戻り値型 |
| causality_audit.py | 全関数 | 全テスト関数 | 戻り値型 |
| tests.py | 全テスト関数 | — | 戻り値型 (pytestでは慣例的に省略可だが付与推奨) |

### 7.2 引数型ヒント欠如

| ファイル | 行 | シンボル | 欠如 |
|----------|-----|----------|------|
| encoder.py | 84 | `make_encoder(**kwargs)` | `**kwargs` の型 |
| converter.py | 338 | `make_vector_field_net(**kwargs)` | `**kwargs` の型 |
| speaker.py | 146 | `make_speaker_encoder(**kwargs)` | `**kwargs` の型 |
| decoder.py | 151 | `make_decoder(speaker_dim, **kwargs)` | `**kwargs` の型 |
| train.py | 35 | `train(args)` | `args` の型 |
| train.py | 19 | `save_checkpoint(models, opt, step, path)` | `models` dict の型詳細 |
| infer.py | 19 | `FlowVCInference.__init__()` | encoder, decoder, vfn 等の型 |

---

## 8. テストカバレッジの抜け

### 8.1 未テストのモジュール/クラス
| 不足項目 | 深刻度 |
|----------|--------|
| `FCPEProsodyExtractor` (prosody.py 全体) | **高** — 外部依存 (torchfcpe) がありバグリスク大 |
| `CausalConvTranspose1d` (blocks.py:44) | **中** — デコーダの重要コンポーネント |
| `DecoderStage` 単体 (decoder.py:59) | **中** — 複合テストのみ |
| `MRFBlock` 単体 (decoder.py:21) | **中** — 複合テストのみ (causality_audit はある) |
| `FlowVCLoss` (cfm_loss.py:77) | **低** — デッドコード (未使用) |
| `CachedDataset` (dataset.py:177) | **低** — デッドコード (未使用) |
| `PromptTokenGenerator` 単体 (speaker.py:39) | **中** — SpeakerEncoder のテストで間接的のみ |
| `TimeMLP` / `SinusoidalEmbedding` (converter.py) | **低** — VectorFieldNet のテストで間接的 |
| `SpeakerCrossAttn` 単体 (converter.py:79) | **中** — causality_audit ではテスト有、tests.py には無 |
| `FlowVCInference.process_chunk` (infer.py:54) | **高** — ストリーミングの中核 |
| `convert_file` / `profile` (infer.py:160,181) | **中** — CLI 関数 |
| 全フェーズ統合テスト (train.py) | **高** — 1 step の forward pass が通るか未検証 |
| 勾配チェック / 数値安定性テスト | **中** — FP16 safety 未確認 |

### 8.2 カバレッジのある項目（tests.py 9件 + causality_audit.py 15件）
- encoder/decoder roundtrip ✅
- CausalConv1d causality ✅ (causality_audit でより詳細)
- GRN normalization ✅
- Zero-init identity ✅
- CFM ODE solver (Euler + RK4) ✅
- Streaming buffer + overlap-add ✅
- Checkpoint save/load ✅
- CFM loss gradient flow ✅
- Speaker encoder shape ✅
- Dataset integration ✅ (VCTK 依存)

**カバレッジ推定**: モジュール単位で ~60%、コード行単位で ~45%

---

## 9. config と実装の二重管理

### 9.1 根本問題: TrainConfig と argparse の二重定義
`config.py` に `TrainConfig` dataclass が定義されているが、`train.py` は `argparse` で全項目を再定義している。ハイパーパラメータの真実が2箇所に分散。
**修正案**: `TrainConfig` を基本とし、argparse で上書きする。または `TrainConfig` を削除し argparse に一元化する。

### 9.2 `ode_steps` の二重管理
`FlowConverterConfig.ode_steps` が定義されているが、全呼び出し箇所がハードコード `n_steps=4` を使っている。
**修正案**: `solve_cfm_euler` と `solve_cfm_rk4` のデフォルト値を削除し、呼び出し側で `cfg.ode_steps` を必ず渡す。

### 9.3 `DecoderConfig` の `mlp_expansion` 欠落
| Config class | mlp_expansion | use_grn |
|-------------|:---:|:---:|
| EncoderConfig | ✅ (L24-26) | ✅ |
| DecoderConfig | ❌ | ✅ (L50) |
| SpeakerEncoderConfig | ❌ | ❌ |
| FlowConverterConfig | ✅ (L88) | N/A |

`DecoderConfig` と `SpeakerEncoderConfig` に `mlp_expansion` と `use_grn` (後者のみ) が欠落しており、実装側でハードコードされている。
**修正案**: 両 Config に `mlp_expansion: int = 4` と `use_grn: bool = True` を追加。

### 9.4 `speaker_dim=192` の分散定義
`192` が SpeakerEncoderConfig, FlowConverterConfig, DecoderConfig (make_decoder デフォルト), PromptTokenGenerator デフォルト等に分散。すべて `SpeakerEncoderConfig.speaker_dim` を参照すべき。
**修正案**: 単一の定数 (`SPEAKER_DIM = 192`) または Config 間の整合性検証を追加。

---

## 集計サマリ

| カテゴリ | 重大 | 警告 | 軽微 | 合計 |
|----------|:---:|:---:|:---:|:---:|
| 1. 未使用インポート | 0 | 5 | 3 | 8 |
| 2. デッドコード | 3 | 2 | 1 | 6 |
| 3. 未使用変数 | 0 | 0 | 2 | 2 |
| 4. コメント不一致 | 0 | 2 | 1 | 3 |
| 5. マジックナンバー | 2 | 5 | 8 | 15 |
| 6. エラーハンドリング | 2 | 3 | 0 | 5 |
| 7. 型ヒント欠如 | 0 | 10 | 8 | 18 |
| 8. テストカバレッジ | 3 | 7 | 2 | 12 |
| 9. config二重管理 | 2 | 2 | 0 | 4 |
| **合計** | **12** | **36** | **25** | **73** |

**総評**: コードベースは機能的に設計されているが、初期開発段階のため多くの「TODO的」な粗さが残る。特に **未使用デッドコード (FlowVCLoss, CachedDataset, TrainConfig)**、**n_steps=4 の重複ハードコード**、**テストカバレッジ不足 (prosody, streaming)** が運用上のリスク。config と実装の二重管理は保守性を著しく損なうため早期解消を推奨。

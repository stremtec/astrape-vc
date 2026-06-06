# 3次深層監査 — Batch B: アーキテクチャ設計 + 学習力学

**対象ファイル**: `encoder.py` (F³Encoder), `decoder.py` (F³Decoder, MRFBlock, DecoderStage), `train.py` (3-Phase 学習)

**分析日付**: 2026-06-06  
**分析範囲**: encoder.py 86【줄】, decoder.py 153【줄】, train.py 174【줄】 + 従属ファイル (blocks.py, config.py, converter.py, cfm_loss.py, speaker.py, prosody.py)

---

## 1. Encoder Downsampling 戦略分析

### 現在の設計

| Stage | Channel | Stride | Kernel (stride×3) | 累積DS | ナイキスト周波数 |
|-------|---------|--------|--------------------|---------|---------------|
| 1 | 32 | 2 | 6 | ×2 | 11,025 Hz |
| 2 | 64 | 2 | 6 | ×4 | 5,512 Hz |
| 3 | 128 | 3 | 9 | ×12 | 1,837 Hz |
| 4 | 256 | 3 | 9 | ×36 | 612 Hz |
| 5 | 512 | 7 | 21 | ×252 | 87 Hz |
| 6 | 768 | 7 | 21 | ×1,764 | 12.5 Hz |

総ダウンサンプリング: 2×2×3×3×7×7 = **1,764【배】** → 44,100Hz → **25Hz**

### anti-aliasing 十分性: `kernel_size = stride × 3`

**理論的基準 (Nyquist-Shannon)**:
- 理想的な anti-aliasing フィルタは cutoff frequency ≤ f_s / (2 × stride) を持つ必要がある
- `kernel_size ≥ 2 × stride` は 最小条件 (カーネルが 1周期 以上のサンプルを カバー)
- `kernel_size = stride × 3` は が 最小条件を 1.5【배】 【초과】 → **最小基準は満足**

**実際のスペクトル観点**:
- stride=2, kernel=6: 【오버랩】 = 4 samples → Nyquist 11kHz. 6-tap フィルタで 11kHz 【이상】を 【감쇠시키】は 【것은】 **【매우】 【거친】 【근사】**. 【가청】 【대역】(20Hz~20kHz)で 【충분한】 【감쇠를】 【보장하지】 【않음】.
- stride=7, kernel=21: 【오버랩】 = 14 samples → Nyquist ~3.15kHz. 21-tap フィルタは 44.1kHzで 【약】 0.48ms 【길이】. が 【정도면】 【중간】 【정도】の フィルタ【링은】 【가능하지만】 スペクトル foldingを 【완전히】 防止【할】 【만큼】 sharp【한】 cutoff【를】 学習【하기】は 【어려움】.

**問題【점】 (HIGH)**:
```
学習【된】 strided convは anti-aliasing LP filterが 【아님】 —
損失 【함수】(L1 reconstruction)【에】 【의해】 【형성되며】 【명시적】 周波数 【제약】が 【없음】.
ConvNeXtV2 【블록】が 【후처리하지만】 【이미】 aliasing【된】 【피처에】は 【소용】 【없음】.
```

### 素数(prime) factor 7の スペクトル 【아티팩트】 【가능성】

stride=7が 【포함된】 【총괄】 downsampling factor 1764 = 2² × 3² × 7².

**アーティファクトメカニズム**:
- stride=7で downsampling 【시】, 入力 信号の 7【샘플마다】 1【샘플】を 【취하게】 【됨】 (convolutional sampling【이므】で weighted average)
- 44.1kHzで 7の 【배수】 周波数 (6.3kHz, 12.6kHz, 18.9kHz, ...)が folding【되어】 【저】周波数 【대역에】 aliasing 【유발】
- 【특히】 音声の fricative, sibilant 【에너지】が 【집중된】 4~8kHz 【대역】が 0~3kHzで folding【될】 【위험】

**Prime strideの 【특수성】**:
- stride 2, 3【은】 harmonic seriesの 【일부이므】で 【배음】 構造と 【자연스럽게】 align【될】 【수】 【있음】
- stride 7【은】 【대부분】の 音声 【배음】 周波数(f₀×n)と 【정수배】 【관계】が 【아님】 → **【비조화적】(non-harmonic) aliasing 【성분】** 【생성】
- 【이】は 【지각적으】で "buzzy"【하거나】 "metallic"【한】 【아티팩트】で 【나타날】 【수】 【있음】

**推奨**:
- Stride 7 段階【에】 **【명시적】 BlurPool (anti-aliased max pooling)** 【또】は low-pass filter pre-convolution 【적용】
- 【또】は stride 7を (2, 2, 2) 【또】は (2, 3)【으】で 【분해】 (stage 【수】 【증가】)
- 【최소한】 stride 7 stage 【이후에】 spectral regularization loss (STFT 【기반】) 【추】が 【검토】

---

## 2. Decoder Upsampling 戦略分析

### TransposedConv + MRFBlock 構造

```
DecoderStage:
  CausalConvTranspose1d(kernel=stride×2+1, stride=stride)
  → MRFBlock(dim, kernels=(3,7,11), dilations=((1,3,5),...))
  → MRFBlock(dim, ...)
  → FiLM(speaker)
```

### Checkerboard 【아티팩트】 【위험성】

**ConvTranspose1dの checkerboard 【메커니즘】**:
- Transposed convolutionで `kernel_size % stride ≠ 0` 【일】 【때】 出力【에】 【주기적】 【진폭】 【변조】 【발생】
- 【모든】 decoder stageで が 条件 【위반】:

| stride | kernel (stride×2+1) | kernel % stride | Checkerboard 【위험】 |
|--------|----------------------|-----------------|-------------------|
| 7 | 15 | 1 ≠ 0 | **HIGH** |
| 3 | 7 | 1 ≠ 0 | **HIGH** |
| 2 | 5 | 1 ≠ 0 | **HIGH** |

**【구체적】 【예시】 (stride=7, kernel=15)**:
- 入力 feature mapの 【각】 【요소】が 15【개】の 出力 【위치에】 "scatter"【됨】
- stride=7【이므】で scatter 【간격은】 7
- 15【를】 7で 【나눈】 【나머지】が 1【이므로】, 【인접한】 出力 【위치들】が 【받】は contribution 【수】が 【다름】
  - 【어떤】 出力 【위치】は 2【개】 inputの overlap 【영역】, 【다른】 【위치】は 3【개】 inputの overlap 【영역】
  - が 【불균일】が 【주기적】 【패턴】 = checkerboard artifact

**MRFBlockの 【완화】 【효과】**:
- MRFBlock【은】 depthwise conv 【기반】 residual block
- Multi-scale カーネル(3, 7, 11)【과】 dilation(1, 3, 5)【으】で 【다양한】 receptive field カバー
- **【완화】 【효과】は 【제한적】** — MRFは 【원래】 HiFi-GANで multi-scale 【패턴】 【캡처용이지】 anti-checkerboard 【용도】が 【아님】
- 【체커보드】 【아티팩트】は transposed conv 【자체】の 構造【적】 問題【이므】で 【후처리】で 【완전히】 【제거하기】 【어려움】

### 【큰】 stride 【먼저】 【적용하】は 【이유】 分析

Decoder stride 【순서】: **(7, 7, 3, 3, 2, 2)** — Encoderの 【역순】.

**【해상도】 【진행】**:
```
25Hz  →(×7)→   175Hz  →(×7)→  1,225Hz  →(×3)→  3,675Hz
      →(×3)→ 11,025Hz →(×2)→ 22,050Hz →(×2)→ 44,100Hz
```

**長所** (【이론적】):
- Autoencoderの encoder-decoder 【대칭성】 【보존】 (mirror architecture)
- 【저해상도】 latentで 【큰】 【폭으】で 【확장】 → global structure【를】 【먼저】 【복원】
- 【후반부】 【작은】 strideで fine detail 【복원】

**短所** (【실제적】):
- **Stage 1 (25Hz → 175Hz)**: 【가장】 【압축된】 latentで 7【배】 【업샘플링】. 【정보】が 【가장】 【적은】 状態で 【가장】 【큰】 【확장】を 【수행】 → **hallucination 【위험】**
- stride 7 transposed convの checkerboardが 【가장】 【낮은】 【해상도】で 【발생】 → 【이후】 【모든】 stageで 【전파됨】
- 【일반적】 【관행은】 **【작은】 stride → 【큰】 stride** 【순서】 (【점진적】 【확장】)

**推奨**:
- `kernel_size = stride * 2` (kernel % stride == 0 【보장】) → checkerboard 【원천】 【제거】
- 【또】は transposed conv 【대신】 **subpixel convolution** (nn.PixelShuffleの 1D【버전】) 【또】は **linear interpolation + Conv1d** 【조합】 【사용】
- Stride 【순서를】 (2,2,3,3,7,7)で 【변경하여】 【점진적】 【확장】 【검토】 (【단】, latent dim=768 → 16chで 【가】は 構造 【변경】 【필요】)

---

## 3. CausalConvTranspose1d 出力 【트리밍】 分析

### 【현재】 実装

```python
class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        self.stride = stride
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        out = self.conv(x)                    # L_out = (L_in - 1) * S + K
        expected_len = x.shape[2] * self.stride  # L_in * S
        return out[:, :, :expected_len]        # 【앞부분만】 【취함】
```

### 【기대】 出力 【길】が 分析

ConvTranspose1d(padding=0)の 出力:
```
L_out = (L_in - 1) × stride + kernel_size
```

【트리밍】 【후】:
```
L_trimmed = L_in × stride
```

**【버려지】は 【샘플】 【수】**: `L_out - L_trimmed = kernel_size - stride`

| stride | kernel (stride×2+1) | 【버려지】は 【샘플】 |
|--------|----------------------|---------------|
| 7 | 15 | 8 |
| 3 | 7 | 4 |
| 2 | 5 | 3 |

総【버려지】は 【샘플】 ≈ 22 (【최종】 44.1kHz 【도메인에서】)

### 【인과성】(Causality) 【보장】 検証

**Encoder 【측】 【인과적】 【연산】**:
- `CausalConv1d` with padding=(kernel-1, 0): `output[t]`は `input[t - (kernel-1)*dilation : t*stride]` 範囲【에만】 【의존】
- Encoderの 【마지막】 出力 【위치】 `i`は 入力 【시간】 `~i*1764`【까지】の 【정보를】 【포함】

**Decoder 【측】 【인과적】 【연산】**:
- `padding=0` ConvTranspose1d: 入力 【위치】 `i`の 【정보】が 出力 【위치】 `[i*S, i*S + K - 1]`で scatter【됨】
- 出力 【위치】 `t` (t < K-1): 入力 【위치】 0【만】 【참조】 → **【인과적】** (【미래】 【정보】 【없음】)
- 出力 【위치】 `t` (t ≥ K-1, t < L_in*S): 【여러】 入力 【위치】の overlap → 【여전히】 【인과적】
- 出力 【위치】 `t` (t ≥ L_in*S): 【가장】 【마지막】 入力の right tail【만】 → 【정보】 【불완전】

**【트리밍】の 【효과】**:
- `out[:, :, :L_in*S]`: 【오른쪽】 tail (【정보】 【불완전】 【구간】) 【제거】 → 【인과성에】 影響 【없음】
- 【왼쪽】 warm-up 【구간】 (t < K-1)【은】 【유지】 → が 【구간은】 入力 0の 【정보만】 【사용하므】で 【인과적이지만】, **【정보】が 【불완전】**【할】 【수】 【있음】

### `padding=kernel_size-1` 提案 評価

"padding=kernel_size-1を 【주고】 【트리밍】を 【없애라】"は 提案:
```
L_out = (L_in - 1)*S + K - 2*(K-1) = L_in*S - S - K + 2
```

- stride=7, K=15: `L_out = L_in*7 - 7 - 15 + 2 = L_in*7 - 20`
- **【필요한】 出力 【길이】(L_in×7)【보다】 【짧아짐】** → reconstruction mismatch 【발생】

**【정확한】 causal inversionを 【위한】 条件**:
- Encoderの `CausalConv1d(kernel=S×3, stride=S)`【를】 inversion【하려면】
- Decoderで `ConvTranspose1d(kernel=S×3, stride=S, padding=kernel-1)` 【사용】 【후】
- `L_out = L_in*S - S + 1` → 【여전히】 【모자람】

**結論**:
【현재】の 【트리밍】 方式【은】 **【실용적】 【타협】**【이다】. 【완벽한】 causal inversion【은】 encoderの left-padding【으】で 【인해】 【불가능하며】, 【현재】 方式【은】 出力 【길이를】 L_in × Sで 【확정하여】 downstream【과】の 【호환성】を 【보장한다】.

**【경미한】 改善【안】**:
- 【트리밍】 【전에】 left side【도】 kernel_size - stride 【만큼】 【제거하여】 出力が 【완전한】 receptive field【를】 【갖도록】 【함】
- 【또】は ConvTranspose1d 【대신】 **linear interpolation + Conv1d**【를】 【사용하여】 checkerboardと causal alignment【를】 【동시에】 解決

---

## 4. 3-Phase 学習力学 分析

### Phase【별】 【개요】

| Phase | 学習 【모듈】 | 【동결】 【모듈】 | Loss | 【목적】 |
|-------|----------|----------|------|------|
| 0 | Encoder, Decoder | — | L1(recon, src) | AE 【사전】学習 |
| 1 | VFN (VectorFieldNet) | Encoder, Decoder, SpeakerEnc, Prosody | CFM MSE | Flow 学習 |
| 2 | Encoder, Decoder, VFN | SpeakerEnc, Prosody | L1(out, tgt) | E2E 【미세조정】 |

### Phase 0: AE 【재구성】の 問題【점】 (CRITICAL)

**Phase 0の loss 【구성】**:
```python
z = encoder(src, training=True)          # src waveform → latent
spk_emb, prompt = speaker_enc(ref)       # REFERENCE speaker embedding
recon = decoder(z, spk_emb)              # latent + ref speaker → waveform
loss = F.l1_loss(recon, src)             # reconstruct SOURCE waveform!
```

**【근본적】 【모순】**:
- `z`は **src 話者**の 音声で 【추출된】 latent
- `spk_emb`は **ref 話者**の speaker embedding (VCTK データ【셋】で refは 【타겟】 話者)
- Decoderは "srcの 【내용】 + refの 話者" 【정보】で **src 【파형】を 【복원】**【해야】 【함】
- 【이】は decoderが **話者 【정보를】 【무시하고】 latent【에】 【의존】**【하도록】 学習【시킴】

**結果【적】 影響**:
1. **Disentanglement 【실패】**: Latent zが 話者 【정보를】 【포함해야만】 decoderが src【를】 【복원】 【가능】 → zは content + speaker 【혼합】 【표현】
2. **Decoderの FiLM 条件【화】 【무력화】**: Decoderは 学習 過程で "FiLMが 【주】は 話者 【정보】 ≠ 出力 話者"【를】 【경험】 → FiLMを 【신뢰하지】 【않게】 【됨】
3. **Phase 1 【악】影響**: VFNが 学習【하】は latent 【공간】が content-onlyが 【아니므로】, flow 【변환】が content 【보존】 + speaker 【변환】を 【동시에】 【수행해야】 【함】 (【난이도】 【상승】)

**Phase 0 改善 【방안】**:
```python
# 提案 1: src = ref (【자기】 【재구성】)
z = encoder(src, training=True)
spk_emb, _ = speaker_enc(src)          # 【동일】 話者!
recon = decoder(z, spk_emb)
loss = F.l1_loss(recon, src)

# 提案 2: ref 【기반】 reconstruction + speaker consistency loss 【추가】
z = encoder(src, training=True)
spk_emb_src, _ = speaker_enc(src)
spk_emb_ref, _ = speaker_enc(ref)
recon_self = decoder(z, spk_emb_src)
loss_recon = F.l1_loss(recon_self, src)
# + speaker consistency: decoder(z, spk_emb_ref)と srcの content 【유사도】
```

### Phase 1: CFM 学習の latent 【적합성】

**CFMの 【동작】**:
- `z_src = encoder.encode(src)`, `z_tgt = encoder.encode(tgt)` (noise 【없음】)
- `z_t = (1-t)*z_src + t*z_tgt + σ_min·ε` (OT path)
- `v_target = z_tgt - z_src` (constant velocity field)
- VFNが `v_θ(z_t, t, c)` → `z_tgt - z_src`【를】 【예측하도록】 学習

**Latent 【공간】 品質 【요구】事項**:
- 【이상적】: `z_src`と `z_tgt`が contentは 【동일하고】 speaker【만】 【다른】 【표현】
- 【현실】 (Phase 0 設計 【결함으로】): `z_src`と `z_tgt`【에】 contentと speaker 【정보】が 【혼재】
- CFM【은】 【직선】 【보간】(OT path)を 仮定【하는데】, 【혼합】 【표현】の 【직선】 【보간】が semantically valid【한지】 【의문】

**【직선】 【보간】の 問題**:
- Content+speaker 【혼합】 【공간】で `(1-t)*z_src + t*z_tgt`は t=0.5【일】 【때】 【두】 話者の 【중간】 【목소리】 + 【중간】 【내용】が 【됨】
- が "【중간】 状態"が 【실제】 voice conversionの 【물리적】 過程を 【대표하는가】? → **【의문】**
- Disentangled 【표현이었다면】: contentは 【불변】, speaker【만】 interpolate → 【훨씬】 【자연스러운】 flow path

### Phase 2: E2E 【미세조정】の gradient path

```python
z_src = encoder.encode(src)           # grad flows (no noise)
z_tgt = solve_cfm_euler(vfn, ...)     # grad flows through 4 Euler steps
out = decoder(z_tgt, spk_emb)         # grad flows
loss = F.l1_loss(out, tgt)
```

**Gradient 【흐름】**: loss → decoder → solver(4 VFN calls) → encoder

**問題【점】**:
1. **ODE solver 【통한】 【역전파】**: 4-step Euler + half-step refinement → 総5【회】の VFN forward → メモリ 【사용량】 5【배】, gradient instability 【위험】
2. **Encoderの 【이중】 【역할】**: Phase 2で encoderは "content 【추출】"【과】 "flow-friendly latent 【생성】"を 【동시에】 学習 → 【목표】 【충돌】 【가능성】
3. **Noise regularization 【부재】**: `encoder.encode()`は noise 【없음】 → Phase 0で 学習【한】 noise robustnessが Phase 2で 【활용되지】 【않음】

---

## 5. Optimizer 分析: betas=(0.8, 0.9), weight_decay=0.01

### betas 【비교】

| パラメータ | FlowVC 【값】 | PyTorch 【기본값】 | 【일반적】 【관행】 |
|----------|----------|---------------|------------|
| β₁ | **0.8** | 0.9 | 0.9 |
| β₂ | **0.9** | 0.999 | 0.999 |

**β₁ = 0.8 (【첫】 【번째】 【모멘텀】 【감쇠율】)**:
- 【낮은】 β₁ → 【최근】 勾配【에】 【더】 【빠르게】 【적응】, 【과거】 【모멘텀】 【빠르게】 【소멸】
- **長所**: batch_size=1の 【고분산】 勾配で 【과거】 【노이즈】が 【현재】 stepを 【오염시키】は 【것】 防止
- **短所**: 【모멘텀】 【효과】 【감소】 → saddle point 【탈출】 【느려짐】, long-term consistency 【저하】
- batch_size=1 【환경에서】は **【합리적】 【선택】**

**β₂ = 0.9 (【두】 【번째】 【모멘텀】 【감쇠율】)**:
- PyTorch 【기본값】 0.999 【대비】 **【급격한】 【적응】 【속도】**
- Adamの adaptive learning rate: `lr / (√v̂ + ε)` — β₂が 【낮을수록】 v̂(【제곱】 勾配 【평균】)が 【빠르게】 【변함】
- **長所**: 3-phase 学習で phase 【전환】 【시】 loss landscape 【변화에】 【빠르게】 【적응】
- **短所**: 
  - 学習 【후반부에】 learning rate 【변동성】 【증】が → 【수렴】 【불안정】
  - Sparse gradient 【환경】で 【분모】が 【너무】 【빨리】 【변해】 step sizeが erratic
  - 94M パラメータ 【규모에서】は 【일부】 パラメータの 勾配が 【드물게】 【발생】 → instability 【위험】

**推奨**: β₂ = 0.95 【또】は 0.98で 【상향】 【조정】 【검토】. Phase 【전환】 【시에만】 optimizer state resetを 【고려】.

### weight_decay = 0.01 分析

- AdamWで weight_decayは L2 正規化と 【유사하지만】 gradientと decoupled
- 94M パラメータ, batch_size=1, steps=200K 【상황에서】:
  - weight_decay=0.01 → 【매】 step【마다】 weightが 0.01 × lr 【만큼】 【감쇠】
  - lr=2e-4 → 【유효】 【감쇠율】 = 2e-6/step → 200K step 【후】 【약】 33% 【감쇠】
- **【적절한】 範囲**: 【일반적으】で 0.0001 ~ 0.1. 0.01【은】 【중간】 【정도】.
- batch_size=1 + 【상대적으】で 【높은】 weight_decay → **【과도한】 正規化 【위험】**
- 【특히】 encoder-decoderが Phase 0【에서만】 200K step 学習【할】 【때】, over-regularization【으】で 【표현력】 【제한】 【가능】

**推奨**: Phase 0【에서】は weight_decay=0.01, Phase 1【에서】は 0.001, Phase 2【에서】は 0.005で phase【별】 【차등】 【적용】 【검토】.

---

## 6. Phase 1 Freeze 戦略分析

### LayerNorm Running Stats 【동결】

```python
encoder.eval()           # → LayerNormが running stats 【사용】, 更新 【중단】
p.requires_grad = False  # → gamma, beta パラメータ 【동결】
```

**Running stats 更新 【메커니즘】** (PyTorch 【내부】):
- `training=True`【일】 【때만】 `running_mean`, `running_var`が momentum 更新【됨】
- `eval()` 【호출】 【시】 running statsは 【고정】, batch statistics 【대신】 【사용】
- `requires_grad=False`は パラメータ(gamma, beta)【만】 【동결】, running statsと 【무관】

**Phase 0 【수렴】 仮定の 【타당성】**:

| 仮定 | 評価 |
|------|------|
| Encoderが 【충분히】 【수렴했음】 | Phase 0が 200K steps【이면】 **【대체】で 【타당】** |
| Running statsが 【안정적임】 | **条件【부】 【타당】** — batch_size=1で 【인한】 【분산】が stats【에】 【누적되었】を 【수】 【있음】 |
| Phase 1 データ 【분포】が Phase 0【과】 【동일】 | **【비타당】** — Phase 0【은】 src=src 【재구성】(?), Phase 1【은】 src→tgt pair 【사용】 |

**【주의점】**:
- Encoderの `ConvNeXtV2Block` 【내부에】 LayerNormが 【있음】 (DWConv 【후】 channel-wise LN)
- が LayerNormの running statsが Phase 0の 入力 【분포】(src 音声)【에】 【맞춰져】 【있음】
- Phase 1【에서】は srcと tgt 【모두】 encode → tgtの 【분포】が 【미묘하게】 【다를】 【수】 【있음】
- 【그러나】 encoder【를】 【통한】 latent 【추출은】 【결정론적】(deterministic)【이므로】, running statsの 【경미한】 mismatchは 【큰】 問題が 【되지】 【않음】

**Phase 2【에서】の LayerNorm 状態**:
```python
# Phase 2: encoderは training mode (eval() 【호출】 【안】 【함】)
z_src = encoder.encode(src)  # encode() → forward(training=False)
```
- `encoder.encode()`は `forward(training=False)` 【호출】 → LayerNorm【은】 running stats 【사용】
- 【하지만】 `encoder` 【자체】は `training=True` 状態 → running statsは **更新【되고】 【있음】**!
- 【이】は **【불일치】**: forward【에서】は running stats【를】 【사용하지만】, backward【에서】は statsが 更新【됨】
- PyTorchの `training` flagと `forward(training=...)` 【인자】の 【상호작용에】 【주】の 【필요】

**【실제】 【동작】 【추적】**:
- `encoder.encode(wav)` → `self.forward(wav, training=False)` → `self.stages(wav)` 【호출】
- `self.stages`は `nn.Sequential` → 【각】 ConvNeXtV2Block.forward() 【호출】
- LayerNorm【은】 `self.training` (【모듈】 状態) 基準【으】で batch stats vs running stats 【결정】
- `encoder.encode()` 【내부에】 `self.eval()` 【또】は context managerが 【없으므로】
- **encoder.trainingが True【면】 LayerNorm【은】 batch stats 【사용】 + running stats 更新**
- training=False 【인자】は noise regularization【에만】 影響 (forward() 【함수】の 【인자】)
- → **【실제로】は training=True 状態で running statsが 更新【됨】**

【이것은】 【의도된】 【동작일】 【수도】 【있지만】, `self.training` 【플래그】と `training` 【함수】 【인자】の 【의미】が 【분리되어】 【있어】 【혼란】を 【야기할】 【수】 【있다】.

---

## 総合 評価

### 設計 【강점】
1. **【완전】 【인과적】 構造**: CausalConv1d, CausalConvTranspose1dで 【실시간】 推論 【가능】
2. **ConvNeXt v2**: GRN + LayerScale + zero-init【으】で 【안정적】 【심층】 学習
3. **FiLM + zero-init**: 学習 【초기에】 speaker conditioningが identityで 【시작】 → 【안정적】 【수렴】
4. **CFM with OT path**: 【직선】 【경로】で 【단순하고】 【효율적인】 flow matching

### 設計 【약점】 (深刻度 【순】)

| # | 問題 | 深刻度 | 影響 |
|---|------|--------|------|
| 1 | Phase 0の src≠ref 【재구성】 | **CRITICAL** | Latent disentanglement 【원천적】 【실패】 |
| 2 | ConvTranspose1d checkerboard (【모든】 stage) | **HIGH** | 出力 【음질】 【저하】, 【주기적】 【아티팩트】 |
| 3 | Stride-7 aliasing without anti-aliasing | **HIGH** | 【고】周波数 folding → 【금속성】 【아티팩트】 |
| 4 | 【큰】 stride 【우선】 【업샘플링】 | **MEDIUM** | 【저해상도】 hallucination 【전파】 |
| 5 | β₂=0.9の 【과도한】 【적응성】 | **MEDIUM** | 【수렴】 安定性 【저하】 【가능】 |
| 6 | Phase 2の noise regularization 【누락】 | **LOW** | Robustness 【저하】, train-inference gap |
| 7 | CausalConvTranspose1d 【왼쪽】 warm-up | **LOW** | 【초기】 【프레임】 品質 【저하】 |

### 【우선】 【조치】 推奨

1. **Phase 0 修正** (【즉시】): `ref → src`で speaker embedding 【변경】 → 【자기】 【재구성】 学習
2. **ConvTranspose1d 改善**: kernel_size【를】 strideの 【정수배】で 【변경】 【또】は interpolation + Conv1dで 【대체】
3. **Anti-aliasing 【추가】**: stride≥3 stage【에】 low-pass filter 【또】は BlurPool 【도입】
4. **β₂ 【조정】**: 0.95で 【변경하고】 Phase 【전환】 【시】 optimizer state 【재】初期化 【검토】

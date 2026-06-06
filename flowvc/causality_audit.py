"""
FlowVC 因果性検証スクリプト — エージェント #2 監査用。

全レイヤーにわたる厳密な因果性検証 + 逸脱の数学的証明。
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from flowvc.blocks import CausalConv1d, CausalConvTranspose1d, ConvNeXtV2Block, GRN
from flowvc.encoder import F3Encoder, make_encoder
from flowvc.decoder import F3Decoder, DecoderStage, MRFBlock
from flowvc.config import DecoderConfig
from flowvc.converter import VectorFieldNet, FlowBlock, SpeakerCrossAttn, solve_cfm_euler

# ── 結果集計 ─────────────────────────────────────────────────────────────────
RESULTS = []

def check(name: str, passed: bool, detail: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    RESULTS.append((name, passed, detail))
    print(f"  {status} | {name}")
    if detail:
        print(f"         {detail}")

def summary():
    print("\n" + "=" * 72)
    total = len(RESULTS)
    passed = sum(1 for _, p, _ in RESULTS if p)
    failed = total - passed
    print(f"  結果: {passed}/{total} PASS, {failed} FAIL")
    print("=" * 72)
    return failed == 0

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CausalConv1d — 左パディングのみ
# ═══════════════════════════════════════════════════════════════════════════════
print("── 1. CausalConv1d 左パディング検証 ──")

def test_causal_conv():
    dim = 32
    for ks in [3, 5, 7, 15]:
        for dil in [1, 2, 4]:
            conv = CausalConv1d(dim, dim, kernel_size=ks, dilation=dil).eval()
            x = torch.randn(1, dim, 200)
            out = conv(x)

            # Modify future samples (t >= 100) and verify past output unchanged
            x_mod = x.clone()
            x_mod[:, :, 100:] = 999.0
            out_mod = conv(x_mod)

            # Past samples: should be identical up to t < 100 - ceil(pad_total)
            # For causal, any t where receptive field doesn't reach t>=100 is safe
            safe_t = 100 - (ks - 1) * dil
            if safe_t > 0:
                diff = (out[:, :, :safe_t] - out_mod[:, :, :safe_t]).abs().max().item()
                if diff > 1e-5:
                    check(f"CausalConv1d(k={ks},d={dil})", False, f"future leak: diff={diff:.6f} at safe_t={safe_t}")
                    return
    check("CausalConv1d 全カーネル/ダイレーション", True,
          f"{sum(1 for ks in [3,5,7,15] for _ in [1,2,4])} 設定すべて因果的")

test_causal_conv()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ConvNeXtV2Block 因果性（DWCconv部分）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 2. ConvNeXtV2Block DWCconv因果性 ──")

def test_block_causality():
    dim = 64
    block = ConvNeXtV2Block(dim, kernel_size=7, use_grn=True).eval()
    x = torch.randn(2, dim, 200)
    out_ref = block(x)

    x_mod = x.clone()
    x_mod[:, :, 150:] = 999.0
    out_mod = block(x_mod)

    # DWConv kernel=7 → pad=6, safe up to t=150-6=144
    diff = (out_ref[:, :, :144] - out_mod[:, :, :144]).abs().max().item()
    check("ConvNeXtV2Block DWCconv因果性", diff < 1e-5,
          f"t<144 の最大差分: {diff:.6e}")
test_block_causality()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. GRN 時間軸演算の因果性違反
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 3. GRN 因果性検証 ──")

def test_grn_causality_violation():
    """
    GRN は L2 norm を時間軸全体にわたって計算するため、
    時刻 t の出力が時刻 t+k の値に依存する（厳密な因果性違反）。

    証明:
      GRN(x)[c, t] = γ_c * x[c,t] * Nx_c + β_c + x[c,t]
      Nx_c = Gx_c / (mean_c(Gx_c) + ε)
      Gx_c = sqrt(Σ_{τ=1}^T x[c,τ]²)

      Nx_c は全時刻 τ に依存するため、
      ∂GRN(x)[c,t] / ∂x[c, t+k] ≠ 0 for all k (within same channel).

      したがって、ストリーミング推論時にチャンク境界で異なる Nx_c が計算され、
      オフライン出力とストリーミング出力が一致しない。
    """
    grn = GRN(dim=16)
    grn.gamma.data = torch.ones(1, 16, 1) * 0.5  # non-zero to amplify effect

    # Short sequence
    x_short = torch.randn(1, 16, 50)
    out_short = grn(x_short)

    # Same sequence + future (extended)
    x_long = torch.cat([x_short, torch.randn(1, 16, 50)], dim=-1)
    out_long = grn(x_long)

    # The first 50 timesteps should be identical IF causal
    diff = (out_short - out_long[:, :, :50]).abs().max().item()

    # With non-zero gamma, GRN normalization differs → output differs
    check("GRN 厳密因果性", diff < 1e-7,
          f"短/長系列で先頭50サンプルの最大差分: {diff:.6e} "
          f"(非ゼロなら因果性違反: GRNのL2ノルムが未来時刻に依存)")

test_grn_causality_violation()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. LayerNorm — 時間軸非混合（チャネル方向のみ）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 4. LayerNorm 因果性 ──")

def test_layernorm_causality():
    ln = torch.nn.LayerNorm(64)
    x = torch.randn(4, 100, 64)  # (B, T, C)
    out_ref = ln(x)

    x_mod = x.clone()
    x_mod[:, 50:, :] = 999.0
    out_mod = ln(x_mod)

    # LayerNorm over last dim → per-timestep → t<50 identical
    diff = (out_ref[:, :50, :] - out_mod[:, :50, :]).abs().max().item()
    check("LayerNorm 因果性（時間非混合）", diff < 1e-7,
          f"t<50 の最大差分: {diff:.6e}")
test_layernorm_causality()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. CausalConvTranspose1d 出力トリミング
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 5. CausalConvTranspose1d トリミング ──")

def test_transpose_trim():
    for stride in [2, 3, 7]:
        in_ch, out_ch = 16, 8
        tconv = CausalConvTranspose1d(in_ch, out_ch, kernel_size=stride*3, stride=stride)
        x = torch.randn(2, in_ch, 10)  # L=10 latent frames
        out = tconv(x)

        expected = x.shape[2] * stride  # 10*stride
        actual = out.shape[2]
        check(f"CausalConvTranspose1d トリミング (stride={stride})",
              actual == expected,
              f"期待長={expected}, 実長={actual}")

        # Verify partial backward: output[:expected] doesn't depend on "future" input
        # For ConvTranspose1d(stride=S, K=3S, pad=0):
        # output[t] depends on input[pos] where pos ∈ [ceil((t-K+1)/S), floor(t/S)]
        # For t < expected (t < L*S): pos ≤ floor((L*S-1)/S) = L-1 ✓ (all within range)
test_transpose_trim()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. エンコーダ全段Strided Convの因果性
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 6. エンコーダ全段因果性 ──")

def test_encoder_causality():
    encoder = make_encoder().eval()
    x = torch.randn(1, 1, 1764 * 10)  # 10 潜在フレーム
    z_ref = encoder.encode(x)

    x_mod = x.clone()
    # 後半のサンプルを破壊
    x_mod[:, :, 1764 * 5:] = 999.0
    z_mod = encoder.encode(x_mod)

    # 最初の数フレームは未来に依存しないはず
    safe_frames = 3  # 安全マージン
    diff = (z_ref[:, :safe_frames, :] - z_mod[:, :safe_frames, :]).abs().max().item()
    check("F3Encoder 因果性 (先頭フレーム)", diff < 1e-5,
          f"先頭{safe_frames}フレームの差分: {diff:.6e}")

    # GRNのため、先頭フレームも「若干」差分が出る可能性を確認
    # （厳密にはNGだが、実用上の影響度を評価）
    if diff > 1e-7:
        print(f"         ⚠ 注: GRNの時間軸正規化により先頭フレームに差分{diff:.6e}")
test_encoder_causality()

# ═══════════════════════════════════════════════════════════════════════════════
# 7. デコーダ因果的アップサンプリング
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 7. デコーダアップサンプリング因果性 ──")

def test_decoder_causality():
    decoder = F3Decoder(DecoderConfig()).eval()
    z = torch.randn(2, 5, 768)   # 5 潜在フレーム
    spk = torch.randn(2, 192)

    with torch.no_grad():
        out_ref = decoder(z, spk)

    # 未来の潜在フレームを改変
    z_mod = z.clone()
    z_mod[:, 3:, :] = 999.0  # フレーム 3,4 を破壊
    out_mod = decoder(z_mod, spk)

    # フレーム0-1に対応する出力サンプルは不変のはず
    # encoder: stride 2,2,3,3,7,7 = 1764
    # フレーム2まで → 2*1764 = 3528 audio samples
    # decoderの受容野で約200サンプル保守的に引く
    safe_samples = 2 * 1764 - 500
    diff = (out_ref[:, :, :safe_samples] - out_mod[:, :, :safe_samples]).abs().max().item()
    check("F3Decoder 因果性", diff < 1e-4,
          f"先頭{safe_samples}サンプルの差分: {diff:.6e}")
test_decoder_causality()

# ═══════════════════════════════════════════════════════════════════════════════
# 8. FlowBlock (VFN) 因果性
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 8. FlowBlock (VFN) 因果性 ──")

def test_flowblock_causality():
    from flowvc.converter import make_vector_field_net
    vfn = make_vector_field_net().eval()
    B, T, dim = 2, 20, 768
    z = torch.randn(B, T, dim)
    t = torch.zeros(B)
    spk = torch.randn(B, 192)
    prosody = torch.randn(B, T, 3)  # need prosody for shape match

    v_ref = vfn(z, t, spk, None, prosody)

    # 未来の潜在を破壊
    z_mod = z.clone()
    z_mod[:, 10:, :] = 999.0
    v_mod = vfn(z_mod, t, spk, None, prosody)

    diff = (v_ref[:, :8, :] - v_mod[:, :8, :]).abs().max().item()
    check("VectorFieldNet DWCconv因果性", diff < 1e-5,
          f"t<8 の差分: {diff:.6e}")

    # GRN within FlowBlock causes some leakage
    if diff > 1e-7:
        print(f"         ⚠ 注: FlowBlock GRNにより差分{diff:.6e}")
test_flowblock_causality()

# ═══════════════════════════════════════════════════════════════════════════════
# 9. ストリーミングバッファ受容野充足
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 9. ストリーミングバッファ受容野検証 ──")

def test_receptive_field_sufficiency():
    """
    エンコーダの総受容野（因果的）を計算し、
    left_ctx_ms が十分かを検証。
    """
    encoder = make_encoder().eval()

    # インパルス応答法で受容野を測定
    x = torch.zeros(1, 1, 1764 * 20)  # 20フレーム分
    x[:, :, -1] = 1.0  # 最終サンプルにインパルス

    z = encoder.encode(x)
    # インパルスの影響が届く最初の潜在フレームを探す
    nonzero_frames = (z.abs() > 1e-7).any(dim=-1).squeeze(0)
    first_affected = nonzero_frames.int().argmax().item() if nonzero_frames.any() else -1

    # 潜在フレーム数
    T_lat = z.shape[1]  # ~20

    # 受容野（サンプル単位）= (T_lat - first_affected) * 1764 が少なくとも
    # 影響範囲
    rf_frames = T_lat - first_affected
    rf_samples = rf_frames * 1764

    sufficient = rf_samples < 14112  # default left_ctx_ms=320 → 14112 samples
    check("受容野 vs リングバッファ",
          sufficient,
          f"受容野 ≈ {rf_samples}サンプル ({rf_samples/44100*1000:.0f}ms) "
          f"vs バッファ 14112サンプル (320ms)")
test_receptive_field_sufficiency()

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Overlap-add 位相連続性
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 10. Overlap-Add 位相連続性 ──")

def test_overlap_add_continuity():
    """
    線形クロスフェードの位相連続性を数学的に解析。

    y_cf[t] = α[t] * y_new[t] + (1-α[t]) * y_prev[t+Δ]
    α[t] = t / ov, t ∈ [0, ov-1]

    ここで y_prev と y_new は隣接する（重複しない）時間セグメント。
    
    線形クロスフェードの性質:
    1. 振幅連続性: lim_{t→0} y_cf[t] = y_prev[Δ] (100% prev)
       lim_{t→ov} y_cf[t] = y_new[ov] (100% new)
       → 振幅の跳躍なし ✓
    
    2. 位相連続性: y_prev[Δ+t] と y_new[t] が異なる波形の場合、
       y_cf[t] はそれらの線形混合となる。
       同一正弦波 sin(ωt) であっても、位相オフセット Δφ があれば
       α·sin(ωt) + (1-α)·sin(ωt + Δφ) ≠ sin(ωt + α·Δφ)
       → 線形クロスフェードは位相連続性を保証しない ✗
    
    3. 振幅相殺: Δφ = π のとき、クロスフェード中央(α=0.5) で完全相殺。
       → コムフィルタリングが発生しうる ✗
    """
    ov = 100
    ramp_out = torch.linspace(0, 1, ov)
    ramp_prev = torch.linspace(1, 0, ov)

    # ケース1: 同一波形 → 完全保存
    y_same = torch.sin(torch.linspace(0, 6.28, ov))
    cf_same = y_same * ramp_out + y_same * ramp_prev
    identity_error = (cf_same - y_same).abs().max().item()

    # ケース2: 位相ずれ
    y_phase = torch.sin(torch.linspace(0, 6.28, ov) + 2.0)
    cf_phase = y_same * ramp_out + y_phase * ramp_prev
    mid_idx = ov // 2
    # 中間点で期待される振幅（完全な正弦波ではない）
    mid_amp = cf_phase[mid_idx].abs().item()

    # ケース3: 逆位相（最悪ケース）
    y_anti = -y_same
    cf_anti = y_same * ramp_out + y_anti * ramp_prev
    mid_anti = cf_anti[mid_idx].abs().item()

    check("Overlap-Add 同一波形復元", identity_error < 1e-6,
          f"完全一致誤差: {identity_error:.2e}")
    check("Overlap-Add 位相連続性",
          mid_anti < 0.5,  # 逆位相で中央がほぼゼロ → 位相連続性なし
          f"逆位相クロスフェード中央振幅: {mid_anti:.4f} "
          f"(完全な位相連続性なら1.0、実測は0に近い)")
test_overlap_add_continuity()

# ═══════════════════════════════════════════════════════════════════════════════
# 11. SpeakerCrossAttn 因果性（時間混合なし）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 11. SpeakerCrossAttn 時間非混合 ──")

def test_cross_attn_causality():
    attn = SpeakerCrossAttn(dim=64, prompt_dim=64, n_heads=2)
    x = torch.randn(2, 20, 64)
    prompt = torch.randn(2, 4, 64)

    out_ref = attn(x, prompt)

    x_mod = x.clone()
    x_mod[:, 10:, :] = 999.0
    out_mod = attn(x_mod, prompt)

    # cross-attn attends to prompt tokens (not time steps of x)
    # Q is x, K/V is prompt → no temporal mixing in x
    diff = (out_ref[:, :10, :] - out_mod[:, :10, :]).abs().max().item()
    check("SpeakerCrossAttn 時間非混合", diff < 1e-7,
          f"t<10 の差分: {diff:.6e}")
test_cross_attn_causality()

# ═══════════════════════════════════════════════════════════════════════════════
# 12. AdaLN-Zero 因果性
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 12. AdaLN-Zero 因果性 ──")

def test_adaln_causality():
    from flowvc.blocks import AdaLNZero
    adaln = AdaLNZero(64, 128)
    x = torch.randn(2, 30, 64)
    cond = torch.randn(2, 30, 128)

    x_mod_adaln, _ = adaln(x, cond)

    x_future = x.clone()
    x_future[:, 15:, :] = 999.0
    x_mod_future, _ = adaln(x_future, cond)

    # LayerNorm per-timestep, then element-wise scale/shift
    # → per-timestep operation, no temporal mixing
    diff = (x_mod_adaln[:, :15, :] - x_mod_future[:, :15, :]).abs().max().item()
    check("AdaLN-Zero 因果性（時間非混合）", diff < 1e-7,
          f"t<15 の差分: {diff:.6e}")
test_adaln_causality()

# ═══════════════════════════════════════════════════════════════════════════════
# 13. CFM ODE 因果性
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 13. CFM ODE ソルバ因果性 ──")

def test_cfm_ode_causality():
    from flowvc.converter import make_vector_field_net
    vfn = make_vector_field_net().eval()
    B, T, dim = 2, 10, 768
    z_src = torch.randn(B, T, dim)
    spk = torch.randn(B, 192)
    prosody = torch.randn(B, T, 3)

    z_ref = solve_cfm_euler(vfn, z_src, spk, None, prosody, n_steps=4)

    z_mod = z_src.clone()
    z_mod[:, 5:, :] = 999.0
    z_out = solve_cfm_euler(vfn, z_mod, spk, None, prosody, n_steps=4)

    diff = (z_ref[:, :3, :] - z_out[:, :3, :]).abs().max().item()
    check("CFM Euler ODE 因果性", diff < 1e-5,
          f"t<3 の差分: {diff:.6e} (GRN影響あり)")
test_cfm_ode_causality()

# ═══════════════════════════════════════════════════════════════════════════════
# 14. エンコーダ・デコーダ完全ラウンドトリップ精度
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 14. ラウンドトリップサンプル数保存 ──")

def test_roundtrip_exact():
    encoder = make_encoder().eval()
    decoder = F3Decoder(DecoderConfig()).eval()
    spk = torch.randn(2, 192)

    for T_audio in [1764, 1764*5, 1764*10, 44100]:
        wav = torch.randn(2, 1, T_audio)
        z = encoder.encode(wav)
        out = decoder(z, spk)
        check(f"ラウンドトリップ T={T_audio}", out.shape[2] == T_audio,
              f"出力={out.shape[2]}, 期待={T_audio}")
test_roundtrip_exact()

# ═══════════════════════════════════════════════════════════════════════════════
# 15. MRFBlock 因果性
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 15. MRFBlock 因果性 ──")

def test_mrf_causality():
    mrf = MRFBlock(32).eval()
    x = torch.randn(1, 32, 200)
    out_ref = mrf(x)

    x_mod = x.clone()
    x_mod[:, :, 150:] = 999.0
    out_mod = mrf(x_mod)

    # MRF uses CausalConv1d internally → causal
    safe_t = 150 - 6  # largest kernel 11 with dil 5: pad = 10*5=50... actually
    # Max receptive field: ks=11, dil=5 → pad = (11-1)*5 = 50
    # After 3 layers of that, it accumulates
    # Conservative: 150-50 = 100
    diff = (out_ref[:, :, :100] - out_mod[:, :, :100]).abs().max().item()
    check("MRFBlock 因果性", diff < 1e-5,
          f"t<100 の差分: {diff:.6e}")
test_mrf_causality()

# ═══════════════════════════════════════════════════════════════════════════════
print()
all_pass = summary()
sys.exit(0 if all_pass else 1)

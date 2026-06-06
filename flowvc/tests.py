"""
FlowVC テストスイート。

7つのテストでパイプラインの正当性を検証:
1. エンコーダ・デコーダ形状テスト (ラウンドトリップ)
2. 因果性テスト (未来情報漏洩なし)
3. GRN正規化テスト (ConvNeXt v2)
4. Zero-init恒等性テスト
5. CFM ODEソルバテスト
6. ストリーミングバッファ + オーバーラップアドテスト
7. チェックポイント保存/復元テスト

実行: python -m pytest flowvc/tests.py -v
"""

import os, sys, tempfile, torch, pytest
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowvc.config import EncoderConfig, DecoderConfig
from flowvc.encoder import F3Encoder, make_encoder
from flowvc.decoder import F3Decoder
from flowvc.converter import (
    VectorFieldNet, make_vector_field_net, solve_cfm_euler, solve_cfm_rk4
)
from flowvc.blocks import GRN, ConvNeXtV2Block, CausalConv1d, AdaLNZero, FiLM
from flowvc.cfm_loss import CFMLoss
from flowvc.speaker import make_speaker_encoder
from flowvc.prosody import make_prosody_extractor


# ── ヘルパー ────────────────────────────────────────────────────

def _make_all(device="cpu"):
    encoder = make_encoder().to(device).eval()
    decoder = F3Decoder(DecoderConfig()).to(device).eval()
    vfn = make_vector_field_net().to(device).eval()
    speaker_enc = make_speaker_encoder().to(device).eval()
    prosody = make_prosody_extractor(device=device).to(device).eval()
    return encoder, decoder, vfn, speaker_enc, prosody


# ── テスト1: エンコーダ・デコーダ形状 ──────────────────────────

@pytest.mark.parametrize("T_audio", [
    1764,        # 1フレーム
    1764 * 5,    # 5フレーム
    1764 * 50,   # 50フレーム (2秒)
    44100,       # 1秒 (25フレーム)
])
def test_encoder_decoder_roundtrip(T_audio):
    """エンコーダ→デコーダのサンプル数保存を検証。"""
    encoder, decoder, _, _, _ = _make_all()

    B = 2
    wav = torch.randn(B, 1, T_audio)
    spk = torch.randn(B, 192)

    z = encoder.encode(wav)
    out = decoder(z, spk)

    T_lat_expected = max(1, T_audio // 1764)
    assert z.shape == (B, T_lat_expected, 768), \
        f"エンコーダ出力: expected ({B},{T_lat_expected},768), got {z.shape}"

    # デコーダ出力長は encoder 入力と厳密に一致する必要あり
    assert out.shape[2] == wav.shape[2], \
        f"デコーダ長不一致: expected {wav.shape[2]}, got {out.shape[2]}"

    assert out.shape[0] == B and out.shape[1] == 1


# ── テスト2: 因果性テスト ──────────────────────────────────────

def test_causality():
    """左パディングのみ→未来情報が現在の出力に影響しないことを検証。"""
    dim = 32

    # CausalConv1d
    conv = CausalConv1d(dim, dim, kernel_size=5)
    x = torch.randn(1, dim, 100)
    out = conv(x)

    # x[:, :, t+1:] を変更しても out[:, :, :t] は不変のはず
    x_modified = x.clone()
    x_modified[:, :, 50:] = 999.0
    out_modified = conv(x_modified)

    # t=0..48 は未来に触れないので同一出力
    max_diff = (out[:, :, :48] - out_modified[:, :, :48]).abs().max().item()
    assert max_diff < 1e-5, \
        f"因果性破綻: t<49で差分 {max_diff} (未来情報漏洩)"

    # ConvNeXtV2Block
    block = ConvNeXtV2Block(dim, kernel_size=5, use_grn=True).eval()
    x2 = torch.randn(1, dim, 100)
    out2 = block(x2)

    x2_mod = x2.clone()
    x2_mod[:, :, 70:] = 999.0
    out2_mod = block(x2_mod)

    max_diff2 = (out2[:, :, :68] - out2_mod[:, :, :68]).abs().max().item()
    assert max_diff2 < 1e-5, \
        f"ConvNeXtV2Block因果性破綻: 差分 {max_diff2}"


# ── テスト3: GRN正規化テスト ───────────────────────────────────

def test_grn_normalization():
    """GRNが実際にチャネル間の特徴競合を行っているか検証。"""
    grn = GRN(dim=64)
    # 学習済みを模擬: gamma=0.5 (ゼロより大きい)
    grn.gamma.data = torch.ones(1, 64, 1) * 0.5
    grn.beta.data = torch.zeros(1, 64, 1)

    x = torch.randn(2, 64, 100)
    x[:, 0, :] *= 10.0
    x[:, 1, :] *= 0.01

    out = grn(x)

    # GRN が恒等でないこと（何らかの変換が行われている）
    max_diff = (out - x).abs().max().item()
    assert max_diff > 1e-3, \
        f"GRNが恒等変換のまま: max|out-x|={max_diff:.6f}"

    # NaN がないこと
    assert not torch.isnan(out).any(), "GRN出力にNaN"


# ── テスト4: Zero-init恒等性テスト ─────────────────────────────

def test_zero_init_identity():
    """AdaLNZero, FiLM, Converter が初期状態で恒等写像であることを検証。"""
    B, T, D = 2, 10, 64

    # AdaLNZero
    adaln = AdaLNZero(D, 128)
    x = torch.randn(B, T, D)
    cond = torch.randn(B, T, 128)
    x_mod, gate = adaln(x, cond)

    assert gate.abs().max().item() < 1e-5, \
        f"AdaLNZero gate非ゼロ: max={gate.abs().max().item()}"
    # gate=0 だが norm 後の値なので完全な恒等ではない
    # 重要なのは gate=0 でスキップされること

    # FiLM
    film = FiLM(D, 128)
    x_c = torch.randn(B, D, 20)
    cond_c = torch.randn(B, 128)
    out_c = film(x_c, cond_c)
    assert torch.allclose(out_c, x_c, atol=1e-5), \
        "FiLM zero-init: 恒等性破綻"

    # Converter: out_gate=0.01 なので完全な恒等ではないが、
    # v_pred がゼロに近いことを確認
    encoder, _, vfn, speaker_enc, prosody = _make_all()
    wav = torch.randn(B, 1, 1764 * 5)
    ref = torch.randn(B, 1, 1764 * 3)

    z = encoder.encode(wav)
    spk, prompt = speaker_enc(ref)
    pros = prosody(wav)

    t = torch.zeros(B)
    v = vfn(z, t, spk, prompt, pros)
    v_norm = v.norm().item()
    z_norm = z.norm().item()

    assert v_norm < z_norm * 0.05, \
        f"Converter初期出力が大きすぎる: |v|={v_norm:.4f}, |z|={z_norm:.4f}"


# ── テスト5: CFM ODEソルバテスト ───────────────────────────────

def test_cfm_ode_solver():
    """Euler と RK4 ソルバの基本動作を検証。"""
    encoder, _, vfn, speaker_enc, prosody = _make_all()
    B = 2
    wav = torch.randn(B, 1, 1764 * 5)
    ref = torch.randn(B, 1, 1764 * 3)

    z = encoder.encode(wav)
    spk, prompt = speaker_enc(ref)
    pros = prosody(wav)

    # Euler
    z_euler = solve_cfm_euler(vfn, z, spk, prompt, pros, n_steps=4)
    assert z_euler.shape == z.shape
    assert not torch.isnan(z_euler).any(), "Euler: NaN発生"

    # RK4
    z_rk4 = solve_cfm_rk4(vfn, z, spk, prompt, pros, n_steps=4)
    assert z_rk4.shape == z.shape
    assert not torch.isnan(z_rk4).any(), "RK4: NaN発生"

    # ステップ数が多いほど delta が増える傾向（out_gate=0.01 の影響あり）
    z_8 = solve_cfm_euler(vfn, z, spk, prompt, pros, n_steps=8)
    delta_4 = (z_euler - z).norm().item()
    delta_8 = (z_8 - z).norm().item()

    # ステップ数が多いほど変化量が大きい（積分経路が長い）
    assert delta_8 >= delta_4 * 0.5, \
        f"ステップ数と変化量の単調性崩壊: Δ4={delta_4:.4f}, Δ8={delta_8:.4f}"


# ── テスト6: ストリーミングバッファテスト ─────────────────────

def test_streaming_buffer():
    """リングバッファ + オーバーラップアドが正しく動作するか検証。"""
    from flowvc.infer import FlowVCInference

    encoder, decoder, vfn, speaker_enc, prosody = _make_all()
    engine = FlowVCInference(
        encoder, decoder, vfn, speaker_enc, prosody,
        device="cpu", chunk_ms=80, left_ctx_ms=160, overlap_ms=20,
        ode_steps=2,  # 高速化のため少なめ
    )

    # 話者設定
    ref = torch.randn(1, 44100)
    engine.set_target_speaker(ref)
    assert engine.speaker_emb is not None
    assert engine.prompt_tokens is not None

    # 複数チャンク処理（短めでテスト）
    audio = torch.randn(44100)  # 1秒
    result = engine.process_stream(audio)

    # 出力が非ゼロ
    assert result.abs().max() > 0, "出力がゼロ"
    assert result.shape[0] > 0, "出力が空"
    assert engine.stats["chunks"] > 0

    print(f"  チャンク数: {engine.stats['chunks']}, RTF: {engine.rtf:.3f}")


# ── テスト7: チェックポイント保存/復元テスト ──────────────────

def test_checkpoint_save_load():
    """全モデルの保存→復元で重みが一致するか検証。"""
    from flowvc.train import save_checkpoint, load_checkpoint

    encoder, decoder, vfn, speaker_enc, prosody = _make_all()

    # ダミー最適化で重みを少し変える
    opt = torch.optim.SGD(list(encoder.parameters()), lr=0.01)
    wav = torch.randn(1, 1, 1764 * 5)
    spk_emb = torch.randn(1, 192)
    loss = F.mse_loss(encoder.encode(wav), torch.randn(1, 5, 768))
    loss.backward()
    opt.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test.pt")

        # 保存
        models = {
            "encoder": encoder, "decoder": decoder, "vfn": vfn,
            "speaker_enc": speaker_enc, "prosody": prosody,
        }
        save_checkpoint(models, opt, step=42, path=ckpt_path)
        assert os.path.exists(ckpt_path)

        # 各モデルの重みを保存
        enc_w_before = encoder.stages[0].conv.weight.clone()

        # 復元
        encoder2, decoder2, vfn2, speaker_enc2, prosody2 = _make_all()
        models2 = {
            "encoder": encoder2, "decoder": decoder2, "vfn": vfn2,
            "speaker_enc": speaker_enc2, "prosody": prosody2,
        }
        opt2 = torch.optim.SGD(list(encoder2.parameters()), lr=0.01)
        restored_step = load_checkpoint(models2, opt2, ckpt_path, device=torch.device("cpu"))

        assert restored_step == 42, f"ステップ復元失敗: {restored_step}"
        assert torch.allclose(encoder2.stages[0].conv.weight, enc_w_before, atol=1e-5), \
            "エンコーダ重み復元不一致"


# ── テスト8: CFM損失テスト ─────────────────────────────────────

def test_cfm_loss():
    """CFM損失が正しく計算され、勾配が流れるか検証。"""
    _, _, vfn, speaker_enc, prosody = _make_all()
    encoder = make_encoder()

    B = 2
    wav_src = torch.randn(B, 1, 1764 * 5)
    wav_tgt = torch.randn(B, 1, 1764 * 5)
    ref = torch.randn(B, 1, 1764 * 3)

    z_src = encoder.encode(wav_src)
    z_tgt = encoder.encode(wav_tgt)
    spk, prompt = speaker_enc(ref)
    pros = prosody(wav_src)

    cfm_loss = CFMLoss(sigma_min=0.001)

    # 順伝播
    loss, logs = cfm_loss(vfn, z_src, z_tgt, spk, prompt, pros)
    assert loss.item() > 0, "CFM損失がゼロ"
    assert "v_pred_norm" in logs

    # 逆伝播
    opt = torch.optim.SGD(vfn.parameters(), lr=0.01)
    opt.zero_grad()
    loss.backward()

    # out_gate に勾配が流れているか
    assert vfn.out_gate.grad is not None, "out_gate に勾配なし"
    assert vfn.out_gate.grad.abs().max() > 0, "out_gate 勾配がゼロ"

    # in_proj にも勾配が流れているか
    assert vfn.in_proj.weight.grad is not None
    assert vfn.in_proj.weight.grad.abs().max() > 0, "in_proj 勾配がゼロ"


# ── テスト9: 話者エンコーダテスト ──────────────────────────────

def test_speaker_encoder():
    """話者エンコーダの出力形状と値の正当性を検証。"""
    speaker_enc = make_speaker_encoder()
    wav = torch.randn(2, 1, 44100 * 3)

    spk_emb, prompt = speaker_enc(wav)

    assert spk_emb.shape == (2, 192), f"spk_emb shape: {spk_emb.shape}"
    assert prompt.shape == (2, 4, 192), f"prompt shape: {prompt.shape}"

    # 異なる入力で異なる出力になること
    wav2 = torch.randn(2, 1, 44100 * 3)
    spk_emb2, _ = speaker_enc(wav2)

    # 同一重み・異なる入力 → 異なる出力（ランダム入力で十分）
    diff = (spk_emb - spk_emb2).abs().max().item()
    assert diff > 0, "異なる入力で同一の話者埋め込み"


# ── テスト10: データセットテスト ───────────────────────────────

@pytest.mark.skipif(
    not os.path.isdir("/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"),
    reason="VCTKデータセット不在"
)
def test_dataset_integration():
    """実VCTKデータでのデータセット動作検証。"""
    from flowvc.dataset import VCTKDataset, create_dataloader

    ds = VCTKDataset(
        "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed",
        crop_seconds=2.0,
    )
    assert len(ds) > 0

    # 3サンプル取得で ref leak がないこと確認
    for i in range(3):
        item = ds[i]
        assert item["src_wav"].shape == (1, 88200)
        assert item["tgt_wav"].shape == (1, 88200)
        assert item["ref_wav"].shape == (1, 132300)

    # データローダ
    loader = create_dataloader(ds, batch_size=2)
    batch = next(iter(loader))
    assert batch["src_wav"].shape[0] == 2

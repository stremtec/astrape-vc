#!/usr/bin/env python3
"""
話者・韻律表現力の品質検証 (Agent #9)

検証項目:
1. SpeakerEncoderの192-dim出力が異なる話者間で十分な分離度を持つか (cosine similarity分布)
2. PromptTokenGeneratorの4トークンが直交しているか
3. FCPEのF0抽出が無声区間で0を出力するか
4. F0→25Hzリサンプル後の値が妥当か (log_f0範囲、voiced遷移)
5. 短い発話(1秒)と長い発話(10秒)でspeaker embeddingの一貫性があるか
6. ECAPA-TDNN との比較で優位性/劣位性があるか

実行: python3 eval_speaker_prosody.py
"""

import os, sys, time, math, random, hashlib
import torch, torch.nn.functional as F
import torchaudio
import numpy as np
import soundfile as sf
from collections import defaultdict

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowvc.speaker import make_speaker_encoder, PromptTokenGenerator
from flowvc.prosody import make_prosody_extractor
from flowvc.config import SpeakerEncoderConfig

SR = 44100
DEVICE = "cpu"
VCTK_DIR = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
PYTHON = "/Users/asill/lfm-venv/bin/python3"

# ─── ユーティリティ ──────────────────────────────────────────────

def load_vctk_audio(speaker_id, utterance_idx, crop_sec=None):
    """Load VCTK FLAC file, return (1, T) tensor @ 44100Hz mono."""
    spk_dir = os.path.join(VCTK_DIR, speaker_id)
    files = sorted([f for f in os.listdir(spk_dir) if f.endswith('.flac') and 'mic1' in f])
    if utterance_idx >= len(files):
        return None
    path = os.path.join(spk_dir, files[utterance_idx])
    wav_np, sr = sf.read(path)
    wav = torch.from_numpy(wav_np).float()
    # Convert to mono if needed
    if wav.dim() == 2:
        wav = wav.mean(dim=-1)
    wav = wav.unsqueeze(0)  # (1, T)
    if sr != SR:
        resampler = torchaudio.transforms.Resample(sr, SR)
        wav = resampler(wav)
    if crop_sec is not None:
        n_samples = int(crop_sec * SR)
        wav = wav[:, :n_samples]
    return wav  # (1, T)


def cosine_sim(a, b):
    a_n = F.normalize(a.float(), dim=-1)
    b_n = F.normalize(b.float(), dim=-1)
    return (a_n * b_n).sum(dim=-1)


# ─── 検証1: SpeakerEncoder 話者分離度 (VCTK実データ) ─────────────

def test_speaker_separation():
    """VCTK実話者間のcosine similarity分布を評価。"""
    print("\n" + "="*70)
    print("検証1: SpeakerEncoder 話者分離度 (VCTK実データ)")
    print("="*70)

    spk_enc = make_speaker_encoder().eval()
    print(f"  SpeakerEncoder params: {sum(p.numel() for p in spk_enc.parameters()):,}")

    # VCTK話者リスト
    all_speakers = sorted([d for d in os.listdir(VCTK_DIR) if d.startswith('p')])
    print(f"  VCTK speakers available: {len(all_speakers)}")

    # ランダムに10話者選択
    random.seed(42)
    selected = random.sample(all_speakers, min(15, len(all_speakers)))
    print(f"  Testing {len(selected)} speakers: {', '.join(selected[:8])}...")

    # 各話者2発話 → 埋め込み
    embeddings = {}
    for spk in selected:
        embs = []
        for i in range(min(3, 5)):  # up to 3 utterances
            wav = load_vctk_audio(spk, i, crop_sec=3.0)
            if wav is None:
                continue
            wav_in = wav.unsqueeze(0)  # (1, 1, T)
            with torch.no_grad():
                emb, _ = spk_enc(wav_in)
            embs.append(emb.squeeze(0))  # (192,)
        if len(embs) >= 2:
            embeddings[spk] = torch.stack(embs)  # (N_utt, 192)

    print(f"\n  Loaded embeddings for {len(embeddings)} speakers")

    # 同話者・異話者間のcosine similarityを集計
    same_sims = []
    diff_sims = []
    spk_centroids = {}

    for spk, embs in embeddings.items():
        centroid = embs.mean(dim=0)  # (192,)
        spk_centroids[spk] = centroid
        # 同話者: 各発話対
        for i in range(embs.shape[0]):
            for j in range(i+1, embs.shape[0]):
                same_sims.append(cosine_sim(embs[i], embs[j]).item())

    spk_list = list(spk_centroids.keys())
    for i in range(len(spk_list)):
        for j in range(i+1, len(spk_list)):
            diff_sims.append(cosine_sim(spk_centroids[spk_list[i]],
                                         spk_centroids[spk_list[j]]).item())

    same_sims = np.array(same_sims)
    diff_sims = np.array(diff_sims)

    print(f"\n  --- Cosine Similarity 分布 ---")
    print(f"  同話者 (N={len(same_sims)}): "
          f"mean={same_sims.mean():.4f}, std={same_sims.std():.4f}, "
          f"min={same_sims.min():.4f}, max={same_sims.max():.4f}")
    print(f"  異話者 (N={len(diff_sims)}): "
          f"mean={diff_sims.mean():.4f}, std={diff_sims.std():.4f}, "
          f"min={diff_sims.min():.4f}, max={diff_sims.max():.4f}")

    # 分離度メトリクス
    separation = same_sims.mean() - diff_sims.mean()
    print(f"  分離度 (same_mean - diff_mean): {separation:.4f}")

    # 判定
    checks = {}
    checks["同話者sim > 異話者sim"] = same_sims.mean() > diff_sims.mean()
    checks["分離度 >= 0.1"] = separation >= 0.1
    checks["異話者sim最大値 < 同話者sim最小値"] = diff_sims.max() < same_sims.min()
    checks["異話者sim平均 < 0.7"] = diff_sims.mean() < 0.7
    checks["同話者sim平均 > 0.8"] = same_sims.mean() > 0.8

    for k, v in checks.items():
        status = "✓" if v else "✗"
        print(f"    {status} {k}: {v}")

    # Embedding norm check
    norms = []
    for embs in embeddings.values():
        norms.extend(embs.norm(dim=-1).tolist())
    norms = np.array(norms)
    print(f"\n  Embedding L2 norm: mean={norms.mean():.3f}, std={norms.std():.3f}, "
          f"min={norms.min():.3f}, max={norms.max():.3f}")

    # Diagnostic: check intermediate feature diversity
    print(f"\n  --- 診断: 中間特徴量の多様性 ---")
    spk_enc_diag = make_speaker_encoder().eval()
    # Test with two very different inputs (sine 220Hz vs noise)
    t = torch.arange(0, int(2 * SR)) / SR
    wav1 = torch.sin(2 * math.pi * 220 * t).unsqueeze(0).unsqueeze(0) * 0.5  # (1,1,T)
    wav2 = torch.randn(1, 1, int(2 * SR)) * 0.5
    
    # Get intermediate features (before pooling)
    with torch.no_grad():
        x1 = spk_enc_diag.stages(wav1)  # (1, C, T)
        x2 = spk_enc_diag.stages(wav2)  # (1, C, T)
    
    # Channel variance
    x1_pooled_std = x1.std(dim=-1).mean().item()  # across time, avg over channels
    x2_pooled_std = x2.std(dim=-1).mean().item()
    x_diff_norm = (x1 - x2).norm().item()
    x1_norm = x1.norm().item()
    
    print(f"  Sine 220Hz feat: temporal std (avg) = {x1_pooled_std:.6f}, norm = {x1_norm:.2f}")
    print(f"  Noise feat:      temporal std (avg) = {x2_pooled_std:.6f}, norm = {x2.norm().item():.2f}")
    print(f"  Feature difference norm: {x_diff_norm:.2f} (vs |x1|={x1_norm:.2f})")
    
    # After pooling: check attention pool variance
    x1_t = x1.transpose(1, 2)  # (1, T, C)
    x1_normed = spk_enc_diag.norm(x1_t)  # LayerNorm
    pooled1 = spk_enc_diag.pool(x1_normed)  # (1, C)
    x2_t = x2.transpose(1, 2)
    x2_normed = spk_enc_diag.norm(x2_t)
    pooled2 = spk_enc_diag.pool(x2_normed)
    
    pool_diff_norm = (pooled1 - pooled2).norm().item()
    pool1_norm = pooled1.norm().item()
    print(f"  Pooled feature diff norm: {pool_diff_norm:.4f} (vs |pool|={pool1_norm:.4f})")
    print(f"  Pooled cosine sim: {cosine_sim(pooled1, pooled2).item():.6f}")
    
    # Final embedding
    emb1 = spk_enc_diag.spk_proj(pooled1)
    emb2 = spk_enc_diag.spk_proj(pooled2)
    print(f"  Final emb diff norm: {(emb1 - emb2).norm().item():.6f} (vs |emb|={emb1.norm().item():.4f})")
    print(f"  Final emb cosine sim: {cosine_sim(emb1, emb2).item():.6f}")

    return {
        "same_sims": same_sims.tolist(),
        "diff_sims": diff_sims.tolist(),
        "separation": float(separation),
        "checks": checks,
        "embedding_norms": {"mean": float(norms.mean()), "std": float(norms.std())},
        "n_speakers": len(embeddings),
    }


# ─── 検証2: PromptTokenGenerator 4トークン直交性 ─────────────────

def test_prompt_orthogonality():
    """PromptTokenGeneratorの4トークンが(近似的に)直交しているか。"""
    print("\n" + "="*70)
    print("検証2: PromptTokenGenerator 4トークン直交性")
    print("="*70)

    spk_enc = make_speaker_encoder().eval()
    prompt_gen = spk_enc.prompt_gen

    # ランダム話者埋め込みでトークン生成
    B = 20  # 多様な話者埋め込み
    speaker_embs = torch.randn(B, 192)
    with torch.no_grad():
        tokens = prompt_gen(speaker_embs)  # (B, 4, 192)

    # 1. ベーストークン間の直交性
    base = prompt_gen.base_tokens  # (4, 192)
    base_cos = torch.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            base_cos[i, j] = cosine_sim(base[i:i+1], base[j:j+1]).item()

    print(f"\n  Base tokens cosine similarity matrix:\n{base_cos.numpy().round(4)}")
    off_diag = base_cos[~torch.eye(4, dtype=bool)]
    print(f"  Base tokens off-diagonal cos_sim: mean={off_diag.mean():.4f}, "
          f"max={off_diag.max():.4f}")

    # 2. 生成後トークン間の平均直交性（20人の話者で評価）
    tok_norms = tokens.norm(dim=-1)  # (B, 4)
    print(f"\n  Token L2 norms (per token, per speaker): "
          f"mean={tok_norms.mean():.3f}, std={tok_norms.std():.3f}")

    # トークン間の相互相関行列 (平均)
    all_corr = torch.zeros((B, 4, 4))
    for b in range(B):
        for i in range(4):
            for j in range(4):
                all_corr[b, i, j] = cosine_sim(tokens[b, i:i+1], tokens[b, j:j+1]).item()
    avg_corr = all_corr.mean(dim=0)
    print(f"\n  Average token cosine similarity (across 20 speakers):\n{avg_corr.numpy().round(4)}")

    off_diag_avg = avg_corr[~torch.eye(4, dtype=bool)]
    diag_avg = torch.diag(avg_corr)
    print(f"  Diagonal (self-sim): mean={diag_avg.mean():.4f}")
    print(f"  Off-diagonal (cross-sim): mean={off_diag_avg.mean():.4f}, max={off_diag_avg.max():.4f}")

    checks = {}
    checks["ベーストークン直交性 (max|off-diag| < 0.5)"] = off_diag.abs().max() < 0.5
    checks["生成トークン直交性 (mean|off-diag| < 0.3)"] = off_diag_avg.abs().mean() < 0.3
    checks["トークン自己類似度 > 0.99"] = diag_avg.min() > 0.99

    for k, v in checks.items():
        status = "✓" if v else "✗"
        print(f"    {status} {k}: {v}")

    # 特異値分解でランク評価
    U, S, V = torch.svd(tokens[0])  # (4, 192) -> S(4,)
    print(f"\n  Token SVD singular values (speaker 0): {S.tolist()}")
    effective_rank = (S / S.max()).sum().item()
    print(f"  Effective rank: {effective_rank:.2f} (out of 4)")

    return {
        "base_off_diag_mean": float(off_diag.mean()),
        "base_off_diag_max": float(off_diag.max()),
        "avg_off_diag_mean": float(off_diag_avg.mean()),
        "avg_off_diag_max": float(off_diag_avg.max()),
        "effective_rank": float(effective_rank),
        "singular_values": S.tolist(),
        "checks": checks,
    }


# ─── 検証3: FCPE F0品質 (無声区間で0出力) ────────────────────────

def test_fcpe_f0_quality():
    """FCPEが無声区間で0を出力し、有声区間で妥当なF0を出力するか。"""
    print("\n" + "="*70)
    print("検証3: FCPE F0抽出品質")
    print("="*70)

    prosody = make_prosody_extractor(device=DEVICE).eval()

    results = {}

    # Test 3a: 合成音 (正弦波) で基本精度確認
    print("\n  --- 3a: 合成正弦波テスト ---")
    for freq, label in [(220, "A3"), (440, "A4"), (880, "A5")]:
        t = torch.arange(0, int(2 * SR)) / SR
        wav = torch.sin(2 * math.pi * freq * t).unsqueeze(0).unsqueeze(0) * 0.5  # (1,1,T)
        with torch.no_grad():
            feat = prosody(wav)  # (1, T25, 3)
        log_f0 = feat[0, :, 0]
        voiced = feat[0, :, 1]
        voiced_mask = voiced > 0.5

        if voiced_mask.sum() > 0:
            f0_vals = torch.exp(log_f0[voiced_mask])
            mean_f0 = f0_vals.mean().item()
            std_f0 = f0_vals.std().item()
        else:
            mean_f0, std_f0 = 0.0, 0.0

        voiced_pct = voiced_mask.float().mean().item() * 100
        print(f"    {label} ({freq}Hz): F0={mean_f0:.1f}±{std_f0:.1f}Hz, "
              f"voiced={voiced_pct:.0f}%, frames={feat.shape[1]}")

        results[label] = {"mean_f0": mean_f0, "std_f0": std_f0, "voiced_pct": voiced_pct}

    # Test 3b: 無音で確実に0出力
    print("\n  --- 3b: 無音入力テスト ---")
    silence = torch.zeros(1, 1, int(3 * SR))
    with torch.no_grad():
        feat_sil = prosody(silence)
    f0_sil = torch.exp(feat_sil[0, :, 0])
    voiced_sil = feat_sil[0, :, 1]
    print(f"    Silence: F0 max={f0_sil.max().item():.2f}Hz, mean={f0_sil.mean().item():.4f}, "
          f"voiced frames={voiced_sil.sum().item()}/{voiced_sil.shape[0]}")
    results["silence"] = {"f0_max": f0_sil.max().item(), "voiced_frames": voiced_sil.sum().item()}

    # Test 3c: ホワイトノイズ (無声のはず)
    print("\n  --- 3c: ホワイトノイズテスト ---")
    noise = torch.randn(1, 1, int(2 * SR)) * 0.2
    with torch.no_grad():
        feat_noise = prosody(noise)
    voiced_noise = feat_noise[0, :, 1]
    voiced_pct_noise = (voiced_noise > 0.5).float().mean().item() * 100
    print(f"    Noise: voiced={voiced_pct_noise:.1f}%")
    results["noise"] = {"voiced_pct": voiced_pct_noise}

    # Test 3d: 実VCTK音声
    print("\n  --- 3d: 実VCTK音声テスト ---")
    speakers = sorted([d for d in os.listdir(VCTK_DIR) if d.startswith('p')])[:3]
    for spk in speakers:
        wav = load_vctk_audio(spk, 0, crop_sec=3.0)
        if wav is None:
            continue
        wav_in = wav.unsqueeze(0)
        with torch.no_grad():
            feat = prosody(wav_in)
        log_f0 = feat[0, :, 0]
        voiced = feat[0, :, 1]
        voiced_mask = voiced > 0.5

        if voiced_mask.sum() > 0:
            f0_vals = torch.exp(log_f0[voiced_mask])
            mean_f0 = f0_vals.mean().item()
            std_f0 = f0_vals.std().item()
            min_f0 = f0_vals.min().item()
            max_f0 = f0_vals.max().item()
        else:
            mean_f0, std_f0, min_f0, max_f0 = 0, 0, 0, 0

        voiced_pct = voiced_mask.float().mean().item() * 100
        unvoiced_zero = (feat[0, :, 0][voiced <= 0.5].abs() < 1e-6).float().mean().item() * 100

        print(f"    {spk}: F0={mean_f0:.1f}±{std_f0:.1f}Hz [{min_f0:.0f}-{max_f0:.0f}], "
              f"voiced={voiced_pct:.0f}%, unvoiced→zero={unvoiced_zero:.0f}%, "
              f"frames={feat.shape[1]}")

        results[spk] = {
            "mean_f0": mean_f0, "std_f0": std_f0,
            "min_f0": min_f0, "max_f0": max_f0,
            "voiced_pct": voiced_pct,
            "unvoiced_zero_pct": unvoiced_zero,
        }

    # 判定
    checks = {}
    checks["正弦波220Hz ±50Hz以内"] = 150 < results.get("A3", {}).get("mean_f0", 0) < 290
    checks["正弦波440Hz ±100Hz以内"] = 300 < results.get("A4", {}).get("mean_f0", 0) < 580
    checks["正弦波880Hz ±200Hz以内"] = 650 < results.get("A5", {}).get("mean_f0", 0) < 1100
    checks["無音F0=0 (max < 1Hz)"] = results.get("silence", {}).get("f0_max", 999) < 1.0
    checks["無音voiced=0"] = results.get("silence", {}).get("voiced_frames", 999) == 0
    checks["ノイズ voiced率 < 20%"] = results.get("noise", {}).get("voiced_pct", 999) < 20

    for k, v in checks.items():
        status = "✓" if v else "✗"
        print(f"    {status} {k}: {v}")

    return {"results": results, "checks": checks}


# ─── 検証4: F0→25Hz リサンプル品質 ──────────────────────────────

def test_f0_resample_quality():
    """log_f0範囲、voiced遷移の連続性を評価。"""
    print("\n" + "="*70)
    print("検証4: F0→25Hz リサンプル品質")
    print("="*70)

    prosody = make_prosody_extractor(device=DEVICE).eval()

    # 4a: 周波数スイープ (80→1000Hz) で連続性評価
    print("\n  --- 4a: 周波数スイープ ---")
    t = torch.arange(0, int(5 * SR)) / SR
    freq = 80 + 920 * t / 5.0
    phase = 2 * math.pi * torch.cumsum(freq / SR, dim=0)
    sweep = torch.sin(phase).unsqueeze(0).unsqueeze(0) * 0.5

    with torch.no_grad():
        feat = prosody(sweep)
    log_f0 = feat[0, :, 0]
    voiced = feat[0, :, 1]
    f0_vals = torch.exp(log_f0)

    voiced_mask = voiced > 0.5
    if voiced_mask.sum() > 1:
        f0_voiced = f0_vals[voiced_mask]
        print(f"    F0 range: [{f0_voiced.min().item():.1f}, {f0_voiced.max().item():.1f}] Hz")
        print(f"    F0 mean: {f0_voiced.mean().item():.1f}, std: {f0_voiced.std().item():.1f}")
        # フレーム間差分のRMS (急激な跳躍がないか)
        f0_diff = torch.diff(f0_voiced)
        print(f"    Frame-to-frame F0 diff RMS: {f0_diff.pow(2).mean().sqrt().item():.1f} Hz")
        # 跳躍フレーム数 (>50Hz変化)
        n_jumps = (f0_diff.abs() > 50).sum().item()
        print(f"    Large jumps (>50Hz): {n_jumps}/{len(f0_diff)} frames")

    voiced_pct = voiced_mask.float().mean().item() * 100
    print(f"    Voiced: {voiced_pct:.1f}%, frames: {feat.shape[1]}")

    # 4b: 有声/無声の遷移確認 (正弦波 → 無音 → 正弦波)
    print("\n  --- 4b: Voiced/Unvoiced 遷移テスト ---")
    t1 = torch.arange(0, int(1 * SR)) / SR
    wav1 = torch.sin(2 * math.pi * 440 * t1) * 0.5
    sil = torch.zeros(int(0.5 * SR))
    wav2 = torch.sin(2 * math.pi * 220 * t1) * 0.5
    hybrid = torch.cat([wav1, sil, wav2])  # (T,)

    with torch.no_grad():
        feat_h = prosody(hybrid.unsqueeze(0).unsqueeze(0))
    voiced_h = feat_h[0, :, 1]

    # 有声→無声→有声の遷移を検出
    voiced_diff = torch.diff((voiced_h > 0.5).int())
    n_transitions = voiced_diff.abs().sum().item()
    print(f"    Voiced/Unvoiced transitions: {n_transitions}")

    # 遷移地点を確認
    transitions = torch.where(voiced_diff != 0)[0]
    if len(transitions) > 0:
        for t_idx in transitions[:8].tolist():
            before = "voiced" if voiced_h[max(0, t_idx-1)] > 0.5 else "unvoiced"
            after = "voiced" if voiced_h[min(len(voiced_h)-1, t_idx+1)] > 0.5 else "unvoiced"
            print(f"      Frame {t_idx}: {before} → {after}")

    # 4c: 実音声のlog_f0分布
    print("\n  --- 4c: 実音声のlog_f0分布 ---")
    spk = sorted([d for d in os.listdir(VCTK_DIR) if d.startswith('p')])[0]
    wav = load_vctk_audio(spk, 0, crop_sec=5.0)
    with torch.no_grad():
        feat_real = prosody(wav.unsqueeze(0))
    log_f0_r = feat_real[0, :, 0]
    voiced_r = feat_real[0, :, 1] > 0.5
    if voiced_r.sum() > 0:
        log_f0_voiced = log_f0_r[voiced_r]
        print(f"    {spk}: log_f0 range [{log_f0_voiced.min().item():.2f}, "
              f"{log_f0_voiced.max().item():.2f}], "
              f"F0 range [{log_f0_voiced.exp().min().item():.0f}, "
              f"{log_f0_voiced.exp().max().item():.0f}] Hz")

    checks = {}
    checks["voiced/unvoiced遷移検出"] = n_transitions >= 2
    checks["log_f0が有限値"] = not torch.isnan(log_f0_r).any()
    checks["SWEEP F0 range > 400Hz"] = (f0_vals.max() - f0_vals.min()) > 400 if voiced_mask.sum() > 0 else False

    for k, v in checks.items():
        status = "✓" if v else "✗"
        print(f"    {status} {k}: {v}")

    return {"checks": checks, "n_transitions": n_transitions}


# ─── 検証5: 短発話vs長発話の一貫性 ──────────────────────────────

def test_short_vs_long_consistency():
    """1秒 vs 10秒発話で SpeakerEncoder embedding の一貫性を評価。"""
    print("\n" + "="*70)
    print("検証5: 短発話(1秒) vs 長発話(10秒) 一貫性")
    print("="*70)

    spk_enc = make_speaker_encoder().eval()

    speakers = sorted([d for d in os.listdir(VCTK_DIR) if d.startswith('p')])[:10]
    results = []
    all_cos_short = []
    all_cos_long = []

    for spk in speakers:
        # 長い発話を探す (>10秒)
        spk_dir = os.path.join(VCTK_DIR, spk)
        files = sorted([f for f in os.listdir(spk_dir) if 'mic1' in f and f.endswith('.flac')])

        # 全ファイル長を確認
        best_long = None
        best_long_dur = 0
        best_short = None

        for f in files:
            path = os.path.join(spk_dir, f)
            info = sf.info(path)
            dur = info.duration
            if dur >= 10.0 and dur > best_long_dur:
                best_long = (f, dur, path)
                best_long_dur = dur
            elif dur >= 3.0 and best_short is None:
                best_short = (f, dur, path)

        if best_long is None or best_short is None:
            continue

        # Load
        wav_long_np, sr_long = sf.read(best_long[2])
        wav_short_np, sr_short = sf.read(best_short[2])
        wav_long = torch.from_numpy(wav_long_np).float()
        wav_short = torch.from_numpy(wav_short_np).float()

        if wav_long.dim() == 2:
            wav_long = wav_long.mean(dim=-1)
        if wav_short.dim() == 2:
            wav_short = wav_short.mean(dim=-1)
        wav_long = wav_long.unsqueeze(0)  # (1, T)
        wav_short = wav_short.unsqueeze(0)

        # Resample if needed
        if sr_long != SR:
            wav_long = torchaudio.functional.resample(wav_long, sr_long, SR)
        if sr_short != SR:
            wav_short = torchaudio.functional.resample(wav_short, sr_short, SR)

        # Short: crop to first 1 second
        wav_1s = wav_short[:, :int(SR)]  # (1, 44100)
        wav_10s = wav_long[:, :int(10 * SR)]  # (1, 441000)

        with torch.no_grad():
            emb_1s, _ = spk_enc(wav_1s.unsqueeze(0).float())
            emb_10s, _ = spk_enc(wav_10s.unsqueeze(0).float())

        cos = cosine_sim(emb_1s, emb_10s).item()
        norm_1s = emb_1s.norm().item()
        norm_10s = emb_10s.norm().item()

        results.append({
            "speaker": spk,
            "dur_long": best_long_dur,
            "cos_sim": cos,
            "norm_1s": norm_1s,
            "norm_10s": norm_10s,
        })
        print(f"  {spk}: long={best_long_dur:.1f}s, cos_sim={cos:.4f}, "
              f"norm(1s)={norm_1s:.2f}, norm(10s)={norm_10s:.2f}")

    if results:
        cos_vals = [r["cos_sim"] for r in results]
        print(f"\n  --- 集計 (N={len(results)}) ---")
        print(f"  Cosine sim (1s vs 10s): mean={np.mean(cos_vals):.4f}, "
              f"std={np.std(cos_vals):.4f}, min={np.min(cos_vals):.4f}, max={np.max(cos_vals):.4f}")

        checks = {}
        checks["短長一貫性 (mean_cos > 0.5)"] = np.mean(cos_vals) > 0.5
        checks["短長一貫性 (min_cos > 0.2)"] = np.min(cos_vals) > 0.2
        checks["全話者NaNなし"] = all(not math.isnan(r["cos_sim"]) for r in results)

        for k, v in checks.items():
            status = "✓" if v else "✗"
            print(f"    {status} {k}: {v}")
    else:
        cos_vals = []
        checks = {"no_long_audio": False}
        print("  ✗ 10秒以上の音声が不足")

    return {
        "results": results,
        "cos_mean": float(np.mean(cos_vals)) if cos_vals else None,
        "cos_std": float(np.std(cos_vals)) if cos_vals else None,
        "checks": checks,
    }


# ─── 検証6: ECAPA-TDNN との比較 ─────────────────────────────────

def test_ecapa_comparison():
    """FlowVC SpeakerEncoder vs ECAPA-TDNN の話者分離性能比較。"""
    print("\n" + "="*70)
    print("検証6: SpeakerEncoder vs ECAPA-TDNN 比較")
    print("="*70)

    try:
        from speechbrain.inference import EncoderClassifier
        ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        print("  ECAPA-TDNN loaded (speechbrain)")
    except Exception as e:
        print(f"  ✗ ECAPA-TDNN load failed: {e}")
        return {"error": str(e), "checks": {}}

    spk_enc = make_speaker_encoder().eval()

    # Use same speakers as test 1
    random.seed(42)
    all_speakers = sorted([d for d in os.listdir(VCTK_DIR) if d.startswith('p')])
    selected = random.sample(all_speakers, min(10, len(all_speakers)))
    print(f"  Testing {len(selected)} speakers")

    def extract_ecapa(wav_44k):
        """wav_44k: (1, T) @ 44100Hz -> (192,) ECAPA embedding"""
        # Resample to 16kHz: (1, T) -> (1, T')
        wav_16k = torchaudio.functional.resample(wav_44k.float(), 44100, 16000)
        # ECAPA encode_batch expects (B, T) — we have (1, T) which is correct
        emb = ecapa.encode_batch(wav_16k)
        return emb.squeeze()  # (192,)

    def extract_flowvc(wav_44k):
        """wav_44k: (1, T) @ 44100Hz -> (192,) FlowVC embedding"""
        wav_in = wav_44k.unsqueeze(0).float()  # (1, 1, T)
        with torch.no_grad():
            emb, _ = spk_enc(wav_in)
        return emb.squeeze()  # (192,)

    # Collect embeddings
    flowvc_embs = {}
    ecapa_embs = {}

    for spk in selected:
        flowvc_list = []
        ecapa_list = []
        for i in range(min(3, 5)):
            wav = load_vctk_audio(spk, i, crop_sec=3.0)
            if wav is None:
                continue
            try:
                # FlowVC: needs 44.1kHz
                emb_f = extract_flowvc(wav)
                flowvc_list.append(emb_f)

                # ECAPA: from 44.1kHz → resample internally
                emb_e = extract_ecapa(wav)
                ecapa_list.append(emb_e)
            except Exception as e:
                print(f"    Error {spk}: {e}")
                continue

        if len(flowvc_list) >= 2:
            flowvc_embs[spk] = torch.stack(flowvc_list)
            ecapa_embs[spk] = torch.stack(ecapa_list)

    print(f"\n  Collected embeddings for {len(flowvc_embs)} speakers")

    # Compute same/diff speaker cosine similarities for both models
    def compute_sims(embeddings_dict):
        same_sims = []
        centroids = {}
        for spk, embs in embeddings_dict.items():
            c = embs.mean(dim=0)
            centroids[spk] = c
            for i in range(embs.shape[0]):
                for j in range(i+1, embs.shape[0]):
                    same_sims.append(cosine_sim(embs[i], embs[j]).item())
        diff_sims = []
        spk_list = list(centroids.keys())
        for i in range(len(spk_list)):
            for j in range(i+1, len(spk_list)):
                diff_sims.append(cosine_sim(centroids[spk_list[i]],
                                            centroids[spk_list[j]]).item())
        return np.array(same_sims), np.array(diff_sims)

    fv_same, fv_diff = compute_sims(flowvc_embs)
    ec_same, ec_diff = compute_sims(ecapa_embs)

    print("\n  --- FlowVC SpeakerEncoder ---")
    print(f"    同話者: mean={fv_same.mean():.4f}, std={fv_same.std():.4f}")
    print(f"    異話者: mean={fv_diff.mean():.4f}, std={fv_diff.std():.4f}")
    fv_sep = fv_same.mean() - fv_diff.mean()

    print("\n  --- ECAPA-TDNN ---")
    print(f"    同話者: mean={ec_same.mean():.4f}, std={ec_same.std():.4f}")
    print(f"    異話者: mean={ec_diff.mean():.4f}, std={ec_diff.std():.4f}")
    ec_sep = ec_same.mean() - ec_diff.mean()

    print("\n  --- 比較 ---")
    print(f"    FlowVC 分離度: {fv_sep:.4f}")
    print(f"    ECAPA  分離度: {ec_sep:.4f}")

    checks = {}
    ratio = fv_sep / (ec_sep + 1e-8)
    if ec_sep > 0:
        print(f"    分離度比 (FlowVC/ECAPA): {ratio:.2f}")
    checks["FlowVC分離度 > 0 (未学習でも会話者分離あり)"] = fv_sep > 0.05
    checks["ECAPA分離度 > FlowVC分離度 (ECAPA既学習優位)"] = ec_sep > fv_sep
    checks["FlowVC分離度 ≥ ECAPAの30%"] = ratio >= 0.3
    checks["ECAPA同話者sim > 異話者sim"] = ec_same.mean() > ec_diff.mean()

    # Embedding correlation between FlowVC and ECAPA
    all_fv = []
    all_ec = []
    for spk in flowvc_embs:
        all_fv.append(flowvc_embs[spk].mean(dim=0))
        all_ec.append(ecapa_embs[spk].mean(dim=0))
    all_fv = torch.stack(all_fv)
    all_ec = torch.stack(all_ec)

    # Pearson correlation
    fv_z = (all_fv - all_fv.mean(0, keepdim=True)) / (all_fv.std(0, keepdim=True) + 1e-8)
    ec_z = (all_ec - all_ec.mean(0, keepdim=True)) / (all_ec.std(0, keepdim=True) + 1e-8)
    pearson = (fv_z * ec_z).mean().item()

    for k, v in checks.items():
        status = "✓" if v else "✗"
        print(f"    {status} {k}: {v}")

    return {
        "flowvc": {"same_mean": float(fv_same.mean()), "diff_mean": float(fv_diff.mean()),
                    "separation": float(fv_sep)},
        "ecapa": {"same_mean": float(ec_same.mean()), "diff_mean": float(ec_diff.mean()),
                   "separation": float(ec_sep)},
        "separation_ratio": float(fv_sep / (ec_sep + 1e-8)),
        "pearson_correlation": float(pearson),
        "checks": checks,
    }


# ─── メイン ──────────────────────────────────────────────────────

def main():
    print("="*70)
    print("FlowVC 話者・韻律表現力 品質検証 (Agent #9)")
    print(f"Device: {DEVICE}, Python: {sys.executable}")
    print("="*70)

    all_results = {}
    all_checks = {}

    # 検証1: SpeakerEncoder 話者分離度
    try:
        r1 = test_speaker_separation()
        all_results["speaker_separation"] = r1
        all_checks.update(r1.get("checks", {}))
    except Exception as e:
        print(f"  ✗ 検証1失敗: {e}")
        import traceback; traceback.print_exc()
        all_results["speaker_separation"] = {"error": str(e)}
        all_checks["speaker_separation"] = False

    # 検証2: PromptToken直交性
    try:
        r2 = test_prompt_orthogonality()
        all_results["prompt_orthogonality"] = r2
        all_checks.update(r2.get("checks", {}))
    except Exception as e:
        print(f"  ✗ 検証2失敗: {e}")
        import traceback; traceback.print_exc()
        all_results["prompt_orthogonality"] = {"error": str(e)}
        all_checks["prompt_orthogonality"] = False

    # 検証3: FCPE F0品質
    try:
        r3 = test_fcpe_f0_quality()
        all_results["fcpe_f0_quality"] = r3
        all_checks.update(r3.get("checks", {}))
    except Exception as e:
        print(f"  ✗ 検証3失敗: {e}")
        import traceback; traceback.print_exc()
        all_results["fcpe_f0_quality"] = {"error": str(e)}
        all_checks["fcpe_f0_quality"] = False

    # 検証4: F0→25Hz リサンプル品質
    try:
        r4 = test_f0_resample_quality()
        all_results["f0_resample"] = r4
        all_checks.update(r4.get("checks", {}))
    except Exception as e:
        print(f"  ✗ 検証4失敗: {e}")
        import traceback; traceback.print_exc()
        all_results["f0_resample"] = {"error": str(e)}
        all_checks["f0_resample"] = False

    # 検証5: 短vs長 一貫性
    try:
        r5 = test_short_vs_long_consistency()
        all_results["short_vs_long"] = r5
        all_checks.update(r5.get("checks", {}))
    except Exception as e:
        print(f"  ✗ 検証5失敗: {e}")
        import traceback; traceback.print_exc()
        all_results["short_vs_long"] = {"error": str(e)}
        all_checks["short_vs_long"] = False

    # 検証6: ECAPA-TDNN比較
    try:
        r6 = test_ecapa_comparison()
        all_results["ecapa_comparison"] = r6
        all_checks.update(r6.get("checks", {}))
    except Exception as e:
        print(f"  ✗ 検証6失敗: {e}")
        import traceback; traceback.print_exc()
        all_results["ecapa_comparison"] = {"error": str(e)}
        all_checks["ecapa_comparison"] = False

    # ─── 総合判定 ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("総合判定")
    print("="*70)

    passed = sum(1 for v in all_checks.values() if v)
    total = len(all_checks)
    print(f"\n  Passed: {passed}/{total}")

    for k, v in all_checks.items():
        status = "✓" if v else "✗"
        print(f"    {status} {k}")

    # Save results
    import json
    out_path = os.path.join(os.path.dirname(__file__), "eval_speaker_prosody_results.json")
    # Convert non-serializable values
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    cleaned = clean(all_results)
    with open(out_path, 'w') as f:
        json.dump(cleaned, f, indent=2, default=str)
    print(f"\n  結果保存: {out_path}")

    return all_results, all_checks


if __name__ == "__main__":
    main()

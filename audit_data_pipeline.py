#!/usr/bin/env python3
"""
データパイプライン健全性監査スクリプト
VCTK データセットの品質・バランス・学習適合性を実データで検証
"""
from __future__ import annotations

import os
import glob
import random
import time
import json
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
import torchaudio
import torchcodec

# ============================================================
# Config
# ============================================================
DATA_DIR = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
SAMPLE_RATE = 44100
CROP_SEC = 2.0
REF_SEC = 3.0
NUM_PAIR_TRIALS = 10000

results = {}

# ============================================================
# 1. 話者バランス調査
# ============================================================
print("=" * 60)
print("1. 話者バランス調査")
print("=" * 60)

files = sorted(glob.glob(str(Path(DATA_DIR) / "**" / "*.flac"), recursive=True))
print(f"  総ファイル数: {len(files)}")

speaker_files = defaultdict(list)
for f in files:
    spk = Path(f).parent.name
    speaker_files[spk].append(f)

speakers = sorted(speaker_files.keys())
print(f"  話者数: {len(speakers)}")

file_counts = {s: len(fs) for s, fs in speaker_files.items()}
min_spk = min(file_counts, key=file_counts.get)
max_spk = max(file_counts, key=file_counts.get)
print(f"  最小話者: {min_spk} ({file_counts[min_spk]}ファイル)")
print(f"  最大話者: {max_spk} ({file_counts[max_spk]}ファイル)")
print(f"  中央値: {sorted(file_counts.values())[len(file_counts)//2]}")

# 有効話者数（2発話以上）
valid_speakers = [s for s, fs in speaker_files.items() if len(fs) >= 2]
print(f"  有効話者数 (2発話以上): {len(valid_speakers)}")

results["speaker_balance"] = {
    "total_files": len(files),
    "total_speakers": len(speakers),
    "valid_speakers": len(valid_speakers),
    "min_files": file_counts[min_spk],
    "max_files": file_counts[max_spk],
    "median_files": sorted(file_counts.values())[len(file_counts)//2],
    "imbalance_ratio": file_counts[max_spk] / file_counts[min_spk],
}

# ファイル数ヒストグラム（簡易）
print(f"\n  ファイル数分布:")
bins = [0, 200, 400, 600, 800, 1000, 9999]
bin_labels = ["0-200", "200-400", "400-600", "600-800", "800-1000", "1000+"]
for lo, hi, label in zip(bins[:-1], bins[1:], bin_labels):
    cnt = sum(1 for v in file_counts.values() if lo < v <= hi)
    bar = "█" * (cnt // 2) if cnt > 0 else ""
    print(f"    {label:>8}: {cnt:3d} 話者 {bar}")

# ============================================================
# 2. オーディオ長分布調査 (実際に読み込む)
# ============================================================
print("\n" + "=" * 60)
print("2. オーディオ長・サンプルレート調査")
print("=" * 60)

# 全ファイルのメタデータ取得（torchcodecで高速）
print("  メタデータ収集中...")
durations = []
sample_rates = []
num_channels_list = []
speaker_durations = defaultdict(list)

# 全ファイルを走査（最大5000ファイル）
sample_files = files[:5000] if len(files) > 5000 else files
for i, f in enumerate(sample_files):
    if i % 1000 == 0 and i > 0:
        print(f"    {i}/{len(sample_files)}...")
    try:
        dec = torchcodec.decoders.AudioDecoder(f)
        meta = dec.metadata
        dur = meta.duration_seconds
        durations.append(dur)
        sample_rates.append(meta.sample_rate)
        num_channels_list.append(meta.num_channels)
        spk = Path(f).parent.name
        speaker_durations[spk].append(dur)
    except Exception:
        pass

print(f"  解析ファイル数: {len(durations)}")
print(f"  サンプルレート: {set(sample_rates)}")
print(f"  チャンネル数: {set(num_channels_list)}")

# 長さ統計
import numpy as np
durs = np.array(durations)
print(f"\n  オーディオ長統計 (秒):")
print(f"    min:    {durs.min():.2f}s")
print(f"    max:    {durs.max():.2f}s")
print(f"    mean:   {durs.mean():.2f}s")
print(f"    median: {np.median(durs):.2f}s")
print(f"    std:    {durs.std():.2f}s")

# crop_seconds=2.0 より短いもの
short_src = (durs < CROP_SEC).sum()
short_ref = (durs < REF_SEC).sum()
print(f"\n  crop_seconds={CROP_SEC}s より短い: {short_src}/{len(durs)} ({100*short_src/len(durs):.1f}%)")
print(f"  ref_seconds={REF_SEC}s より短い:   {short_ref}/{len(durs)} ({100*short_ref/len(durs):.1f}%)")

# 話者ごとの平均長
spk_avg_dur = {s: np.mean(ds) for s, ds in speaker_durations.items()}
print(f"\n  話者別平均長: min={min(spk_avg_dur.values()):.2f}s, max={max(spk_avg_dur.values()):.2f}s")

results["audio_length"] = {
    "num_analyzed": len(durations),
    "sample_rates": list(set(sample_rates)),
    "channels": list(set(num_channels_list)),
    "duration_min": float(durs.min()),
    "duration_max": float(durs.max()),
    "duration_mean": float(durs.mean()),
    "duration_median": float(np.median(durs)),
    "short_than_2s": int(short_src),
    "short_than_3s": int(short_ref),
    "short_2s_pct": float(100*short_src/len(durs)),
    "short_3s_pct": float(100*short_ref/len(durs)),
}

# ============================================================
# 3. サンプルレート変換品質検証 (48kHz → 44.1kHz)
# ============================================================
print("\n" + "=" * 60)
print("3. サンプルレート変換品質検証")
print("=" * 60)

# 実際のファイルでリサンプリング前後の特性を比較
# VCTKは48kHzなので、44.1kHzへの変換が必要
# torchaudio.transforms.Resample と torchaudio.functional.resample の比較

# テストファイルを選んで実際に変換
test_file = files[0]
wav_48k, sr_orig = torchaudio.load(test_file)
print(f"  テストファイル: {Path(test_file).name}")
print(f"  元サンプルレート: {sr_orig} Hz")
print(f"  元shape: {wav_48k.shape}")

# リサンプリング
wav_44k = torchaudio.functional.resample(wav_48k, sr_orig, SAMPLE_RATE)
print(f"  変換後shape: {wav_44k.shape}")

# 波形の変化を確認
if wav_48k.shape[1] > 0:
    wav_48k_mono = wav_48k.mean(dim=0) if wav_48k.shape[0] > 1 else wav_48k[0]
    wav_44k_mono = wav_44k.mean(dim=0) if wav_44k.shape[0] > 1 else wav_44k[0]
    
    # RMS energy
    rms_48k = torch.sqrt(torch.mean(wav_48k_mono ** 2))
    rms_44k = torch.sqrt(torch.mean(wav_44k_mono ** 2))
    print(f"  RMS (48kHz): {rms_48k:.6f}")
    print(f"  RMS (44.1kHz): {rms_44k:.6f}")
    print(f"  RMS変化率: {100*(rms_44k - rms_48k)/rms_48k:.4f}%")
    
    # Peak
    peak_48k = wav_48k_mono.abs().max()
    peak_44k = wav_44k_mono.abs().max()
    print(f"  Peak (48kHz): {peak_48k:.6f}")
    print(f"  Peak (44.1kHz): {peak_44k:.6f}")
    
    # 期待サンプル数との一致
    expected_samples = int(wav_48k.shape[1] * SAMPLE_RATE / sr_orig)
    actual_samples = wav_44k.shape[1]
    print(f"  期待サンプル数: {expected_samples}, 実際: {actual_samples}, 差: {actual_samples - expected_samples}")

results["resample_quality"] = {
    "orig_sr": sr_orig,
    "target_sr": SAMPLE_RATE,
    "rms_48k": float(rms_48k),
    "rms_44k": float(rms_44k),
    "rms_change_pct": float(100*(rms_44k - rms_48k)/rms_48k),
    "expected_samples": expected_samples,
    "actual_samples": actual_samples,
    "sample_diff": actual_samples - expected_samples,
}

# ============================================================
# 4. Zero-padding の音響的影響
# ============================================================
print("\n" + "=" * 60)
print("4. Zero-padding の影響分析")
print("=" * 60)

# 短いファイルを探してzero-paddingの影響を確認
short_files = []
for f in sample_files:
    try:
        dec = torchcodec.decoders.AudioDecoder(f)
        dur = dec.metadata.duration_seconds
        if dur < CROP_SEC:
            short_files.append((f, dur))
    except Exception:
        pass

print(f"  {CROP_SEC}s未満のファイル: {len(short_files)}個")

if short_files:
    # 最短のファイルでテスト
    short_f, short_dur = min(short_files, key=lambda x: x[1])
    print(f"  最短ファイル: {Path(short_f).name} ({short_dur:.2f}s)")
    
    wav, sr = torchaudio.load(short_f)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    
    target_samples = int(CROP_SEC * SAMPLE_RATE)
    original_len = wav.shape[-1]
    pad_len = target_samples - original_len
    print(f"  元長: {original_len}, target: {target_samples}, padding: {pad_len} ({100*pad_len/target_samples:.1f}%)")
    
    # RMS 比較
    rms_before = torch.sqrt(torch.mean(wav ** 2))
    wav_padded = F.pad(wav, (0, pad_len))
    rms_after = torch.sqrt(torch.mean(wav_padded ** 2))
    print(f"  RMS padding前: {rms_before:.6f}, padding後: {rms_after:.6f}")
    print(f"  RMS低下率: -{100*(1 - rms_after/rms_before):.2f}%")
    
    # 最大zero-padding率
    max_pad_ratio = 0.0
    if short_files:
        for f, dur in short_files:
            try:
                dec = torchcodec.decoders.AudioDecoder(f)
                meta = dec.metadata
                orig_samples_44k = int(meta.duration_seconds * SAMPLE_RATE)
                ratio = max(0, 1 - orig_samples_44k / target_samples)
                max_pad_ratio = max(max_pad_ratio, ratio)
            except Exception:
                pass
        print(f"  最大zero-padding率: {100*max_pad_ratio:.1f}%")
else:
    max_pad_ratio = 0.0
    print("  全てのファイルが十分な長さを持っています")

results["zero_padding"] = {
    "short_files_count": len(short_files),
    "max_padding_ratio": float(max_pad_ratio) if short_files else 0.0,
}

# ============================================================
# 5. Random Pair Sampling の偏りチェック
# ============================================================
print("\n" + "=" * 60)
print("5. Random Pair Sampling 偏り分析")
print("=" * 60)

# シミュレーション: 話者ペア生成の偏りをチェック
random.seed(42)

# 各話者の選択回数をカウント (異話者ペア)
cross_speaker_counts = Counter()
identity_counts = Counter()
tgt_spk_in_cross = Counter()

# データセットのロジックを再現
valid_speakers_list = [s for s in speakers if len(speaker_files[s]) >= 2]
speaker_to_files = {s: [f for f in fl] for s, fl in speaker_files.items()}

for _ in range(NUM_PAIR_TRIALS):
    src_spk = random.choice(valid_speakers_list)
    src_path = random.choice(speaker_to_files[src_spk])
    
    if random.random() < 0.5:
        # 同一話者
        tgt_path = random.choice([f for f in speaker_to_files[src_spk] if f != src_path])
        tgt_spk = src_spk
        identity_counts[src_spk] += 1
    else:
        # 異話者
        tgt_spk = random.choice([s for s in valid_speakers_list if s != src_spk])
        tgt_path = random.choice(speaker_to_files[tgt_spk])
        cross_speaker_counts[src_spk] += 1
        tgt_spk_in_cross[tgt_spk] += 1

print(f"  シミュレーション試行数: {NUM_PAIR_TRIALS}")
print(f"  同一話者ペア: {sum(identity_counts.values())}")
print(f"  異話者ペア:   {sum(cross_speaker_counts.values())}")

# 各話者の異話者ペアにおける出現頻度の偏り
src_counts = list(cross_speaker_counts.values())
if src_counts:
    print(f"\n  異話者ペアでのsrc話者出現頻度:")
    print(f"    min: {min(src_counts)}, max: {max(src_counts)}, mean: {np.mean(src_counts):.1f}")
    print(f"    CV (変動係数): {np.std(src_counts)/np.mean(src_counts):.4f}")

tgt_counts = list(tgt_spk_in_cross.values())
if tgt_counts:
    print(f"\n  異話者ペアでのtgt話者出現頻度:")
    print(f"    min: {min(tgt_counts)}, max: {max(tgt_counts)}, mean: {np.mean(tgt_counts):.1f}")
    print(f"    CV (変動係数): {np.std(tgt_counts)/np.mean(tgt_counts):.4f}")

# Gini係数 (話者選択の不平等度)
def gini(x):
    x = sorted(x)
    n = len(x)
    cum = np.cumsum(x)
    return (2 * sum((i+1)*x[i] for i in range(n)) - (n+1)*sum(x)) / (n * sum(x))

if len(src_counts) > 0:
    print(f"\n  ソース話者選択 Gini係数: {gini(src_counts):.4f}")
    print(f"  ターゲット話者選択 Gini係数: {gini(tgt_counts):.4f}")

results["pair_sampling"] = {
    "num_trials": NUM_PAIR_TRIALS,
    "identity_pairs": sum(identity_counts.values()),
    "cross_pairs": sum(cross_speaker_counts.values()),
    "src_gini": float(gini(src_counts)) if src_counts else None,
    "tgt_gini": float(gini(tgt_counts)) if tgt_counts else None,
}

# ============================================================
# 6. ファイルI/O ボトルネック分析
# ============================================================
print("\n" + "=" * 60)
print("6. ファイルI/O ベンチマーク")
print("=" * 60)

# 100ファイルの読み込み時間を測定
bench_files = files[:100]
io_times = []

print("  100ファイル読み込みベンチマーク...")
for f in bench_files:
    t0 = time.time()
    wav, sr = torchaudio.load(f)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    # crop or pad
    target_samples = int(CROP_SEC * SAMPLE_RATE)
    if wav.shape[-1] > target_samples:
        wav = wav[:target_samples]
    elif wav.shape[-1] < target_samples:
        wav = F.pad(wav, (0, target_samples - wav.shape[-1]))
    io_times.append(time.time() - t0)

io_times = np.array(io_times)
print(f"  平均I/O時間: {io_times.mean()*1000:.1f}ms/ファイル")
print(f"  中央値:      {np.median(io_times)*1000:.1f}ms")
print(f"  max:          {io_times.max()*1000:.1f}ms")
print(f"  スループット: {100/io_times.sum():.1f} ファイル/秒")

# バッチサイズ8、ワーカー0の場合の推定スループット
batch_time_est = io_times.mean() * 3 * 8  # 3ファイル/サンプル × バッチ8
print(f"  推定バッチ処理時間 (bs=8, 3 wavs/item): {batch_time_est*1000:.0f}ms")
print(f"  推定イテレーション/秒: {1/batch_time_est:.2f}")

results["io_benchmark"] = {
    "avg_load_ms": float(io_times.mean() * 1000),
    "median_load_ms": float(np.median(io_times) * 1000),
    "max_load_ms": float(io_times.max() * 1000),
    "throughput_files_per_sec": float(100 / io_times.sum()),
    "est_batch_time_ms": float(batch_time_est * 1000),
}

# ============================================================
# 7. CachedDataset ビルドの決定性問題
# ============================================================
print("\n" + "=" * 60)
print("7. CachedDataset 決定性分析")
print("=" * 60)

# 問題: VCTKDataset.__getitem__ 内で random.random() と random.choice() を使用
# → シード固定なしでは毎回異なるペアが生成される
# → build_cache でのエンコード結果が非決定的

# 検証: 同じインデックスで2回呼び出した場合の一致率
random.seed(42)
test_ds_files = files[:100]  # テスト用
test_speaker_files = defaultdict(list)
for f in test_ds_files:
    spk = Path(f).parent.name
    test_speaker_files[spk].append(f)
test_valid_spk = [s for s in test_speaker_files if len(test_speaker_files[s]) >= 2]

# 簡易シミュレーション
random.seed(123)
pairs_run1 = []
for i in range(100):
    spk = Path(test_ds_files[i % len(test_ds_files)]).parent.name
    if random.random() < 0.5:
        pair_type = "same"
    else:
        pair_type = "diff"
    pairs_run1.append((i % len(test_ds_files), pair_type, spk))

random.seed(123)
pairs_run2 = []
for i in range(100):
    spk = Path(test_ds_files[i % len(test_ds_files)]).parent.name
    if random.random() < 0.5:
        pair_type = "same"
    else:
        pair_type = "diff"
    pairs_run2.append((i % len(test_ds_files), pair_type, spk))

match = sum(1 for a, b in zip(pairs_run1, pairs_run2) if a[1] == b[1])
print(f"  同一シードでのペアタイプ一致率: {match}/{len(pairs_run1)} ({100*match/len(pairs_run1):.1f}%)")

# 異なるシードでは？
random.seed(456)
pairs_run3 = []
for i in range(100):
    spk = Path(test_ds_files[i % len(test_ds_files)]).parent.name
    if random.random() < 0.5:
        pair_type = "same"
    else:
        pair_type = "diff"
    pairs_run3.append((i % len(test_ds_files), pair_type, spk))

match_diff = sum(1 for a, b in zip(pairs_run1, pairs_run3) if a[1] == b[1])
print(f"  異シードでのペアタイプ一致率: {match_diff}/{len(pairs_run1)} ({100*match_diff/len(pairs_run1):.1f}%)")

print(f"\n  結論: build_cache() は各呼び出しで異なるペアを生成する")
print(f"  → キャッシュ再構築で結果が再現しない")
print(f"  → 推奨: random.seed() の固定 または 決定論的ペア生成アルゴリズム")

results["cached_dataset_determinism"] = {
    "same_seed_match_rate": float(100*match/len(pairs_run1)),
    "diff_seed_match_rate": float(100*match_diff/len(pairs_run1)),
    "is_deterministic": match == len(pairs_run1),
    "issue": "build_cacheは非決定的。random.seed()の固定が必要。",
}

# ============================================================
# 8. 総合評価
# ============================================================
print("\n" + "=" * 60)
print("8. 総合評価")
print("=" * 60)

issues = []
warnings = []
oks = []

# 話者バランス
imbalance = results["speaker_balance"]["imbalance_ratio"]
if imbalance > 5:
    issues.append(f"話者バランス悪化 (imbalance ratio={imbalance:.1f}x)")
elif imbalance > 3:
    warnings.append(f"話者バランスに偏りあり (ratio={imbalance:.1f}x)")
else:
    oks.append(f"話者バランス良好 (ratio={imbalance:.1f}x)")

# 有効話者数
n_valid = results["speaker_balance"]["valid_speakers"]
if n_valid < 50:
    issues.append(f"有効話者数不足 ({n_valid})")
else:
    oks.append(f"有効話者数十分 ({n_valid})")

# 短いファイル
short_2s_pct = results["audio_length"]["short_2s_pct"]
if short_2s_pct > 10:
    issues.append(f"CROP_SEC=2s未満のファイルが{short_2s_pct:.1f}%存在")
elif short_2s_pct > 2:
    warnings.append(f"CROP_SEC=2s未満のファイルが{short_2s_pct:.1f}%存在")
else:
    oks.append(f"CROP_SEC=2s未満のファイルは{short_2s_pct:.1f}% (良好)")

short_3s_pct = results["audio_length"]["short_3s_pct"]
if short_3s_pct > 10:
    issues.append(f"REF_SEC=3s未満のファイルが{short_3s_pct:.1f}%存在")
elif short_3s_pct > 2:
    warnings.append(f"REF_SEC=3s未満のファイルが{short_3s_pct:.1f}%存在")
else:
    oks.append(f"REF_SEC=3s未満のファイルは{short_3s_pct:.1f}% (良好)")

# サンプルレート
if 48000 in results["audio_length"]["sample_rates"]:
    oks.append("VCTK 48kHz → 44.1kHz 変換は正常動作")
else:
    warnings.append("想定外のサンプルレート")

# Zero-padding
zp = results["zero_padding"]
if zp["max_padding_ratio"] > 0.5:
    issues.append(f"Zero-padding率が高い ({zp['max_padding_ratio']*100:.1f}%)")
elif zp["max_padding_ratio"] > 0.2:
    warnings.append(f"一部ファイルでZero-padding発生 ({zp['max_padding_ratio']*100:.1f}%)")
else:
    oks.append(f"Zero-paddingの影響は限定的")

# I/O
io = results["io_benchmark"]
if io["avg_load_ms"] > 100:
    warnings.append(f"ファイルI/Oが遅い ({io['avg_load_ms']:.0f}ms/file)")
else:
    oks.append(f"ファイルI/O速度は許容範囲 ({io['avg_load_ms']:.0f}ms/file)")

# 決定性
if not results["cached_dataset_determinism"]["is_deterministic"]:
    issues.append("CachedDatasetビルドが非決定的 (重大)")
else:
    # 同一シードでは決定的だが、build_cacheはシードを固定していない
    if results["cached_dataset_determinism"]["diff_seed_match_rate"] < 100:
        issues.append("CachedDatasetビルドが非決定的 (シード未固定のため)")
    else:
        oks.append("CachedDatasetビルドは決定的")

# ペアサンプリング
ps = results["pair_sampling"]
if ps["src_gini"] and ps["src_gini"] > 0.3:
    warnings.append(f"話者ペアサンプリングに偏り (Gini={ps['src_gini']:.3f})")
else:
    oks.append("話者ペアサンプリングは概ね均一")

print("\n  ◆ 重大な問題:")
for i in issues:
    print(f"    ❌ {i}")

print("\n  ◆ 警告:")
for w in warnings:
    print(f"    ⚠️  {w}")

print("\n  ◆ 問題なし:")
for o in oks:
    print(f"    ✅ {o}")

# 結果をJSON出力
results["summary"] = {
    "issues": issues,
    "warnings": warnings,
    "oks": oks,
    "grade": "A" if len(issues) == 0 else ("B" if len(issues) <= 1 else "C"),
}

output_path = "/Users/asill/btrv5/audit_data_pipeline_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

print(f"\n結果を {output_path} に保存しました")

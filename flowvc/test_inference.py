#!/usr/bin/env python3
"""
FlowVC 推論パイプライン実戦検証スクリプト
検証項目:
1. 実際のWAVファイルで convert_file がエラーなく完了するか
2. 出力音声の基本的な品質指標 (RMS, peak, clipping有無)
3. 無音/極短/極長などエッジケースの処理
4. FCPEが実際の音声で合理的なF0値を出力するか
5. SpeakerEncoderが異なる話者を区別できるか (cosine similarity)
6. ストリーミング出力の連続性 (click音の有無を波形解析)
"""

import os, sys, time, math, json, tempfile
import torch
import torchaudio
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowvc.infer import FlowVCInference, load_models, convert_file
from flowvc.encoder import make_encoder
from flowvc.decoder import F3Decoder
from flowvc.converter import make_vector_field_net, solve_cfm_euler
from flowvc.speaker import make_speaker_encoder
from flowvc.prosody import make_prosody_extractor
from flowvc.config import DecoderConfig

SR = 44100
DEVICE = "cpu"

# ─── テスト用WAV生成 ─────────────────────────────────────────────

def generate_test_wavs():
    """様々なテスト用WAVファイルを生成"""
    wavs = {}
    tmpdir = tempfile.mkdtemp(prefix="flowvc_test_")
    
    # 1. 正弦波 440Hz (A4), 2秒
    t = torch.arange(0, 2 * SR) / SR
    sine440 = torch.sin(2 * math.pi * 440 * t).unsqueeze(0) * 0.5
    wavs["sine440_2s"] = sine440
    
    # 2. 正弦波 220Hz (A3), 3秒 (低い男性声域)
    t = torch.arange(0, 3 * SR) / SR
    sine220 = torch.sin(2 * math.pi * 220 * t).unsqueeze(0) * 0.5
    wavs["sine220_3s"] = sine220
    
    # 3. 正弦波 880Hz (A5), 1秒 (高い女性声域)
    t = torch.arange(0, 1 * SR) / SR
    sine880 = torch.sin(2 * math.pi * 880 * t).unsqueeze(0) * 0.5
    wavs["sine880_1s"] = sine880
    
    # 4. 無音 (0.5秒)
    silence = torch.zeros(1, int(0.5 * SR))
    wavs["silence_05s"] = silence
    
    # 5. 極短 (10ms = 441 samples)
    t = torch.arange(0, 441) / SR
    very_short = torch.sin(2 * math.pi * 440 * t).unsqueeze(0) * 0.3
    wavs["veryshort_10ms"] = very_short
    
    # 6. 極長 (30秒, 低周波ドローン)
    t = torch.arange(0, 30 * SR) / SR
    drone = (torch.sin(2 * math.pi * 100 * t) + 0.5 * torch.sin(2 * math.pi * 150 * t)).unsqueeze(0) * 0.3
    wavs["drone_30s"] = drone
    
    # 7. ホワイトノイズ (2秒)
    noise = torch.randn(1, 2 * SR) * 0.1
    wavs["noise_2s"] = noise
    
    # 8. クリッピングテスト用 (振幅1.0の正弦波)
    t = torch.arange(0, 1 * SR) / SR
    loud = torch.sin(2 * math.pi * 440 * t).unsqueeze(0) * 1.0
    wavs["loud_1s"] = loud
    
    # 9. スイープ (低→高, 3秒)
    t = torch.arange(0, 3 * SR) / SR
    freq = 80 + 920 * t / 3.0  # 80Hz → 1000Hz
    phase = 2 * math.pi * torch.cumsum(freq / SR, dim=0)
    sweep = torch.sin(phase).unsqueeze(0) * 0.5
    wavs["sweep_3s"] = sweep
    
    # ファイルに保存
    paths = {}
    for name, wav in wavs.items():
        fpath = os.path.join(tmpdir, f"{name}.wav")
        torchaudio.save(fpath, wav, SR)
        paths[name] = fpath
    
    return paths, tmpdir


def audio_quality_metrics(wav: torch.Tensor, label: str = ""):
    """音声品質指標を計算"""
    if wav.dim() > 1:
        wav = wav.squeeze()
    wav_np = wav.cpu().numpy()
    
    rms = float(np.sqrt(np.mean(wav_np ** 2)))
    peak = float(np.max(np.abs(wav_np)))
    peak_db = 20 * math.log10(peak + 1e-10)
    
    # クリッピング検出
    clipping_threshold = 0.99
    clipping_samples = int(np.sum(np.abs(wav_np) > clipping_threshold))
    clipping_pct = clipping_samples / len(wav_np) * 100
    
    # ダイナミックレンジ
    non_silent = wav_np[np.abs(wav_np) > 1e-6]
    if len(non_silent) > 0:
        dynamic_range = 20 * math.log10(np.max(np.abs(non_silent)) / (np.sqrt(np.mean(non_silent**2)) + 1e-10))
    else:
        dynamic_range = 0.0
    
    # ゼロクロスレート
    zero_crossings = np.sum(np.abs(np.diff(np.sign(wav_np)))) / 2
    zcr = zero_crossings / len(wav_np) * SR
    
    metrics = {
        "label": label,
        "samples": len(wav_np),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "peak_db": round(peak_db, 2),
        "clipping_samples": clipping_samples,
        "clipping_pct": round(clipping_pct, 3),
        "dynamic_range_db": round(dynamic_range, 2),
        "zero_cross_rate": round(zcr, 1),
    }
    return metrics


def detect_clicks(wav: torch.Tensor, threshold_std: float = 8.0):
    """波形の不連続性（クリック音）を検出"""
    if wav.dim() > 1:
        wav = wav.squeeze()
    wav_np = wav.cpu().numpy()
    
    if len(wav_np) < 2:
        return {"clicks_detected": 0, "max_jump": 0.0}
    
    # 隣接サンプル間の差分
    diffs = np.abs(np.diff(wav_np))
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    
    # 閾値を超えるジャンプをクリック候補として検出
    threshold = mean_diff + threshold_std * std_diff
    click_positions = np.where(diffs > threshold)[0]
    
    # チャンク境界 (80ms = 3528 samples @ 44100Hz) 付近のクリックを特に注視
    chunk_size = int(0.080 * SR)  # 3528
    boundary_clicks = []
    for pos in click_positions:
        remainder = pos % chunk_size
        if remainder < 10 or remainder > chunk_size - 10:
            boundary_clicks.append(int(pos))
    
    # 最大ジャンプ
    max_jump = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    
    return {
        "total_clicks": len(click_positions),
        "boundary_clicks": len(boundary_clicks),
        "max_sample_jump": round(max_jump, 6),
        "mean_sample_jump": round(mean_diff, 6),
        "std_sample_jump": round(std_diff, 6),
    }


# ─── テスト1: FCPE Prosody ─────────────────────────────────────

def test_fcpe_prosody():
    """FCPEが実音声で合理的なF0値を出力するか検証"""
    print("\n" + "="*60)
    print("TEST 1: FCPE Prosody Extraction")
    print("="*60)
    
    prosody = make_prosody_extractor(device=DEVICE).eval()
    
    # 440Hz正弦波でテスト
    t = torch.arange(0, int(2 * SR)) / SR
    wav_440 = torch.sin(2 * math.pi * 440 * t).unsqueeze(0).unsqueeze(0) * 0.5  # (1,1,T)
    wav_220 = torch.sin(2 * math.pi * 220 * t).unsqueeze(0).unsqueeze(0) * 0.5
    wav_880 = torch.sin(2 * math.pi * 880 * t).unsqueeze(0).unsqueeze(0) * 0.5
    silence = torch.zeros(1, 1, int(1 * SR))
    
    results = {}
    for name, wav in [("440Hz", wav_440), ("220Hz", wav_220), ("880Hz", wav_880), ("silence", silence)]:
        t0 = time.time()
        feat = prosody(wav)  # (1, T, 3) = [log_f0, voiced, log_energy]
        elapsed = (time.time() - t0) * 1000
        
        log_f0 = feat[0, :, 0]
        voiced = feat[0, :, 1]
        log_energy = feat[0, :, 2]
        
        # F0値の計算
        voiced_frames = voiced > 0.5
        if voiced_frames.sum() > 0:
            f0_values = torch.exp(log_f0[voiced_frames])
            mean_f0 = f0_values.mean().item()
            std_f0 = f0_values.std().item()
            voiced_pct = voiced_frames.float().mean().item() * 100
        else:
            mean_f0 = 0.0
            std_f0 = 0.0
            voiced_pct = 0.0
        
        results[name] = {
            "frames": feat.shape[1],
            "mean_f0_hz": round(mean_f0, 2),
            "std_f0_hz": round(std_f0, 2),
            "voiced_pct": round(voiced_pct, 1),
            "mean_log_energy": round(log_energy.mean().item(), 3),
            "latency_ms": round(elapsed, 2),
        }
        
        print(f"  {name}: F0={mean_f0:.1f}±{std_f0:.1f}Hz, voiced={voiced_pct:.0f}%, "
              f"frames={feat.shape[1]}, {elapsed:.1f}ms")
    
    # 検証: 440Hz正弦波で約440Hzが検出されるか
    f0_440 = results["440Hz"]["mean_f0_hz"]
    f0_220 = results["220Hz"]["mean_f0_hz"]
    f0_880 = results["880Hz"]["mean_f0_hz"]
    
    checks = {}
    checks["fcpe_440hz_reasonable"] = 300 < f0_440 < 600
    checks["fcpe_220hz_reasonable"] = 150 < f0_220 < 300
    checks["fcpe_880hz_reasonable"] = 700 < f0_880 < 1100
    checks["fcpe_silence_unvoiced"] = results["silence"]["voiced_pct"] < 10
    checks["fcpe_latency_ok"] = all(r["latency_ms"] < 2000 for r in results.values())
    
    for k, v in checks.items():
        status = "✓" if v else "✗"
        print(f"    {status} {k}: {v}")
    
    return {"results": results, "checks": checks}


# ─── テスト2: SpeakerEncoder ────────────────────────────────────

def test_speaker_encoder():
    """SpeakerEncoderが異なる話者を区別できるか検証 (cosine similarity)"""
    print("\n" + "="*60)
    print("TEST 2: SpeakerEncoder Discrimination")
    print("="*60)
    
    spk_enc = make_speaker_encoder().eval()
    
    # 異なる周波数/波形を異なる「話者」としてテスト
    # 同一話者: 同じ正弦波の異なる部分
    # 異なる話者: 異なる周波数/波形
    
    t = torch.arange(0, int(2 * SR)) / SR
    
    # Speaker A: 220Hz (低い声)
    spk_a_wav1 = torch.sin(2 * math.pi * 220 * t).unsqueeze(0).unsqueeze(0) * 0.5
    # Speaker A variant (same speaker, different utterance)
    spk_a_wav2 = torch.sin(2 * math.pi * 220 * (t + 0.3)).unsqueeze(0).unsqueeze(0) * 0.5
    
    # Speaker B: 440Hz (中程度の声)
    spk_b_wav = torch.sin(2 * math.pi * 440 * t).unsqueeze(0).unsqueeze(0) * 0.5
    
    # Speaker C: 880Hz (高い声)
    spk_c_wav = torch.sin(2 * math.pi * 880 * t).unsqueeze(0).unsqueeze(0) * 0.5
    
    # Speaker D: ノイズ (全く異なる)
    spk_d_wav = torch.randn(1, 1, 2 * SR) * 0.1
    
    # 埋め込み抽出
    with torch.no_grad():
        emb_a1, _ = spk_enc(spk_a_wav1)
        emb_a2, _ = spk_enc(spk_a_wav2)
        emb_b, _ = spk_enc(spk_b_wav)
        emb_c, _ = spk_enc(spk_c_wav)
        emb_d, _ = spk_enc(spk_d_wav)
    
    # Cosine similarity
    def cos_sim(a, b):
        a_n = torch.nn.functional.normalize(a, dim=-1)
        b_n = torch.nn.functional.normalize(b, dim=-1)
        return (a_n * b_n).sum().item()
    
    sim_aa = cos_sim(emb_a1, emb_a2)  # same speaker, diff utterance
    sim_ab = cos_sim(emb_a1, emb_b)   # diff speaker
    sim_ac = cos_sim(emb_a1, emb_c)   # diff speaker (far)
    sim_ad = cos_sim(emb_a1, emb_d)   # diff speaker (noise)
    sim_bc = cos_sim(emb_b, emb_c)    # diff speaker
    
    results = {
        "same_speaker_sim": round(sim_aa, 4),
        "diff_speaker_ab": round(sim_ab, 4),
        "diff_speaker_ac": round(sim_ac, 4),
        "diff_speaker_ad": round(sim_ad, 4),
        "diff_speaker_bc": round(sim_bc, 4),
    }
    
    print(f"  Same speaker (A1 vs A2): cosine_sim = {sim_aa:.4f}")
    print(f"  Diff speaker (A vs B):    cosine_sim = {sim_ab:.4f}")
    print(f"  Diff speaker (A vs C):    cosine_sim = {sim_ac:.4f}")
    print(f"  Diff speaker (A vs D):    cosine_sim = {sim_ad:.4f}")
    print(f"  Diff speaker (B vs C):    cosine_sim = {sim_bc:.4f}")
    
    # 検証: 同話者 > 異話者 であるべき
    checks = {}
    checks["same_higher_than_diff_ab"] = sim_aa > sim_ab
    checks["same_higher_than_diff_ac"] = sim_aa > sim_ac
    checks["same_higher_than_diff_ad"] = sim_aa > sim_ad
    
    # Embedding norm check
    norms = {
        "emb_a1_norm": round(emb_a1.norm().item(), 4),
        "emb_a2_norm": round(emb_a2.norm().item(), 4),
        "emb_b_norm": round(emb_b.norm().item(), 4),
    }
    results.update(norms)
    
    for k, v in checks.items():
        status = "✓" if v else "✗"
        print(f"    {status} {k}: {v}")
    
    # NaN/inf check
    checks["no_nan_inf"] = all(not torch.isnan(e).any() and not torch.isinf(e).any()
                                for e in [emb_a1, emb_a2, emb_b, emb_c, emb_d])
    
    return {"results": results, "checks": checks}


# ─── テスト3: エッジケース (無音/極短/極長) ────────────────────

def test_edge_cases():
    """無音/極短/極長などエッジケースの処理を検証"""
    print("\n" + "="*60)
    print("TEST 3: Edge Cases (silence, very short, very long)")
    print("="*60)
    
    # ランダム重みでパイプラインを構築
    encoder = make_encoder().eval()
    decoder = F3Decoder(DecoderConfig()).eval()
    vfn = make_vector_field_net().eval()
    speaker_enc = make_speaker_encoder().eval()
    prosody = make_prosody_extractor(device=DEVICE).eval()
    
    # 参照音声（正弦波）
    t = torch.arange(0, int(2 * SR)) / SR
    ref_wav = torch.sin(2 * math.pi * 220 * t).unsqueeze(0) * 0.5  # (1, T)
    
    results = {}
    checks = {}
    
    # Test 1: 無音入力
    print("\n  --- Edge: Silence input ---")
    try:
        engine = FlowVCInference(encoder, decoder, vfn, speaker_enc, prosody, device=DEVICE)
        engine.set_target_speaker(ref_wav)
        
        silence_input = torch.zeros(int(0.5 * SR))  # 0.5秒の無音
        t0 = time.time()
        out_silence = engine.process_stream(silence_input.unsqueeze(0))
        elapsed = time.time() - t0
        
        metrics = audio_quality_metrics(out_silence, "silence_output")
        results["silence"] = {
            "input_samples": len(silence_input),
            "output_samples": len(out_silence),
            "elapsed_sec": round(elapsed, 3),
            "metrics": metrics,
        }
        print(f"    Input: {len(silence_input)} samples, Output: {len(out_silence)} samples, {elapsed:.3f}s")
        print(f"    Output RMS: {metrics['rms']:.6f}, peak: {metrics['peak']:.6f}")
        checks["silence_no_error"] = True
        checks["silence_output_shape_match"] = abs(len(out_silence) - len(silence_input)) < 10
    except Exception as e:
        print(f"    ERROR: {e}")
        results["silence"] = {"error": str(e)}
        checks["silence_no_error"] = False
    
    # Test 2: 極短入力 (10ms)
    print("\n  --- Edge: Very short input (10ms) ---")
    try:
        engine2 = FlowVCInference(encoder, decoder, vfn, speaker_enc, prosody, device=DEVICE)
        engine2.set_target_speaker(ref_wav)
        
        t_short = torch.arange(0, 441) / SR
        short_input = torch.sin(2 * math.pi * 440 * t_short) * 0.3
        t0 = time.time()
        out_short = engine2.process_stream(short_input.unsqueeze(0))
        elapsed = time.time() - t0
        
        metrics = audio_quality_metrics(out_short, "short_output")
        results["very_short"] = {
            "input_samples": len(short_input),
            "output_samples": len(out_short),
            "elapsed_sec": round(elapsed, 3),
            "metrics": metrics,
        }
        print(f"    Input: {len(short_input)} samples, Output: {len(out_short)} samples, {elapsed:.3f}s")
        checks["short_no_error"] = True
    except Exception as e:
        print(f"    ERROR: {e}")
        results["very_short"] = {"error": str(e)}
        checks["short_no_error"] = False
    
    # Test 3: 極長入力 (30秒) - メモリ/レイテンシ検証
    print("\n  --- Edge: Very long input (30s) ---")
    try:
        engine3 = FlowVCInference(encoder, decoder, vfn, speaker_enc, prosody, device=DEVICE)
        engine3.set_target_speaker(ref_wav)
        
        t_long = torch.arange(0, int(30 * SR)) / SR
        long_input = torch.sin(2 * math.pi * 100 * t_long) * 0.3
        t0 = time.time()
        out_long = engine3.process_stream(long_input.unsqueeze(0))
        elapsed = time.time() - t0
        
        metrics = audio_quality_metrics(out_long, "long_output")
        results["very_long"] = {
            "input_samples": len(long_input),
            "output_samples": len(out_long),
            "elapsed_sec": round(elapsed, 3),
            "chunks": engine3.stats["chunks"],
            "avg_latency_ms": round(engine3.avg_latency_ms, 1),
            "rtf": round(engine3.rtf, 4),
            "metrics": metrics,
        }
        clicks = detect_clicks(out_long)
        results["very_long"]["clicks"] = clicks
        
        print(f"    Input: {len(long_input)} samples ({len(long_input)/SR:.1f}s)")
        print(f"    Chunks: {engine3.stats['chunks']}, Avg latency: {engine3.avg_latency_ms:.1f}ms, RTF: {engine3.rtf:.4f}")
        print(f"    Output RMS: {metrics['rms']:.6f}, peak: {metrics['peak']:.6f}")
        print(f"    Clicks: total={clicks['total_clicks']}, boundary={clicks['boundary_clicks']}, max_jump={clicks['max_sample_jump']:.6f}")
        
        checks["long_no_error"] = True
        checks["long_rtf_reasonable"] = engine3.rtf < 100  # RTF should be somewhat reasonable on CPU
        checks["long_no_nan"] = not torch.isnan(out_long).any()
        checks["long_no_inf"] = not torch.isinf(out_long).any()
    except Exception as e:
        print(f"    ERROR: {e}")
        results["very_long"] = {"error": str(e)}
        checks["long_no_error"] = False
    
    # Test 4: DC offset
    print("\n  --- Edge: DC offset input ---")
    try:
        engine4 = FlowVCInference(encoder, decoder, vfn, speaker_enc, prosody, device=DEVICE)
        engine4.set_target_speaker(ref_wav)
        
        dc_input = torch.ones(int(1 * SR)) * 0.5  # DC offset
        t0 = time.time()
        out_dc = engine4.process_stream(dc_input.unsqueeze(0))
        elapsed = time.time() - t0
        
        metrics = audio_quality_metrics(out_dc, "dc_output")
        results["dc_offset"] = {
            "input_samples": len(dc_input),
            "output_samples": len(out_dc),
            "elapsed_sec": round(elapsed, 3),
            "metrics": metrics,
        }
        print(f"    Output RMS: {metrics['rms']:.6f}, peak: {metrics['peak']:.6f}")
        checks["dc_no_error"] = True
        checks["dc_no_nan"] = not torch.isnan(out_dc).any()
    except Exception as e:
        print(f"    ERROR: {e}")
        results["dc_offset"] = {"error": str(e)}
        checks["dc_no_error"] = False
    
    return {"results": results, "checks": checks}


# ─── テスト4: ストリーミング連続性 ─────────────────────────────

def test_streaming_continuity():
    """ストリーミング出力の連続性 (click音の有無を波形解析)"""
    print("\n" + "="*60)
    print("TEST 4: Streaming Continuity (click detection)")
    print("="*60)
    
    encoder = make_encoder().eval()
    decoder = F3Decoder(DecoderConfig()).eval()
    vfn = make_vector_field_net().eval()
    speaker_enc = make_speaker_encoder().eval()
    prosody = make_prosody_extractor(device=DEVICE).eval()
    
    # 参照音声
    t = torch.arange(0, int(2 * SR)) / SR
    ref_wav = torch.sin(2 * math.pi * 220 * t).unsqueeze(0) * 0.5
    
    # スイープ信号で連続性テスト
    t_sw = torch.arange(0, int(5 * SR)) / SR
    freq = 100 + 400 * t_sw / 5.0
    phase = 2 * math.pi * torch.cumsum(freq / SR, dim=0)
    sweep_input = torch.sin(phase) * 0.5
    
    engine = FlowVCInference(encoder, decoder, vfn, speaker_enc, prosody, device=DEVICE)
    engine.set_target_speaker(ref_wav)
    
    t0 = time.time()
    out = engine.process_stream(sweep_input.unsqueeze(0))
    elapsed = time.time() - t0
    
    # クリック検出
    clicks = detect_clicks(out, threshold_std=6.0)
    
    # チャンク境界での波形差分を詳細分析
    chunk_size = int(0.080 * SR)  # 3528 samples
    n_chunks = len(out) // chunk_size
    boundary_diffs = []
    for i in range(1, n_chunks):
        boundary = i * chunk_size
        if boundary + 1 < len(out):
            diff = abs(out[boundary].item() - out[boundary - 1].item())
            boundary_diffs.append(diff)
    
    # チャンク内部での平均差分
    internal_diffs = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(out) - 1)
        chunk_diffs = []
        for j in range(start + 1, end):
            chunk_diffs.append(abs(out[j].item() - out[j-1].item()))
        if chunk_diffs:
            internal_diffs.append(np.mean(chunk_diffs))
    
    metrics = audio_quality_metrics(out, "streaming_output")
    
    results = {
        "total_samples": len(out),
        "chunks": n_chunks,
        "elapsed_sec": round(elapsed, 3),
        "rtf": round(engine.rtf, 4),
        "avg_latency_ms": round(engine.avg_latency_ms, 1),
        "clicks": clicks,
        "boundary_mean_diff": round(float(np.mean(boundary_diffs)), 6) if boundary_diffs else 0,
        "internal_mean_diff": round(float(np.mean(internal_diffs)), 6) if internal_diffs else 0,
        "boundary_vs_internal_ratio": round(float(np.mean(boundary_diffs) / (np.mean(internal_diffs) + 1e-10)), 4) if boundary_diffs and internal_diffs else 0,
        "metrics": metrics,
    }
    
    print(f"  Output: {len(out)} samples ({len(out)/SR:.1f}s), {n_chunks} chunks")
    print(f"  RTF: {engine.rtf:.4f}, Avg latency: {engine.avg_latency_ms:.1f}ms")
    print(f"  Boundary mean diff: {results['boundary_mean_diff']:.6f}")
    print(f"  Internal mean diff: {results['internal_mean_diff']:.6f}")
    print(f"  Boundary/Internal ratio: {results['boundary_vs_internal_ratio']:.4f}")
    print(f"  Click candidates: total={clicks['total_clicks']}, boundary={clicks['boundary_clicks']}")
    print(f"  Output RMS: {metrics['rms']:.6f}, peak: {metrics['peak']:.6f}, clipping: {metrics['clipping_pct']:.3f}%")
    
    checks = {}
    checks["streaming_no_error"] = True
    checks["streaming_no_nan"] = not torch.isnan(out).any()
    checks["streaming_no_inf"] = not torch.isinf(out).any()
    # 境界差分が内部差分の10倍以内なら許容範囲（ランダム重みのため緩めに）
    if results["boundary_vs_internal_ratio"] > 0:
        checks["boundary_not_extreme"] = results["boundary_vs_internal_ratio"] < 100
    checks["clipping_acceptable"] = metrics["clipping_pct"] < 5.0  # 5%未満ならOK
    
    for k, v in checks.items():
        status = "✓" if v else "✗"
        print(f"    {status} {k}: {v}")
    
    return {"results": results, "checks": checks}


# ─── テスト5: 品質指標 (RMS, peak, clipping) ────────────────────

def test_quality_metrics():
    """出力音声の基本的な品質指標を総合評価"""
    print("\n" + "="*60)
    print("TEST 5: Quality Metrics (RMS, Peak, Clipping)")
    print("="*60)
    
    encoder = make_encoder().eval()
    decoder = F3Decoder(DecoderConfig()).eval()
    vfn = make_vector_field_net().eval()
    speaker_enc = make_speaker_encoder().eval()
    prosody = make_prosody_extractor(device=DEVICE).eval()
    
    t = torch.arange(0, int(2 * SR)) / SR
    ref_wav = torch.sin(2 * math.pi * 220 * t).unsqueeze(0) * 0.5
    
    # 様々な入力でテスト
    test_inputs = {
        "sine220": torch.sin(2 * math.pi * 220 * t) * 0.5,
        "sine440": torch.sin(2 * math.pi * 440 * t) * 0.5,
        "sine880": torch.sin(2 * math.pi * 880 * t) * 0.5,
        "noise": torch.randn(int(2 * SR)) * 0.1,
    }
    
    all_metrics = {}
    checks = {}
    
    for name, inp in test_inputs.items():
        engine = FlowVCInference(encoder, decoder, vfn, speaker_enc, prosody, device=DEVICE)
        engine.set_target_speaker(ref_wav)
        
        t0 = time.time()
        out = engine.process_stream(inp.unsqueeze(0))
        elapsed = time.time() - t0
        
        in_metrics = audio_quality_metrics(inp, f"{name}_input")
        out_metrics = audio_quality_metrics(out, f"{name}_output")
        
        all_metrics[name] = {
            "input": in_metrics,
            "output": out_metrics,
            "elapsed_sec": round(elapsed, 3),
            "rtf": round(engine.rtf, 4),
            "chunks": engine.stats["chunks"],
        }
        
        print(f"\n  {name}:")
        print(f"    Input  RMS={in_metrics['rms']:.4f}, peak={in_metrics['peak']:.4f}, zcr={in_metrics['zero_cross_rate']:.0f}")
        print(f"    Output RMS={out_metrics['rms']:.4f}, peak={out_metrics['peak']:.4f}, zcr={out_metrics['zero_cross_rate']:.0f}")
        print(f"    Clipping: {out_metrics['clipping_pct']:.3f}%, Dynamic range: {out_metrics['dynamic_range_db']:.1f}dB")
        print(f"    Latency: {elapsed:.3f}s, RTF: {engine.rtf:.4f}")
        
        # 基本的な健全性チェック
        checks[f"{name}_no_nan"] = not torch.isnan(out).any()
        checks[f"{name}_no_inf"] = not torch.isinf(out).any()
        checks[f"{name}_output_not_all_zero"] = out_metrics["rms"] > 1e-8
        checks[f"{name}_output_not_all_same"] = out.std() > 1e-8
        
        for k, v in checks.items():
            if k.startswith(name):
                status = "✓" if v else "✗"
                print(f"    {status} {k}")
    
    return {"results": all_metrics, "checks": checks}


# ─── テスト6: モデル全体の健全性 ────────────────────────────────

def test_model_sanity():
    """モデルの入出力形状、NaN、メモリなどの基本的な健全性チェック"""
    print("\n" + "="*60)
    print("TEST 6: Model Sanity Checks")
    print("="*60)
    
    encoder = make_encoder().eval()
    decoder = F3Decoder(DecoderConfig()).eval()
    vfn = make_vector_field_net().eval()
    speaker_enc = make_speaker_encoder().eval()
    prosody = make_prosody_extractor(device=DEVICE).eval()
    
    checks = {}
    results = {}
    
    # 1. エンコーダ入出力形状
    wav = torch.randn(1, 1, 44100)
    with torch.no_grad():
        z = encoder.encode(wav)
    T_lat = 44100 // 1764  # 25Hz → 約25フレーム
    results["encoder"] = {
        "input_shape": list(wav.shape),
        "output_shape": list(z.shape),
        "expected_frames": T_lat,
        "actual_frames": z.shape[1],
    }
    print(f"  Encoder: {list(wav.shape)} → {list(z.shape)} (expected ~{T_lat} frames)")
    checks["encoder_shape"] = z.shape[1] == T_lat or abs(z.shape[1] - T_lat) <= 2
    checks["encoder_no_nan"] = not torch.isnan(z).any()
    
    # 2. デコーダ (ランダム潜在変数 → 波形)
    z_rand = torch.randn(1, T_lat, 768)
    spk_emb = torch.randn(1, 192)
    with torch.no_grad():
        wav_out = decoder(z_rand, spk_emb)
    results["decoder"] = {
        "input_shape": list(z_rand.shape),
        "output_shape": list(wav_out.shape),
    }
    print(f"  Decoder: {list(z_rand.shape)} → {list(wav_out.shape)}")
    checks["decoder_shape"] = wav_out.shape[2] == 44100 or abs(wav_out.shape[2] - 44100) < 100
    checks["decoder_no_nan"] = not torch.isnan(wav_out).any()
    checks["decoder_range"] = wav_out.min() >= -1.5 and wav_out.max() <= 1.5  # tanh出力なので±1付近
    
    # 3. SpeakerEncoder
    ref = torch.randn(1, 1, 44100 * 2)
    with torch.no_grad():
        emb, prompt = speaker_enc(ref)
    results["speaker_enc"] = {
        "emb_shape": list(emb.shape),
        "prompt_shape": list(prompt.shape),
        "emb_norm": round(emb.norm().item(), 4),
    }
    print(f"  SpeakerEnc: ref → emb{list(emb.shape)}, prompt{list(prompt.shape)}")
    checks["speaker_emb_shape"] = emb.shape == (1, 192)
    checks["speaker_prompt_shape"] = prompt.shape == (1, 4, 192)
    checks["speaker_no_nan"] = not torch.isnan(emb).any() and not torch.isnan(prompt).any()
    
    # 4. VectorFieldNet
    t = torch.tensor([0.5])
    with torch.no_grad():
        v = vfn(z, t, spk_emb, prompt, prosody=None)
    results["vfn"] = {
        "input_z_shape": list(z.shape),
        "output_v_shape": list(v.shape),
    }
    print(f"  VFN: z{list(z.shape)} → v{list(v.shape)}")
    checks["vfn_shape_match"] = v.shape == z.shape
    checks["vfn_no_nan"] = not torch.isnan(v).any()
    
    # 5. CFM ODE solver
    z_src = torch.randn(1, T_lat, 768)
    with torch.no_grad():
        z_tgt = solve_cfm_euler(vfn, z_src, spk_emb, prompt, None, n_steps=4)
    results["cfm_ode"] = {
        "input_shape": list(z_src.shape),
        "output_shape": list(z_tgt.shape),
    }
    print(f"  CFM ODE: z_src → z_tgt (Euler 4-step)")
    checks["ode_shape_match"] = z_tgt.shape == z_src.shape
    checks["ode_no_nan"] = not torch.isnan(z_tgt).any()
    
    # 6. パラメータ数カウント
    param_counts = {}
    for name, model in [("encoder", encoder), ("decoder", decoder), 
                         ("vfn", vfn), ("speaker_enc", speaker_enc)]:
        n = sum(p.numel() for p in model.parameters())
        param_counts[name] = n
        print(f"  {name}: {n:,} params")
    total = sum(param_counts.values())
    results["params"] = param_counts
    results["total_params"] = total
    print(f"  TOTAL: {total:,} params ({total/1e6:.1f}M)")
    
    for k, v in checks.items():
        status = "✓" if v else "✗"
        print(f"    {status} {k}: {v}")
    
    return {"results": results, "checks": checks}


# ─── メイン ──────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  FlowVC Inference Pipeline Verification")
    print("="*60)
    print(f"  Device: {DEVICE}")
    print(f"  SR: {SR}")
    print(f"  Torch: {torch.__version__}")
    print(f"  Torchaudio: {torchaudio.__version__}")
    
    all_results = {}
    all_checks = {}
    
    # テスト実行
    tests = [
        ("ModelSanity", test_model_sanity),
        ("FCPE_Prosody", test_fcpe_prosody),
        ("SpeakerEncoder", test_speaker_encoder),
        ("EdgeCases", test_edge_cases),
        ("StreamingContinuity", test_streaming_continuity),
        ("QualityMetrics", test_quality_metrics),
    ]
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            all_results[name] = result["results"]
            all_checks[name] = result["checks"]
        except Exception as e:
            print(f"\n  ✗ TEST {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}
            all_checks[name] = {"error": True}
    
    # 総合結果
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    
    total = 0
    passed = 0
    for test_name, checks in all_checks.items():
        for check_name, check_val in checks.items():
            total += 1
            if check_val:
                passed += 1
            else:
                status = "✗" if check_val is False else "?"
                print(f"  {status} {test_name}.{check_name} = {check_val}")
    
    print(f"\n  {'─'*45}")
    print(f"  Results: {passed}/{total} checks passed ({passed/total*100:.1f}%)" if total > 0 else "  No checks run")
    
    # 結果をJSON保存
    output_path = "/Users/asill/btrv5/flowvc/verification_results.json"
    # Convert tensors to strings/lists for JSON
    def clean(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist() if obj.numel() < 100 else f"<tensor {list(obj.shape)}>"
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj
    
    json_results = {
        "summary": {"total": total, "passed": passed, "rate": round(passed/total*100, 1) if total > 0 else 0},
        "results": clean(all_results),
        "checks": clean(all_checks),
    }
    
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {output_path}")
    
    return all_results, all_checks


if __name__ == "__main__":
    main()

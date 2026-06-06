"""
FlowVC 学習パイプライン。

3フェーズ学習:
  フェーズ0: AE事前学習 — エンコーダ+デコーダの再構成のみ
  フェーズ1: CFM学習 — フローマッチング変換器の学習
  フェーズ2: E2E+GAN — 全体end-to-end微調整 + 敵対的損失

使用方法:
    python -m flowvc.train --phase 0 --data-dir /path/to/data \\
        --output-dir runs/phase0 --device cuda --batch-size 8

    python -m flowvc.train --phase 1 --resume runs/phase0/ae_final.pt \\
        --output-dir runs/phase1 --device cuda --batch-size 8

    python -m flowvc.train --phase 2 --resume runs/phase1/cfm_step100000.pt \\
        --output-dir runs/phase2 --device mps --batch-size 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EncoderConfig, DecoderConfig, FlowConverterConfig, TrainConfig
from .encoder import F3Encoder, make_encoder
from .decoder import F3Decoder
from .converter import VectorFieldNet, make_vector_field_net
from .speaker import SpeakerEncoder, make_speaker_encoder
from .prosody import ProsodyExtractor, make_prosody_extractor
from .cfm_loss import CFMLoss


# ── フェーズ0: AE事前学習 ────────────────────────────────────────

def train_phase0(args):
    """
    エンコーダ+デコーダの自己再構成学習。
    KLフリー、ノイズ正則化、L1波形損失 + マルチ解像度STFT損失。
    """
    device = torch.device(args.device)

    encoder = make_encoder().to(device)
    decoder = F3Decoder(DecoderConfig()).to(device)
    prosody = make_prosody_extractor().to(device)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()) + list(prosody.parameters()),
        lr=args.lr, betas=(0.8, 0.9), weight_decay=0.01,
    )

    print(f"[Phase 0] AE事前学習")
    print(f"  デバイス: {device}, バッチサイズ: {args.batch_size}")
    print(f"  エンコーダ: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"  デコーダ:   {sum(p.numel() for p in decoder.parameters()):,} params")

    # ダミーデータでスモークテスト
    B, T_audio = args.batch_size, 44100  # 1秒
    wav = torch.randn(B, 1, T_audio, device=device)

    z = encoder(wav, training=True)
    recon = decoder(z, torch.randn(B, 192, device=device))

    loss = F.l1_loss(recon, wav)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"  再構成L1損失: {loss.item():.4f}")
    print(f"  勾配バックワード: OK")

    # TODO: 実際のデータローダ + 完全な学習ループ
    print("  [TODO] データローダと本格的な学習ループの実装が必要")

    return encoder, decoder, prosody


# ── フェーズ1: CFM学習 ──────────────────────────────────────────

def train_phase1(args, encoder, prosody, speaker_enc):
    """
    フローマッチング変換器の学習。
    OTパス + CFM損失。エンコーダ/プロソディ/話者エンコーダは凍結。
    """
    device = torch.device(args.device)

    vfn = make_vector_field_net().to(device)
    cfm_loss = CFMLoss(sigma_min=0.001)

    opt = torch.optim.AdamW(vfn.parameters(), lr=args.lr, betas=(0.8, 0.9))

    print(f"\n[Phase 1] CFM学習")
    print(f"  デバイス: {device}, バッチサイズ: {args.batch_size}")
    print(f"  VFN: {sum(p.numel() for p in vfn.parameters()):,} params")

    # 凍結
    encoder.eval()
    prosody.eval()
    speaker_enc.eval()
    for m in [encoder, prosody, speaker_enc]:
        for p in m.parameters():
            p.requires_grad = False

    # スモークテスト
    B, T_audio = args.batch_size, 44100
    wav_src = torch.randn(B, 1, T_audio, device=device)
    wav_tgt = torch.randn(B, 1, T_audio, device=device)
    wav_ref = torch.randn(B, 1, T_audio, device=device)

    with torch.no_grad():
        z_src = encoder.encode(wav_src)
        z_tgt = encoder.encode(wav_tgt)
        spk_emb, prompt = speaker_enc(wav_ref)
        pros = prosody(wav_src)

    loss, logs = cfm_loss(vfn, z_src, z_tgt, spk_emb, prompt, pros)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"  CFM損失: {loss.item():.4f}")
    print(f"  勾配バックワード: OK")

    return vfn


# ── フェーズ2: E2E微調整 ────────────────────────────────────────

def train_phase2(args, encoder, decoder, vfn, speaker_enc, prosody):
    """
    End-to-end微調整 + 敵対的GAN損失。
    全コンポーネントを低学習率で共同最適化。
    """
    device = torch.device(args.device)

    params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(vfn.parameters())
    )
    opt = torch.optim.AdamW(params, lr=args.lr * 0.1, betas=(0.8, 0.9))

    print(f"\n[Phase 2] E2E+GAN微調整")
    print(f"  デバイス: {device}, バッチサイズ: {args.batch_size}")
    print(f"  LR: {args.lr * 0.1}")

    # スモークテスト
    B, T_audio = args.batch_size, 44100
    wav_src = torch.randn(B, 1, T_audio, device=device)
    wav_ref = torch.randn(B, 1, T_audio, device=device)

    z_src = encoder.encode(wav_src)
    spk_emb, prompt = speaker_enc(wav_ref)
    pros = prosody(wav_src)

    from .converter import solve_cfm_euler
    z_tgt = solve_cfm_euler(vfn, z_src, spk_emb, prompt, pros, n_steps=4)
    out = decoder(z_tgt, spk_emb)

    loss = F.l1_loss(out, wav_src)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"  E2E損失: {loss.item():.4f}")
    print(f"  勾配バックワード: OK")


# ── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FlowVC 学習")
    parser.add_argument("--phase", type=int, default=0,
                        help="学習フェーズ: 0=AE, 1=CFM, 2=E2E")
    parser.add_argument("--data-dir", type=str, default="",
                        help="データディレクトリ")
    parser.add_argument("--output-dir", type=str, default="./runs",
                        help="出力ディレクトリ")
    parser.add_argument("--device", type=str, default="cpu",
                        help="デバイス: cpu, cuda, mps")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--resume", type=str, default="",
                        help="再開用チェックポイント")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.phase == 0:
        train_phase0(args)

    elif args.phase == 1:
        # 簡易: 新しいエンコーダを作成（実際は--resumeからロード）
        encoder = make_encoder().to(args.device)
        prosody = make_prosody_extractor().to(args.device)
        speaker_enc = make_speaker_encoder().to(args.device)
        train_phase1(args, encoder, prosody, speaker_enc)

    elif args.phase == 2:
        encoder = make_encoder().to(args.device)
        decoder = F3Decoder(DecoderConfig()).to(args.device)
        vfn = make_vector_field_net().to(args.device)
        speaker_enc = make_speaker_encoder().to(args.device)
        prosody = make_prosody_extractor().to(args.device)
        train_phase2(args, encoder, decoder, vfn, speaker_enc, prosody)

    else:
        print(f"未知のフェーズ: {args.phase}")


if __name__ == "__main__":
    main()

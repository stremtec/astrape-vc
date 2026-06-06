"""
FlowVC データセットローダ。

VCTK / LibriTTS 対応。話者ペア生成 + 前処理 + キャッシュ。

使用方法:
    from flowvc.dataset import VCTKDataset, create_dataloader

    ds = VCTKDataset("/path/to/vctk/wav48_silence_trimmed")
    loader = create_dataloader(ds, batch_size=8)
"""

from __future__ import annotations

import os
import glob
import random
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from collections import defaultdict


SAMPLE_RATE = 44100


class VCTKDataset(torch.utils.data.Dataset):
    """
    VCTK データセット。

    話者ペア生成:
      - 同一話者ペア (identity, 50%)
      - 異話者ペア (cross-speaker, 50%)

    出力:
      src_wav: ソース音声 (crop_seconds)
      tgt_wav: ターゲット音声 (crop_seconds, 同一発話)
      ref_wav: 参照音声 (ターゲット話者の別発話, ref_seconds)
    """

    def __init__(
        self,
        data_dir: str,
        crop_seconds: float = 2.0,
        ref_seconds: float = 2.0,
        sample_rate: int = SAMPLE_RATE,
        sr_orig: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.crop_samples = int(crop_seconds * sample_rate)
        self.ref_samples = int(ref_seconds * sample_rate)
        self.sample_rate = sample_rate

        # 音声ファイル収集
        self.files = sorted(glob.glob(str(self.data_dir / "**" / "*.wav"), recursive=True))
        if not self.files:
            self.files = sorted(glob.glob(str(self.data_dir / "**" / "*.flac"), recursive=True))

        if not self.files:
            raise FileNotFoundError(f"音声ファイルが見つかりません: {data_dir}")

        # 話者ごとにグループ化
        self.speaker_files = defaultdict(list)
        for f in self.files:
            # VCTK形式: p225/p225_001.wav → speaker_id = "p225"
            spk = Path(f).parent.name
            self.speaker_files[spk].append(f)

        # 有効な話者（2発話以上）
        self.speakers = [s for s, files in self.speaker_files.items() if len(files) >= 2]

        if len(self.speakers) < 2:
            raise ValueError(f"話者が不足しています ({len(self.speakers)}人)。最低2人必要です。")

        # 元のサンプルレート検出
        if sr_orig is None:
            try:
                info = torchaudio.info(self.files[0])
                self.sr_orig = info.sample_rate
            except Exception:
                self.sr_orig = sample_rate
        else:
            self.sr_orig = sr_orig

        print(f"  VCTK: {len(self.files)}ファイル, {len(self.speakers)}話者")

    def __len__(self):
        return len(self.files)

    def _load_audio(self, path: str, target_samples: int) -> torch.Tensor:
        """音声読み込み + リサンプリング + トリミング/繰り返し。"""
        wav, sr = torchaudio.load(path)

        # リサンプリング
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # モノラル化
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)  # (T,)

        # Truncate or zero-pad (no repeat to avoid phase discontinuity)
        if wav.shape[-1] > target_samples:
            wav = wav[:target_samples]
        elif wav.shape[-1] < target_samples:
            wav = F.pad(wav, (0, target_samples - wav.shape[-1]))

        return wav

    def __getitem__(self, idx: int):
        src_path = self.files[idx]
        src_spk = Path(src_path).parent.name

        # 50% 同一話者 / 50% 異話者
        if random.random() < 0.5:
            # 同一話者 (identity)
            tgt_path = random.choice(
                [f for f in self.speaker_files[src_spk] if f != src_path]
            )
            tgt_spk = src_spk
        else:
            # 異話者 (cross-speaker)
            tgt_spk = random.choice([s for s in self.speakers if s != src_spk])
            tgt_path = random.choice(self.speaker_files[tgt_spk])

        # ref audio: different from both src and tgt, and different utterance ID
        src_utt = Path(src_path).stem.replace("_mic1", "").replace("_mic2", "")
        tgt_utt = Path(tgt_path).stem.replace("_mic1", "").replace("_mic2", "")
        ref_opts = [f for f in self.speaker_files[tgt_spk]
                    if f != tgt_path and f != src_path
                    and Path(f).stem.replace("_mic1", "").replace("_mic2", "") not in (src_utt, tgt_utt)]
        if not ref_opts:
            ref_opts = [f for f in self.speaker_files[tgt_spk] if f != src_path]
        if not ref_opts:
            ref_opts = [tgt_path]
        ref_path = random.choice(ref_opts)

        # 読み込み
        src_wav = self._load_audio(src_path, self.crop_samples)
        tgt_wav = self._load_audio(tgt_path, self.crop_samples)
        ref_wav = self._load_audio(ref_path, self.ref_samples)

        return {
            "src_wav": src_wav.unsqueeze(0),  # (1, T)
            "tgt_wav": tgt_wav.unsqueeze(0),
            "ref_wav": ref_wav.unsqueeze(0),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """バッチ統合。"""
    return {
        "src_wav": torch.stack([b["src_wav"] for b in batch]),
        "tgt_wav": torch.stack([b["tgt_wav"] for b in batch]),
        "ref_wav": torch.stack([b["ref_wav"] for b in batch]),
    }


def create_dataloader(
    dataset: VCTKDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )


# ── キャッシュデータセット（事前エンコード済み）──────────────────

class CachedDataset(torch.utils.data.Dataset):
    """
    事前エンコード済み潜在キャッシュから読み込むデータセット。

    キャッシュ構築:
        python -m flowvc.dataset --cache --data-dir /vctk --cache-dir ./cache
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(glob.glob(str(self.cache_dir / "*.pt")))

        if not self.files:
            raise FileNotFoundError(f"キャッシュファイルが見つかりません: {cache_dir}")

        print(f"  CachedDataset: {len(self.files)} サンプル")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        return torch.load(self.files[idx], weights_only=True)


# ── CLI: キャッシュ構築 ─────────────────────────────────────────

def build_cache(data_dir: str, cache_dir: str, device: str = "cpu"):
    """
    データセット全体をエンコードして潜在キャッシュを構築。
    """
    from .encoder import make_encoder
    from .speaker import make_speaker_encoder
    from .prosody import make_prosody_extractor

    os.makedirs(cache_dir, exist_ok=True)
    dev = torch.device(device)

    random.seed(42)  # deterministic cache building

    encoder = make_encoder().to(dev).eval()
    speaker_enc = make_speaker_encoder().to(dev).eval()
    prosody_ext = make_prosody_extractor(device=str(dev)).to(dev).eval()

    ds = VCTKDataset(data_dir)

    for i in range(len(ds)):
        item = ds[i]
        src = item["src_wav"].unsqueeze(0).to(dev)
        tgt = item["tgt_wav"].unsqueeze(0).to(dev)
        ref = item["ref_wav"].unsqueeze(0).to(dev)

        with torch.no_grad():
            z_src = encoder.encode(src)
            z_tgt = encoder.encode(tgt)
            spk_emb, _ = speaker_enc(ref)
            prosody = prosody_ext(src)

        cache = {
            "z_src": z_src.cpu(),
            "z_tgt": z_tgt.cpu(),
            "spk_emb": spk_emb.cpu(),
            "prosody": prosody.cpu(),
        }
        torch.save(cache, os.path.join(cache_dir, f"{i:06d}.pt"))

        if (i + 1) % 100 == 0:
            print(f"  キャッシュ: {i+1}/{len(ds)}")

    print(f"  キャッシュ完了: {cache_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FlowVC データセット")
    parser.add_argument("--cache", action="store_true",
                        help="潜在キャッシュを構築")
    parser.add_argument("--data-dir", type=str, default="",
                        help="データディレクトリ (VCTK)")
    parser.add_argument("--cache-dir", type=str, default="./cache",
                        help="キャッシュ出力先")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.cache:
        if not args.data_dir:
            print("ERROR: --data-dir が必要です")
            return
        build_cache(args.data_dir, args.cache_dir, args.device)
    else:
        # データセット読み込みテスト
        ds = VCTKDataset(args.data_dir)
        item = ds[0]
        print(f"  src_wav: {item['src_wav'].shape}")
        print(f"  tgt_wav: {item['tgt_wav'].shape}")
        print(f"  ref_wav: {item['ref_wav'].shape}")


if __name__ == "__main__":
    main()

"""Decoder-side dataset + audio helpers (Phase-0 / v5 decoder training).

Moved out of the old `train_decoder.py` so the training script stays a thin CLI.
`Phase0Dataset` loads the WavLM cache (rate-aware), the original audio, and a
per-speaker centroid embedding; `gaussian_blur_wave` is the cdecoder teacher
smoothing.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

import math
import time
from dataclasses import dataclass, field

S = 44100  # decoder / audio sample rate


def _io_retry(fn, tries: int = 10, delay: float = 0.5):
    """Retry a flaky read. The WavLM/content caches AND the VCTK audio live on an external
    USB drive that occasionally drops an I/O for <1 s — surfacing as FileNotFoundError /
    OSError (np.load) OR soundfile.LibsndfileError (sf.read), which is NOT an OSError. We
    catch any read exception and retry; one blip would otherwise kill the whole run.
    Genuine errors still surface (re-raised after `tries`)."""
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(delay)


def gaussian_blur_wave(wave: torch.Tensor, sigma_ms: float = 2.0) -> torch.Tensor:
    """Causal (left-padded) time-domain Gaussian blur of the teacher waveform."""
    if sigma_ms <= 0:
        return wave
    sigma_samples = int(sigma_ms / 1000 * S)
    if sigma_samples < 1:
        return wave
    radius = min(4 * sigma_samples, 512)
    kernel_size = 2 * radius + 1
    t = torch.arange(-radius, radius + 1, dtype=torch.float32, device=wave.device)
    kernel = torch.exp(-0.5 * (t / sigma_samples) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    wave_3d = wave.unsqueeze(1)  # (B, 1, T)
    padded = F.pad(wave_3d, (kernel_size - 1, 0), mode='reflect')  # left-only → causal
    blurred = F.conv1d(padded, kernel.expand(1, 1, -1))
    return blurred.squeeze(1)


class Phase0Dataset(Dataset):
    """Loads WavLM CNN features (rate-aware), original audio, and speaker embedding."""

    def __init__(self, indices, wavlm_dir, source_files, spk_embeds,
                 spk_names, max_content_frames=50, seed=42, wavlm_rate=50,
                 speaker_emb_map=None, content_dir=None, load_audio=True):
        self.indices = [int(i) for i in indices]
        # load_audio=False: skip the original-audio read (only its LENGTH is needed for
        # teacher-distillation targets) → training depends ONLY on the local content cache,
        # surviving the flaky external audio drive disconnecting mid-run.
        self.load_audio = load_audio
        self.wavlm_dir = Path(wavlm_dir)
        # If set, load pre-cached FULL-context content (Tc,768) and crop content windows
        # — the decoder then sees the same context as streaming inference, and the frozen
        # encoder is not re-run every step.  Else load WavLM (encoded in the train loop).
        self.content_dir = Path(content_dir) if content_dir else None
        self.source_files = source_files
        self.spk_names = spk_names       # (N,) array of speaker IDs (e.g., 'p315')
        self.spk_embeds = spk_embeds     # (N_spk, 128) float32 tensor (legacy path)
        self.max_cf = max_content_frames
        # WavLM frames per 25Hz content frame: 50Hz cache=2, 200Hz L4 cache=8.
        self.R = max(1, wavlm_rate // 25)
        self.max_wavlm = max_content_frames * self.R
        self.max_samples = max_content_frames * 1764
        self.rng = random.Random(seed)

        # Preferred: per-speaker centroid map {speaker_id: (128,) tensor}
        # (from `astrape.cache --what speakers` — covers ALL speakers).
        self.speaker_emb_map = speaker_emb_map
        if speaker_emb_map is not None:
            self._spk_fallback = torch.stack(list(speaker_emb_map.values())).mean(0)
            print(f"  Speaker mapping: {len(speaker_emb_map)} per-speaker centroids")
        else:
            # Legacy: speaker ID → first-occurrence index in spk_embeds.
            n_emb = len(self.spk_embeds)
            self.spk_to_emb = {}
            for i in range(min(n_emb, len(spk_names))):
                spk_id = str(spk_names[i])
                if spk_id not in self.spk_to_emb:
                    self.spk_to_emb[spk_id] = i
            print(f"  Speaker mapping: {len(self.spk_to_emb)} unique speakers → "
                  f"{n_emb} embeddings")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        import soundfile as sf
        idx = self.indices[i]

        # ── features: pre-cached full-context content, or raw WavLM ──
        # Crop content-aligned: 1 content frame = R wavlm frames = 1764 audio samples.
        if self.content_dir is not None:
            feat = torch.from_numpy(_io_retry(
                lambda: np.load(self.content_dir / f"s_{idx:05d}.npy", allow_pickle=False).astype(np.float32)))  # (Tc,768)
            avail_cf = feat.shape[0]
            if avail_cf < self.max_cf:
                feat = F.pad(feat, (0, 0, 0, self.max_cf - feat.shape[0])); cf_start = 0
            else:
                cf_start = self.rng.randint(0, avail_cf - self.max_cf)
                feat = feat[cf_start:cf_start + self.max_cf]
        else:
            feat = torch.from_numpy(_io_retry(
                lambda: np.load(self.wavlm_dir / f"s_{idx:05d}.npy", allow_pickle=False).astype(np.float32)))  # (T,512)
            avail_cf = feat.shape[0] // self.R
            if avail_cf < self.max_cf:
                feat = F.pad(feat, (0, 0, 0, self.max_wavlm - feat.shape[0])); cf_start = 0
            else:
                cf_start = self.rng.randint(0, avail_cf - self.max_cf)
                feat = feat[cf_start * self.R: cf_start * self.R + self.max_wavlm]

        # ── Original audio (skippable — only the length matters for distillation) ──
        if not self.load_audio:
            wave = torch.zeros(self.max_samples)
            spk_id = str(self.spk_names[idx])
            if self.speaker_emb_map is not None:
                spk = self.speaker_emb_map.get(spk_id, self._spk_fallback).clone()
            else:
                spk = self.spk_embeds[self.spk_to_emb.get(spk_id, 0)].clone()
            return {"wavlm": feat, "audio": wave, "speaker": spk, "idx": idx}
        src_path = str(self.source_files[idx])
        wave, sr = _io_retry(lambda: sf.read(src_path, dtype="float32"))
        wave = torch.from_numpy(np.asarray(wave))
        if wave.ndim == 2:
            wave = wave.mean(1)
        if sr != S:
            wave = torchaudio.functional.resample(wave.unsqueeze(0), sr, S).squeeze(0)

        # Crop audio to the SAME content window (1764 samples per content frame)
        audio_start = cf_start * 1764
        audio_end = audio_start + self.max_samples
        if audio_end > wave.shape[0]:
            wave = F.pad(wave, (0, audio_end - wave.shape[0]))
            wave = wave[audio_start:audio_end]
        elif audio_start + self.max_samples > wave.shape[0]:
            wave = wave[audio_start:]
            wave = F.pad(wave, (0, self.max_samples - wave.shape[0]))
        else:
            wave = wave[audio_start:audio_end]

        # ── Speaker embedding (match actual speaker) ──
        spk_id = str(self.spk_names[idx])
        if self.speaker_emb_map is not None:
            spk = self.speaker_emb_map.get(spk_id, self._spk_fallback).clone()
        else:
            spk_idx = self.spk_to_emb.get(spk_id, 0)  # fallback to first
            spk = self.spk_embeds[spk_idx].clone()

        return {"wavlm": feat, "audio": wave, "speaker": spk, "idx": idx}


def collate_phase0(batch):
    """Stack batch items → (wavlm, audio, speaker, indices)."""
    wavlm = torch.stack([b["wavlm"] for b in batch])      # (B, T_wl, 512)
    audio = torch.stack([b["audio"] for b in batch])      # (B, samples)
    speaker = torch.stack([b["speaker"] for b in batch])  # (B, 128)
    indices = [b["idx"] for b in batch]
    return wavlm, audio, speaker, indices


# ════════════════════════════════════════════════════════════════════
# Encoder-side dataset (compact VCTK cache → Batch), moved from mcs_common.py
# ════════════════════════════════════════════════════════════════════

DEFAULT_DATA_DIR = Path("data/mio_vctk_full_compact")
DEFAULT_PROJECTION = Path("checkpoints/teacher_fsq_proj_out.pt")


@dataclass
class Batch:
    mel: torch.Tensor            # (B, 80, M)
    content: torch.Tensor        # (B, L, 768)  teacher target
    tokens: torch.Tensor         # (B, L)        integer codes
    mask: torch.Tensor           # (B, L)        bool
    speakers: list[str]
    indices: torch.Tensor
    crop_starts: torch.Tensor
    ssl_L0: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # (B, Lx2, 768)
    ssl_L4: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    ssl_L8: torch.Tensor = field(default_factory=lambda: torch.empty(0))


class MioCompactDataset(Dataset):
    def __init__(self, root: Path, indices: np.ndarray, speakers: np.ndarray):
        self.root = root
        self.indices = [int(i) for i in indices.tolist()]
        self.speakers = speakers

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        idx = self.indices[item]
        with np.load(self.root / f"s_{idx:05d}.npz", allow_pickle=False) as data:
            ssl0 = data.get("ssl_L0")
            ssl4 = data.get("ssl_L4")
            ssl8 = data.get("ssl_L8")
            return {
                "idx": idx,
                "speaker": str(self.speakers[idx]),
                "mel": torch.from_numpy(data["logmel"].astype(np.float32)),
                "content": torch.from_numpy(data["ce_768"].astype(np.float32)),
                "tokens": torch.from_numpy(data["ct"].astype(np.int64)),
                "ssl_L0": torch.from_numpy(ssl0.astype(np.float32)) if ssl0 is not None else torch.empty(0,768),
                "ssl_L4": torch.from_numpy(ssl4.astype(np.float32)) if ssl4 is not None else torch.empty(0,768),
                "ssl_L8": torch.from_numpy(ssl8.astype(np.float32)) if ssl8 is not None else torch.empty(0,768),
            }


class ContentCollator:
    def __init__(self, mel_frames: int | None, seed: int, pad_mel_multiple: int = 2,
                 frames_per_token: int = 2):
        self.mel_frames = mel_frames
        self.rng = random.Random(seed)
        self.pad_mel_multiple = pad_mel_multiple
        # Frontend ("mel") frames per teacher token (25Hz content).  Mel and the
        # 50Hz WavLM cache are 2:1; the 200Hz L4 raw cache is 8:1.  Crops must use
        # this ratio to keep the cropped frontend window aligned with the cropped
        # content/token window (a hard-coded 2 overruns and mis-pairs at 200Hz).
        self.frames_per_token = frames_per_token

    def _crop(self, sample: dict) -> tuple:
        mel = sample["mel"]
        content = sample["content"]
        tokens = sample["tokens"]
        ssl0 = sample.get("ssl_L0", torch.empty(0,768))
        ssl4 = sample.get("ssl_L4", torch.empty(0,768))
        ssl8 = sample.get("ssl_L8", torch.empty(0,768))
        idx = int(sample["idx"])
        if self.mel_frames is None or mel.shape[1] <= self.mel_frames:
            return mel, content, tokens, ssl0, ssl4, ssl8, 0, idx

        R = self.frames_per_token
        max_start = mel.shape[1] - self.mel_frames
        start = self.rng.randint(0, max_start)
        start -= start % R                       # align crop to a token boundary
        mel = mel[:, start : start + self.mel_frames]
        token_start = start // R
        token_len = math.ceil(mel.shape[1] / R)
        ssl_start = token_start * 2              # SSL features are 50Hz = 2× token rate
        ssl_len = token_len * 2
        return (
            mel,
            content[token_start : token_start + token_len],
            tokens[token_start : token_start + token_len],
            ssl0[ssl_start : ssl_start + ssl_len] if ssl0.numel()>0 else ssl0,
            ssl4[ssl_start : ssl_start + ssl_len] if ssl4.numel()>0 else ssl4,
            ssl8[ssl_start : ssl_start + ssl_len] if ssl8.numel()>0 else ssl8,
            start,
            idx,
        )

    def __call__(self, samples: list[dict]) -> Batch:
        cropped = [self._crop(sample) for sample in samples]
        max_mel = max(mel.shape[1] for mel, _, _, _, _, _, _, _ in cropped)
        if self.pad_mel_multiple > 1:
            max_mel = ((max_mel + self.pad_mel_multiple - 1) // self.pad_mel_multiple) * self.pad_mel_multiple
        max_tokens = max(tokens.shape[0] for _, _, tokens, _, _, _, _, _ in cropped)
        max_ssl = max_tokens * 2

        mels, contents, tokens_out, masks = [], [], [], []
        ssl0s, ssl4s, ssl8s = [], [], []
        crop_starts, indices = [], []
        for mel, content, tokens, ssl0, ssl4, ssl8, crop_start, idx in cropped:
            token_len = min(tokens.shape[0], content.shape[0])
            mels.append(F.pad(mel, (0, max_mel - mel.shape[1])))
            contents.append(F.pad(content[:token_len], (0, 0, 0, max_tokens - token_len)))
            tokens_out.append(F.pad(tokens[:token_len], (0, max_tokens - token_len)))
            mask = torch.zeros(max_tokens, dtype=torch.bool)
            mask[:token_len] = True
            masks.append(mask)
            # SSL features: pad to max_ssl
            if ssl0.numel() > 0:
                sl = min(ssl0.shape[0], max_ssl)
                ssl0s.append(F.pad(ssl0[:sl], (0,0,0,max_ssl-sl)))
                ssl4s.append(F.pad(ssl4[:sl], (0,0,0,max_ssl-sl)))
                ssl8s.append(F.pad(ssl8[:sl], (0,0,0,max_ssl-sl)))
            else:
                ssl0s.append(torch.zeros(max_ssl, 768))
                ssl4s.append(torch.zeros(max_ssl, 768))
                ssl8s.append(torch.zeros(max_ssl, 768))
            crop_starts.append(crop_start)
            indices.append(idx)

        return Batch(
            mel=torch.stack(mels),
            content=torch.stack(contents),
            tokens=torch.stack(tokens_out),
            mask=torch.stack(masks),
            ssl_L0=torch.stack(ssl0s),
            ssl_L4=torch.stack(ssl4s),
            ssl_L8=torch.stack(ssl8s),
            speakers=[sample["speaker"] for sample in samples],
            indices=torch.tensor(indices, dtype=torch.long),
            crop_starts=torch.tensor(crop_starts, dtype=torch.long),
        )


def split_by_speaker(speakers: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.array(sorted(set(speakers.astype(str).tolist())), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    val_speakers = set(str(s) for s in unique[:n_val].tolist())
    train_idx, val_idx = [], []
    for idx, speaker in enumerate(speakers.astype(str).tolist()):
        (val_idx if speaker in val_speakers else train_idx).append(idx)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def speaker_balanced_subset(indices: np.ndarray, speakers: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    by_speaker: dict[str, list[int]] = {}
    for idx in indices.tolist():
        by_speaker.setdefault(str(speakers[idx]), []).append(int(idx))
    for values in by_speaker.values():
        rng.shuffle(values)
    selected: list[int] = []
    names = sorted(by_speaker)
    cursor = 0
    while len(selected) < min(n, len(indices)) and names:
        name = names[cursor % len(names)]
        if by_speaker[name]:
            selected.append(by_speaker[name].pop())
        else:
            names.remove(name)
            cursor -= 1
        cursor += 1
    return np.asarray(selected, dtype=np.int64)


def move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        mel=batch.mel.to(device),
        content=batch.content.to(device),
        tokens=batch.tokens.to(device),
        mask=batch.mask.to(device),
        speakers=batch.speakers,
        indices=batch.indices.to(device),
        crop_starts=batch.crop_starts.to(device),
        ssl_L0=batch.ssl_L0.to(device),
        ssl_L4=batch.ssl_L4.to(device),
        ssl_L8=batch.ssl_L8.to(device),
    )

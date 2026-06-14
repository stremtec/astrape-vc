from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .audio import StreamingLogMel
from .text import encode_transcript


@dataclass(frozen=True)
class OriginalRecord:
    audio_path: Path
    transcript_path: Path
    speaker: str


@dataclass
class OriginalSample:
    mel: torch.Tensor
    transcript: torch.Tensor


@dataclass
class OriginalBatch:
    mel: torch.Tensor
    input_lengths: torch.Tensor
    transcripts: torch.Tensor
    transcript_lengths: torch.Tensor


def minimum_ctc_frames(transcript: torch.Tensor) -> int:
    if transcript.numel() == 0:
        return 0
    repeated = (transcript[1:] == transcript[:-1]).sum().item()
    return int(transcript.numel() + repeated)


def scan_vctk(
    audio_root: str | Path,
    transcript_root: str | Path,
    allowed_speakers: Optional[Sequence[str]] = None,
) -> list[OriginalRecord]:
    audio_root = Path(audio_root)
    transcript_root = Path(transcript_root)
    allowed = set(map(str, allowed_speakers)) if allowed_speakers is not None else None
    records = []
    for speaker_dir in sorted(path for path in audio_root.iterdir() if path.is_dir()):
        speaker = speaker_dir.name
        if not speaker.startswith("p") or (allowed is not None and speaker not in allowed):
            continue
        for audio_path in sorted(speaker_dir.glob(f"{speaker}_*_mic1.flac")):
            utterance = audio_path.name.removesuffix("_mic1.flac")
            transcript_path = transcript_root / speaker / f"{utterance}.txt"
            if transcript_path.exists():
                records.append(OriginalRecord(audio_path, transcript_path, speaker))
    return records


class OriginalVCTKDataset(Dataset[OriginalSample]):
    def __init__(self, records: Sequence[OriginalRecord]):
        self.records = list(records)
        self.mel_extractor = StreamingLogMel()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> OriginalSample:
        record = self.records[index]
        audio, sample_rate = sf.read(record.audio_path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            divisor = math.gcd(sample_rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor)
        mel = self.mel_extractor(torch.from_numpy(np.asarray(audio)).float()).squeeze(0)
        transcript = encode_transcript(record.transcript_path.read_text())
        return OriginalSample(mel=mel, transcript=transcript)


class OriginalCollator:
    def __call__(self, samples: list[OriginalSample]) -> OriginalBatch:
        valid = [
            sample
            for sample in samples
            if 0 < len(sample.transcript)
            and minimum_ctc_frames(sample.transcript) <= sample.mel.shape[1]
        ]
        if not valid:
            raise ValueError("Batch contains no CTC-compatible original samples")
        input_lengths = torch.tensor(
            [sample.mel.shape[1] for sample in valid], dtype=torch.long
        )
        transcript_lengths = torch.tensor(
            [len(sample.transcript) for sample in valid], dtype=torch.long
        )
        max_length = int(input_lengths.max())
        mel = torch.stack(
            [
                F.pad(sample.mel, (0, max_length - sample.mel.shape[1]))
                for sample in valid
            ]
        )
        transcripts = torch.cat([sample.transcript for sample in valid])
        return OriginalBatch(
            mel=mel,
            input_lengths=input_lengths,
            transcripts=transcripts,
            transcript_lengths=transcript_lengths,
        )

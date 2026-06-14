from __future__ import annotations

import re

import torch


SYMBOLS = "abcdefghijklmnopqrstuvwxyz' "
BLANK_INDEX = 0
CHAR_TO_INDEX = {character: index + 1 for index, character in enumerate(SYMBOLS)}
VOCAB_SIZE = len(SYMBOLS) + 1


def normalize_transcript(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z' ]+", "", text)
    return " ".join(text.split())


def encode_transcript(text: str) -> torch.Tensor:
    normalized = normalize_transcript(text)
    return torch.tensor(
        [CHAR_TO_INDEX[character] for character in normalized],
        dtype=torch.long,
    )

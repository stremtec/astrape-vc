from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


DEFAULT_LEVELS = (8, 8, 8, 5, 5)


def indices_to_level_indices(
    indices: torch.Tensor, levels: Sequence[int] = DEFAULT_LEVELS
) -> torch.Tensor:
    basis = []
    product = 1
    for level in levels:
        basis.append(product)
        product *= level
    basis_tensor = torch.tensor(basis, device=indices.device, dtype=torch.long)
    levels_tensor = torch.tensor(levels, device=indices.device, dtype=torch.long)
    return (indices.unsqueeze(-1) // basis_tensor) % levels_tensor


def indices_to_codes(
    indices: torch.Tensor, levels: Sequence[int] = DEFAULT_LEVELS
) -> torch.Tensor:
    level_indices = indices_to_level_indices(indices, levels)
    half_width = torch.tensor(
        [level // 2 for level in levels],
        device=indices.device,
        dtype=torch.float32,
    )
    return (level_indices.to(torch.float32) - half_width) / half_width


def fit_fsq_projection(
    token_indices: torch.Tensor,
    embeddings: torch.Tensor,
    levels: Sequence[int] = DEFAULT_LEVELS,
) -> dict[str, torch.Tensor]:
    if token_indices.ndim != 1:
        raise ValueError("token_indices must be one-dimensional")
    if embeddings.ndim != 2 or embeddings.shape[0] != token_indices.shape[0]:
        raise ValueError("embeddings must have shape [frames, embedding_dim]")
    codes = indices_to_codes(token_indices, levels).to(torch.float64)
    design = torch.cat(
        (
            codes,
            torch.ones(
                codes.shape[0],
                1,
                device=codes.device,
                dtype=torch.float64,
            ),
        ),
        dim=1,
    )
    if torch.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("FSQ samples do not span an affine projection")
    solution = torch.linalg.lstsq(design, embeddings.to(torch.float64)).solution
    return {
        "weight": solution[:-1].T.to(torch.float32).contiguous(),
        "bias": solution[-1].to(torch.float32).contiguous(),
    }


def masked_fsq_cross_entropy(
    logits: tuple[torch.Tensor, ...],
    token_indices: torch.Tensor,
    mask: torch.Tensor,
    levels: Sequence[int] = DEFAULT_LEVELS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = indices_to_level_indices(token_indices, levels)
    losses = []
    accuracies = []
    predictions = []
    for axis, axis_logits in enumerate(logits):
        prediction = axis_logits.argmax(dim=1)
        predictions.append(prediction)
        losses.append(
            F.cross_entropy(
                axis_logits.transpose(1, 2)[mask],
                targets[:, :, axis][mask],
            )
        )
        accuracies.append(
            (prediction[mask] == targets[:, :, axis][mask]).float().mean()
        )
    predicted_levels = torch.stack(predictions, dim=-1)
    exact = (predicted_levels == targets).all(dim=-1)
    return (
        torch.stack(losses).mean(),
        torch.stack(accuracies).mean(),
        exact[mask].float().mean(),
    )

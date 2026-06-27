"""Encoder training losses: Q2D2 content objective + GRL + SSL distillation +
multi-resolution STFT (decoder-in-loop).

Moved out of `train_mcs_q2d2.py` / `mcs_common.py`.  `grad_reverse` is the GRL
gradient op applied at loss time; the `SpeakerClassifier` module itself lives
with the model in `encoder.py`.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Batch
from .quantizer import Q2D2Quantizer, compute_q2d2_perplexity


class GradientReversal(torch.autograd.Function):
    """Reverses gradient sign during backward pass.  Forward is identity."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lambda_)


def _voiced_weights(mel: torch.Tensor, length: int, voiced_boost: float) -> torch.Tensor:
    if voiced_boost <= 1.0:
        return mel.new_ones(mel.shape[0], length)
    t_mel = mel.shape[2]
    # Frontend frames per content frame.  Mel and the 50Hz WavLM cache run at
    # 2× the content rate (factor=2); the 200Hz L4 raw cache runs at 8×.
    # Deriving the factor from the actual lengths keeps the voiced mask
    # time-aligned with the content frames for every frontend rate.  (Was
    # hard-coded to 2, which mis-mapped — and silently dropped 3/4 of — the
    # utterance when the 200Hz StridingAdapter frontend was used.)
    factor = max(1, int(round(t_mel / length)))
    t_tok = min(length, t_mel // factor)
    mel_groups = mel[:, :, : t_tok * factor].reshape(
        mel.shape[0], mel.shape[1], t_tok, factor
    )
    rms = mel_groups.pow(2).mean(dim=(1, 3)).sqrt()
    threshold = rms.mean(dim=1, keepdim=True).clamp(min=1e-5)
    voiced = (rms > threshold * 0.5).float()
    weights = 1.0 + (voiced_boost - 1.0) * voiced
    if t_tok < length:
        weights = torch.cat([weights, weights.new_ones(weights.shape[0], length - t_tok)], dim=1)
    return weights


def contrastive_loss(
    pred_768: torch.Tensor,
    tgt_768: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss to prevent content centroid hedging.

    For each frame, the positive is the matching teacher frame; all other
    frames in the batch are negatives.  Operates per-frame over masked
    positions only.

    Args:
        pred_768: (B, 768, L) student projected content.
        tgt_768: (B, L, 768) teacher content.
        mask: (B, L) bool mask of valid frames.
        tau: temperature.

    Returns:
        scalar contrastive loss.
    """
    pred = pred_768.permute(0, 2, 1)[mask]            # (N, 768)
    tgt = tgt_768[mask]                                 # (N, 768)
    if pred.shape[0] < 2:
        return pred.sum() * 0.0
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(tgt, dim=-1)
    # (N, N) similarity; diagonal = positive
    sim = pred_n @ tgt_n.t() / tau
    labels = torch.arange(pred.shape[0], device=pred.device)
    return F.cross_entropy(sim, labels)


def ssl_distill_loss(
    hidden: torch.Tensor | None,
    batch: Batch,
    mask: torch.Tensor,
    ssl_heads: nn.ModuleList | None,
    ssl_layers: tuple[int, ...] = (0, 4, 8),
    ts: int = 0,
) -> torch.Tensor:
    """WavLM multi-target distillation (Mimi-style).

    The student's pre-quantization hidden state is projected through
    ``ssl_heads`` (one per target layer) and matched to the cached WavLM
    layer outputs by cosine similarity.

    Args:
        hidden: (B, trans_dim, T) student pre-quantization state.
        batch: training batch carrying ssl_L* targets.
        mask: (B, L) valid-frame mask.
        ssl_heads: ModuleList of Linear(trans_dim → 768), one per target.
        ssl_layers: which WavLM layers to target (used to build attr names).
        ts: time-shift offset.

    Returns:
        scalar distillation loss (mean of 1 - cos over masked frames & layers).
    """
    if hidden is None or ssl_heads is None or len(ssl_heads) == 0:
        return torch.tensor(0.0, device=hidden.device if hidden is not None else "cpu")
    h = hidden[:, :, ts:ts + mask.shape[1]].permute(0, 2, 1)  # (B, L, trans_dim)
    L = h.shape[1]
    ssl_keys = [f"ssl_L{lv}" for lv in ssl_layers[:len(ssl_heads)]]
    cos_terms: list[torch.Tensor] = []
    for head, k in zip(ssl_heads, ssl_keys):
        tgt = getattr(batch, k, None)
        if tgt is None or tgt.numel() == 0:
            continue
        tgt = tgt[:, :L]                                   # (B, L, 768)
        pred = head(h)                                     # (B, L, 768)
        a = F.normalize(pred, dim=-1)
        b = F.normalize(tgt, dim=-1)
        cos = (a * b).sum(dim=-1)                          # (B, L)
        cos_terms.append(
            (1.0 - cos * mask.float()).sum() / mask.float().sum().clamp(min=1)
        )
    if not cos_terms:
        return hidden.sum() * 0.0
    return torch.stack(cos_terms).mean()


def q2d2_losses(
    output: dict,
    batch: Batch,
    args: argparse.Namespace,
    quantizer: Q2D2Quantizer | None = None,
    speaker_classifier: nn.Module | None = None,
    speaker_ids: torch.Tensor | None = None,
    time_shift: int = 0,
    ssl_heads: nn.ModuleList | None = None,
    ssl_layers: tuple[int, ...] = (0, 4, 8),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute losses for Q2D2 quantized output.

    Since Q2D2 has no per-axis ordinal structure, losses are:
      - content_cos:  cosine similarity between projected 768d and teacher content
      - content_l1:   L1 between projected and teacher content
      - q2d2_perplexity (metrics only): codebook utilization

    Args:
        output: Model output dict with 'projected' and 'q2d2_codes'.
        batch: Training batch.
        args: Training arguments.
        quantizer: Optional Q2D2Quantizer for utilization stats.

    Returns:
        loss, metrics dict.
    """
    projected = output["projected"]                     # (B, 768, T)
    q2d2_codes = output.get("q2d2_codes")               # (B, T, 6) or None

    ts = time_shift
    length = min(projected.shape[2] - ts, batch.content.shape[1] - ts,
                 batch.mask.shape[1] - ts)
    if length < 2:
        zero = projected.sum() * 0.0
        return zero, {"loss": float(zero.detach().cpu()), "cos768": 0.0}
    mask = batch.mask[:, ts:ts + length]

    # ── time-shifted alignment ──
    # student[t] compares with teacher[t-ts]
    pred_768 = projected[:, :, ts:ts + length]           # (B, 768, L)
    if ts > 0:
        tgt_768 = batch.content[:, :length]               # student[ts..] ↔ teacher[0..]
    else:
        tgt_768 = batch.content[:, :length]               # (B, L, 768)

    # voiced weighting
    voiced_boost = getattr(args, "voiced_boost", 1.0)
    vw = _voiced_weights(batch.mel, length, voiced_boost)  # (B, L)
    weighted_mask_sum = (vw * mask.float()).sum().clamp(min=1)

    # ── content cosine (primary quality metric) ──
    # Compute over masked frames: cos per batch item then average
    pred_masked = pred_768.permute(0, 2, 1)[mask]       # (N_valid, 768)
    tgt_masked = tgt_768[mask]                            # (N_valid, 768)
    cos768 = F.cosine_similarity(pred_masked, tgt_masked, dim=-1).mean()
    cos768_loss = 1.0 - cos768

    # ── content L1 ──
    pred_flat = pred_768.permute(0, 2, 1)                # (B, L, 768)
    l1_per_frame = (pred_flat - tgt_768).abs().mean(dim=-1)  # (B, L)
    content_l1 = ((l1_per_frame * vw * mask.float()).sum() / weighted_mask_sum)

    # ── delta (temporal smoothness) ──
    if length >= 2:
        delta_mask = mask[:, 1:] & mask[:, :-1]
        pred_delta = pred_flat[:, 1:] - pred_flat[:, :-1]
        tgt_delta = tgt_768[:, 1:] - tgt_768[:, :-1]
        delta_weights = 0.5 * (vw[:, 1:] + vw[:, :-1])
        delta = F.smooth_l1_loss(
            pred_delta[delta_mask], tgt_delta[delta_mask], reduction="mean"
        )
    else:
        delta = projected.sum() * 0.0

    # ── delta2 (2nd-order temporal smoothness) ──
    delta2 = projected.sum() * 0.0
    if length >= 3:
        d2_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        if d2_mask.any():
            pred_d2 = pred_flat[:, 2:] - 2 * pred_flat[:, 1:-1] + pred_flat[:, :-2]
            tgt_d2 = tgt_768[:, 2:] - 2 * tgt_768[:, 1:-1] + tgt_768[:, :-2]
            delta2 = F.smooth_l1_loss(
                pred_d2[d2_mask], tgt_d2[d2_mask], reduction="mean"
            )

    # ── total loss ──
    loss = (args.content_cos_weight * cos768_loss +
            args.content_l1_weight * content_l1 +
            args.delta_weight * delta +
            getattr(args, "delta2_weight", 0.0) * delta2)

    # ── forecast loss ──
    forecast_weight = getattr(args, "forecast_weight", 0.0)
    forecast_loss_val: float = 0.0
    if forecast_weight > 0:
        fc1 = output.get("forecast_1")
        fc2 = output.get("forecast_2")
        # Lf bounded so the target/mask shifts (t+1, t+2) stay in range.
        Lf = min(length - 2, batch.content.shape[1] - 2)
        if fc1 is not None and fc2 is not None and length >= 3 and Lf >= 1:
            fc1_flat = fc1[:, :, ts:ts + length].permute(0, 2, 1)[:, :Lf, :]
            fc2_flat = fc2[:, :, ts:ts + length].permute(0, 2, 1)[:, :Lf, :]
            tgt_fc1 = batch.content[:, 1:1 + Lf]
            tgt_fc2 = batch.content[:, 2:2 + Lf]
            # Mask: predict only where BOTH the source frame and the future
            # target frame are valid (excludes right-padding).
            m1 = (mask[:, :Lf] & mask[:, 1:1 + Lf]).float()
            m2 = (mask[:, :Lf] & mask[:, 2:2 + Lf]).float()
            fl1 = (F.mse_loss(fc1_flat, tgt_fc1, reduction="none").mean(-1) * m1
                   ).sum() / m1.sum().clamp(min=1)
            fl2 = (F.mse_loss(fc2_flat, tgt_fc2, reduction="none").mean(-1) * m2
                   ).sum() / m2.sum().clamp(min=1)
            fl = (fl1 + fl2) * 0.5
            forecast_loss_val = float(fl.detach().cpu())
            loss = loss + forecast_weight * fl

    # ── GRL speaker disentanglement loss ──
    grl_loss_val: float = 0.0
    grl_acc_val: float = 0.0
    if speaker_classifier is not None and speaker_ids is not None:
        grl_weight = getattr(args, "grl_weight", 0.0)
        if grl_weight > 0:
            # Reverse gradient: classifier tries to predict speaker,
            # but encoder gets reversed gradient → strips speaker info.
            # Pool over the valid (masked) loss region only.
            grl_content = grad_reverse(projected[:, :, ts:ts + length], grl_weight)
            speaker_logits = speaker_classifier(grl_content, mask)
            grl_loss = F.cross_entropy(speaker_logits, speaker_ids)
            loss = loss + grl_loss
            grl_loss_val = float(grl_loss.detach().cpu())
            grl_acc_val = float(
                (speaker_logits.argmax(dim=-1) == speaker_ids).float().mean().cpu()
            )

    # ── InfoNCE contrastive loss ──
    contrastive_loss_val: float = 0.0
    contrastive_weight = getattr(args, "contrastive_weight", 0.0)
    if contrastive_weight > 0:
        c_loss = contrastive_loss(
            pred_768, tgt_768, mask,
            tau=getattr(args, "contrastive_tau", 0.1),
        )
        contrastive_loss_val = float(c_loss.detach().cpu())
        loss = loss + contrastive_weight * c_loss

    # ── WavLM SSL multi-target distillation ──
    ssl_loss_val: float = 0.0
    ssl_weight = getattr(args, "ssl_weight", 0.0)
    if ssl_weight > 0:
        hidden = output.get("hidden")              # (B, trans_dim, T)
        s_loss = ssl_distill_loss(hidden, batch, mask, ssl_heads, ssl_layers, ts)
        ssl_loss_val = float(s_loss.detach().cpu())
        loss = loss + ssl_weight * s_loss

    # ── metrics ──
    metrics: dict[str, float] = {
        "loss": float(loss.detach().cpu()),
        "cos768": float(cos768.detach().cpu()),
        "content_l1": float(content_l1.detach().cpu()),
        "delta": float(delta.detach().cpu()),
        "delta2": float(delta2.detach().cpu()),
        "grl_loss": grl_loss_val,
        "grl_acc": grl_acc_val,
        "forecast_loss": forecast_loss_val,
        "contrastive_loss": contrastive_loss_val,
        "ssl_loss": ssl_loss_val,
    }

    # Q2D2 utilization stats (diagnostic, no gradient)
    if quantizer is not None and q2d2_codes is not None:
        with torch.no_grad():
            stats = compute_q2d2_perplexity(quantizer, q2d2_codes)
            metrics["q2d2_usage"] = stats["overall_usage"]
            for i in range(quantizer.num_pairs):
                metrics[f"q2d2_pair{i}_usage"] = stats[f"pair_{i}_usage"]

    return loss, metrics


# ── multi-resolution STFT (decoder-in-loop wave loss) ──

def stft_mag(wave: torch.Tensor, n_fft: int) -> torch.Tensor:
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=wave.device, dtype=wave.dtype)
    spec = torch.stft(
        wave, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, return_complex=True,
    )
    return spec.abs().clamp_min(1e-7)


def multi_resolution_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_ffts: tuple[int, ...],
) -> torch.Tensor:
    pred = pred.squeeze(0) if pred.dim() == 2 else pred
    target = target.squeeze(0) if target.dim() == 2 else target
    length = min(pred.shape[-1], target.shape[-1])
    pred = pred[:length]
    target = target[:length]
    losses = []
    for n_fft in n_ffts:
        pred_mag = stft_mag(pred, n_fft)
        target_mag = stft_mag(target, n_fft)
        spectral_convergence = torch.linalg.vector_norm(pred_mag - target_mag) / (
            torch.linalg.vector_norm(target_mag).clamp_min(1e-7))
        log_mag = F.l1_loss(pred_mag.log(), target_mag.log())
        losses.append(spectral_convergence + log_mag)
    return torch.stack(losses).mean()

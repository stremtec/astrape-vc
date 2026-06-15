"""Unit tests for the FFL training module."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import CTCLoss

from astrape.data import ContentBatch, ContentCollator, ContentSample, MioContentDataset
from astrape.ffl_training import (
    FflTrainingConfig,
    _masked_output_effect_loss,
    _set_phase_trainability,
    _training_phase,
    evaluate_ffl,
    ffl_loss,
    train_ffl_student,
    validate_ffl_config,
)
from astrape.flat_ctc_training import speaker_balanced_subset
from astrape.model import ContentStudent, ContentStudentConfig
from astrape.text import VOCAB_SIZE


def _small_ffl_student(hidden: int = 32) -> ContentStudent:
    return ContentStudent(
        ContentStudentConfig(
            architecture="mio_ffl",
            hidden=hidden,
            n_layers=2,
            n_heads=4,
            mio_ff_hidden=hidden * 4,
            content_dim=hidden,
            text_vocab_size=VOCAB_SIZE,
            max_attention_context=16,
            ffl_hidden=hidden // 2,
            ffl_horizon=4,
            ffl_history=8,
            ffl_heads=4,
            ffl_coarse_layers=1,
            ffl_reverse_layers=1,
            ffl_summary_layers=2,
        )
    )


class FflConfigTests(unittest.TestCase):
    def test_validate_rejects_bad_weight(self) -> None:
        with self.assertRaises(ValueError):
            validate_ffl_config(
                FflTrainingConfig(
                    data_dir=Path("/tmp"),
                    output_dir=Path("/tmp"),
                    ctc_weight=-1.0,
                )
            )

    def test_training_phase_schedule(self) -> None:
        config = FflTrainingConfig(
            data_dir=Path("/tmp"),
            output_dir=Path("/tmp"),
            epochs=4,
            causal_warmup_epochs=1,
            effect_warmup_epochs=1,
        )
        self.assertEqual(_training_phase(0, config), "causal")
        self.assertEqual(_training_phase(1, config), "effect")
        self.assertEqual(_training_phase(2, config), "joint")

    def test_collator_quantizes_padded_mel_length(self) -> None:
        sample = ContentSample(
            mel=torch.randn(80, 65),
            content=torch.randn(33, 32),
            pre_fsq=None,
            token_indices=None,
            speaker="s",
            index=0,
            transcript=torch.tensor([1, 2]),
        )
        batch = ContentCollator(
            None,
            seed=1,
            include_transcripts=True,
            pad_mel_multiple=64,
        )([sample])
        self.assertEqual(batch.mel.shape[-1], 128)
        self.assertEqual(batch.input_lengths.item(), 65)


def _synthetic_batch(device: torch.device) -> ContentBatch:
    """Build a small deterministic batch for fast loss/eval testing."""
    length = 16
    n_mels = 80
    hidden = 32
    mel = torch.randn(2, n_mels, length, device=device)
    content = torch.randn(2, length // 2, hidden, device=device)
    target_mask = torch.ones(2, length // 2, dtype=torch.bool, device=device)
    transcripts = torch.tensor([1, 2, 3, 4, 5, 6, 0, 0], dtype=torch.long, device=device)
    transcript_lengths = torch.tensor([6, 2], dtype=torch.long, device=device)
    return ContentBatch(
        mel=mel,
        content=content,
        pre_fsq=None,
        token_indices=None,
        input_lengths=torch.tensor([length, length], dtype=torch.long, device=device),
        target_lengths=torch.tensor([length // 2, length // 2], dtype=torch.long, device=device),
        target_mask=target_mask,
        transcripts=transcripts,
        transcript_lengths=transcript_lengths,
    )


class FflLossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.model = _small_ffl_student().to(self.device)
        self.criterion = CTCLoss(blank=0, zero_infinity=True)
        self.config = FflTrainingConfig(
            data_dir=Path("/nonexistent"),
            output_dir=Path("/tmp/_unused"),
            run_name="unit",
            batch_size=2,
            epochs=1,
            steps_per_epoch=1,
            log_every=1,
        )

    def test_returns_loss_and_diagnostics(self) -> None:
        batch = _synthetic_batch(self.device)
        output = self.model(batch.mel, batch.input_lengths)
        loss, parts = ffl_loss(output, batch, self.config, self.criterion)
        self.assertTrue(torch.isfinite(loss))
        for key in (
            "content_loss", "content_cosine", "delta_loss", "ctc_loss",
            "gate_l2", "gate_mean", "effect_ratio",
        ):
            self.assertIn(key, parts)
        self.assertGreater(parts["content_loss"], 0.0)
        self.assertGreaterEqual(parts["gate_mean"], 0.0)
        self.assertLessEqual(parts["gate_mean"], 1.0)

    def test_backward_reaches_generator_and_gate(self) -> None:
        batch = _synthetic_batch(self.device)
        output = self.model(batch.mel, batch.input_lengths)
        with torch.no_grad():
            baseline = self.model(
                batch.mel,
                batch.input_lengths,
                enable_false_future=False,
            ).content
        loss, _ = ffl_loss(
            output,
            batch,
            self.config,
            self.criterion,
            baseline,
        )
        loss.backward()
        self.assertIsNotNone(
            self.model.false_future_generator.input_projection.weight.grad
        )
        self.assertIsNotNone(self.model.false_future_adapters[0].gate.weight.grad)
        self.assertGreater(
            self.model.false_future_generator.input_projection.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            self.model.false_future_adapters[0].gate.weight.grad.abs().sum().item(),
            0.0,
        )

    def test_output_effect_loss_rewards_teacher_residual(self) -> None:
        baseline = torch.zeros(1, 4, 3)
        target = torch.ones(1, 3, 4)
        mask = torch.ones(1, 3, dtype=torch.bool)
        good, cosine = _masked_output_effect_loss(
            torch.ones(1, 4, 3),
            baseline,
            target,
            mask,
            cosine_weight=0.1,
        )
        bad, _ = _masked_output_effect_loss(
            torch.zeros(1, 4, 3),
            baseline,
            target,
            mask,
            cosine_weight=0.1,
        )
        self.assertLess(good.item(), bad.item())
        self.assertAlmostEqual(cosine.item(), 1.0, places=5)

    def test_effect_phase_freezes_backbone(self) -> None:
        _set_phase_trainability(self.model, "effect")
        self.assertFalse(self.model.content_head.weight.requires_grad)
        self.assertTrue(
            self.model.false_future_generator.input_projection.weight.requires_grad
        )


class FflEvaluateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.model = _small_ffl_student().to(self.device)
        self.criterion = CTCLoss(blank=0, zero_infinity=True)

    def test_evaluator_returns_full_metrics(self) -> None:
        batch = _synthetic_batch(self.device)

        class _L:
            def __init__(self, b): self.b = b
            def __iter__(self): yield self.b

        metrics = evaluate_ffl(self.model, _L(batch), self.device, self.criterion)
        for key in (
            "val_frame_cosine",
            "val_frame_cosine_p05",
            "val_sequence_cosine",
            "val_ctc_loss",
            "val_character_error_rate",
            "val_gate_mean",
            "val_effect_ratio",
            "val_gate_sparsity_frac",
        ):
            self.assertIn(key, metrics)
        self.assertTrue(np.isfinite(metrics["val_frame_cosine"]))
        self.assertAlmostEqual(metrics["val_gate_mean"], 0.0180, places=3)
        self.assertEqual(metrics["val_gate_sparsity_frac"], 1.0)


class FflTrainerTests(unittest.TestCase):
    """A small end-to-end run driven against a mocked cache."""

    def test_train_runs_one_epoch_on_synthetic_dataset(self) -> None:
        device = torch.device("cpu")
        with TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"
            output_dir = Path(directory) / "ckpt"
            data_dir.mkdir()
            meta = {
                "spk_names": np.asarray(["spk0", "spk1", "spk2"]),
                "utterance_ids": np.asarray(["u0", "u1", "u2"]),
                "source_files": np.asarray(["s0.wav", "s1.wav", "s2.wav"]),
                "transcript_files": np.asarray(["t0.txt", "t1.txt", "t2.txt"]),
                "n_samples": np.asarray(3),
                "cache_format": np.asarray("compact-fp16-ctc-v2"),
            }
            np.savez(data_dir / "meta.npz", **meta)
            for index in range(3):
                np.savez(
                    data_dir / f"s_{index:05d}.npz",
                    logmel=np.random.randn(80, 32).astype(np.float16),
                    ce_768=np.random.randn(16, 32).astype(np.float16),
                    pre_fsq_768=np.random.randn(32, 16).astype(np.float16),
                    ct=np.random.randint(0, 8, size=(32,)).astype(np.int64),
                    transcript=np.random.randint(0, 28, size=(10,)).astype(np.int64),
                )

            model = ContentStudentConfig(
                architecture="mio_ffl",
                hidden=32,
                n_layers=2,
                n_heads=4,
                mio_ff_hidden=128,
                content_dim=32,
                text_vocab_size=VOCAB_SIZE,
                max_attention_context=16,
                ffl_hidden=16,
                ffl_horizon=4,
                ffl_history=8,
                ffl_heads=4,
                ffl_coarse_layers=1,
                ffl_reverse_layers=1,
                ffl_summary_layers=2,
            )
            config = FflTrainingConfig(
                data_dir=data_dir,
                output_dir=output_dir,
                run_name="unit",
                device="cpu",
                batch_size=2,
                epochs=1,
                steps_per_epoch=1,
                log_every=1,
                probe_samples=2,
                full_validation_every=1,
                causal_warmup_epochs=1,
                effect_warmup_epochs=0,
            )
            train_ffl_student(model, config)
            self.assertTrue((output_dir / "unit.last.pt").exists())
            self.assertTrue((output_dir / "unit.probe-best.pt").exists())
            self.assertTrue((output_dir / "unit.best.pt").exists())


if __name__ == "__main__":
    unittest.main()

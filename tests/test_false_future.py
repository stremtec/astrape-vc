import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from astrape.checkpoint import load_content_checkpoint, save_checkpoint
from astrape.false_future import FalseFutureConfig, FalseFutureSlotGenerator
from astrape.model import ContentStudent, ContentStudentConfig
from diagnose_false_future import make_false_future


class FalseFutureDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.observed = torch.arange(1, 6, dtype=torch.float32).view(1, 5, 1)
        self.full = torch.arange(1, 9, dtype=torch.float32).view(1, 8, 1)

    def test_replay_maps_recent_history_forward(self):
        future = make_false_future(self.observed, self.full, 3, "replay")
        torch.testing.assert_close(
            future,
            torch.tensor([[[3.0], [4.0], [5.0]]]),
        )

    def test_reverse_maps_nearest_history_to_nearest_future(self):
        future = make_false_future(self.observed, self.full, 3, "reverse")
        torch.testing.assert_close(
            future,
            torch.tensor([[[5.0], [4.0], [3.0]]]),
        )

    def test_oracle_uses_real_following_frames(self):
        future = make_false_future(self.observed, self.full, 3, "oracle")
        torch.testing.assert_close(
            future,
            torch.tensor([[[6.0], [7.0], [8.0]]]),
        )

    def test_no_future_returns_empty_sequence(self):
        future = make_false_future(self.observed, self.full, 3, "no_future")
        self.assertEqual(future.shape, (1, 0, 1))


class FalseFutureGeneratorTests(unittest.TestCase):
    def small_generator(self) -> FalseFutureSlotGenerator:
        return FalseFutureSlotGenerator(
            FalseFutureConfig(
                input_dim=16,
                hidden_dim=8,
                horizon=4,
                history=16,
                n_heads=2,
                coarse_layers=1,
                reverse_layers=1,
                summary_layers=1,
                ff_mult=2,
            )
        )

    def test_output_shape(self):
        generator = self.small_generator().eval()
        output = generator(torch.randn(2, 10, 16))
        self.assertEqual(output.shape, (2, 10, 4, 8))

    def test_generator_has_no_future_leakage(self):
        generator = self.small_generator().eval()
        prefix = torch.randn(1, 7, 16)
        suffix = torch.randn(1, 5, 16)
        expected = generator(prefix)
        actual = generator(torch.cat((prefix, suffix), dim=1))[:, : prefix.shape[1]]
        torch.testing.assert_close(actual, expected)

    def test_generator_streaming_matches_full_sequence(self):
        generator = self.small_generator().eval()
        x = torch.randn(1, 13, 16)
        expected = generator(x)
        history = None
        chunks = []
        for start, length in ((0, 2), (2, 5), (7, 1), (8, 5)):
            slots, history = generator.forward_stream(
                x[:, start : start + length],
                history,
            )
            chunks.append(slots)
        torch.testing.assert_close(torch.cat(chunks, dim=1), expected)

    def test_rejects_empty_history(self):
        generator = self.small_generator()
        with self.assertRaises(ValueError):
            generator(torch.empty(1, 0, 16))


class MioFalseFutureStudentTests(unittest.TestCase):
    def small_student(self) -> ContentStudent:
        return ContentStudent(
            ContentStudentConfig(
                architecture="mio_ffl",
                hidden=64,
                n_layers=2,
                n_heads=4,
                mio_ff_hidden=128,
                content_dim=32,
                max_attention_context=8,
                ffl_hidden=32,
                ffl_horizon=4,
                ffl_history=16,
                ffl_heads=4,
                ffl_coarse_layers=1,
                ffl_reverse_layers=1,
                ffl_summary_layers=2,
            )
        )

    def test_student_has_no_future_leakage(self):
        model = self.small_student().eval()
        prefix = torch.randn(1, 80, 13)
        suffix = torch.randn(1, 80, 10)
        prefix_output = model(prefix).content
        full_output = model(torch.cat((prefix, suffix), dim=-1)).content
        torch.testing.assert_close(
            prefix_output,
            full_output[:, :, : prefix_output.shape[-1]],
            atol=2e-6,
            rtol=2e-6,
        )

    def test_streaming_matches_full_sequence(self):
        model = self.small_student().eval()
        x = torch.randn(1, 80, 23)
        expected = model(x).content
        state = None
        chunks = []
        for start, length in ((0, 3), (3, 4), (7, 1), (8, 7), (15, 8)):
            output, state = model.forward_stream(
                x[:, :, start : start + length],
                state,
            )
            if output.content.shape[-1]:
                chunks.append(output.content)
        output, _ = model.forward_stream(x[:, :, :0], state, flush=True)
        chunks.append(output.content)
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1),
            expected,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_outputs_layer_diagnostics(self):
        model = self.small_student().eval()
        output = model(torch.randn(2, 80, 12))
        self.assertEqual(output.false_future_effects.shape, (2, 2, 12, 64))
        self.assertEqual(output.false_future_corrections.shape, (2, 2, 12, 64))
        self.assertEqual(output.false_future_gates.shape, (2, 2, 12))
        self.assertEqual(output.false_future_benefit.shape, (2, 2, 12))
        self.assertTrue(torch.all(output.false_future_gates < 0.02))
        torch.testing.assert_close(
            output.false_future_corrections,
            output.false_future_effects
            * output.false_future_gates.unsqueeze(-1),
        )

    def test_content_loss_reaches_generator_and_gate(self):
        model = self.small_student()
        output = model(torch.randn(2, 80, 12))
        output.content.square().mean().backward()
        generator_gradient = model.false_future_generator.input_projection.weight.grad
        gate_gradient = model.false_future_adapters[0].gate.weight.grad
        self.assertIsNotNone(generator_gradient)
        self.assertIsNotNone(gate_gradient)
        self.assertGreater(generator_gradient.abs().sum().item(), 0.0)
        self.assertGreater(gate_gradient.abs().sum().item(), 0.0)

    def test_checkpoint_round_trip(self):
        model = self.small_student().eval()
        x = torch.randn(1, 80, 12)
        expected = model(x).content
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ffl.pt"
            save_checkpoint(path, model, epoch=0, metrics={})
            loaded, _ = load_content_checkpoint(path)
        torch.testing.assert_close(loaded.eval()(x).content, expected)


if __name__ == "__main__":
    unittest.main()

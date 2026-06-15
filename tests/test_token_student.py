import unittest
from pathlib import Path

import torch

from astrape.data import ContentBatch
from astrape.fsq import indices_to_codes
from astrape.token_student import TokenStudentConfig, TokenSynchronousStudent
from astrape.token_student_training import TokenPhase0Config, token_phase0_loss


class TokenStudentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def small_model(self) -> TokenSynchronousStudent:
        model = TokenSynchronousStudent(
            TokenStudentConfig(
                edge_dim=32,
                core_dim=64,
                edge_layers=2,
                core_layers=4,
                n_heads=4,
                ff_hidden=128,
                attention_context=8,
                edge_kernel=3,
                core_kernel=3,
                content_dim=16,
                text_vocab_size=12,
            )
        )
        model.load_fsq_projection(
            {
                "weight": torch.randn(16, 5),
                "bias": torch.randn(16),
            }
        )
        return model

    def test_end_of_cell_output_length(self):
        model = self.small_model().eval()
        self.assertEqual(model(torch.randn(1, 80, 1)).content.shape[-1], 0)
        self.assertEqual(model(torch.randn(1, 80, 2)).content.shape[-1], 1)
        self.assertEqual(model(torch.randn(1, 80, 9)).content.shape[-1], 4)

    def test_model_has_no_future_leakage(self):
        model = self.small_model().eval()
        prefix = torch.randn(1, 80, 12)
        suffix = torch.randn(1, 80, 8)
        expected = model(prefix).content
        actual = model(torch.cat((prefix, suffix), dim=-1)).content
        torch.testing.assert_close(
            expected,
            actual[:, :, : expected.shape[-1]],
            atol=2e-6,
            rtol=2e-6,
        )

    def test_streaming_matches_full_sequence(self):
        model = self.small_model().eval()
        mel = torch.randn(1, 80, 23)
        expected = model(mel).content
        state = None
        chunks = []
        start = 0
        for length in (1, 4, 2, 7, 3, 6):
            output, state = model.forward_stream(
                mel[:, :, start : start + length],
                state,
            )
            if output.content.shape[-1]:
                chunks.append(output.content)
            start += length
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1),
            expected,
            atol=3e-6,
            rtol=3e-6,
        )

    def test_phase0_loss_is_finite(self):
        model = self.small_model()
        mel = torch.randn(2, 80, 20)
        tokens = torch.randint(0, 12800, (2, 10))
        with torch.no_grad():
            codes = indices_to_codes(tokens)
            content = model.fsq_projection(codes)
        batch = ContentBatch(
            mel=mel,
            content=content,
            pre_fsq=None,
            token_indices=tokens,
            input_lengths=torch.tensor([20, 18]),
            target_lengths=torch.tensor([10, 9]),
            target_mask=torch.tensor(
                [
                    [True] * 10,
                    [True] * 9 + [False],
                ]
            ),
        )
        loss, metrics = token_phase0_loss(
            model,
            batch,
            TokenPhase0Config(
                data_dir=Path("data"),
                projection_path=Path("projection.pt"),
                output_dir=Path("checkpoints"),
            ),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()

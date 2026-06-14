import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from astrape.audio import StreamingLogMel
from astrape.checkpoint import load_content_checkpoint, save_checkpoint
from astrape.curriculum import (
    CurriculumConfig,
    original_loss,
    phase_weights,
    validate_curriculum,
)
from astrape.data import (
    ContentSample,
    crop_aligned,
    masked_content_loss,
    speaker_disjoint_split,
)
from astrape.mel_decoder import CausalMelDecoder, MelDecoderConfig, load_mel_decoder
from astrape.model import ContentStudent, ContentStudentConfig
from astrape.fsq import (
    fit_fsq_projection,
    indices_to_codes,
    indices_to_level_indices,
)
from astrape.original_data import OriginalBatch, minimum_ctc_frames
from astrape.text import VOCAB_SIZE
from tiers import TIERS, get_tier


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def small_student(self) -> ContentStudent:
        return ContentStudent(
            ContentStudentConfig(
                hidden=64,
                n_layers=2,
                n_heads=4,
                content_dim=32,
            )
        ).eval()

    def test_content_model_has_no_future_leakage(self):
        model = self.small_student()
        prefix = torch.randn(1, 80, 12)
        suffix = torch.randn(1, 80, 8)
        prefix_output = model(prefix).content
        full_output = model(torch.cat((prefix, suffix), dim=-1)).content
        torch.testing.assert_close(prefix_output, full_output[:, :, :6])

    def test_content_streaming_matches_full_sequence(self):
        model = self.small_student()
        x = torch.randn(1, 80, 20)
        expected = model(x).content
        state = None
        chunks = []
        for start in range(0, x.shape[-1], 2):
            output, state = model.forward_stream(x[:, :, start : start + 2], state)
            chunks.append(output.content)
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1), expected, atol=2e-6, rtol=2e-6
        )

    def test_content_streaming_buffers_odd_chunks_and_flushes(self):
        model = self.small_student()
        x = torch.randn(1, 80, 21)
        expected = model(x).content
        state = None
        chunks = []
        for start, length in ((0, 3), (3, 4), (7, 1), (8, 7), (15, 6)):
            output, state = model.forward_stream(
                x[:, :, start : start + length], state
            )
            if output.content.shape[-1]:
                chunks.append(output.content)
        output, state = model.forward_stream(x[:, :, :0], state, flush=True)
        chunks.append(output.content)
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1), expected, atol=2e-6, rtol=2e-6
        )

    def test_limited_attention_streaming_matches_full_sequence(self):
        model = ContentStudent(
            ContentStudentConfig(
                hidden=64,
                n_layers=2,
                n_heads=4,
                content_dim=32,
                max_attention_context=4,
            )
        ).eval()
        x = torch.randn(1, 80, 20)
        expected = model(x).content
        state = None
        chunks = []
        for start in range(0, x.shape[-1], 2):
            output, state = model.forward_stream(x[:, :, start : start + 2], state)
            chunks.append(output.content)
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1), expected, atol=2e-6, rtol=2e-6
        )

    def test_mel_decoder_streaming_matches_full_sequence(self):
        decoder = CausalMelDecoder(
            MelDecoderConfig(hidden=64, n_layers=2, n_heads=4, dropout=0.0)
        ).eval()
        content = torch.randn(1, 10, 768)
        global_embedding = torch.randn(1, 128)
        expected = decoder(content, global_embedding)
        state = None
        chunks = []
        for index in range(content.shape[1]):
            output, state = decoder.forward_stream(
                content[:, index : index + 1], global_embedding, state
            )
            chunks.append(output)
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1), expected, atol=2e-6, rtol=2e-6
        )

    def test_streaming_logmel_matches_full_sequence(self):
        extractor = StreamingLogMel()
        waveform = torch.randn(1, 16000)
        expected = extractor(waveform)
        state = None
        chunks = []
        for start in range(0, waveform.shape[-1], 777):
            output, state = extractor.forward_stream(
                waveform[:, start : start + 777], state
            )
            chunks.append(output)
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1), expected, atol=1e-6, rtol=1e-6
        )

    def test_speaker_split_is_disjoint_and_deterministic(self):
        speakers = np.array(["a"] * 4 + ["b"] * 3 + ["c"] * 5 + ["d"] * 2)
        train_a, validation_a = speaker_disjoint_split(speakers, 0.25, 42)
        train_b, validation_b = speaker_disjoint_split(speakers, 0.25, 42)
        self.assertTrue(np.array_equal(train_a, train_b))
        self.assertTrue(np.array_equal(validation_a, validation_b))
        self.assertFalse(set(speakers[train_a]) & set(speakers[validation_a]))

    def test_crop_uses_even_grid_and_includes_last_window(self):
        mel = torch.arange(20).repeat(80, 1).float()
        content = torch.arange(10).unsqueeze(1).repeat(1, 4).float()
        sample = ContentSample(mel, content, content.clone(), None, "p001", 0)

        class LastChoice(random.Random):
            def choice(self, sequence):
                return sequence[-1]

        cropped = crop_aligned(sample, 8, LastChoice())
        self.assertEqual(cropped.mel[0, 0].item(), 12)
        self.assertEqual(cropped.content[0, 0].item(), 6)
        self.assertEqual(cropped.mel.shape[1], 8)
        self.assertEqual(cropped.content.shape[0], 4)

    def test_masked_loss_ignores_padding(self):
        prediction = torch.zeros(2, 3, 4)
        target = torch.zeros(2, 4, 3)
        target[1, 2:] = 1000
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        loss, cosine = masked_content_loss(prediction, target, mask)
        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertAlmostEqual(cosine.item(), 0.0)

    def test_versioned_checkpoint_roundtrip_and_legacy_gate(self):
        model = self.small_student()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            versioned = directory / "model.pt"
            legacy = directory / "legacy.pt"
            save_checkpoint(versioned, model, epoch=3, metrics={"val_cosine": 0.5})
            loaded, metadata = load_content_checkpoint(versioned)
            self.assertEqual(metadata["epoch"], 3)
            self.assertEqual(loaded.config, model.config)
            torch.save(model.state_dict(), legacy)
            with self.assertRaises(ValueError):
                load_content_checkpoint(legacy)
            loaded_legacy, metadata = load_content_checkpoint(
                legacy, allow_legacy=True
            )
            self.assertEqual(metadata["format_version"], 1)
            self.assertEqual(loaded_legacy.config.hidden, model.config.hidden)

    def test_mel_decoder_checkpoint_roundtrip(self):
        model = CausalMelDecoder(
            MelDecoderConfig(hidden=64, n_layers=2, n_heads=4, dropout=0.0)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decoder.pt"
            torch.save(
                {
                    "model_type": "causal_mel_decoder",
                    "config": {
                        "hidden": 64,
                        "n_layers": 2,
                        "n_heads": 4,
                        "dropout": 0.0,
                    },
                    "state_dict": model.state_dict(),
                },
                path,
            )
            loaded = load_mel_decoder(path)
            self.assertEqual(loaded.config.hidden, 64)

    def test_structured_fsq_output_matches_frozen_projection(self):
        config = ContentStudentConfig(
            hidden=64,
            n_layers=2,
            n_heads=4,
            content_dim=16,
            structured_fsq=True,
            text_vocab_size=VOCAB_SIZE,
        )
        model = ContentStudent(config).eval()
        projection = {
            "weight": torch.randn(16, 5),
            "bias": torch.randn(16),
        }
        model.load_fsq_projection(projection)
        output = model(torch.randn(1, 80, 20))
        self.assertIsNotNone(output.fsq_codes)
        codes = []
        for axis, levels in enumerate(config.fsq_levels):
            values = (
                output.fsq_codes[:, :, axis].float() - levels // 2
            ) / (levels // 2)
            codes.append(values)
        codes = torch.stack(codes, dim=-1)
        expected = torch.nn.functional.linear(
            codes, projection["weight"], projection["bias"]
        ).transpose(1, 2)
        torch.testing.assert_close(output.content, expected)

    def test_fsq_index_roundtrip_targets(self):
        indices = torch.tensor([[0, 930, 12799]])
        levels = indices_to_level_indices(indices)
        codes = indices_to_codes(indices)
        self.assertEqual(tuple(levels.shape), (1, 3, 5))
        self.assertEqual(tuple(codes.shape), (1, 3, 5))
        self.assertTrue(torch.equal(levels[0, 0], torch.zeros(5, dtype=torch.long)))

    def test_fsq_projection_fit_recovers_affine_teacher(self):
        indices = torch.arange(12800)
        codes = indices_to_codes(indices)
        weight = torch.randn(16, 5)
        bias = torch.randn(16)
        embeddings = torch.nn.functional.linear(codes, weight, bias)
        fitted = fit_fsq_projection(indices, embeddings)
        torch.testing.assert_close(fitted["weight"], weight, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(fitted["bias"], bias, atol=2e-6, rtol=2e-6)

    def test_structured_fsq_streaming_matches_full_sequence(self):
        model = ContentStudent(
            ContentStudentConfig(
                hidden=64,
                n_layers=2,
                n_heads=4,
                content_dim=16,
                structured_fsq=True,
                max_attention_context=8,
            )
        ).eval()
        model.load_fsq_projection(
            {"weight": torch.randn(16, 5), "bias": torch.randn(16)}
        )
        x = torch.randn(1, 80, 20)
        expected = model(x).content
        state = None
        chunks = []
        for start in range(0, x.shape[-1], 2):
            output, state = model.forward_stream(x[:, :, start : start + 2], state)
            chunks.append(output.content)
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1), expected, atol=2e-6, rtol=2e-6
        )

    def test_ctc_minimum_frames_accounts_for_repeated_labels(self):
        self.assertEqual(minimum_ctc_frames(torch.tensor([1, 1, 2, 2, 2])), 8)

    def test_curriculum_phase_schedule(self):
        config = CurriculumConfig(
            data_dir=Path("."),
            mel_dir=Path("."),
            audio_root=Path("."),
            transcript_root=Path("."),
            fsq_projection=Path("."),
            output_dir=Path("."),
            original_epochs=2,
            blend_epochs=2,
            teacher_epochs=2,
        )
        self.assertEqual(phase_weights(0, config)[0], "original")
        self.assertEqual(phase_weights(2, config)[0], "blend")
        self.assertEqual(phase_weights(4, config)[0], "teacher")
        validate_curriculum(config)

    def test_curriculum_rejects_odd_teacher_crop(self):
        config = CurriculumConfig(
            data_dir=Path("."),
            mel_dir=Path("."),
            audio_root=Path("."),
            transcript_root=Path("."),
            fsq_projection=Path("."),
            output_dir=Path("."),
            max_teacher_mel_frames=79,
        )
        with self.assertRaises(ValueError):
            validate_curriculum(config)

    def test_original_ctc_loss_backpropagates(self):
        model = ContentStudent(
            ContentStudentConfig(
                hidden=32,
                n_layers=1,
                n_heads=4,
                content_dim=16,
                structured_fsq=True,
                text_vocab_size=VOCAB_SIZE,
            )
        )
        batch = OriginalBatch(
            mel=torch.randn(1, 80, 12),
            input_lengths=torch.tensor([12]),
            transcripts=torch.tensor([1, 2, 3]),
            transcript_lengths=torch.tensor([3]),
        )
        loss = original_loss(
            model,
            batch,
            torch.device("cpu"),
            torch.nn.CTCLoss(blank=0, zero_infinity=True),
        )
        loss.backward()
        self.assertIsNotNone(model.text_head.weight.grad)
        self.assertGreater(model.text_head.weight.grad.abs().sum().item(), 0.0)

    def test_all_tiers_construct(self):
        for name in TIERS:
            tier = get_tier(name)
            model = ContentStudent(tier.model)
            self.assertEqual(model.config, tier.model)


if __name__ == "__main__":
    unittest.main()

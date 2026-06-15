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
    MioContentDataset,
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
from astrape.hybrid_training import (
    HybridTrainingConfig,
    hybrid_teacher_loss,
)
from astrape.flat_ctc_training import (
    FlatCtcTrainingConfig,
    flat_ctc_loss,
    speaker_balanced_subset,
    validate_flat_ctc_config,
)
from astrape.original_data import OriginalBatch, minimum_ctc_frames
from astrape.streaming_pipeline import OutputRingBuffer, StreamingVoiceConverter
from astrape.text import VOCAB_SIZE
from astrape.two_phase_training import (
    TwoPhaseTrainingConfig,
    phase2_source_schedule,
    phase_for_epoch,
    validate_two_phase_config,
)
from astrape.voicebank import (
    ASTRAPE_EMBEDDING_BYTES,
    ASTRAPE_EMBEDDING_DIM,
    MIN_REFERENCE_SECONDS,
    MIO_GLOBAL_MODEL,
    VOICEBANK_FORMAT_VERSION,
    VoiceBank,
    analyze_reference,
    detect_format,
    header_peek,
    open_embedding_mmap,
    parse_astrape_header,
)
from astrape.wave_decoder import (
    DirectWaveDecoder,
    WaveDecoderConfig,
    load_wave_decoder,
    save_wave_decoder_checkpoint,
)
from tiers import TIERS, get_tier
from train_wave_decoder import WaveDataset


def _deterministic_embedding() -> torch.Tensor:
    """128-dim reference embedding used by the astrape round-trip tests.

    A fixed-seed generator guarantees that ``torch.testing.assert_close(atol=0)``
    validates a lossless round trip rather than a tolerance-bound approximation.
    """
    generator = torch.Generator().manual_seed(1729)
    return torch.randn(128, generator=generator)


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

    def small_mio_causal_student(self) -> ContentStudent:
        return ContentStudent(
            ContentStudentConfig(
                architecture="mio_causal",
                hidden=64,
                n_layers=2,
                n_heads=4,
                content_dim=32,
                max_attention_context=8,
                mio_ff_hidden=128,
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

    def test_content_streaming_emits_first_frame_without_pair_buffer(self):
        model = self.small_student()
        output, state = model.forward_stream(torch.randn(1, 80, 1))
        self.assertEqual(output.content.shape[-1], 1)
        output, state = model.forward_stream(torch.randn(1, 80, 1), state)
        self.assertEqual(output.content.shape[-1], 0)
        output, _ = model.forward_stream(torch.randn(1, 80, 1), state)
        self.assertEqual(output.content.shape[-1], 1)

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

    def test_mio_causal_student_has_no_future_leakage(self):
        model = self.small_mio_causal_student()
        prefix = torch.randn(1, 80, 13)
        suffix = torch.randn(1, 80, 8)
        prefix_output = model(prefix).content
        full_output = model(torch.cat((prefix, suffix), dim=-1)).content
        torch.testing.assert_close(
            prefix_output,
            full_output[:, :, : prefix_output.shape[-1]],
            atol=2e-6,
            rtol=2e-6,
        )

    def test_mio_causal_student_streaming_matches_irregular_chunks(self):
        model = self.small_mio_causal_student()
        x = torch.randn(1, 80, 23)
        expected = model(x).content
        state = None
        chunks = []
        start = 0
        for length in (1, 4, 2, 7, 3, 6):
            output, state = model.forward_stream(
                x[:, :, start : start + length],
                state,
                flush=start + length == x.shape[-1],
            )
            chunks.append(output.content)
            start += length
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1),
            expected,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_two_phase_schedule_is_exact_and_reproducible(self):
        first = phase2_source_schedule(1000, 0.5, 44)
        second = phase2_source_schedule(1000, 0.5, 44)
        self.assertEqual(first, second)
        self.assertEqual(sum(first), 500)
        self.assertEqual(len(first) - sum(first), 500)

    def test_two_phase_boundaries(self):
        config = TwoPhaseTrainingConfig(
            data_dir=Path("cache"),
            audio_root=Path("audio"),
            transcript_root=Path("text"),
            output_dir=Path("checkpoints"),
            phase1_epochs=2,
            phase2_epochs=3,
        )
        validate_two_phase_config(config)
        self.assertEqual(phase_for_epoch(0, config), "teacher")
        self.assertEqual(phase_for_epoch(1, config), "teacher")
        self.assertEqual(phase_for_epoch(2, config), "teacher_original")
        self.assertEqual(phase_for_epoch(4, config), "teacher_original")
        with self.assertRaises(ValueError):
            phase_for_epoch(5, config)

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

    def small_wave_decoder(self) -> DirectWaveDecoder:
        return DirectWaveDecoder(
            WaveDecoderConfig(
                content_dim=8,
                condition_dim=4,
                sample_rate=600,
                content_rate=100,
                initial_channels=16,
                stage_channels=(12, 8),
                upsample_factors=(2, 3),
                mrf_kernel_sizes=(3,),
                mrf_dilations=((1, 2),),
                output_kernel_size=3,
            )
        ).eval()

    def test_wave_decoder_output_length_and_no_future_leakage(self):
        decoder = self.small_wave_decoder()
        prefix = torch.randn(1, 3, 8)
        suffix = torch.randn(1, 2, 8)
        global_embedding = torch.randn(1, 4)
        prefix_audio = decoder(prefix, global_embedding)
        full_audio = decoder(
            torch.cat((prefix, suffix), dim=1),
            global_embedding,
        )
        self.assertEqual(prefix_audio.shape[-1], 3 * 6)
        torch.testing.assert_close(
            prefix_audio,
            full_audio[:, : prefix_audio.shape[-1]],
            atol=2e-6,
            rtol=2e-6,
        )

    def test_wave_decoder_streaming_matches_irregular_chunks(self):
        decoder = self.small_wave_decoder()
        content = torch.randn(1, 7, 8)
        global_embedding = torch.randn(1, 4)
        expected = decoder(content, global_embedding)
        state = None
        chunks = []
        for start, length in ((0, 1), (1, 3), (4, 2), (6, 1)):
            output, state = decoder.forward_stream(
                content[:, start : start + length],
                global_embedding,
                state,
            )
            chunks.append(output)
        actual = torch.cat(chunks, dim=-1)
        self.assertEqual(state.content_frames, content.shape[1])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_wave_decoder_checkpoint_roundtrip(self):
        decoder = self.small_wave_decoder()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wave.pt"
            save_wave_decoder_checkpoint(
                path,
                decoder,
                step=12,
                metrics={"loss": 0.5},
            )
            loaded = load_wave_decoder(path)
            self.assertEqual(loaded.config, decoder.config)
            content = torch.randn(1, 2, 8)
            condition = torch.randn(1, 4)
            torch.testing.assert_close(
                loaded(content, condition),
                decoder(content, condition),
            )

    def test_voicebank_requires_one_reference_of_at_least_five_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voicebank.npz"
            bank = VoiceBank(
                global_embedding=torch.randn(128),
                duration_seconds=MIN_REFERENCE_SECONDS,
                source_sample_rate=44100,
                source_path="/tmp/reference.wav",
            )
            bank.save(path)
            loaded = VoiceBank.load(path)
            torch.testing.assert_close(
                loaded.global_embedding,
                bank.global_embedding,
            )
            self.assertEqual(loaded.duration_seconds, MIN_REFERENCE_SECONDS)
            with self.assertRaises(ValueError):
                VoiceBank(
                    global_embedding=torch.randn(128),
                    duration_seconds=MIN_REFERENCE_SECONDS - 0.01,
                    source_sample_rate=44100,
                    source_path="/tmp/short.wav",
                ).validate()

    def test_voicebank_v2_metadata_and_v1_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            versioned = directory / "v2.npz"
            legacy = directory / "v1.npz"
            bank = VoiceBank(
                global_embedding=torch.randn(128),
                duration_seconds=6.0,
                source_sample_rate=44100,
                source_path="/tmp/reference.wav",
                reference_sha256="abc",
                created_utc="2026-06-14T00:00:00+00:00",
                peak_amplitude=0.8,
                rms_dbfs=-20.0,
                clipping_fraction=0.0,
                active_speech_ratio=0.9,
                dc_offset=0.001,
                quality_warnings=("test_warning",),
            )
            bank.save(versioned)
            loaded = VoiceBank.load(versioned)
            self.assertEqual(loaded.reference_sha256, "abc")
            self.assertEqual(loaded.quality_warnings, ("test_warning",))
            np.savez_compressed(
                legacy,
                format_version=np.asarray(1, dtype=np.int64),
                global_embedding=bank.global_embedding.numpy(),
                duration_seconds=np.asarray(6.0, dtype=np.float32),
                source_sample_rate=np.asarray(44100, dtype=np.int64),
                source_path=np.asarray("/tmp/legacy.wav"),
            )
            migrated = VoiceBank.load(legacy)
            self.assertEqual(migrated.embedding_model, MIO_GLOBAL_MODEL)
            self.assertTrue(np.isnan(migrated.rms_dbfs))

    def test_voicebank_v3_astrape_round_trip_is_lossless(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "profile.astrape"
            original = VoiceBank(
                global_embedding=_deterministic_embedding(),
                duration_seconds=MIN_REFERENCE_SECONDS + 0.5,
                source_sample_rate=44100,
                source_path="/tmp/profile-reference.wav",
                reference_sha256="a" * 64,
                created_utc="2026-06-15T00:00:00+00:00",
                peak_amplitude=0.85,
                rms_dbfs=-18.0,
                clipping_fraction=0.0,
                active_speech_ratio=0.91,
                dc_offset=-0.0008,
                quality_warnings=("reference_too_loud",),
            )
            original.save(path)
            self.assertEqual(detect_format(path), "astrape")
            header = parse_astrape_header(path.read_bytes()[:48])
            self.assertEqual(header["embedding_length"], ASTRAPE_EMBEDDING_BYTES)
            self.assertEqual(header["embedding_length"] // 4, ASTRAPE_EMBEDDING_DIM)
            loaded = VoiceBank.load(path)
            torch.testing.assert_close(
                loaded.global_embedding, original.global_embedding, atol=0, rtol=0,
            )
            self.assertEqual(loaded.reference_sha256, original.reference_sha256)
            self.assertEqual(loaded.created_utc, original.created_utc)
            self.assertEqual(loaded.quality_warnings, original.quality_warnings)
            self.assertAlmostEqual(loaded.peak_amplitude, original.peak_amplitude)
            self.assertAlmostEqual(loaded.rms_dbfs, original.rms_dbfs)
            self.assertAlmostEqual(loaded.dc_offset, original.dc_offset)

    def test_astrape_header_peek_is_lossless_metadata_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "profile.astrape"
            bank = VoiceBank(
                global_embedding=_deterministic_embedding(),
                duration_seconds=6.0,
                source_sample_rate=44100,
                source_path="/tmp/x.wav",
            )
            bank.save(path)
            peek = header_peek(path)
            self.assertEqual(peek["format"], "astrape")
            self.assertEqual(peek["version"], VOICEBANK_FORMAT_VERSION)
            self.assertEqual(peek["embedding_length_bytes"], ASTRAPE_EMBEDDING_BYTES)
            self.assertEqual(peek["embedding_dim"], ASTRAPE_EMBEDDING_DIM)
            self.assertGreater(peek["file_size_bytes"], 0)

    def test_astrape_embedding_mmap_is_zero_copy_view(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "profile.astrape"
            torch.manual_seed(123)
            original = VoiceBank(
                global_embedding=torch.randn(128),
                duration_seconds=6.0,
                source_sample_rate=44100,
                source_path="/tmp/x.wav",
            )
            original.save(path)
            handle = open_embedding_mmap(path)
            try:
                self.assertEqual(handle.array.shape, (ASTRAPE_EMBEDDING_DIM,))
                self.assertEqual(handle.array.dtype, np.float32)
                np.testing.assert_array_equal(
                    handle.array,
                    original.global_embedding.numpy().astype(np.float32),
                )
            finally:
                handle.close()

    def test_astrape_force_format_writes_legacy_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            # The legacy `.npz` writer appends `.npz` to the file name when the
            # supplied suffix is not already `.npz`; ask explicitly for one.
            npz_path = directory / "profile.npz"
            bank = VoiceBank(
                global_embedding=_deterministic_embedding(),
                duration_seconds=6.0,
                source_sample_rate=44100,
                source_path="/tmp/x.wav",
            )
            bank.save(npz_path, force_format="npz")
            self.assertTrue(npz_path.exists())
            self.assertEqual(detect_format(npz_path), "npz")
            round_trip = VoiceBank.load(npz_path)
            torch.testing.assert_close(
                round_trip.global_embedding, bank.global_embedding, atol=0, rtol=0,
            )

    def test_astrape_v2_npz_still_loads_losslessly(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            v2_path = directory / "legacy.npz"
            np.savez_compressed(
                v2_path,
                format_version=np.asarray(2, dtype=np.int64),
                global_embedding=_deterministic_embedding().numpy(),
                duration_seconds=np.asarray(7.0, dtype=np.float32),
                source_sample_rate=np.asarray(44100, dtype=np.int64),
                source_path=np.asarray("/tmp/x.wav"),
                embedding_model=np.asarray("Aratako/MioCodec-25Hz-44.1kHz-v2"),
                reference_sha256=np.asarray("b" * 64),
                created_utc=np.asarray("2026-06-15T00:00:00+00:00"),
                peak_amplitude=np.asarray(0.7, dtype=np.float32),
                rms_dbfs=np.asarray(-18.0, dtype=np.float32),
                clipping_fraction=np.asarray(0.0, dtype=np.float32),
                active_speech_ratio=np.asarray(0.92, dtype=np.float32),
                dc_offset=np.asarray(0.001, dtype=np.float32),
                quality_warnings=np.asarray(["a"], dtype="<U"),
            )
            self.assertEqual(detect_format(v2_path), "npz")
            loaded = VoiceBank.load(v2_path)
            torch.testing.assert_close(
                loaded.global_embedding, _deterministic_embedding(), atol=0, rtol=0,
            )
            self.assertEqual(loaded.reference_sha256, "b" * 64)
            self.assertEqual(loaded.quality_warnings, ("a",))
            peek = header_peek(v2_path)
            self.assertEqual(peek["format"], "npz")
            self.assertEqual(peek["version"], 2)
            self.assertEqual(peek["embedding_dim"], ASTRAPE_EMBEDDING_DIM)

    def test_astrape_mmap_rejects_bad_magic(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bad_path = directory / "broken.astrape"
            bad_path.write_bytes(b"NOPE" + b"\x00" * 60)
            with self.assertRaises(ValueError):
                parse_astrape_header(bad_path.read_bytes()[:48])

    def test_astrape_detect_format_writes_in_default_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            astrape_path = directory / "auto.astrape"
            bank = VoiceBank(
                global_embedding=_deterministic_embedding(),
                duration_seconds=6.0,
                source_sample_rate=44100,
                source_path="/tmp/x.wav",
            )
            bank.save(astrape_path)
            self.assertEqual(detect_format(astrape_path), "astrape")


    def test_reference_quality_detects_clipping_and_low_activity(self):
        audio = np.zeros(16000 * 6, dtype=np.float32)
        audio[:12] = 1.0
        quality = analyze_reference(audio, 16000)
        self.assertIn("clipping_detected", quality.warnings)
        self.assertIn("reference_too_quiet", quality.warnings)
        self.assertIn("low_active_speech_ratio", quality.warnings)

    def test_e2e_streaming_pipeline_matches_full_models(self):
        content_model = ContentStudent(
            ContentStudentConfig(
                hidden=32,
                n_layers=2,
                n_heads=4,
                content_dim=8,
            )
        ).eval()
        wave_model = DirectWaveDecoder(
            WaveDecoderConfig(
                content_dim=8,
                condition_dim=4,
                sample_rate=150,
                content_rate=25,
                initial_channels=16,
                stage_channels=(12, 8),
                upsample_factors=(2, 3),
                mrf_kernel_sizes=(3,),
                mrf_dilations=((1, 2),),
                output_kernel_size=3,
            )
        ).eval()
        condition = torch.randn(4)
        frontend = StreamingLogMel()
        waveform = torch.randn(6031)
        expected_mel = frontend(waveform)
        expected_content = content_model(expected_mel).content
        expected = wave_model(
            expected_content.transpose(1, 2),
            condition.unsqueeze(0),
        )
        pipeline = StreamingVoiceConverter(
            content_model,
            wave_model,
            condition,
        )
        output_chunks = []
        start = 0
        chunk_sizes = (157, 641, 83, 1000, 319)
        chunk_index = 0
        while start < waveform.numel():
            size = chunk_sizes[chunk_index % len(chunk_sizes)]
            chunk = pipeline.process(waveform[start : start + size])
            if chunk.output_samples:
                output_chunks.append(chunk.audio)
            start += size
            chunk_index += 1
        final = pipeline.flush()
        if final.output_samples:
            output_chunks.append(final.audio)
        actual = torch.cat(output_chunks, dim=-1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertEqual(
            pipeline.counters.output_samples,
            expected.shape[-1],
        )
        with self.assertRaises(RuntimeError):
            pipeline.process(torch.zeros(1))

    def test_output_ring_buffer_tracks_underruns(self):
        buffer = OutputRingBuffer(capacity_samples=4)
        buffer.write(torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(
            buffer.read(2),
            torch.tensor([1.0, 2.0]),
        )
        torch.testing.assert_close(
            buffer.read(3),
            torch.tensor([3.0, 0.0, 0.0]),
        )
        self.assertEqual(buffer.buffered_samples, 0)
        self.assertEqual(buffer.underrun_samples, 2)
        buffer.write(torch.tensor([1.0, 2.0, 3.0]))
        buffer.write(torch.tensor([4.0, 5.0, 6.0]))
        torch.testing.assert_close(
            buffer.read(4),
            torch.tensor([3.0, 4.0, 5.0, 6.0]),
        )
        self.assertEqual(buffer.overrun_samples, 2)

    def test_wave_validation_crop_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            np.savez(
                data_dir / "s_00000.npz",
                ce_768=np.arange(12 * 768, dtype=np.float32).reshape(12, 768),
                ge_128=np.zeros(128, dtype=np.float32),
                audio=np.arange(12 * 1764, dtype=np.float32),
            )
            dataset = WaveDataset(
                data_dir,
                np.asarray([0]),
                crop_frames=4,
                target_dir=None,
                seed=123,
                random_crops=False,
            )
            first = dataset[0]
            second = dataset[0]
            torch.testing.assert_close(first.content, second.content)
            torch.testing.assert_close(first.waveform, second.waveform)

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

    def test_crop_with_history_masks_warmup_frames(self):
        mel = torch.arange(32).repeat(80, 1).float()
        content = torch.arange(16).unsqueeze(1).repeat(1, 4).float()
        sample = ContentSample(mel, content, content.clone(), None, "p001", 0)

        class MiddleChoice(random.Random):
            def choice(self, sequence):
                return sequence[len(sequence) // 2]

        cropped = crop_aligned(sample, 8, MiddleChoice(), history_mel_frames=8)
        self.assertGreater(cropped.supervision_start, 0)
        self.assertEqual(cropped.mel.shape[1], 16)
        self.assertEqual(cropped.content.shape[0], 8)

    def test_compact_content_cache_loads_mel_from_target_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(
                root / "meta.npz",
                spk_names=np.asarray(["p001"]),
                n_samples=np.asarray(1),
            )
            np.savez(
                root / "s_00000.npz",
                logmel=np.zeros((80, 12), dtype=np.float16),
                ce_768=np.zeros((6, 16), dtype=np.float16),
                pre_fsq_768=np.zeros((6, 16), dtype=np.float16),
                ct=np.zeros(6, dtype=np.uint16),
                transcript=np.asarray([1, 2, 3], dtype=np.uint8),
            )
            sample = MioContentDataset(root, root)[0]
            self.assertEqual(tuple(sample.mel.shape), (80, 12))
            self.assertEqual(tuple(sample.content.shape), (6, 16))
            self.assertEqual(sample.token_indices.dtype, torch.long)
            self.assertTrue(torch.equal(sample.transcript, torch.tensor([1, 2, 3])))

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

    def test_mio_causal_checkpoint_roundtrip(self):
        model = self.small_mio_causal_student()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mio-causal.pt"
            save_checkpoint(path, model, epoch=2, metrics={"val_cosine": 0.4})
            loaded, metadata = load_content_checkpoint(path)
            self.assertEqual(metadata["epoch"], 2)
            self.assertEqual(loaded.config, model.config)
            self.assertEqual(
                loaded(torch.randn(1, 80, 9)).content.shape,
                model(torch.randn(1, 80, 9)).content.shape,
            )

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

    def test_hybrid_content_keeps_continuous_and_fsq_outputs(self):
        config = ContentStudentConfig(
            hidden=64,
            n_layers=2,
            n_heads=4,
            content_dim=16,
            auxiliary_prefsq=True,
            structured_fsq=True,
            hybrid_content=True,
        )
        model = ContentStudent(config)
        model.load_fsq_projection(
            {"weight": torch.randn(16, 5), "bias": torch.randn(16)}
        )
        output = model(torch.randn(2, 80, 20))
        self.assertEqual(tuple(output.content.shape), (2, 16, 10))
        self.assertEqual(tuple(output.soft_fsq_content.shape), (2, 16, 10))
        self.assertEqual(tuple(output.hard_content.shape), (2, 16, 10))
        self.assertEqual(tuple(output.soft_fsq_codes.shape), (2, 10, 5))

    def test_hybrid_teacher_loss_updates_both_heads(self):
        config = ContentStudentConfig(
            hidden=32,
            n_layers=1,
            n_heads=4,
            content_dim=16,
            auxiliary_prefsq=True,
            structured_fsq=True,
            hybrid_content=True,
        )
        model = ContentStudent(config)
        model.load_fsq_projection(
            {"weight": torch.randn(16, 5), "bias": torch.randn(16)}
        )
        from astrape.data import ContentBatch

        batch = ContentBatch(
            mel=torch.randn(2, 80, 12),
            content=torch.randn(2, 6, 16),
            pre_fsq=torch.randn(2, 6, 16),
            token_indices=torch.randint(0, 12800, (2, 6)),
            input_lengths=torch.tensor([12, 12]),
            target_lengths=torch.tensor([6, 6]),
            target_mask=torch.ones(2, 6, dtype=torch.bool),
        )
        training = HybridTrainingConfig(
            data_dir=Path("."),
            mel_dir=Path("."),
            fsq_projection=Path("."),
            output_dir=Path("."),
        )
        loss, metrics = hybrid_teacher_loss(model, batch, training)
        loss.backward()
        self.assertGreater(model.content_head.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.fsq_head.weight.grad.abs().sum().item(), 0.0)
        self.assertIn("direct_cosine", metrics)

    def test_flat_ctc_loss_updates_content_and_text_heads(self):
        model = ContentStudent(
            ContentStudentConfig(
                hidden=32,
                n_layers=1,
                n_heads=4,
                content_dim=16,
                text_vocab_size=VOCAB_SIZE,
            )
        )
        from astrape.data import ContentBatch

        batch = ContentBatch(
            mel=torch.randn(2, 80, 12),
            content=torch.randn(2, 6, 16),
            pre_fsq=None,
            token_indices=None,
            input_lengths=torch.tensor([12, 12]),
            target_lengths=torch.tensor([6, 6]),
            target_mask=torch.ones(2, 6, dtype=torch.bool),
            transcripts=torch.tensor([1, 2, 3, 2, 3, 4]),
            transcript_lengths=torch.tensor([3, 3]),
        )
        config = FlatCtcTrainingConfig(
            data_dir=Path("."),
            output_dir=Path("."),
        )
        loss, metrics = flat_ctc_loss(
            model,
            batch,
            config,
            torch.nn.CTCLoss(blank=0, zero_infinity=True),
        )
        loss.backward()
        self.assertGreater(model.content_head.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.text_head.weight.grad.abs().sum().item(), 0.0)
        self.assertIn("ctc_loss", metrics)

    def test_flat_ctc_probe_is_deterministic_and_speaker_balanced(self):
        speakers = np.array(["a"] * 5 + ["b"] * 5 + ["c"] * 5)
        indices = np.arange(len(speakers))
        first = speaker_balanced_subset(indices, speakers, 6, 42)
        second = speaker_balanced_subset(indices, speakers, 6, 42)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(set(speakers[first]), {"a", "b", "c"})
        self.assertEqual(len(first), 6)

    def test_flat_ctc_rejects_conflicting_checkpoint_modes(self):
        config = FlatCtcTrainingConfig(
            data_dir=Path("."),
            output_dir=Path("."),
            resume=Path("resume.pt"),
            init_checkpoint=Path("init.pt"),
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_flat_ctc_config(config)

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

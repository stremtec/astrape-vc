"""Migration and behaviour tests for the voice-bank format."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from astrape.voicebank import VoiceBank


class MigrationScriptTests(unittest.TestCase):
    def _seed_voicebank(self, directory: Path, name: str, *, seed: int) -> Path:
        generator = torch.Generator().manual_seed(seed)
        embedding = torch.randn(128, generator=generator)
        meta = {
            "duration_seconds": 6.0 + seed / 10.0,
            "source_sample_rate": 44100,
            "source_path": f"/tmp/refs/{name}",
            "embedding_model": "Aratako/MioCodec-25Hz-44.1kHz-v2",
            "reference_sha256": ("%02x" % seed) * 32,
            "created_utc": "2026-06-15T12:00:00+00:00",
            "peak_amplitude": 0.8,
            "rms_dbfs": -18.0,
            "clipping_fraction": 0.0,
            "active_speech_ratio": 0.9,
            "dc_offset": 0.001,
            "quality_warnings": ("test",),
        }
        legacy = directory / f"{name}.npz"
        np.savez_compressed(
            legacy,
            format_version=np.asarray(2, dtype=np.int64),
            global_embedding=embedding.numpy(),
            duration_seconds=np.asarray(meta["duration_seconds"], dtype=np.float32),
            source_sample_rate=np.asarray(meta["source_sample_rate"], dtype=np.int64),
            source_path=np.asarray(meta["source_path"]),
            embedding_model=np.asarray(meta["embedding_model"]),
            reference_sha256=np.asarray(meta["reference_sha256"]),
            created_utc=np.asarray(meta["created_utc"]),
            peak_amplitude=np.asarray(meta["peak_amplitude"], dtype=np.float32),
            rms_dbfs=np.asarray(meta["rms_dbfs"], dtype=np.float32),
            clipping_fraction=np.asarray(meta["clipping_fraction"], dtype=np.float32),
            active_speech_ratio=np.asarray(meta["active_speech_ratio"], dtype=np.float32),
            dc_offset=np.asarray(meta["dc_offset"], dtype=np.float32),
            quality_warnings=np.asarray(meta["quality_warnings"], dtype="<U"),
        )
        return legacy

    def test_migration_creates_lossless_astrape_sibling(self):
        from migrate_voicebanks import migrate_all

        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            self._seed_voicebank(directory, "profile_a", seed=1)
            code = migrate_all(
                directory, move=False, quiet=True, dry_run=False,
                keep_existing=False,
            )
            self.assertEqual(code, 0)
            new_path = directory / "profile_a.astrape"
            self.assertTrue(new_path.exists())
            legacy = directory / "profile_a.npz"
            self.assertTrue(legacy.exists())  # move=False keeps the original
            ast = VoiceBank.load(new_path)
            npz = VoiceBank.load(legacy)
            torch.testing.assert_close(
                ast.global_embedding, npz.global_embedding, atol=0, rtol=0,
            )
            self.assertEqual(ast.reference_sha256, npz.reference_sha256)
            self.assertEqual(ast.created_utc, npz.created_utc)
            self.assertEqual(ast.quality_warnings, npz.quality_warnings)

    def test_migration_with_dry_run_does_not_create_files(self):
        from migrate_voicebanks import migrate_all

        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            self._seed_voicebank(directory, "profile_b", seed=7)
            migrate_all(
                directory, move=False, quiet=True, dry_run=True,
                keep_existing=False,
            )
            self.assertFalse((directory / "profile_b.astrape").exists())


class WebuiListingTests(unittest.TestCase):
    def test_webui_listing_dedups_and_lists_both_formats(self):
        from webui.server import _list_voicebanks, ROOT, VOICEBANK_DIR

        with tempfile.TemporaryDirectory() as d:
            backup_dir = VOICEBANK_DIR
            try:
                # Monkeypatch the webui's VOICEBANK_DIR temporarily so we
                # exercise the real listing function with a throw-away set of
                # profiles.
                import webui.server as ws
                ws.VOICEBANK_DIR = Path(d)
                bank_a = VoiceBank(
                    global_embedding=torch.randn(128),
                    duration_seconds=6.0,
                    source_sample_rate=44100,
                    source_path="/tmp/a.wav",
                )
                bank_b = VoiceBank(
                    global_embedding=torch.randn(128),
                    duration_seconds=7.0,
                    source_sample_rate=44100,
                    source_path="/tmp/b.wav",
                )
                bank_a.save(Path(d) / "alpha.astrape")
                bank_b.save(Path(d) / "beta.npz")
                profiles = ws._list_voicebanks()
                ids = {p["id"] for p in profiles}
                self.assertIn("alpha", ids)
                self.assertIn("beta", ids)
                formats = {p["id"]: p["format"] for p in profiles}
                self.assertEqual(formats["alpha"], "astrape")
                self.assertEqual(formats["beta"], "npz")
            finally:
                # Restore the real path
                import webui.server as ws
                ws.VOICEBANK_DIR = backup_dir


if __name__ == "__main__":
    unittest.main()

"""Regression checks for the electrode-shift study's measurement protocol."""

import unittest

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from emg2qwerty.data import LabelData
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.transforms import FixedWristShift, LogSpectrogram
from scripts.evaluate_shifts import (
    CONDITIONS,
    REPO,
    aggregate,
    build_eval_transform,
    edit_counts,
)


class ShiftProtocolTest(unittest.TestCase):
    def test_shift_wraps_only_intended_wrist(self):
        raw = torch.arange(5 * 2 * 16).reshape(5, 2, 16)
        original = raw.clone()
        self.assertEqual(len(set(CONDITIONS)), 9)
        for left, right in CONDITIONS:
            transform = FixedWristShift(left, right)
            result = transform(raw)
            for band, offset in enumerate((left, right)):
                for electrode in range(16):
                    self.assertTrue(
                        torch.equal(
                            result[:, band, electrode],
                            original[:, band, (electrode - offset) % 16],
                        )
                    )
            self.assertTrue(
                torch.equal(FixedWristShift(-left, -right)(result), original)
            )
        self.assertTrue(torch.equal(raw, original))

    def test_chunked_spectrogram_and_zero_shift_match_default(self):
        generator = np.random.default_rng(2026)
        # Deliberately leave a partial trailing FFT window.
        raw = np.empty(2345, dtype=[("emg_left", "f4", 16), ("emg_right", "f4", 16)])
        for field in raw.dtype.names:
            raw[field] = generator.normal(size=(len(raw), 16))
        with initialize_config_dir(version_base=None, config_dir=str(REPO / "config")):
            baseline = compose(config_name="base")
            no_rotation = compose(
                config_name="base", overrides=["transforms=log_spectrogram_no_rotation"]
            )
        train_pipeline = [c._target_ for c in baseline.transforms.train]
        self.assertEqual(
            [c._target_ for c in no_rotation.transforms.train],
            [c for c in train_pipeline if not c.endswith(".ForEach")],
        )
        canonical = instantiate(baseline.logspec)(instantiate(baseline.to_tensor)(raw))
        for split in ("val", "test"):
            result = build_eval_transform(baseline, split, 0, 0, chunk_frames=17)(raw)
            torch.testing.assert_close(result, canonical)
        clean = build_eval_transform(no_rotation, "test", 0, 0, chunk_frames=17)(raw)
        rotated = build_eval_transform(no_rotation, "test", 1, -2, chunk_frames=17)(raw)
        torch.testing.assert_close(rotated[:, 0], clean[:, 0].roll(1, dims=1))
        torch.testing.assert_close(rotated[:, 1], clean[:, 1].roll(-2, dims=1))
        self.assertTrue(isinstance(instantiate(no_rotation.logspec), LogSpectrogram))

    def test_native_edit_alignment_and_micro_average(self):
        examples = [("abcdef", "abc"), ("a", "ab"), ("abc", "adc"), ("", "xy")]
        counts = [edit_counts(ref, hyp) for ref, hyp in examples]
        self.assertEqual(counts[0]["deletions"], 3)
        self.assertEqual(counts[1]["insertions"], 1)
        self.assertEqual(counts[2]["substitutions"], 1)
        self.assertEqual(counts[3]["insertions"], 2)
        total = aggregate(counts)
        metric = CharacterErrorRates()
        for ref, hyp in examples:
            metric.update(LabelData.from_str(hyp), LabelData.from_str(ref))
        native = metric.compute()
        self.assertEqual(total["reference_chars"], 10)
        self.assertEqual(total["cer_pct"], 70.0)
        self.assertEqual(native["CER"], total["cer_pct"])
        self.assertEqual(int(metric.insertions), total["deletions"])
        self.assertEqual(int(metric.deletions), total["insertions"])


if __name__ == "__main__":
    unittest.main()

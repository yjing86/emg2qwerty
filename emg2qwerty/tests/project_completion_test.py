"""Verify interruption and restart behavior without training or accessing records."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf

from emg2qwerty import train as native
from emg2qwerty.fit_progress import IncompleteFitError, checkpoint_completion
from scripts import run_study


class FitCompletionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = Path(tmp.name) / "seed-0" / "E1"
        self.ckpts = self.run / "checkpoints"
        self.ckpts.mkdir(parents=True)
        self.last = self.ckpts / "last.ckpt"
        self.best = self.ckpts / "epoch=0-step=6.ckpt"
        self.best.write_bytes(b"selected checkpoint fixture")
        self.args = SimpleNamespace(seed=0, batch_size=16, num_workers=2,
                                    accelerator="gpu", epochs=4, data_root=Path(tmp.name),
                                    dry_run=False)
        opts = run_study.training_overrides(self.args, "E1", self.run)
        self.config = OmegaConf.create(run_study.effective_config(opts))
        self.config.trainer.default_root_dir = str(self.run)
        self.summary = {"best_checkpoint": str(self.best), "val_metrics": [{"val/CER": 22}]}

    def save_progress(self, epoch_count, is_last=True):
        batch = {"is_last_batch": is_last, "current": {"ready": 6, "completed": 6}}
        steps = epoch_count * 6
        fit = {
            "epoch_progress": {"current": {"processed": epoch_count, "completed": epoch_count-1}},
            "epoch_loop.batch_progress": batch,
            "epoch_loop.val_loop.epoch_loop.batch_progress": batch,
            "epoch_loop.batch_loop.optimizer_loop.optim_progress": {"optimizer": {
                "step": {"total": {"ready": steps, "completed": steps}, "current": {"completed": 6}}
            }},
        }
        torch.save({
            "epoch": epoch_count - 1, "global_step": steps, "loops": {"fit_loop": fit},
            "callbacks": {"ModelCheckpoint{}": {"monitor": "val/CER", "best_model_path": str(self.best)}},
        }, self.last)

    def save_legacy_marker(self):
        d = self.run / "hydra_configs"
        d.mkdir()
        opts = run_study.training_overrides(self.args, "E1", self.run)
        OmegaConf.save(OmegaConf.create(run_study.effective_config(opts)), d / "config.yaml")
        run_study.write_json(self.run / "run_summary.json", self.summary)

    def test_checkpoint_at_epoch_end_and_middle(self):
        self.save_progress(4)
        self.assertEqual(checkpoint_completion(self.last, 4)["global_step"], 24)
        self.save_progress(4, is_last=False)
        with self.assertRaises(IncompleteFitError):
            checkpoint_completion(self.last, 4)
        self.save_progress(2)
        with self.assertRaises(IncompleteFitError):
            checkpoint_completion(self.last, 4)

    def test_swallowed_interrupt_never_validates_or_writes_summary(self):
        trainer = Mock(interrupted=True, max_epochs=4, max_steps=-1)
        trainer.checkpoint_callback.dirpath = self.ckpts
        with patch.object(native.HydraConfig, "get", return_value=SimpleNamespace(runtime=SimpleNamespace(output_dir=self.run))), \
             patch.object(native, "get_original_cwd", return_value=str(self.run)), \
             patch.object(native, "instantiate", return_value=Mock()), \
             patch.object(native.pl, "Trainer", return_value=trainer), \
             patch.object(native.pl, "seed_everything"):
            with self.assertRaises(IncompleteFitError):
                native.main.__wrapped__(self.config)
        trainer.fit.assert_called_once()
        trainer.validate.assert_not_called()
        self.assertFalse((self.run / "run_summary.json").exists())

    def test_legacy_incomplete_marker_archived_before_resume(self):
        self.save_progress(2)
        self.save_legacy_marker()
        old_last = self.last.read_bytes()
        with self.assertRaises(IncompleteFitError):
            run_study.artifact_paths(self.run)
        with patch.object(run_study, "launch", side_effect=InterruptedError("mocked launch")) as launch:
            with self.assertRaises(InterruptedError):
                run_study.train(self.args, "E1", self.run)
        launch.assert_called_once()
        self.assertFalse((self.run / "run_summary.json").exists())
        (archived,) = self.run.glob("run_summary.incomplete-*.json")
        self.assertEqual(json.loads(archived.read_text()), self.summary)
        self.assertEqual(self.last.read_bytes(), old_last)
        self.assertTrue(self.best.is_file())

    def test_complete_legacy_marker_skips_training(self):
        self.save_progress(4)
        self.save_legacy_marker()
        with patch.object(run_study, "launch") as launch:
            run_study.train(self.args, "E1", self.run)
        launch.assert_not_called()

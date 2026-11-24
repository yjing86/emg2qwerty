# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pprint
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd, instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.fit_progress import IncompleteFitError, checkpoint_completion
from emg2qwerty.transforms import Transform


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)

    # Helper to instantiate full paths for dataset sessions
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    # Helper to instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    # Instantiate LightningModule
    log.info(f"Instantiating LightningModule {config.module}")
    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    if config.checkpoint is not None:
        log.info(f"Loading module from checkpoint {config.checkpoint}")
        module = module.load_from_checkpoint(
            config.checkpoint,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
            decoder=config.decoder,
        )

    # Instantiate LightningDataModule
    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]

    # Initialize trainer
    trainer_settings = dict(config.trainer)
    if isinstance(trainer_settings.get("logger"), DictConfig):
        trainer_settings["logger"] = instantiate(trainer_settings["logger"])
    trainer = pl.Trainer(**trainer_settings, callbacks=callbacks)

    best_checkpoint = config.checkpoint
    fit_seconds = 0.0
    progress = None
    if config.train:
        checkpoint_dir = Path(trainer.checkpoint_callback.dirpath)
        last = checkpoint_dir / "last.ckpt"
        expected_epochs = trainer.max_epochs
        verify_study = (
            not config.evaluate_test and expected_epochs is not None and expected_epochs > 0
            and trainer.max_steps == -1 and config.monitor_metric == "val/CER"
        )
        # If an earlier process finished fitting but stopped in post-fit validation,
        # recover the selected model without replaying the last epoch.
        if verify_study and last.is_file():
            try:
                progress = checkpoint_completion(last, expected_epochs)
            except IncompleteFitError:
                pass
        if progress is None:
            resume = last if last.is_file() else utils.get_last_checkpoint(checkpoint_dir)
            if resume is not None:
                log.info(f"Resuming training from checkpoint {resume}")
            fit_start = time.monotonic()
            trainer.fit(module, datamodule, ckpt_path=resume)
            fit_seconds = time.monotonic() - fit_start
            # Lightning 1.8 catches Ctrl+C and returns normally from fit().
            # A subsequent validate() resets trainer.state and conceals that event.
            if trainer.interrupted:
                raise IncompleteFitError("Training interrupted; the latest checkpoint is available for resume.")
            if verify_study:
                progress = checkpoint_completion(last, expected_epochs)
        best_checkpoint = (
            str(checkpoint_dir / progress["best_checkpoint_name"]) if progress is not None
            else trainer.checkpoint_callback.best_model_path
        )
        module = module.load_from_checkpoint(best_checkpoint)

    val_metrics = trainer.validate(module, datamodule)
    if trainer.interrupted or not val_metrics:
        raise IncompleteFitError("Post-fit validation was interrupted or produced no metrics.")
    results = {
        "val_metrics": val_metrics,
        "best_checkpoint": best_checkpoint,
        "parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
        "fit_seconds_this_invocation": fit_seconds,
    }
    if progress is not None:
        results["fit_completion"] = progress
    if config.evaluate_test:
        results["test_metrics"] = trainer.test(module, datamodule)
        if trainer.interrupted or not results["test_metrics"]:
            raise IncompleteFitError("Test evaluation was interrupted or produced no metrics.")
    summary = output_dir / "run_summary.json"
    temp = summary.with_suffix(".json.tmp")
    temp.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    temp.replace(summary)
    pprint.pprint(results, sort_dicts=False)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()

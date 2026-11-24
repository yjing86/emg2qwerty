# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    CausalGRUEncoder,
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = ConcatDataset(
                [
                    WindowedEMGDataset(
                        hdf5_path,
                        transform=self.train_transform,
                        window_length=self.window_length,
                        padding=self.padding,
                        jitter=True,
                    )
                    for hdf5_path in self.train_sessions
                ]
            )
        if stage in (None, "fit", "validate"):
            self.val_dataset = ConcatDataset(
                [
                    WindowedEMGDataset(
                        hdf5_path,
                        transform=self.val_transform,
                        window_length=self.window_length,
                        padding=self.padding,
                        jitter=False,
                    )
                    for hdf5_path in self.val_sessions
                ]
            )
        if stage in (None, "test"):
            self.test_dataset = ConcatDataset(
                [
                    WindowedEMGDataset(
                        hdf5_path,
                        transform=self.test_transform,
                        # Feed the entire session at once without windowing/padding
                        # at test time for more realism
                        window_length=None,
                        padding=(0, 0),
                        jitter=False,
                    )
                    for hdf5_path in self.test_sessions
                ]
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class _WristCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def _frontend(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        mlp_offsets: Sequence[int],
    ) -> list[nn.Module]:
        return [
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                offsets=mlp_offsets,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        ]

    def _setup_ctc(self, decoder: DictConfig) -> None:
        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Both encoders preserve frame stride after their initial context.
        # Subtract the measured reduction from each unpadded sequence length.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class TDSConvCTCModule(_WristCTCModule):
    """Course TDS baseline, optionally using only the unshifted wrist MLP."""

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        mlp_offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        num_features = self.NUM_BANDS * mlp_features[-1]
        # Preserve the course checkpoint keys (model.0 through model.5).
        self.model = nn.Sequential(
            *self._frontend(in_features, mlp_features, mlp_offsets),
            TDSConvEncoder(num_features, block_channels, kernel_width),
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )
        self._setup_ctc(decoder)

    @torch.inference_mode()
    def forward_in_chunks(self, inputs: torch.Tensor, chunk_frames: int) -> torch.Tensor:
        """Evaluate on contiguous output chunks with exact convolutional overlap."""
        if self.training or chunk_frames < 1:
            raise ValueError("Chunked evaluation requires eval mode and positive chunk_frames.")
        context = self.model[3].left_context
        num_outputs = len(inputs) - context
        if num_outputs <= 0:
            raise ValueError("The input is shorter than the TDS receptive field.")
        emissions = []
        for start in range(0, num_outputs, chunk_frames):
            segment = inputs[start : start + chunk_frames + context].to(self.device)
            emissions.append(self(segment))
        return torch.cat(emissions)


class GRUCTCModule(_WristCTCModule):
    """Unidirectional GRU with the course wrist frontend and CTC decoder."""

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        hidden_size: int,
        num_layers: int,
        emission_warmup_frames: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        mlp_offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            *self._frontend(in_features, mlp_features, mlp_offsets),
            CausalGRUEncoder(
                self.NUM_BANDS * mlp_features[-1],
                hidden_size,
                num_layers,
                emission_warmup_frames,
            ),
            nn.Linear(hidden_size, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )
        self._setup_ctc(decoder)

    @torch.inference_mode()
    def forward_in_chunks(self, inputs: torch.Tensor, chunk_frames: int) -> torch.Tensor:
        """Carry the recurrent state across chunks; warm up once per sequence."""
        if self.training or chunk_frames < 1:
            raise ValueError("Chunked evaluation requires eval mode and positive chunk_frames.")
        frontend = self.model[:3]
        encoder = self.model[3]
        readout = self.model[4:]
        if len(inputs) <= encoder.left_context:
            raise ValueError("The input is shorter than the GRU emission warm-up.")
        state = None
        emissions = []
        for start in range(0, len(inputs), chunk_frames):
            segment = inputs[start : start + chunk_frames].to(self.device)
            features = frontend(segment)
            sequence, state = encoder.gru(features, state)
            first = max(encoder.left_context - start, 0)
            if first < len(sequence):
                emissions.append(readout(sequence[first:]))
        return torch.cat(emissions)

"""Regression checks for paired TDS/GRU emission grids and whole-session decoding."""

import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from emg2qwerty.lightning import GRUCTCModule, TDSConvCTCModule, WindowedEMGDataModule

COMMON = dict(
    in_features=48,
    mlp_features=[8],
    optimizer={},
    lr_scheduler={},
    decoder=OmegaConf.create({"_target_": "emg2qwerty.decoder.CTCGreedyDecoder"}),
)


def make_model(encoder, offsets=(-1, 0, 1)):
    if encoder == "tds":
        return TDSConvCTCModule(
            **COMMON, mlp_offsets=offsets, block_channels=[2, 2], kernel_width=4
        )
    return GRUCTCModule(
        **COMMON,
        mlp_offsets=offsets,
        hidden_size=10,
        num_layers=2,
        emission_warmup_frames=6,
    )


class ProjectModelTest(unittest.TestCase):
    def test_tds(self):
        self.check_encoder("tds")

    def test_gru(self):
        self.check_encoder("gru")

    def check_encoder(self, encoder):
        torch.manual_seed(42)
        model = make_model(encoder).eval()
        inputs = torch.randn(29, 2, 2, 16, 3)
        inputs[22:, 1] = 0
        logits = model(inputs)
        assert logits.shape[:2] == (23, 2)
        individual = model(inputs[:22, 1:2])
        torch.testing.assert_close(logits[:16, 1], individual[:, 0], atol=2e-5, rtol=1e-5)
        for chunk_frames in (4, 7, 32):
            chunks = model.forward_in_chunks(inputs, chunk_frames)
            torch.testing.assert_close(logits, chunks, atol=2e-5, rtol=1e-5)
        model.train()
        log_probs = model(inputs)
        loss = model.ctc_loss(
            log_probs,
            torch.tensor([[1, 2], [2, 3]]),
            torch.tensor([23, 16]),
            torch.tensor([2, 2]),
        )
        loss.backward()
        assert torch.isfinite(loss)
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()

    def test_pooling_preserves_parameter_count(self):
        for encoder in ("tds", "gru"):
            off = make_model(encoder, (0,))
            on = make_model(encoder)
            off_count = sum(p.numel() for p in off.parameters())
            self.assertEqual(off_count, sum(p.numel() for p in on.parameters()))
            for band in off.model[1].mlps:
                assert tuple(band.offsets) == (0,)

    def test_stage_opens_only_needed_sessions(self):
        data = WindowedEMGDataModule(
            window_length=8000, padding=(1800, 200), batch_size=2, num_workers=0,
            train_sessions=["train"], val_sessions=["val"], test_sessions=["test"],
            train_transform=None, val_transform=None, test_transform=None,
        )
        for stage, expected in (("fit", ["train", "val"]), ("validate", ["val"]), ("test", ["test"])):
            with patch("emg2qwerty.lightning.WindowedEMGDataset", return_value=[0]) as dataset:
                data.setup(stage)
                self.assertEqual([call.args[0] for call in dataset.call_args_list], expected)
        self.assertFalse(data.train_dataloader().persistent_workers)

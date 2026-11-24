"""Inspect a course Lightning 1.8 epoch-end checkpoint before accepting a run."""

from pathlib import Path


class IncompleteFitError(RuntimeError):
    """The latest persistent epoch boundary has not met the training budget."""


def checkpoint_completion(path: Path, expected_epochs: int) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    fit = checkpoint["loops"]["fit_loop"]
    epoch = int(checkpoint["epoch"])  # zero-based
    epoch_progress = fit["epoch_progress"]["current"]
    batches = fit["epoch_loop.batch_progress"]
    val_batches = fit["epoch_loop.val_loop.epoch_loop.batch_progress"]
    optimizer = fit["epoch_loop.batch_loop.optimizer_loop.optim_progress"]["optimizer"]["step"]
    global_step = int(checkpoint["global_step"])
    completed = int(optimizer["total"]["completed"])
    # Lightning 1.8 saves in on_train_epoch_end after increment_processed()
    # but before increment_completed(). Do not count standalone validation CSV rows.
    boundary = (
        epoch_progress["processed"] == epoch + 1
        and epoch_progress["completed"] in (epoch, epoch + 1)
        and batches["is_last_batch"]
        and batches["current"]["ready"] == batches["current"]["completed"] > 0
        and val_batches["is_last_batch"]
        and val_batches["current"]["ready"] == val_batches["current"]["completed"] > 0
        and optimizer["total"]["ready"] == completed == global_step > 0
    )
    if not boundary or epoch + 1 != expected_epochs:
        raise IncompleteFitError(
            f"{path}: expected {expected_epochs} complete epochs; checkpoint epoch index "
            f"{epoch}, optimizer steps {completed}, full training/validation boundary {boundary}."
        )
    # The prescribed study has fixed-length epochs, one optimizer and constant
    # accumulation. This guards against a missing or replayed epoch on resume.
    if optimizer["current"]["completed"] * expected_epochs != completed:
        raise IncompleteFitError(f"{path}: completed optimizer steps disagree with the epoch budget.")
    callback = next(
        (v for k, v in checkpoint["callbacks"].items()
         if k.startswith("ModelCheckpoint") and v.get("monitor") == "val/CER"),
        None,
    )
    if callback is None or not callback.get("best_model_path"):
        raise ValueError(f"{path}: missing best-validation checkpoint selection")
    return {
        "completed_epochs": epoch + 1,
        "global_step": global_step,
        "optimizer_steps_per_epoch": int(optimizer["current"]["completed"]),
        "best_checkpoint_name": Path(callback["best_model_path"]).name,
    }

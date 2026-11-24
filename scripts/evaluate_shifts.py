"""Evaluate one frozen checkpoint under clean and eight fixed electrode layouts.

From the repository root:
    python -m scripts.evaluate_shifts --checkpoint run/epoch.ckpt \
        --config run/hydra_configs/config.yaml --data-root data --split val \
        --out results/validation.json

Use --split test only after the checkpoints and analysis choices are frozen.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import Levenshtein
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.decoder import CTCGreedyDecoder
from emg2qwerty.transforms import Compose, FixedWristShift, LogSpectrogram, ToTensor


REPO = Path(__file__).resolve().parents[1]
CONDITIONS = [
    (0, 0),
    *((k, 0) for k in (-2, -1, 1, 2)),
    *((0, k) for k in (-2, -1, 1, 2)),
]
COUNT_KEYS = ("insertions", "deletions", "substitutions", "reference_chars")


def edit_counts(reference: str, prediction: str) -> dict[str, int]:
    # Reuse the course metric's optimal alignment, including its tie-breaking.
    # That implementation repairs the prediction into the reference. Label the
    # repairs in the conventional reference-to-prediction direction for reporting.
    repairs = Counter(op for op, _, _ in Levenshtein.editops(prediction, reference))
    return {
        "insertions": repairs["delete"],  # extra characters in the hypothesis
        "deletions": repairs["insert"],  # missing characters in the hypothesis
        "substitutions": repairs["replace"],
        "reference_chars": len(reference),
    }


def aggregate(items):
    counts = {key: sum(item[key] for item in items) for key in COUNT_KEYS}
    n = counts["reference_chars"]
    if n == 0:
        raise ValueError("No reference characters in this evaluation split")
    edits = counts["insertions"] + counts["deletions"] + counts["substitutions"]
    return {**counts, "cer_pct": 100 * edits / n}


def build_eval_transform(config, split, left, right, chunk_frames):
    transforms = [instantiate(spec) for spec in config.transforms[split]]
    if not (
        len(transforms) == 2
        and isinstance(transforms[0], ToTensor)
        and isinstance(transforms[1], LogSpectrogram)
    ):
        raise ValueError(
            "Evaluation requires the deterministic ToTensor → LogSpectrogram pipeline"
        )
    if (
        tuple(transforms[0].fields) != ("emg_left", "emg_right")
        or transforms[0].stack_dim != 1
    ):
        raise ValueError("Expected left and right wrists stacked along axis 1")
    if split == "test":
        transforms[1].chunk_frames = chunk_frames
    return Compose([transforms[0], FixedWristShift(left, right), transforms[1]])


def evaluate_condition(model, config, args, left, right):
    transform = build_eval_transform(config, args.split, left, right, args.chunk_frames)
    window_length = config.datamodule.window_length if args.split == "val" else None
    padding = tuple(config.datamodule.padding) if args.split == "val" else (0, 0)
    decoder = CTCGreedyDecoder()
    records = []
    for entry in config.dataset[args.split]:
        path = args.data_root / (entry.session + ".hdf5")
        dataset = WindowedEMGDataset(
            path,
            window_length=window_length,
            padding=padding,
            jitter=False,
            transform=transform,
        )
        n = len(dataset)
        if args.limit_val_windows is not None:
            n = min(n, args.limit_val_windows)
        loader = DataLoader(
            Subset(dataset, range(n)),
            shuffle=False,
            batch_size=(
                (args.batch_size or config.batch_size) if args.split == "val" else 1
            ),
            num_workers=args.workers,
            collate_fn=WindowedEMGDataset.collate,
        )
        index = 0
        for batch in loader:
            inputs = batch["inputs"]
            if args.split == "test":
                emissions = model.forward_in_chunks(inputs, args.chunk_frames)
            else:
                emissions = model(inputs.to(args.device))
            t_diff = inputs.shape[0] - emissions.shape[0]
            lengths = batch["input_lengths"] - t_diff
            if (lengths <= 0).any() or not torch.isfinite(emissions).all().item():
                raise ValueError(
                    f"Invalid emissions for {entry.session}, window {index}"
                )
            predictions = decoder.decode_batch(emissions.cpu().numpy(), lengths.numpy())
            for b, prediction in enumerate(predictions):
                labels = batch["targets"][: int(batch["target_lengths"][b]), b]
                reference = LabelData.from_labels(labels.numpy()).text
                start = index * dataset.stride
                records.append(
                    {
                        "session": entry.session,
                        "window": index,
                        "start_sample": start,
                        "stop_sample": start + dataset.window_length,
                        "reference": reference,
                        "prediction": prediction.text,
                        **edit_counts(reference, prediction.text),
                    }
                )
                index += 1
        if index != n:
            raise RuntimeError(f"Missing windows in {entry.session}: {index} of {n}")
    return {"left": left, "right": right, **aggregate(records), "items": records}


def load_config(path):
    if path is not None:
        return OmegaConf.load(path)
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "config")):
        return compose(config_name="base")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    if args.limit_val_windows is not None and (
        args.split != "val" or args.limit_val_windows < 1
    ):
        raise ValueError(
            "--limit-val-windows requires a positive count and --split val"
        )
    if args.chunk_frames < 1 or (args.batch_size is not None and args.batch_size < 1):
        raise ValueError("Frame and batch sizes must be positive")
    config = load_config(args.config)
    if config.decoder._target_ != "emg2qwerty.decoder.CTCGreedyDecoder":
        raise ValueError("This evaluation protocol uses the course greedy CTC decoder")
    model = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    # Lightning checkpoints from the local experiment or the official release.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    # Changing pooling offsets has no effect on parameter shapes. Detect a
    # mismatched run config before an apparently successful state-dict load.
    hp = checkpoint.get("hyper_parameters", {})
    if tuple(hp.get("mlp_offsets", (-1, 0, 1))) != tuple(
        config.module.get("mlp_offsets", (-1, 0, 1))
    ):
        raise ValueError(
            "Checkpoint pooling offsets differ from the supplied run configuration"
        )
    if (
        "emission_warmup_frames" in hp
        and hp["emission_warmup_frames"] != config.module.emission_warmup_frames
    ):
        raise ValueError(
            "Checkpoint GRU warmup differs from the supplied run configuration"
        )
    model.eval().to(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    conditions = []
    paired_keys = None
    start = time.perf_counter()
    with torch.inference_mode():
        for left, right in CONDITIONS:
            result = evaluate_condition(model, config, args, left, right)
            keys = [
                (
                    r["session"],
                    r["window"],
                    r["start_sample"],
                    r["stop_sample"],
                    r["reference"],
                )
                for r in result["items"]
            ]
            if paired_keys is None:
                paired_keys = keys
            elif keys != paired_keys:
                raise ValueError(
                    "Evaluation conditions do not share identical references and windows"
                )
            conditions.append(result)
            result["delta_pp"] = result["cer_pct"] - conditions[0]["cer_pct"]
            print(
                f"{args.split} ({left:+d}, {right:+d}) CER {result['cer_pct']:.3f}% "
                f"over {result['reference_chars']} characters",
                flush=True,
            )

    clean = conditions[0]["cer_pct"]
    mean_shifted = sum(c["cer_pct"] for c in conditions[1:]) / (len(conditions) - 1)
    worst = max(conditions[1:], key=lambda c: c["cer_pct"])
    result = {
        "run_id": args.run_id or args.checkpoint.stem,
        "split": args.split,
        "complete_split": args.limit_val_windows is None,
        "device": args.device,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "torch_version": str(torch.__version__),
        "seed": config.seed,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "module": OmegaConf.to_container(config.module, resolve=True),
        "train_transforms": OmegaConf.to_container(
            config.transforms.train, resolve=True
        ),
        "window_length": (
            config.datamodule.window_length if args.split == "val" else None
        ),
        "padding": list(config.datamodule.padding) if args.split == "val" else [0, 0],
        "seconds": time.perf_counter() - start,
        "edit_convention": "Extra predicted character = insertion; omitted reference character = deletion. "
        "Alignment uses course Levenshtein.editops(prediction, reference).",
        "summary": {
            "clean_cer_pct": clean,
            "mean_shifted_cer_pct": mean_shifted,
            "mean_delta_pp": mean_shifted - clean,
            "worst_shifted_cer_pct": worst["cer_pct"],
            "worst_shift": [worst["left"], worst["right"]],
        },
        "conditions": conditions,
    }
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    tmp.replace(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Saved Hydra config for this run; defaults to the course TDS baseline",
    )
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--batch-size", type=int, help="Validation batch size; otherwise use run config"
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=1024,
        help="Full-session STFT/inference block size; decoder still sees one sequence",
    )
    parser.add_argument(
        "--limit-val-windows",
        type=int,
        help="Development only: prefix per validation session",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())

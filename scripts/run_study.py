"""Train, validate, freeze, and test the paired wrist-EMG study.

Examples (run in the course repository):
    python -m scripts.run_study train --accelerator gpu
    python -m scripts.run_study evaluate --split val --device cuda
    python -m scripts.run_study freeze
    python -m scripts.run_study evaluate --split test --device cuda

Each seed has an independent directory. The primary (seed 0) freeze covers all
five models. For a secondary seed, --runs can limit the paired replication.
"""

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
# (temporal encoder, training rotation, shared MLP position averaging)
RUNS = {
    "E1": ("tds_conv_ctc", True, True),
    "E2": ("gru_ctc", True, True),
    "E3": ("tds_conv_ctc", False, True),
    "E4": ("tds_conv_ctc", True, False),
    "E5": ("tds_conv_ctc", False, False),
}
CONDITIONS = [(0, 0), *((i, 0) for i in (-2, -1, 1, 2)), *((0, i) for i in (-2, -1, 1, 2))]
EVAL_SOURCES = (
    "emg2qwerty/charset.py", "emg2qwerty/data.py", "emg2qwerty/decoder.py",
    "emg2qwerty/lightning.py", "emg2qwerty/modules.py", "emg2qwerty/transforms.py",
    "scripts/evaluate_shifts.py",
)


def sha256(path):
    with path.open("rb") as f:
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


@contextlib.contextmanager
def lock(directory, filename):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / filename).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {directory}/{filename}") from exc
        yield


def training_overrides(args, run_id, run_dir):
    model, augmentation, pooling = RUNS[run_id]
    return [
        f"model={model}",
        "transforms=" + ("log_spectrogram" if augmentation else "log_spectrogram_no_rotation"),
        "module.mlp_offsets=" + ("[-1,0,1]" if pooling else "[0]"),
        f"seed={args.seed}", f"batch_size={args.batch_size}", f"num_workers={args.num_workers}",
        f"trainer.accelerator={args.accelerator}", "trainer.devices=1",
        f"trainer.max_epochs={args.epochs}", "evaluate_test=false",
        "optimizer.lr=0.001", "lr_scheduler.scheduler.warmup_epochs=10",
        f"dataset.root={args.data_root}", f"hydra.run.dir={run_dir}",
        "+trainer.logger._target_=pytorch_lightning.loggers.CSVLogger",
        f"+trainer.logger.save_dir={run_dir}", "+trainer.logger.name=curves",
        "+trainer.log_every_n_steps=1",
    ]


def effective_config(overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(REPO / "config")):
        cfg = compose(config_name="base", overrides=overrides)
    # Preserve runtime interpolations as Hydra does in its saved task config.
    return OmegaConf.to_container(cfg, resolve=False)


def launch(cmd, dry_run):
    print(shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=REPO, check=True)


def artifact_paths(run_dir):
    summary_path = run_dir / "run_summary.json"
    summary = read_json(summary_path)
    if not summary.get("val_metrics") or not summary.get("best_checkpoint"):
        raise ValueError(f"Incomplete run summary: {summary_path}")
    if "test_metrics" in summary:
        raise ValueError(f"Development run already evaluated test: {summary_path}")
    # The native summary may have been generated at a different absolute path.
    checkpoint = run_dir / "checkpoints" / Path(summary["best_checkpoint"]).name
    config = run_dir / "hydra_configs" / "config.yaml"
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError(f"Selected checkpoint/config missing in {run_dir}")
    from omegaconf import OmegaConf
    from emg2qwerty.fit_progress import IncompleteFitError, checkpoint_completion

    expected_epochs = int(OmegaConf.load(config).trainer.max_epochs)
    progress = checkpoint_completion(run_dir / "checkpoints" / "last.ckpt", expected_epochs)
    if progress["best_checkpoint_name"] != checkpoint.name or (
        "fit_completion" in summary and summary["fit_completion"] != progress
    ):
        raise IncompleteFitError(f"{summary_path} disagrees with the final checkpoint's training state")
    return checkpoint, config, summary


def train(args, run_id, run_dir):
    overrides = training_overrides(args, run_id, run_dir)
    cmd = [sys.executable, "-m", "emg2qwerty.train", *overrides]
    if args.dry_run:
        launch(cmd, True)
        return
    from omegaconf import OmegaConf
    from emg2qwerty.fit_progress import IncompleteFitError

    with lock(run_dir, ".train.lock"):
        expected = effective_config(overrides)
        spec_path = run_dir / "training_config.json"
        hydra_cfg = run_dir / "hydra_configs" / "config.yaml"
        if spec_path.exists() and read_json(spec_path) != expected:
            raise ValueError(f"Training settings differ from {spec_path}; use another output directory.")
        if hydra_cfg.exists() and OmegaConf.to_container(OmegaConf.load(hydra_cfg), resolve=False) != expected:
            raise ValueError(f"Training settings differ from {hydra_cfg}; use another output directory.")
        if not spec_path.exists() and not hydra_cfg.exists() and (run_dir / "checkpoints").exists():
            raise RuntimeError(f"Found checkpoints without a corresponding training configuration: {run_dir}")
        if not spec_path.exists():
            write_json(spec_path, expected)
        summary_path = run_dir / "run_summary.json"
        frozen = (run_dir.parent / "freeze.json").exists()
        if summary_path.exists():
            try:
                artifact_paths(run_dir)
            except IncompleteFitError as exc:
                if frozen:
                    raise
                # Preserve the old post-interrupt validation and all checkpoints.
                # A successful resumed invocation writes a new canonical summary.
                stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                archived = run_dir / f"run_summary.incomplete-{stamp}.json"
                summary_path.rename(archived)
                print(f"Archiving incomplete marker {archived.name}: {exc}", flush=True)
            else:
                print(f"Already trained: {run_dir}", flush=True)
                return
        if frozen:
            raise RuntimeError(f"Seed is frozen; cannot resume or start training in {run_dir}")
        launch(cmd, False)
        artifact_paths(run_dir)


def verify_evaluation(result, run_id, seed, split, checkpoint_hash):
    if (result.get("run_id") != run_id or result.get("seed") != seed
            or result.get("split") != split or result.get("complete_split") is not True
            or result.get("checkpoint_sha256") != checkpoint_hash):
        raise ValueError(f"Evaluation identity mismatch for seed {seed}, {run_id}, {split}")
    conditions = result["conditions"]
    if [(c["left"], c["right"]) for c in conditions] != CONDITIONS:
        raise ValueError(f"Missing or reordered electrode conditions in {run_id}/{split}")
    counts = {c["reference_chars"] for c in conditions}
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError(f"Reference lengths disagree in {run_id}/{split}")


def training_protocol(run_id, seed, path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    model, rotation, pooling = RUNS[run_id]
    names = {"tds_conv_ctc": "TDSConvCTCModule", "gru_ctc": "GRUCTCModule"}
    if (cfg.seed != seed or cfg.module._target_.split(".")[-1] != names[model]
            or tuple(cfg.module.mlp_offsets) != ((-1, 0, 1) if pooling else (0,))
            or cfg.evaluate_test or cfg.monitor_metric != "val/CER" or cfg.monitor_mode != "min"):
        raise ValueError(f"Run {run_id} does not match its planned model, seed or selection settings")
    transforms = OmegaConf.to_container(cfg.transforms.train, resolve=True)
    rotations = [t for t in transforms if t.get("transform", {}).get("_target_", "").endswith(".RandomBandRotation")]
    if len(rotations) != int(rotation):
        raise ValueError(f"Training rotation differs from {run_id}'s specified condition")
    # Each paired run must have the same sessions, data processing (apart from
    # the rotation ablation), optimiser, schedule, training budget and decoder.
    return {
        "batch_size": cfg.batch_size, "epochs": cfg.trainer.max_epochs,
        "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
        "scheduler": OmegaConf.to_container(cfg.lr_scheduler, resolve=True),
        "datamodule": OmegaConf.to_container(cfg.datamodule, resolve=True),
        "splits": {s: OmegaConf.to_container(cfg.dataset[s], resolve=True) for s in ("train", "val", "test")},
        "train_transforms_without_rotation": [t for t in transforms if t not in rotations],
        "validation_transforms": OmegaConf.to_container(cfg.transforms.val, resolve=True),
        "test_transforms": OmegaConf.to_container(cfg.transforms.test, resolve=True),
        "decoder": OmegaConf.to_container(cfg.decoder, resolve=True),
    }


def frozen_run(run_dir, seed):
    run_id = run_dir.name
    weights, config, summary = artifact_paths(run_dir)
    val = run_dir / "val.json"
    result = read_json(val)
    training_protocol(run_id, seed, config)
    weights_hash = sha256(weights)
    verify_evaluation(result, run_id, seed, "val", weights_hash)
    return {
        "checkpoint": str(weights.relative_to(run_dir)), "checkpoint_sha256": weights_hash,
        "config_sha256": sha256(config), "validation_sha256": sha256(val),
        "best_epoch": result["epoch"],
        "validation_clean_cer_pct": result["summary"]["clean_cer_pct"],
        "native_best_validation": summary["val_metrics"],
    }


def eval_sources():
    return {file: sha256(REPO / file) for file in EVAL_SOURCES}


def freeze(args, seed_dir):
    if args.seed == 0 and set(args.runs) != set(RUNS):
        raise ValueError("The primary seed 0 freeze requires E1 through E5.")
    if args.dry_run:
        print(f"Would verify completed validation and freeze {args.runs} in {seed_dir}/freeze.json")
        return
    with lock(seed_dir, ".freeze.lock"):
        runs = {run: frozen_run(seed_dir / run, args.seed) for run in args.runs}
        paired = [training_protocol(run, args.seed, seed_dir / run / "hydra_configs" / "config.yaml")
                  for run in args.runs]
        if any(settings != paired[0] for settings in paired[1:]):
            raise ValueError("Paired runs use different common training or evaluation settings")
        protocol = {
            "shared_training_protocol": paired[0],
            "seed": args.seed, "conditions": [list(c) for c in CONDITIONS],
            "selection": "Minimum unshifted validation CER, first epoch in case of a tie",
            "evaluation_source_sha256": eval_sources(), "runs": runs,
        }
        manifest = seed_dir / "freeze.json"
        if manifest.exists():
            existing = read_json(manifest)
            if any(existing.get(k) != v for k, v in protocol.items()):
                raise ValueError(f"Artifacts differ from existing immutable freeze: {manifest}")
            print(f"Already frozen: {manifest}")
            return
        if any((seed_dir / run / "test.json").exists() for run in args.runs):
            raise ValueError("An existing test result predates this freeze.")
        protocol["frozen_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(manifest, protocol)
        print(f"Frozen: {manifest}")


def evaluate(args, run_id, run_dir):
    out = run_dir / (args.split + ".json")
    if args.dry_run and not (run_dir / "run_summary.json").is_file():
        print(f"Would evaluate {run_dir} best checkpoint on {args.split} -> {out}")
        return
    weights, config, _ = artifact_paths(run_dir)
    cmd = [
        sys.executable, "-m", "scripts.evaluate_shifts",
        "--checkpoint", str(weights), "--config", str(config), "--data-root", str(args.data_root),
        "--split", args.split, "--device", args.device, "--workers", str(args.num_workers),
        "--chunk-frames", str(args.chunk_frames), "--run-id", run_id, "--out", str(out),
    ]
    if args.dry_run:
        launch(cmd, True)
        return
    with lock(run_dir, ".evaluate.lock"):
        manifest = run_dir.parent / "freeze.json"
        if args.split == "test":
            if not manifest.is_file():
                raise ValueError(f"Complete validation and freeze this seed before test: {manifest}")
            protocol = read_json(manifest)
            if (protocol["seed"] != args.seed or protocol["runs"].get(run_id) != frozen_run(run_dir, args.seed)
                    or protocol["evaluation_source_sha256"] != eval_sources()):
                raise ValueError(f"Current validation, code, or weights differ from {manifest}")
        weights_hash = sha256(weights)
        if out.exists():
            result = read_json(out)
            verify_evaluation(result, run_id, args.seed, args.split, weights_hash)
            if args.split == "val" and manifest.exists():
                if read_json(manifest)["runs"][run_id]["validation_sha256"] != sha256(out):
                    raise ValueError(f"Validation output changed since freeze: {out}")
            print(f"Already evaluated: {out}", flush=True)
            return
        if args.split == "val" and manifest.exists():
            raise ValueError(f"Cannot regenerate validation under an existing freeze: {manifest}")
        launch(cmd, False)
        verify_evaluation(read_json(out), run_id, args.seed, args.split, weights_hash)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--output-dir", type=Path, default=REPO / "logs")
    shared.add_argument("--seed", type=int, default=0)
    shared.add_argument("--runs", nargs="+", choices=RUNS, default=list(RUNS))
    shared.add_argument("--dry-run", action="store_true", help="Show steps without writing files or running models")
    fit = commands.add_parser("train", parents=[shared])
    ev = commands.add_parser("evaluate", parents=[shared])
    commands.add_parser("freeze", parents=[shared])
    for entry in (fit, ev):
        entry.add_argument("--data-root", type=Path, default=REPO / "data")
    fit.add_argument("--accelerator", choices=("mps", "cpu", "gpu"), default="gpu")
    fit.add_argument("--epochs", type=int, default=40)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--num-workers", type=int, default=2)
    ev.add_argument("--split", choices=("val", "test"), required=True)
    ev.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    ev.add_argument("--num-workers", "--workers", type=int, default=0)
    ev.add_argument("--chunk-frames", type=int, default=1024)
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs) or args.seed < 0:
        parser.error("Run IDs must be unique and seed must be nonnegative.")
    for key in ("epochs", "batch_size", "chunk_frames"):
        if getattr(args, key, 1) < 1:
            parser.error(f"{key} must be positive.")
    if getattr(args, "num_workers", 0) < 0:
        parser.error("num_workers must be nonnegative.")
    seed_dir = args.output_dir.resolve() / f"seed-{args.seed}"
    if args.command == "freeze":
        freeze(args, seed_dir)
        return
    args.data_root = args.data_root.resolve()
    for run_id in args.runs:
        if args.command == "train":
            train(args, run_id, seed_dir / run_id)
        else:
            evaluate(args, run_id, seed_dir / run_id)


if __name__ == "__main__":
    main()

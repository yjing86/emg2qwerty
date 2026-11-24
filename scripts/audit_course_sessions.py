"""Validate the course recordings and write a split manifest with locally computed hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def audit(path, user, split):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024**2), b""):
            digest.update(block)
    with h5py.File(path) as file:
        group = file["emg2qwerty"]
        signal = group["timeseries"]
        if str(group.attrs["user"]) != str(user) or group.attrs["session_name"] != path.stem:
            raise ValueError(f"Session or participant mismatch: {path.name}")
        if any(signal.dtype[side].shape != (16,) for side in ("emg_left", "emg_right")):
            raise ValueError(f"Unexpected channel dimensions: {path.name}")
        if float(group.attrs["daq_sample_rate"]) != 2000:
            raise ValueError(f"Unexpected acquisition rate: {path.name}")
        timestamps = signal["time"]
        if not len(timestamps) or not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
            raise ValueError(f"Invalid timestamps: {path.name}")
        for start in range(0, len(signal), 65536):
            chunk = signal[start:start + 65536]
            if not all(np.isfinite(chunk[side]).all() for side in ("emg_left", "emg_right")):
                raise ValueError(f"Nonfinite EMG in {path.name} at sample block {start}")
        return {
            "split": split, "session": path.stem, "user": str(user),
            "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
            "samples": len(signal),
            "sample_rate_hz": float(group.attrs["daq_sample_rate"]),
            "recorded_keystroke_events": len(json.loads(group.attrs["keystrokes"])),
            "elapsed_seconds": float(timestamps[-1] - timestamps[0]),
            "condition": str(group.attrs["condition"]),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = ROOT / "config/user/single_user.yaml"
    splits = yaml.safe_load(config.read_text())["dataset"]
    sessions = []
    seen = set()
    for split, required in (("train", 16), ("val", 1), ("test", 1)):
        entries = splits[split]
        if len(entries) != required:
            raise ValueError(f"Expected {required} recordings for {split}")
        for entry in entries:
            name = entry["session"]
            if name in seen:
                raise ValueError(f"Duplicate session in split configuration: {name}")
            seen.add(name)
            sessions.append(audit(args.data / f"{name}.hdf5", entry["user"], split))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generic_config = ROOT / "config/user/generic.yaml"
    generic = yaml.safe_load(generic_config.read_text())["dataset"]
    generic_train = {entry["session"] for entry in generic["train"]}
    result = {
        "source": "https://github.com/facebookresearch/emg2qwerty#data",
        "split_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "generic_training_set_overlap": {
            split: sum(entry["session"] in generic_train for entry in entries)
            for split, entries in splits.items()
        },
        "sessions": sessions,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for split in splits:
        records = [r for r in sessions if r["split"] == split]
        duration = sum(r["elapsed_seconds"] for r in records)
        print(f"{split}: {len(records)} sessions, {duration / 60:.2f} recorded minutes")


if __name__ == "__main__":
    main()

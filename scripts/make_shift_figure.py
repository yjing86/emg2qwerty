"""Build the paired shift plot and its verified source table from evaluation JSON.

Example: python make_shift_figure.py --out results E1=E1.json E2=E2.json
Each invocation compares complete evaluations of the same split and seed.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = {
    "E1": ("TDS, both", "#0072B2", "o", "-"),
    "E2": ("GRU, both", "#9B488B", "D", "--"),
    "E3": ("TDS, pooling", "#009E73", "s", "-."),
    "E4": ("TDS, augmentation", "#D55E00", "v", ":"),
    "E5": ("TDS, neither", "#555555", "X", "--"),
}
OFFSETS = (-2, -1, 0, 1, 2)
EXPECTED = {(d, 0) for d in OFFSETS} | {(0, d) for d in OFFSETS}
COUNTS = ("insertions", "deletions", "substitutions", "reference_chars")


def item_refs(condition):
    return [(r["session"], r["start_sample"], r["stop_sample"], r["reference"])
            for r in condition["items"]]


def check_edits(condition, source):
    totals = dict.fromkeys(COUNTS, 0)
    for item in condition["items"]:
        if any(type(item[key]) is not int or item[key] < 0 for key in COUNTS):
            raise ValueError(f"{source}: invalid edit count in a prediction")
        if (len(item["reference"]) != item["reference_chars"] or
            len(item["prediction"]) - len(item["reference"]) != item["insertions"] - item["deletions"]):
            raise ValueError(f"{source}: reference and prediction lengths disagree with edits")
        for key in COUNTS:
            totals[key] += item[key]
    if any(condition[key] != totals[key] for key in COUNTS):
        raise ValueError(f"{source}: per-window counts do not sum to the reported counts")
    length = totals["reference_chars"]
    edits = totals["insertions"] + totals["deletions"] + totals["substitutions"]
    if length <= 0 or not math.isclose(condition["cer_pct"], 100 * edits / length, abs_tol=1e-7):
        raise ValueError(f"{source}: CER and counts disagree")


def read_runs(specs):
    runs = {}
    cohort = None
    for spec in specs:
        model_id, path = spec.split("=", 1)
        if model_id not in MODELS or model_id in runs:
            raise ValueError(f"Use distinct labels from {list(MODELS)}: {model_id}")
        raw = json.loads(Path(path).read_text())
        indexed = {(c["left"], c["right"]): c for c in raw["conditions"]}
        if not raw["complete_split"] or set(indexed) != EXPECTED or len(raw["conditions"]) != 9:
            raise ValueError(f"{path}: need a complete split with nine distinct paired conditions")
        baseline = item_refs(indexed[0, 0])
        this_cohort = raw["split"], raw["seed"], baseline
        if cohort is not None and cohort != this_cohort:
            raise ValueError(f"{path}: different reference, evaluation windows, split, or seed")
        cohort = this_cohort
        for condition in indexed.values():
            if item_refs(condition) != baseline:
                raise ValueError(f"{path}: references differ across conditions")
            check_edits(condition, path)
        runs[model_id] = raw, indexed
    return {k: runs[k] for k in MODELS if k in runs}


def figures(runs, out):
    width = 6.4
    with plt.rc_context({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "legend.frameon": False}):
        fig, axes = plt.subplots(1, 2, figsize=(width, 2.8), sharey=True)
        for ax, wrist in zip(axes, ("Left wrist", "Right wrist")):
            for model_id, (_, indexed) in runs.items():
                name, color, marker, line = MODELS[model_id]
                ys = [indexed[d, 0]["cer_pct"] if wrist == "Left wrist" else indexed[0, d]["cer_pct"]
                      for d in OFFSETS]
                ax.plot(OFFSETS, ys, label=name, color=color, marker=marker,
                        lw=1.35, ms=4, linestyle=line)
            ax.set_title(f"{wrist} shifted", loc="left", fontsize=10)
            ax.set_xticks(OFFSETS)
            ax.axvline(0, color="#AAAAAA", lw=0.7, zorder=0)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xlabel("Offset (electrode positions)")
        axes[0].set_ylabel("Character error rate (%)")
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=min(3, len(runs)), fontsize=8)
        fig.subplots_adjust(bottom=0.36, left=0.1, top=0.89, right=0.985, wspace=0.1)
        fig.savefig(out / "shifts.pdf", bbox_inches="tight", pad_inches=0.035)
        plt.close(fig)


def summary(runs, out):
    with (out / "condition_counts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seed", "split", "checkpoint_sha256", "left", "right", "reference_chars",
                         "substitutions", "deletions", "insertions", "cer_pct", "delta_pp"])
        for key, (raw, indexed) in runs.items():
            for (left, right), c in indexed.items():
                writer.writerow([key, raw["seed"], raw["split"], raw["checkpoint_sha256"], left, right,
                                 c["reference_chars"], c["substitutions"], c["deletions"], c["insertions"],
                                 c["cer_pct"], c["cer_pct"] - indexed[0, 0]["cer_pct"]])
    with (out / "model_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seed", "split", "parameters", "checkpoint_epoch_1based",
                         "clean_cer_pct", "mean_shifted_cer_pct", "mean_delta_pp", "worst_cer_pct"])
        for key, (raw, indexed) in runs.items():
            clean = indexed[0, 0]["cer_pct"]
            shifted = [c["cer_pct"] for d, c in indexed.items() if d != (0, 0)]
            mean = sum(shifted)/len(shifted)
            epoch = raw["epoch"] + 1 if raw.get("epoch") is not None else ""
            writer.writerow([key, raw["seed"], raw["split"], raw["parameters"], epoch,
                             clean, mean, mean - clean, max(shifted)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("runs", nargs="+", help="E1=path.json through E5=path.json")
    args = parser.parse_args()
    loaded = read_runs(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    figures(loaded, args.out)
    summary(loaded, args.out)

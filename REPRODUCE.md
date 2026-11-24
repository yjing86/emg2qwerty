# Reproduce the electrode-shift experiments

The five models use the same 16 training sessions, one validation session and one test session. They train from random initialization and share a greedy CTC decoder. The measured run used Python 3.10, PyTorch 2.3.0, one NVIDIA T4 and seed 0. Each model received 40 epochs (4,800 optimizer updates) with 32 windows per batch.

## 1. Prepare the environment

Run these commands from a Linux shell with an NVIDIA driver compatible with CUDA 12.1. Skip the first two commands if you are already in the fork's root directory. The experiment initializes its own weights, so cloning can skip the upstream Git LFS checkpoints.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/yjing86/emg2qwerty.git
cd emg2qwerty
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install uv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
uv pip install --python .venv/bin/python -r requirements-study.txt
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
```

## 2. Extract and verify the recordings

**The full upstream download is 308.4 GB. If you already have the 18 course HDF5 files, place them directly under `data/`, run `mkdir -p logs/reproduction`, and skip to the audit command.** The archive covers 1,135 recordings. The extraction below retains just the 18 files in this study (4.73 GB), so allow about 314 GB of free space for the download and the extracted files. [Upstream dataset](https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz) · [Split configuration](config/user/single_user.yaml)

```bash
mkdir -p data logs/reproduction
curl --fail --location --continue-at - --output emg2qwerty-data-2021-08.tar.gz \
    https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
python - <<'PY' > logs/reproduction/session-files.txt
from pathlib import Path
import yaml

split = yaml.safe_load(Path("config/user/single_user.yaml").read_text())["dataset"]
for partition in ("train", "val", "test"):
    for entry in split[partition]:
        print(entry["session"] + ".hdf5")
PY
# GNU tar matches each basename anywhere in the archive. Keep only the basename on extraction.
tar --extract --gzip --file=emg2qwerty-data-2021-08.tar.gz --directory=data \
    --no-anchored --wildcards --transform='s|.*/||' --files-from=logs/reproduction/session-files.txt
```

The audit checks recording identities, array shapes, timestamps and the split. The digest below combines the split assignment, the recording name and the SHA-256 of each original file. Matching it confirms all 18 inputs used in the reported run.

```bash
python scripts/audit_course_sessions.py --data data --output logs/reproduction/input-manifest.json
python - <<'PY'
from pathlib import Path
import hashlib
import json

manifest = json.loads(Path("logs/reproduction/input-manifest.json").read_text())
assert manifest["split_config_sha256"] == "bf3de56634dbe24063fe19649c48b156608b9e10b6386ab3c67f48aedca1fa63"
entries = sorted((s["split"], s["session"], s["sha256"]) for s in manifest["sessions"])
text = "".join(f"{split} {session} {digest}\n" for split, session, digest in entries)
assert hashlib.sha256(text.encode("utf-8")).hexdigest() == "5bd851a03ad6d0d13d3556c70b88a573b2dd94b73799910512d7bd4dfe5dd003"
print(f"Verified the split and all {len(entries)} input files.")
PY
```

## 3. Train, validate, freeze and test

The four commands below operate on a shared output directory. The first command trains E1 through E5. The second evaluates their selected checkpoints on validation under all nine conditions. The third records the checkpoint, configuration and validation fingerprints. The final command verifies that freeze before evaluating the held-out session. The course validation and test sessions both appear in the upstream generic training split, so pretrained weights would compromise this comparison.

```bash
python -m scripts.run_study train --data-root data --output-dir logs/reproduction --seed 0 \
    --accelerator gpu --epochs 40 --batch-size 32 --num-workers 2
python -m scripts.run_study evaluate --data-root data --output-dir logs/reproduction --seed 0 \
    --device cuda --split val
python -m scripts.run_study freeze --output-dir logs/reproduction --seed 0
python -m scripts.run_study evaluate --data-root data --output-dir logs/reproduction --seed 0 \
    --device cuda --split test
```

The clean condition preserves channel order. Each shifted condition moves one wrist's electrode indices by −2, −1, +1 or +2 positions, leaving the other wrist fixed. All models evaluate exactly the same reference characters within a split. Validation counts 4,514 characters across 212 windows. Test counts 4,627 in one continuous transcript.

## 4. Rebuild the tables and compare the results

The plotter checks per-window and per-condition edit counts before it writes `model_summary.csv`, `condition_counts.csv` and `shifts.pdf`. This Bash loop generates both validation and test summaries. [Published test summary](results/test/model_summary.csv) · [Full test edit counts](results/test/condition_counts.csv)

```bash
for split in val test; do
    inputs=()
    for model in E1 E2 E3 E4 E5; do
        inputs+=("$model=logs/reproduction/seed-0/$model/$split.json")
    done
    python scripts/make_shift_figure.py --out "logs/reproduction/summary/$split" "${inputs[@]}"
done
```

Use the following comparison to print differences in percentage points between a rerun and the archived test scores. Floating-point kernels and training hardware can affect the trained parameters.

```bash
python - <<'PY'
import csv
from pathlib import Path

def read_scores(path):
    with Path(path).open(newline="") as stream:
        return {row["model"]: row for row in csv.DictReader(stream)}

published = read_scores("results/test/model_summary.csv")
rerun = read_scores("logs/reproduction/summary/test/model_summary.csv")
assert set(published) == set(rerun) == {"E1", "E2", "E3", "E4", "E5"}
print("Model   clean delta (pp)   mean shifted delta (pp)")
for model in sorted(published):
    fields = ("clean_cer_pct", "mean_shifted_cer_pct")
    clean, shifted = [float(rerun[model][f]) - float(published[model][f]) for f in fields]
    print(f"{model:5s}   {clean:+16.4f}   {shifted:+23.4f}")
PY
```

CER is 100 times the sum of substitutions, deletions and insertions divided by the reference length. The shifted score in the table is the arithmetic mean of the eight individual shifted CERs. These eight inputs replay one recording. One participant and one training seed limit the scope of the comparison.

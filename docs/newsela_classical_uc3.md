# Newsela classical models on UC3

This guide trains and evaluates logistic regression, linear SVM, and random
forest text-simplification models on the encrypted Newsela dataset. It assumes
access to bwUniCluster 3.0 (UC3), the licensed Newsela cache and its Fernet key,
and the team's NVIDIA/PyTorch Apptainer image.

The dataset and key are private inputs. Never commit, print, or share the key.
The repository ignores `private_data/`, `.venv-newsela/`, encrypted pickle
files, run outputs, and Newsela Slurm logs.

## What the run guarantees

- Deterministic 80/10/10 train, validation, and test assignment by Newsela
  article ID.
- No article appears in more than one split.
- Repeated source sentences crossing split boundaries are removed.
- The same split is reused for all three classifiers.
- `split_manifest.json` records document IDs, split counts, and overlap checks.
- Validation and test predictions and metrics are written separately.

The metric suite includes classifier accuracy, macro precision, macro recall,
macro F1, weighted F1, BERTScore, BLEU, token F1, Flesch-Kincaid grade, SARI,
ROUGE-L, named-entity preservation, and number preservation.

## Expected UC3 layout

The checked-in batch script uses these defaults:

```text
$HOME/NLP2/
├── .venv-newsela/
├── private_data/
│   ├── _env
│   └── newsela_articles_20150302.aligned.sents.pkl.enc
├── scripts/run_newsela_classical_slurm.sh
└── runs/

$HOME/project-data-science-2026/
└── nvidia+pytorch+26.04-py3.sqsh
```

The `_env` file must contain a single variable named `NEWSELA_CACHE_KEY`. Do
not display its value in terminal logs or support messages.

## 1. Clone the correct branch

Run on the UC3 login node (`uc3n...`), not on a compute node:

```bash
cd "$HOME"
git clone \
  --branch dev \
  --single-branch \
  https://github.com/safreu/NLP.git \
  NLP2

cd "$HOME/NLP2"
git log -1 --oneline
```

For an existing clone:

```bash
cd "$HOME/NLP2"
git status --short --branch
git pull --ff-only
```

Do not pull over local source-code edits. Commit or preserve them separately
first.

## 2. Upload the encrypted inputs

On the login node:

```bash
mkdir -p "$HOME/NLP2/private_data"
chmod 700 "$HOME/NLP2/private_data"
```

From a separate terminal on the local computer, replace the local paths and UC3
username as needed:

```bash
scp \
  /local/path/newsela_articles_20150302.aligned.sents.pkl.enc \
  /local/path/_env \
  USER@uc3.scc.kit.edu:/home/path/to/USER/NLP2/private_data/
```

Back on the login node:

```bash
chmod 600 "$HOME/NLP2/private_data/_env"
chmod 600 "$HOME/NLP2/private_data/newsela_articles_20150302.aligned.sents.pkl.enc"
ls -lh "$HOME/NLP2/private_data"
```

Never paste an OTP, password, private key, or the contents of `_env` into an
issue or chat.

## 3. Create the persistent environment once

Confirm the team container exists:

```bash
test -f "$HOME/project-data-science-2026/nvidia+pytorch+26.04-py3.sqsh" \
  && echo "Container image ready"
```

If it is absent, obtain the approved image location from the project team. Do
not create another roughly 19 GB copy without checking storage and provenance.

Request a short setup allocation from the login node:

```bash
salloc \
  --partition=gpu_a100_short \
  --gres=gpu:1 \
  --time=00:30:00 \
  --cpus-per-task=8 \
  --mem=64G
```

After the prompt changes to `uc2n...`, enter the container:

```bash
cd "$HOME/NLP2"
apptainer shell \
  --nv \
  --bind "$PWD:/work" \
  --pwd /work \
  "$HOME/project-data-science-2026/nvidia+pytorch+26.04-py3.sqsh"
```

Inside Apptainer, set the container paths before running other commands:

```bash
export PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/.singularity.d/libs"
hash -r
```

Create and populate the persistent environment:

```bash
python -m venv --system-site-packages /work/.venv-newsela
source /work/.venv-newsela/bin/activate
python -m pip install --upgrade uv

cd /work
uv sync --frozen --active
```

The environment is stored under `$HOME/NLP2` and survives the allocation.
Reuse it; do not reinstall dependencies for each job.

## 4. Validate the environment and encrypted data

Still inside Apptainer with `.venv-newsela` active:

```bash
cd /work
python - <<'PY'
from pathlib import Path

import spacy
import torch

from data.corpus.newsela_corpus import NewselaCorpus

print("CUDA ready:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("spaCy model:", spacy.load("en_core_web_sm").meta["name"])

corpus = NewselaCorpus.load_from_disk(
    encrypted_cache_path=Path(
        "/work/private_data/newsela_articles_20150302.aligned.sents.pkl.enc"
    ),
    env_file=Path("/work/private_data/_env"),
)
print("Newsela entries:", len(corpus.entries))
print("Newsela articles:", len({entry.doc_id for entry in corpus.entries}))
PY
```

The supplied cache contains 141,582 aligned entries from 1,131 articles.

## 5. Run the smoke test

Run all three model paths with small caps before submitting the full job:

```bash
python -m pipeline.newsela_classical_pipeline \
  --encrypted-cache /work/private_data/newsela_articles_20150302.aligned.sents.pkl.enc \
  --env-file /work/private_data/_env \
  --output-path /work/runs/newsela_classical_smoke \
  --models logistic_regression svm random_forest \
  --max-train-samples 100 \
  --max-eval-samples 20 \
  --random-forest-estimators 5 \
  --skip-generation-metrics
```

Success ends with `Newsela classical experiment finished` and creates three
score files:

```bash
find /work/runs/newsela_classical_smoke -maxdepth 2 -name scores.json -print
```

Exit Apptainer, then release the setup allocation:

```bash
exit
exit
```

The final prompt must be back on `uc3n...` before submitting the batch job.

## 6. Submit the full batch job

From `$HOME/NLP2` on the login node:

```bash
bash -n scripts/run_newsela_classical_slurm.sh
sbatch scripts/run_newsela_classical_slurm.sh
```

The script requests `gpu_a100_il`, one A100, 16 CPU cores, 120 GB RAM, and 12
hours. It enters Apptainer, activates `.venv-newsela`, and writes the full run to
`runs/newsela_classical_full`.

If the repository or image lives elsewhere, submit with overrides:

```bash
sbatch \
  --export=ALL,PROJECT_DIR=/absolute/project/path,CONTAINER_IMAGE=/absolute/image.sqsh \
  scripts/run_newsela_classical_slurm.sh
```

Do not submit a second copy merely because the job is pending.

## 7. Monitor the job

Replace `JOB_ID` with the number printed by `sbatch`:

```bash
squeue -j JOB_ID \
  -o "%.18i %.12P %.10T %.10M %.19S %.20R"
```

- `PENDING (Resources)` is normal; wait for an A100.
- `RUNNING` means the batch script has started.
- No row means the job ended; use `sacct` to determine its outcome.

Follow the log after the job starts:

```bash
tail -f "$HOME/NLP2/newsela-classical-JOB_ID.out"
```

Pressing `Ctrl+C` stops `tail`; it does not cancel the job. Check completed or
failed jobs with:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Only use `scancel JOB_ID` when the specific job should be cancelled.

## 8. Confirm results

A successful run ends with `Newsela classical experiment finished` in the log
and has all three score files:

```bash
grep -F "Newsela classical experiment finished" \
  "$HOME/NLP2/newsela-classical-JOB_ID.out"

find "$HOME/NLP2/runs/newsela_classical_full" \
  -maxdepth 2 -name scores.json -print
```

The output layout is:

```text
runs/newsela_classical_full/
├── run_config.json
├── split_manifest.json
├── logistic_regression/
│   ├── model/
│   ├── predictions.json
│   ├── validation_predictions.json
│   └── scores.json
├── svm/
│   └── ...
└── random_forest/
    └── ...
```

Confirm the leakage checks before reporting metrics:

```bash
cd "$HOME/NLP2"
source .venv-newsela/bin/activate
python - <<'PY'
import json
from pathlib import Path

root = Path("runs/newsela_classical_full")
manifest = json.loads((root / "split_manifest.json").read_text())
print("documents:", manifest["documents"])
print("document overlap:", manifest["document_overlap"])
print("source overlap:", manifest["cross_split_source_overlap"])
for model in ("logistic_regression", "svm", "random_forest"):
    scores = json.loads((root / model / "scores.json").read_text())
    print(model, "test metrics:", sorted(scores["test"]))
PY
```

Both overlap values must be `False`.

## 9. Download results

From the local computer, not from inside UC3:

```bash
scp -r \
  USER@uc3.scc.kit.edu:/home/path/to/USER/NLP2/runs/newsela_classical_full \
  /local/destination/
```

The trained models and predictions may be large. To retrieve only evaluation
files, use `rsync` with exclusions:

```bash
rsync -av \
  --exclude='model/' \
  USER@uc3.scc.kit.edu:/home/path/to/USER/NLP2/runs/newsela_classical_full/ \
  /local/destination/newsela_classical_full/
```

## Troubleshooting

### Standard commands missing when Apptainer starts

Warnings such as `sed: executable file not found in $PATH` are corrected with:

```bash
export PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/.singularity.d/libs"
hash -r
```

The batch script applies these values automatically.

### `sinfo` reports permission denied

General `sinfo` access is administrator-only on UC3. Use `sinfo_t_idle` for a
cluster-provided availability summary or inspect only the submitted job with
`squeue -j JOB_ID`.

### Job remains pending with `(Resources)`

The requested GPU is busy. The batch job remains valid and survives SSH
disconnects. Wait; do not submit duplicates.

### Job stops at the time limit

Inspect `sacct` and the Slurm log. Completed model directories remain on disk.
Do not delete them. Before resubmitting, either choose a new output directory or
explicitly decide which model should be rerun using the runner's `--models`
option.

### BERTScore downloads weights

The first full-metric run may download model weights from Hugging Face. The
cache under the user's home directory persists for later runs. If compute-node
network access is unavailable, arrange the cache on the login node according to
UC3 policy instead of removing BERTScore from the reported metric suite.

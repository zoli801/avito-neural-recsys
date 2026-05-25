#!/usr/bin/env bash
set -euo pipefail

python scripts/eda.py
EPOCHS="${EPOCHS:-6}" \
BATCH_SIZE="${BATCH_SIZE:-1024}" \
MAX_TRAIN_POSITIVES="${MAX_TRAIN_POSITIVES:-2000000}" \
python train.py

CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-4096}" \
python predict.py --out "${SUBMISSION_PATH:-/kaggle/working/submission.csv}"

python - <<'PY'
import os
import pandas as pd

path = os.getenv("SUBMISSION_PATH", "/kaggle/working/submission.csv")
sub = pd.read_csv(path)
assert list(sub.columns) == ["user_id", "item_id"]
assert sub.duplicated(["user_id", "item_id"]).sum() == 0
assert sub.groupby("user_id").size().max() <= 160
print(f"submission ok: {path}, rows={len(sub):,}, users={sub['user_id'].nunique():,}")
PY

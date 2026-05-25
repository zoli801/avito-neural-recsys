#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-https://storage.yandexcloud.net/datafest2026/datafest_2026_v2_v4}"
DATA_DIR="${DATA_DIR:-data}"
mkdir -p "$DATA_DIR"

files=(
  item_features.parquet
  contact_eids.csv
  eval_users.csv
  eval_user_events.zip
)

for file in "${files[@]}"; do
  if [[ ! -f "$DATA_DIR/$file" ]]; then
    curl -L "$BASE/$file" -o "$DATA_DIR/$file"
  fi
done

echo "Core inference/training files are in $DATA_DIR"

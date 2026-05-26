# AvitoTech ML CUP 2026: Neural-Only Recommender

Public repository for the additional nomination **"Лучшее нейросетевое решение"**.

This solution uses a single end-to-end PyTorch neural recommender. It trains on user event sequences and item metadata, predicts next contact items, and writes `submission.csv` in the official `user_id,item_id` format.

## Status

The Kaggle notebook run produced a valid submission:

- rows: `15,105,280`
- eval users: `94,408`
- recommendations per user: exactly `160`
- duplicate `user_id,item_id` pairs: `0`
- reported score from the improved run: `0.003482787346320407`

Validation details are in `reports/submission_validation.json`. The generated CSV is large and ignored by git, but the local artifact is available at `submissions/submission.csv`.

Improved run artifacts:

- training notebook: `notebooks/avito_neural_solution_submit.ipynb`
- feature-selection report: `reports/feature_selection.json`
- EDA figures: `reports/contact_item_tail.png`, `reports/event_id_frequency.png`
- model weights: [GitHub release `v0.2-improved-neural`](https://github.com/zoli801/avito-neural-recsys/releases/tag/v0.2-improved-neural)

## Neural-Only Claim

There is one trainable model in the improved notebook: `FeatureSeqNeuralRecommender`. The script pipeline under `src/avito_nn/` also uses one PyTorch neural recommender for Docker reproducibility.

No CatBoost, LightGBM, matrix-factorization ensemble, blended ranker, or hand-tuned second-stage model is used. Candidate selection is deterministic and memory-bounded; final ranking is produced by the neural network.

## Repository Layout

```text
.
├── train.py
├── predict.py
├── Dockerfile
├── notebooks/avito_neural_solution_submit.ipynb
├── src/avito_nn/
├── scripts/
├── reports/submission_validation.json
├── reports/feature_selection.json
├── models/          # generated, git-ignored
└── submissions/     # generated, git-ignored
```

## Data

Kaggle dataset path used by the notebook:

```text
/kaggle/input/datasets/nikitakuznetsof/avito-ml-cup-2026
```

For Docker/local runs, mount the official data directory as `/data`. It should contain:

```text
item_features.parquet
contact_eids.csv
eval_users.csv
eval_user_events/eval_user_events.pq
train_*/train_data/part_*.parquet  # optional for larger runs
```

## Required Commands

Inside the container:

```bash
python train.py      # creates model artifacts in /app/models
python predict.py    # creates /app/submission.csv
```

## Docker

```bash
docker build -t avito-neural-recsys .
docker run --rm --gpus all \
  -v /path/to/avito-data:/data \
  -v "$PWD/models:/app/models" \
  -v "$PWD:/app/out" \
  avito-neural-recsys python train.py

docker run --rm --gpus all \
  -v /path/to/avito-data:/data \
  -v "$PWD/models:/app/models" \
  -v "$PWD:/app/out" \
  avito-neural-recsys python predict.py --out /app/out/submission.csv
```

## Kaggle Notebook

The exact notebook used for the successful run is:

```text
notebooks/avito_neural_solution_submit.ipynb
```

It is intentionally RAM-bounded:

- streams fresh train partitions and eval events;
- builds a bounded item vocabulary;
- reads `item_features.parquet` in batches only for vocabulary items;
- uses event sequence features plus item metadata features;
- trains one neural recommender (`FeatureSeqNeuralRecommender`);
- writes `submission.csv` in the current directory.

## Reproducibility Knobs

Useful environment variables for script runs:

```bash
DATA_DIR=/data
MODEL_DIR=/app/models
SUBMISSION_PATH=/app/submission.csv
EPOCHS=4
BATCH_SIZE=512
CANDIDATE_POOL_SIZE=4096
DEVICE=cuda
```

## Submission Format

`predict.py` and the notebook both write:

```csv
user_id,item_id
33,172768884
33,5569823
...
```

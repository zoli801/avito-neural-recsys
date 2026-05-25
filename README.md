# AvitoTech ML CUP 2026: Neural Retrieval

One reproducible end-to-end neural solution for the **"Лучшее нейросетевое решение"** nomination.

The model is a single PyTorch sequential recommender. It encodes a user's pre-cutoff event history with a Transformer encoder, encodes item metadata (`vertical_id`, category, geo ids and semantic ids) with learned embeddings, and trains with sampled-softmax next-contact retrieval. `predict.py` scores the catalog with the same network and writes `submission.csv` in the competition format.

## Competition Facts

- Task: predict up to 160 relevant `item_id` values for each eval `user_id`.
- Submission format: two CSV columns, `user_id,item_id`; pairs must be unique.
- Metric: mean user-level `Recall@160`.
- Public data: `item_features.parquet`, `contact_eids.csv`, `eval_users.csv`, `eval_user_events.zip`, and optional train partitions.
- Official pages: [task](https://ods.ai/competitions/avitotechmlcup2026), [dataset](https://ods.ai/competitions/avitotechmlcup2026/dataset).

## Repository Layout

```text
.
├── train.py                 # python train.py -> model artifacts
├── predict.py               # python predict.py -> submissions/submission.csv
├── src/avito_nn/            # data loading, model, train/predict implementation
├── scripts/download_data.sh # downloads public core files
├── scripts/eda.py           # produces reports/figures/*.png
├── reports/                 # EDA outputs
├── models/                  # generated model artifacts, ignored by git
└── Dockerfile
```

## Quick Start

```bash
bash scripts/download_data.sh
python scripts/eda.py
python train.py
python predict.py
```

The generated submission is written to `submissions/submission.csv`. In Kaggle the scripts automatically use `/kaggle/input/datasets/nikitakuznetsof/avito-ml-cup-2026` when `DATA_DIR` is not set.

For the Kaggle GPU run used to create a submit-ready artifact:

```bash
bash scripts/run_kaggle_40m.sh
```

## Docker

```bash
docker build -t avito-neural-recsys .
docker run --rm --gpus all -v "$PWD/data:/data" -v "$PWD/models:/app/models" -v "$PWD/submissions:/app/submissions" avito-neural-recsys python train.py
docker run --rm --gpus all -v "$PWD/data:/data" -v "$PWD/models:/app/models" -v "$PWD/submissions:/app/submissions" avito-neural-recsys python predict.py
```

CPU also works, but full-catalog prediction is intended for a GPU runtime.

## Data

Put the official files under `data/`:

```text
data/
├── item_features.parquet
├── contact_eids.csv
├── eval_users.csv
└── eval_user_events.zip or eval_user_events.pq
```

On Kaggle, the attached dataset path from `main.ipynb` is supported directly:

```text
/kaggle/input/datasets/nikitakuznetsof/avito-ml-cup-2026/
```

Optional train partitions can be sampled by setting `MAX_TRAIN_ROWS`, but the default solution uses the published eval-user pre-cutoff history for a compact reproducible run.

## EDA

Run:

```bash
python scripts/eda.py
```

It writes:

- `reports/figures/items_by_vertical.png`
- `reports/figures/history_length.png`
- `reports/figures/event_id_frequency.png`
- `reports/figures/contacts_by_vertical.png`
- `reports/eda_summary.csv`

## Reproducibility Knobs

Environment variables:

- `DATA_DIR=/data`
- `MODEL_DIR=/app/models`
- `SUBMISSION_PATH=/app/submissions/submission.csv`
- `EPOCHS=5`
- `BATCH_SIZE=768`
- `MAX_TRAIN_POSITIVES=2000000`
- `CANDIDATE_POOL_SIZE=4096`
- `DEVICE=cuda`

## Neural-Only Claim

There is one trainable model: `AvitoNeuralRecommender`. No CatBoost, LightGBM, matrix-factorization ensemble, handcrafted ranker, or blended post-processing is used. Candidate scores in the final CSV come directly from the neural model.

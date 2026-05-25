from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import encode_existing_items, load_eval_users, load_events, load_item_features, load_json
from .model import AvitoNeuralRecommender
from .paths import default_data_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate submission.csv with the trained neural recommender.")
    p.add_argument("--data-dir", default=default_data_dir())
    p.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "models"))
    p.add_argument("--out", default=os.getenv("SUBMISSION_PATH", "submissions/submission.csv"))
    p.add_argument("--device", default=os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--candidate-pool-size", type=int, default=int(os.getenv("CANDIDATE_POOL_SIZE", "4096")))
    return p.parse_args()


def make_user_batches(events: pd.DataFrame, eval_users: np.ndarray, item_to_row: Dict[int, int], max_seq_len: int):
    encoded = encode_existing_items(events, item_to_row)
    grouped = {int(u): g for u, g in encoded.groupby("user_id", sort=False)}
    for user_id in eval_users:
        g = grouped.get(int(user_id))
        item_arr = np.full(max_seq_len, -1, dtype=np.int64)
        eid_arr = np.zeros(max_seq_len, dtype=np.int64)
        mask = np.zeros(max_seq_len, dtype=np.float32)
        seen = set()
        if g is not None and len(g):
            tail = g.tail(max_seq_len)
            rows = tail["item_row"].to_numpy(np.int64)
            eids = tail["eid"].to_numpy(np.int64)
            pad = max_seq_len - len(rows)
            item_arr[pad:] = rows
            eid_arr[pad:] = eids
            mask[pad:] = 1.0
            seen = set(g["item_row"].astype(int).tolist())
        yield int(user_id), item_arr, eid_arr, mask, seen


def build_candidate_pool(events: pd.DataFrame, items: pd.DataFrame, item_to_row: Dict[int, int], pool_size: int) -> np.ndarray:
    counts = events.groupby("item_id", sort=False).size().sort_values(ascending=False)
    pool: List[int] = [item_to_row[int(i)] for i in counts.index if int(i) in item_to_row]

    if len(pool) < pool_size and "vertical_id" in items.columns:
        allowed_verticals = {0, 2, 3, 4, 5, 7}
        extra = items[items["vertical_id"].isin(allowed_verticals)]["item_id"].to_numpy()
        pool.extend(item_to_row[int(i)] for i in extra if int(i) in item_to_row)
    if len(pool) < pool_size:
        pool.extend(range(len(items)))

    deduped = list(dict.fromkeys(int(x) for x in pool))
    return np.asarray(deduped[:pool_size], dtype=np.int64)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    meta = load_json(model_dir / "metadata.json")
    item_ids = np.load(model_dir / "item_ids.npy")
    item_features = np.load(model_dir / "item_features.npy")
    item_to_row = {int(item_id): i for i, item_id in enumerate(item_ids)}

    model = AvitoNeuralRecommender(
        cardinalities=meta["cardinalities"],
        item_feature_matrix=torch.from_numpy(item_features),
        emb_dim=meta["emb_dim"],
        hidden_dim=meta["hidden_dim"],
        max_eid=int(meta.get("max_eid", max(max(meta["contact_eids"]), 1))),
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.eval()

    eval_users = load_eval_users(data_dir)
    events = load_events(data_dir, max_train_rows=0)
    k = int(meta.get("submission_k", 160))
    user_batch_size = int(meta.get("user_batch_size", 128))
    items_df = load_item_features(data_dir)
    candidate_rows_np = build_candidate_pool(events, items_df, item_to_row, int(args.candidate_pool_size))
    candidate_rows = torch.tensor(candidate_rows_np, dtype=torch.long, device=device)
    if len(candidate_rows_np) < k:
        raise RuntimeError(f"Candidate pool has only {len(candidate_rows_np)} items, need at least {k}.")
    print(f"Scoring neural candidate pool: {len(candidate_rows_np):,} items")

    rows: List[tuple[int, int]] = []
    stream = list(make_user_batches(events, eval_users, item_to_row, int(meta["max_seq_len"])))
    with torch.no_grad():
        temperature = model.temperature.abs().clamp_min(0.02)
        item_vec = model.item_encode(candidate_rows)
        item_bias = model.item_bias(candidate_rows).squeeze(-1)
        for start in tqdm(range(0, len(stream), user_batch_size), desc="scoring users"):
            chunk = stream[start : start + user_batch_size]
            users = [x[0] for x in chunk]
            seq_items = torch.tensor(np.stack([x[1] for x in chunk]), dtype=torch.long, device=device)
            seq_eids = torch.tensor(np.stack([x[2] for x in chunk]), dtype=torch.long, device=device)
            mask = torch.tensor(np.stack([x[3] for x in chunk]), dtype=torch.float32, device=device)
            user_vec = model.user_encode(seq_items, seq_eids, mask)
            scores = (user_vec @ item_vec.T) / temperature + item_bias.unsqueeze(0)
            for i, (_, _, _, _, seen) in enumerate(chunk):
                if seen:
                    seen_idx = np.flatnonzero(np.isin(candidate_rows_np, np.fromiter(seen, dtype=np.int64)))
                    if len(seen_idx):
                        scores[i, torch.tensor(seen_idx, dtype=torch.long, device=device)] = -1e30
            _, idx = torch.topk(scores, k=k, dim=1)
            best_rows = candidate_rows[idx]
            pred_rows = best_rows.cpu().numpy()
            for user_id, rec_rows in zip(users, pred_rows):
                rows.extend((user_id, int(item_ids[r])) for r in rec_rows)

    sub = pd.DataFrame(rows, columns=["user_id", "item_id"]).drop_duplicates(["user_id", "item_id"])
    if sub.groupby("user_id").size().max() > k:
        raise RuntimeError("More than 160 recommendations for at least one user.")
    sub.to_csv(out_path, index=False)
    print(f"Saved {len(sub):,} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()

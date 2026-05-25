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
    all_item_rows = torch.arange(len(item_ids), dtype=torch.long, device=device)
    k = int(meta.get("submission_k", 160))
    item_chunk_size = int(meta.get("item_chunk_size", 65536))
    user_batch_size = int(meta.get("user_batch_size", 128))

    rows: List[tuple[int, int]] = []
    stream = list(make_user_batches(events, eval_users, item_to_row, int(meta["max_seq_len"])))
    with torch.no_grad():
        for start in tqdm(range(0, len(stream), user_batch_size), desc="scoring users"):
            chunk = stream[start : start + user_batch_size]
            users = [x[0] for x in chunk]
            seq_items = torch.tensor(np.stack([x[1] for x in chunk]), dtype=torch.long, device=device)
            seq_eids = torch.tensor(np.stack([x[2] for x in chunk]), dtype=torch.long, device=device)
            mask = torch.tensor(np.stack([x[3] for x in chunk]), dtype=torch.float32, device=device)
            user_vec = model.user_encode(seq_items, seq_eids, mask)
            best_scores = torch.full((len(users), k), -1e30, device=device)
            best_rows = torch.zeros((len(users), k), dtype=torch.long, device=device)
            temperature = model.temperature.abs().clamp_min(0.02)
            for item_start in range(0, len(item_ids), item_chunk_size):
                item_rows = all_item_rows[item_start : item_start + item_chunk_size]
                item_vec = model.item_encode(item_rows)
                scores = user_vec @ item_vec.T
                scores = scores / temperature
                scores = scores + model.item_bias(item_rows).squeeze(-1).unsqueeze(0)
                for i, (_, _, _, _, seen) in enumerate(chunk):
                    if seen:
                        local_seen = [r - item_start for r in seen if item_start <= r < item_start + len(item_rows)]
                        if local_seen:
                            scores[i, torch.tensor(local_seen, dtype=torch.long, device=device)] = -1e30
                merged_scores = torch.cat([best_scores, scores], dim=1)
                merged_rows = torch.cat([best_rows, item_rows.unsqueeze(0).expand(len(users), -1)], dim=1)
                best_scores, idx = torch.topk(merged_scores, k=k, dim=1)
                best_rows = torch.gather(merged_rows, 1, idx)
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

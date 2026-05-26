from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset


EVENT_COLS = ["timestamp", "eid", "user_id", "item_id"]
ITEM_ID_COL = "item_id"
PREFERRED_ITEM_FEATURES = [
    "vertical_id",
    "category_ext_y",
    "region_id_y",
    "loc_id_y",
    "sid_0_y",
    "sid_1_y",
    "sid_2_y",
    "sid_3_y",
]
EVAL_EVENTS_NAMES = ["eval_user_events.pq", "eval_user_events.parquet", "eval_user_events.zip"]
EVAL_USERS_NAMES = ["eval_users.csv", "eval_users.pq", "eval_users.parquet"]
ITEM_FEATURE_NAMES = ["item_features.parquet", "item_features.pq"]
CONTACT_EIDS_NAMES = ["contact_eids.csv"]
TRAIN_NAMES = ["train_data", "train.parquet", "train.pq"]


def find_existing(data_dir: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = data_dir / name
        if path.exists():
            return path
    for name in names:
        hits = list(data_dir.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not find any of {names} under {data_dir}")


def maybe_unzip_eval_events(path: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path
    out_dir = path.with_suffix("")
    out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        parquet_names = [n for n in zf.namelist() if n.endswith((".pq", ".parquet"))]
        if not parquet_names:
            raise FileNotFoundError(f"{path} has no parquet files")
        target = out_dir / Path(parquet_names[0]).name
        if not target.exists():
            zf.extract(parquet_names[0], out_dir)
            extracted = out_dir / parquet_names[0]
            if extracted != target:
                extracted.rename(target)
    return target


def read_table(path: Path, columns: Sequence[str] | None = None, max_rows: int | None = None) -> pd.DataFrame:
    if path.is_dir():
        table = ds.dataset(path, format="parquet").to_table(columns=columns)
        df = table.to_pandas()
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=columns)
    else:
        df = pd.read_parquet(path, columns=columns)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).sort_index()
    return df


def load_contact_eids(data_dir: Path) -> List[int]:
    path = find_existing(data_dir, CONTACT_EIDS_NAMES)
    df = pd.read_csv(path)
    col = "mapped_eid" if "mapped_eid" in df.columns else df.columns[0]
    return sorted(df[col].astype(int).unique().tolist())


def load_events(data_dir: Path, max_train_rows: int = 0) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if max_train_rows > 0:
        try:
            train_path = find_existing(data_dir, TRAIN_NAMES)
            frames.append(read_table(train_path, columns=EVENT_COLS, max_rows=max_train_rows))
        except FileNotFoundError:
            pass
    eval_path = maybe_unzip_eval_events(find_existing(data_dir, EVAL_EVENTS_NAMES))
    frames.append(read_table(eval_path, columns=EVENT_COLS, max_rows=None))
    events = pd.concat(frames, ignore_index=True)
    events = events.dropna(subset=EVENT_COLS).copy()
    for col in EVENT_COLS:
        events[col] = events[col].astype("int64")
    return events.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def load_eval_users(data_dir: Path) -> np.ndarray:
    path = find_existing(data_dir, EVAL_USERS_NAMES)
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    user_col = "user_id" if "user_id" in df.columns else df.columns[0]
    return df[user_col].astype("int64").to_numpy()


def load_item_features(data_dir: Path) -> pd.DataFrame:
    path = find_existing(data_dir, ITEM_FEATURE_NAMES)
    schema_cols = pd.read_parquet(path, columns=[]).columns.tolist()
    if not schema_cols:
        import pyarrow.parquet as pq

        schema_cols = pq.ParquetFile(path).schema.names
    cols = [ITEM_ID_COL] + [c for c in PREFERRED_ITEM_FEATURES if c in schema_cols]
    if len(cols) == 1:
        # Fallback for unexpected schemas: keep a compact prefix, never read the full 1.85GB payload.
        cols = [ITEM_ID_COL] + [c for c in schema_cols if c != ITEM_ID_COL][:8]
    df = pd.read_parquet(path, columns=cols)
    if ITEM_ID_COL not in df.columns:
        raise ValueError(f"{path} must contain {ITEM_ID_COL}")
    df = df.drop_duplicates(ITEM_ID_COL).reset_index(drop=True)
    df[ITEM_ID_COL] = df[ITEM_ID_COL].astype("int64")
    return df


def choose_feature_columns(items: pd.DataFrame) -> List[str]:
    cols = [c for c in PREFERRED_ITEM_FEATURES if c in items.columns]
    if cols:
        return cols
    banned = {ITEM_ID_COL, "title", "description", "image", "text"}
    out: List[str] = []
    for col in items.columns:
        low = str(col).lower()
        if low in banned or any(x in low for x in ["embedding", "vector"]):
            continue
        if pd.api.types.is_numeric_dtype(items[col]) or items[col].dtype == "object":
            out.append(col)
    return out[:16]


def build_item_matrix(items: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Dict[str, int]], Dict[int, int]]:
    vocabs: Dict[str, Dict[str, int]] = {}
    matrix = np.zeros((len(items), len(feature_cols)), dtype=np.int64)
    for j, col in enumerate(feature_cols):
        values = items[col].fillna("__NA__").astype(str)
        uniques = pd.Index(values.unique())
        vocab = {v: i + 1 for i, v in enumerate(uniques)}
        vocabs[col] = vocab
        matrix[:, j] = values.map(vocab).fillna(0).astype("int64").to_numpy()
    item_to_row = {int(v): i for i, v in enumerate(items[ITEM_ID_COL].to_numpy())}
    return matrix, vocabs, item_to_row


def encode_existing_items(events: pd.DataFrame, item_to_row: Dict[int, int]) -> pd.DataFrame:
    events = events.copy()
    events["item_row"] = events["item_id"].map(item_to_row)
    events = events.dropna(subset=["item_row"])
    events["item_row"] = events["item_row"].astype("int64")
    return events


def build_training_examples(
    events: pd.DataFrame,
    max_seq_len: int,
    max_examples: int,
    contact_eids: Iterable[int],
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    contact_eids = set(int(x) for x in contact_eids)
    examples: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for _, user_events in events.groupby("user_id", sort=False):
        rows = user_events["item_row"].to_numpy(np.int64)
        eids = user_events["eid"].to_numpy(np.int64)
        if len(rows) < 2:
            continue
        for pos in range(1, len(rows)):
            if int(eids[pos]) not in contact_eids:
                continue
            start = max(0, pos - max_seq_len)
            examples.append((rows[start:pos], eids[start:pos], int(rows[pos])))
            if len(examples) >= max_examples:
                random.Random(42).shuffle(examples)
                return examples
    random.Random(42).shuffle(examples)
    return examples


class NextContactDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Tuple[np.ndarray, np.ndarray, int]],
        n_items: int,
        max_seq_len: int,
        negatives: int,
        seed: int,
    ) -> None:
        self.examples = list(examples)
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.negatives = negatives
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_items, seq_eids, pos_item = self.examples[idx]
        seq_items = seq_items[-self.max_seq_len:]
        seq_eids = seq_eids[-self.max_seq_len:]
        pad = self.max_seq_len - len(seq_items)
        item_arr = np.full(self.max_seq_len, -1, dtype=np.int64)
        eid_arr = np.zeros(self.max_seq_len, dtype=np.int64)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        if len(seq_items):
            item_arr[pad:] = seq_items
            eid_arr[pad:] = seq_eids
            mask[pad:] = 1.0
        neg = self.rng.integers(0, self.n_items, size=self.negatives, dtype=np.int64)
        neg[neg == pos_item] = (pos_item + 1) % self.n_items
        return {
            "seq_items": torch.from_numpy(item_arr),
            "seq_eids": torch.from_numpy(eid_arr),
            "mask": torch.from_numpy(mask),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "neg_items": torch.from_numpy(neg),
        }


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

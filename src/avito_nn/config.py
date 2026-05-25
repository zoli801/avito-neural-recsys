from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path
    model_dir: Path
    max_seq_len: int = 64
    emb_dim: int = 128
    hidden_dim: int = 192
    batch_size: int = 768
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    negatives: int = 64
    max_train_rows: int = 0
    max_train_positives: int = 2_000_000
    submission_k: int = 160
    item_chunk_size: int = 65536
    user_batch_size: int = 128
    candidate_pool_size: int = 4096
    seed: int = 42

from __future__ import annotations

import os
from pathlib import Path


KAGGLE_DATA_DIR = Path("/kaggle/input/datasets/nikitakuznetsof/avito-ml-cup-2026")


def default_data_dir() -> str:
    env = os.getenv("DATA_DIR")
    if env:
        return env
    if KAGGLE_DATA_DIR.exists():
        return str(KAGGLE_DATA_DIR)
    return "data"

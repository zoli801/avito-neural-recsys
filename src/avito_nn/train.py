from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .config import Config
from .data import (
    NextContactDataset,
    build_item_matrix,
    build_training_examples,
    choose_feature_columns,
    encode_existing_items,
    load_contact_eids,
    load_events,
    load_item_features,
    save_json,
)
from .model import AvitoNeuralRecommender
from .paths import default_data_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one end-to-end neural recommender for AvitoTech ML Cup 2026.")
    p.add_argument("--data-dir", default=default_data_dir())
    p.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "models"))
    p.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "5")))
    p.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "768")))
    p.add_argument("--max-train-rows", type=int, default=int(os.getenv("MAX_TRAIN_ROWS", "0")))
    p.add_argument("--max-train-positives", type=int, default=int(os.getenv("MAX_TRAIN_POSITIVES", "2000000")))
    p.add_argument("--device", default=os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    cfg = Config(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_rows=args.max_train_rows,
        max_train_positives=args.max_train_positives,
    )
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(args.device)

    print("Loading data...")
    items = load_item_features(cfg.data_dir)
    feature_cols = choose_feature_columns(items)
    item_matrix, vocabs, item_to_row = build_item_matrix(items, feature_cols)
    events = load_events(cfg.data_dir, max_train_rows=cfg.max_train_rows)
    events = encode_existing_items(events, item_to_row)
    contact_eids = load_contact_eids(cfg.data_dir)

    print(f"Items: {len(items):,}; events: {len(events):,}; contact eids: {contact_eids}")
    examples = build_training_examples(events, cfg.max_seq_len, cfg.max_train_positives, contact_eids)
    if not examples:
        raise RuntimeError("No positive contact examples were found. Check contact_eids.csv and event data.")
    print(f"Training examples: {len(examples):,}")

    dataset = NextContactDataset(examples, len(items), cfg.max_seq_len, cfg.negatives, cfg.seed)
    val_size = max(1, min(len(dataset) // 20, 50_000))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")

    cardinalities = [len(v) for v in vocabs.values()]
    item_tensor = torch.from_numpy(item_matrix)
    max_eid = max(max(contact_eids), int(events["eid"].max()), 1)
    model = AvitoNeuralRecommender(
        cardinalities=cardinalities,
        item_feature_matrix=item_tensor,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        max_eid=max_eid,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = torch.zeros(batch["pos_item"].shape[0], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(batch)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=np.mean(losses[-50:]))

        model.eval()
        val_losses, top1 = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                labels = torch.zeros(batch["pos_item"].shape[0], dtype=torch.long, device=device)
                logits = model(batch)
                val_losses.append(float(F.cross_entropy(logits, labels).cpu()))
                top1.append(float((logits.argmax(1) == 0).float().mean().cpu()))
        val_loss = float(np.mean(val_losses))
        val_top1 = float(np.mean(top1))
        print(f"epoch={epoch} train_loss={np.mean(losses):.5f} val_loss={val_loss:.5f} val_top1={val_top1:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.model_dir / "model.pt")

    np.save(cfg.model_dir / "item_ids.npy", items["item_id"].astype("int64").to_numpy())
    np.save(cfg.model_dir / "item_features.npy", item_matrix)
    save_json(
        cfg.model_dir / "metadata.json",
        {
            "feature_cols": feature_cols,
            "cardinalities": cardinalities,
            "max_seq_len": cfg.max_seq_len,
            "emb_dim": cfg.emb_dim,
            "hidden_dim": cfg.hidden_dim,
            "contact_eids": contact_eids,
            "max_eid": max_eid,
            "submission_k": cfg.submission_k,
            "item_chunk_size": cfg.item_chunk_size,
            "user_batch_size": cfg.user_batch_size,
            "candidate_pool_size": cfg.candidate_pool_size,
            "best_val_loss": best_val,
            "source": "AvitoTech ML CUP 2026 public data",
        },
    )
    print(f"Saved model artifacts to {cfg.model_dir.resolve()}")


if __name__ == "__main__":
    main()

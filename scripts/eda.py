from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

KAGGLE_DATA_DIR = Path("/kaggle/input/datasets/nikitakuznetsof/avito-ml-cup-2026")


def main() -> None:
    data_dir = KAGGLE_DATA_DIR if KAGGLE_DATA_DIR.exists() else Path("data")
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(data_dir / "item_features.parquet")
    events_path = data_dir / "eval_user_events.pq"
    if not events_path.exists():
        events_path = next((data_dir / "eval_user_events").glob("*.pq"))
    events = pd.read_parquet(events_path, columns=["timestamp", "eid", "user_id", "item_id"])
    contact = pd.read_csv(data_dir / "contact_eids.csv")
    contact_col = "mapped_eid" if "mapped_eid" in contact.columns else contact.columns[0]
    contact_eids = set(contact[contact_col].astype(int).tolist())

    ax = items["vertical_id"].value_counts().sort_index().plot(kind="bar", color="#2f6f73")
    ax.set_title("Item count by vertical_id")
    ax.set_xlabel("vertical_id")
    ax.set_ylabel("items")
    plt.tight_layout()
    plt.savefig(out_dir / "items_by_vertical.png", dpi=160)
    plt.close()

    per_user = events.groupby("user_id").size()
    ax = per_user.clip(upper=per_user.quantile(0.99)).plot(kind="hist", bins=50, color="#8b5e3c")
    ax.set_title("Eval-user history length, clipped at p99")
    ax.set_xlabel("events per user")
    plt.tight_layout()
    plt.savefig(out_dir / "history_length.png", dpi=160)
    plt.close()

    eid_counts = events["eid"].value_counts().head(30).sort_values()
    colors = ["#b23a48" if int(eid) in contact_eids else "#4d6a8a" for eid in eid_counts.index]
    ax = eid_counts.plot(kind="barh", color=colors)
    ax.set_title("Top event ids in eval-user history")
    ax.set_xlabel("events")
    plt.tight_layout()
    plt.savefig(out_dir / "event_id_frequency.png", dpi=160)
    plt.close()

    merged = events.merge(items[["item_id", "vertical_id"]], on="item_id", how="inner")
    contact_events = merged[merged["eid"].isin(contact_eids)]
    ax = contact_events["vertical_id"].value_counts().sort_index().plot(kind="bar", color="#596f2f")
    ax.set_title("Historical contact events by vertical_id")
    ax.set_xlabel("vertical_id")
    ax.set_ylabel("contacts")
    plt.tight_layout()
    plt.savefig(out_dir / "contacts_by_vertical.png", dpi=160)
    plt.close()

    summary = {
        "items": int(len(items)),
        "eval_history_events": int(len(events)),
        "eval_history_users": int(events["user_id"].nunique()),
        "historical_contact_events": int(events["eid"].isin(contact_eids).sum()),
        "contact_eids": sorted(int(x) for x in contact_eids),
    }
    pd.Series(summary).to_csv("reports/eda_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()

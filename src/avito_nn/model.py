from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn


class AvitoNeuralRecommender(nn.Module):
    def __init__(
        self,
        cardinalities: Iterable[int],
        item_feature_matrix: torch.Tensor,
        emb_dim: int = 128,
        hidden_dim: int = 192,
        max_eid: int = 512,
    ) -> None:
        super().__init__()
        self.register_buffer("item_feature_matrix", item_feature_matrix.long())
        self.feature_embeddings = nn.ModuleList(
            [nn.Embedding(int(card) + 1, emb_dim, padding_idx=0) for card in cardinalities]
        )
        self.item_norm = nn.LayerNorm(emb_dim)
        self.eid_embedding = nn.Embedding(max_eid + 1, emb_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.position_embedding = nn.Embedding(256, emb_dim)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.user_proj = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, emb_dim), nn.Tanh())
        self.item_proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.LayerNorm(emb_dim))
        self.item_bias = nn.Embedding(item_feature_matrix.shape[0], 1)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def item_encode(self, item_rows: torch.Tensor) -> torch.Tensor:
        flat = item_rows.reshape(-1).clamp_min(0)
        feats = self.item_feature_matrix[flat]
        emb = 0
        for j, layer in enumerate(self.feature_embeddings):
            emb = emb + layer(feats[:, j])
        emb = self.item_proj(self.item_norm(emb))
        return emb.reshape(*item_rows.shape, -1)

    def user_encode(self, seq_items: torch.Tensor, seq_eids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe_items = seq_items.clamp_min(0)
        item_emb = self.item_encode(safe_items)
        eid_emb = self.eid_embedding(seq_eids.clamp(0, self.eid_embedding.num_embeddings - 1))
        positions = torch.arange(seq_items.shape[1], device=seq_items.device).unsqueeze(0)
        x = (item_emb + eid_emb + self.position_embedding(positions)) * mask.unsqueeze(-1)
        key_padding = mask.eq(0)
        empty_rows = mask.sum(1).eq(0)
        if empty_rows.any():
            key_padding = key_padding.clone()
            key_padding[empty_rows] = False
        encoded = self.encoder(x, src_key_padding_mask=key_padding)
        denom = mask.sum(1, keepdim=True).clamp_min(1.0)
        pooled = (encoded * mask.unsqueeze(-1)).sum(1) / denom
        return self.user_proj(pooled)

    def score_items(self, user_vec: torch.Tensor, item_rows: torch.Tensor) -> torch.Tensor:
        item_vec = self.item_encode(item_rows)
        score = (user_vec.unsqueeze(-2) * item_vec).sum(-1)
        score = score / self.temperature.abs().clamp_min(0.02)
        score = score + self.item_bias(item_rows.clamp_min(0)).squeeze(-1)
        return score

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_vec = self.user_encode(batch["seq_items"], batch["seq_eids"], batch["mask"])
        items = torch.cat([batch["pos_item"].unsqueeze(1), batch["neg_items"]], dim=1)
        return self.score_items(user_vec, items)

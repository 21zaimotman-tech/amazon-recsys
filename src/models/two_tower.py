"""Two-tower retrieval model (PyTorch) with in-batch negatives and softmax
cross-entropy. This is the heaviest required component (10-15h) and 25% of the
grade together with MF-BPR — start it early.

================================ READ THIS ================================
Implement from scratch and EXPLAIN at the defense. The two ideas the examiners
will probe: (1) why in-batch negatives, (2) why softmax cross-entropy over the
batch. Make sure everyone can answer.
==========================================================================

Architecture
------------
USER TOWER  : takes the user's recent interaction history (a set of item ids),
              embeds + pools them into a user vector. (An id-embedding for the
              user works too, but a history-based tower is what lets the API
              serve users using TEST-PERIOD history the model never trained on.)
ITEM TOWER  : embeds an item id (optionally + side features) into an item vector.

In-batch negatives: in a batch of B (user, positive-item) pairs, each user's
positive is everyone else's negative. We score every user against every item in
the batch (B x B matrix) and apply cross-entropy so the diagonal (the true pair)
wins. Cheap, effective, and the standard YouTube/Google two-tower recipe.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTower(nn.Module):
    def __init__(self, n_items, dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(n_items, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        nn.init.normal_(self.emb.weight, std=0.01)

    def forward(self, item_ids):
        return self.mlp(self.emb(item_ids))


class UserTower(nn.Module):
    """Pools the user's history item-embeddings (mean pooling) then an MLP.
    Shares the item embedding table with ItemTower so history and candidates
    live in the same space."""
    def __init__(self, item_emb: nn.Embedding, dim=64, hidden=128):
        super().__init__()
        self.item_emb = item_emb
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, history_ids, history_mask):
        # history_ids: (B, L) padded; history_mask: (B, L) 1 for real items
        emb = self.item_emb(history_ids)                      # (B, L, dim)
        summed = (emb * history_mask.unsqueeze(-1)).sum(1)
        counts = history_mask.sum(1, keepdim=True).clamp(min=1)
        pooled = summed / counts                              # mean pool
        return self.mlp(pooled)


class TwoTower(nn.Module):
    def __init__(self, n_items, dim=64, hidden=128, temperature=0.05):
        super().__init__()
        self.item_tower = ItemTower(n_items, dim, hidden)
        self.user_tower = UserTower(self.item_tower.emb, dim, hidden)
        self.temperature = temperature

    def in_batch_loss(self, history_ids, history_mask, pos_item_ids):
        """----- THE CORE LINES YOU MUST BE ABLE TO EXPLAIN -----
        u = self.user_tower(history_ids, history_mask)        # (B, dim)
        v = self.item_tower(pos_item_ids)                     # (B, dim)
        u, v = F.normalize(u, dim=-1), F.normalize(v, dim=-1) # cosine geometry
        logits = (u @ v.T) / self.temperature                 # (B, B) sims
        labels = torch.arange(len(u), device=u.device)        # diagonal is correct
        return F.cross_entropy(logits, labels)
        ------------------------------------------------------"""
        u = F.normalize(self.user_tower(history_ids, history_mask), dim=-1)
        v = F.normalize(self.item_tower(pos_item_ids), dim=-1)
        logits = (u @ v.T) / self.temperature
        labels = torch.arange(len(u), device=u.device)
        return F.cross_entropy(logits, labels)


def make_history_batches(train_df, max_hist=20, batch_size=1024, seed=0):
    """Yield (history_ids, history_mask, pos_item) batches.
    For each (user, target) we use the user's *earlier* items as history — never
    the target or anything after it (no leakage)."""
    rng = np.random.default_rng(seed)
    df = train_df.sort_values("timestamp")
    seqs = df.groupby("u")["i"].agg(list).to_dict()
    samples = []
    for u, items in seqs.items():
        for t in range(1, len(items)):                 # predict items[t] from items[:t]
            samples.append((items[max(0, t - max_hist):t], items[t]))
    rng.shuffle(samples)
    for s in range(0, len(samples), batch_size):
        chunk = samples[s:s + batch_size]
        L = max(len(h) for h, _ in chunk)
        hist = np.zeros((len(chunk), L), dtype=np.int64)
        mask = np.zeros((len(chunk), L), dtype=np.float32)
        pos = np.zeros(len(chunk), dtype=np.int64)
        for r, (h, p) in enumerate(chunk):
            hist[r, :len(h)] = h; mask[r, :len(h)] = 1.0; pos[r] = p
        yield hist, mask, pos


def train_two_tower(train_df, n_items, dim=64, lr=1e-3, epochs=10,
                    batch_size=1024, max_hist=20, device="cpu"):
    model = TwoTower(n_items, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train(); running = n = 0
        for hist, mask, pos in make_history_batches(train_df, max_hist, batch_size, seed=ep):
            hist = torch.as_tensor(hist, device=device)
            mask = torch.as_tensor(mask, device=device)
            pos = torch.as_tensor(pos, device=device)
            opt.zero_grad()
            loss = model.in_batch_loss(hist, mask, pos)
            loss.backward(); opt.step()
            running += loss.item(); n += 1
        print(f"epoch {ep+1}/{epochs}  loss={running/max(n,1):.4f}")
    return model


@torch.no_grad()
def export_item_embeddings(model, n_items, device="cpu"):
    ids = torch.arange(n_items, device=device)
    v = F.normalize(model.item_tower(ids), dim=-1)
    return v.cpu().numpy()


@torch.no_grad()
def user_vector(model, history_idx, device="cpu"):
    """Online: forward-pass a user's TEST-PERIOD history through the user tower.
    history_idx: list[int] of encoded item ids. Returns a (dim,) vector for FAISS."""
    if not history_idx:
        return None                              # cold user -> popularity fallback
    h = torch.as_tensor([history_idx], device=device)
    m = torch.ones_like(h, dtype=torch.float32)
    return F.normalize(model.user_tower(h, m), dim=-1)[0].cpu().numpy()

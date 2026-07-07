"""Matrix Factorization with BPR pairwise loss (PyTorch), trained with SGD.

================================ READ THIS ================================
The brief requires you to implement this FROM SCRATCH and EXPLAIN EVERY LINE
at the defense. Use this as a reference for the structure, then make sure each
of you can derive the BPR loss and explain the sampling. Do not submit code you
cannot defend — the examiners will ask.
==========================================================================

BPR idea: for a user u, an observed (positive) item i should score higher than
an unobserved (negative) item j. We optimise the pairwise ranking:

    L = - sum over (u,i,j) of  log sigmoid( s_ui - s_uj )  +  reg * ||params||^2

where s_ui = <p_u, q_i> + b_i  is the predicted score (dot product of user and
item embeddings plus an item bias).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class MFBPR(nn.Module):
    def __init__(self, n_users, n_items, dim=64, reg=1e-5):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)
        self.reg = reg

    def score(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(-1) + self.item_bias(i).squeeze(-1)

    def bpr_loss(self, u, i, j):
        """Pairwise BPR loss for a batch of (user, pos_item, neg_item) triplets.

        ----- THE CORE LINES YOU MUST BE ABLE TO EXPLAIN -----
        s_ui = self.score(u, i)
        s_uj = self.score(u, j)
        # maximise margin (s_ui - s_uj) => minimise -log(sigmoid(margin))
        loss = -torch.log(torch.sigmoid(s_ui - s_uj) + 1e-9).mean()
        # L2 regularisation on the embeddings used in this batch
        reg = self.reg * (self.user_emb(u).pow(2).sum()
                          + self.item_emb(i).pow(2).sum()
                          + self.item_emb(j).pow(2).sum())
        return loss + reg / len(u)
        ------------------------------------------------------
        """
        s_ui = self.score(u, i)
        s_uj = self.score(u, j)
        loss = -torch.log(torch.sigmoid(s_ui - s_uj) + 1e-9).mean()
        reg = self.reg * (self.user_emb(u).pow(2).sum()
                          + self.item_emb(i).pow(2).sum()
                          + self.item_emb(j).pow(2).sum())
        return loss + reg / len(u)


def sample_triplets(train_df, n_items, rng, batch_size, pos=None, user_items=None):
    """Uniformly sample (u, pos_i, neg_j). neg_j is a random item the user has
    not interacted with. BONUS: try popularity-based negative sampling and
    compare (see the bonus list).

    `pos`/`user_items` can be precomputed once by the caller and passed in —
    rebuilding `user_items` (a groupby + per-user set) from scratch on every
    call is fine for a handful of calls, but train_mfbpr calls this once per
    *step* (thousands of times per epoch), where re-deriving it every time
    dominates the runtime and starves the GPU of work. Passing it in once
    keeps the actual sampling logic below identical."""
    if pos is None:
        pos = train_df[["u", "i"]].values
    if user_items is None:
        user_items = train_df.groupby("u")["i"].agg(set).to_dict()
    idx = rng.integers(0, len(pos), size=batch_size)
    u = pos[idx, 0]
    i = pos[idx, 1]
    j = rng.integers(0, n_items, size=batch_size)
    for k in range(batch_size):           # resample collisions with positives
        while j[k] in user_items[u[k]]:
            j[k] = rng.integers(0, n_items)
    return u, i, j


def train_mfbpr(train_df, n_users, n_items, dim=64, lr=0.05, reg=1e-5,
                epochs=20, batch_size=4096, steps_per_epoch=2000, device="cpu"):
    rng = np.random.default_rng(0)
    model = MFBPR(n_users, n_items, dim, reg).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    pos = train_df[["u", "i"]].values                             # built once, not per step
    user_items = train_df.groupby("u")["i"].agg(set).to_dict()    # built once, not per step
    for ep in range(epochs):
        model.train(); running = 0.0
        for _ in range(steps_per_epoch):
            u, i, j = sample_triplets(train_df, n_items, rng, batch_size, pos, user_items)
            u = torch.as_tensor(u, device=device); i = torch.as_tensor(i, device=device)
            j = torch.as_tensor(j, device=device)
            opt.zero_grad()
            loss = model.bpr_loss(u, i, j)
            loss.backward(); opt.step()
            running += loss.item()
        print(f"epoch {ep+1}/{epochs}  loss={running/steps_per_epoch:.4f}")
    return model


def export_item_embeddings(model):
    """Item matrix for FAISS = item_emb. (User vectors come from user_emb at
    serving time; for cold users fall back to popularity.)"""
    return model.item_emb.weight.detach().cpu().numpy()

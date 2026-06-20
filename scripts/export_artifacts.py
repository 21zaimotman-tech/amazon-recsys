"""Run AFTER training to write everything the API needs into ./artifacts.
The API loads these read-only at startup.

Produces:
  popularity.json     ranked item_ids (popularity baseline + cold fallback)
  item_encoder.json   contiguous item_id order (matches item_emb rows)
  item_emb.npy        item embedding matrix for FAISS (your best retriever)
  user_tower.pt       TorchScripted user tower -> robust, no model class needed in API
  lgbm.pkl            trained LightGBM ranker
  feature_store.pkl   {'store':..., 'now_ts':...} for ONLINE re-rank features
  similar_items.json  precomputed item -> [neighbours] (static; similar/BYL rows)
"""
import json, pickle
import numpy as np
import sys; sys.path.insert(0, ".")
from src import config as C


def script_user_tower(two_tower_model, path):
    """Export ONLY the user tower as TorchScript so the API needs neither the
    class definition nor the full checkpoint. Robust to variable-length history."""
    import torch
    two_tower_model.eval()
    scripted = torch.jit.script(two_tower_model.user_tower)
    scripted.save(str(path))


def precompute_similar(index, item_ids, n=10):
    """index = EmbeddingIndex. Static -> compute once, serve from JSON."""
    return {iid: index.similar_items(iid, n) for iid in item_ids}


def export(popularity_ids, item_encoder, item_emb, two_tower_model,
           lgbm_model, similar_map, feature_store=None, now_ts=None):
    A = C.ARTIFACTS
    json.dump(popularity_ids, open(A / "popularity.json", "w"))
    json.dump(item_encoder, open(A / "item_encoder.json", "w"))
    np.save(A / "item_emb.npy", item_emb)
    json.dump(similar_map, open(A / "similar_items.json", "w"))
    if lgbm_model is not None:
        pickle.dump(lgbm_model, open(A / "lgbm.pkl", "wb"))
    if two_tower_model is not None:
        script_user_tower(two_tower_model, A / "user_tower.pt")
    if feature_store is not None:
        pickle.dump({"store": feature_store, "now_ts": now_ts},
                    open(A / "feature_store.pkl", "wb"))
    print("artifacts exported to", A)

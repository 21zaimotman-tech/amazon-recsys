"""FAISS wrapper for top-N retrieval from learned item embeddings.
Works for both MF-BPR and two-tower: give it item vectors + a way to get a
user vector, it returns top-N item ids.

Use IndexFlatIP for exact inner-product search (good up to ~100k items).
For larger catalogs switch to IVF (see build_ivf)."""
from __future__ import annotations
import numpy as np
import faiss


class EmbeddingIndex:
    def __init__(self, item_embeddings: np.ndarray, item_ids: list, normalize=True):
        self.item_ids = np.asarray(item_ids)
        emb = item_embeddings.astype("float32")
        if normalize:                       # cosine == inner product on L2-normed vecs
            faiss.normalize_L2(emb)
        self.normalize = normalize
        self.dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)
        self._emb = emb

    @classmethod
    def build_ivf(cls, item_embeddings, item_ids, nlist=256, nprobe=16):
        obj = cls.__new__(cls)
        obj.item_ids = np.asarray(item_ids)
        emb = item_embeddings.astype("float32")
        faiss.normalize_L2(emb)
        obj.normalize = True
        obj.dim = emb.shape[1]
        quant = faiss.IndexFlatIP(obj.dim)
        obj.index = faiss.IndexIVFFlat(quant, obj.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        obj.index.train(emb)
        obj.index.add(emb)
        obj.index.nprobe = nprobe
        obj._emb = emb
        return obj

    def search(self, user_vecs: np.ndarray, n=100):
        """user_vecs: (B, dim). Returns list[list[item_id]] of length B."""
        q = np.atleast_2d(user_vecs).astype("float32")
        if self.normalize:
            faiss.normalize_L2(q)
        _, idx = self.index.search(q, n)
        return [self.item_ids[row].tolist() for row in idx]

    def similar_items(self, item_id, n=10):
        """Cosine nearest neighbours of one item — powers 'Similar items' and
        'Because you liked X' in the webapp. These are static -> precompute."""
        pos = int(np.where(self.item_ids == item_id)[0][0])
        vec = self._emb[pos:pos+1]
        _, idx = self.index.search(vec, n + 1)
        return [self.item_ids[j].tolist() for j in idx[0] if self.item_ids[j] != item_id][:n]

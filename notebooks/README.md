# Notebooks — your main deliverable

The notebook **is** the report. Readable top-to-bottom by someone who hasn't seen
the code. Every code cell has a markdown cell before it (what + why) and an
interpretation cell after each result. Split into 5 self-contained notebooks;
each imports from `src/` so logic lives in one place.

Save checkpoints + the frozen split to Google Drive so Colab timeouts don't cost you.

## 01_data_prep_eda.ipynb  (after S1)  — owner: Person A
1. **Why Electronics** — rich metadata + real image URLs for the webapp; head-concentrated, so popularity is a strong baseline (a finding, not a bug).
2. Load reviews via `src.data.load.load_reviews` (streaming — never load all of Electronics into RAM).
3. **EDA**: sparsity, rating distribution, long-tail (plot the head/tail), interactions-per-user and per-item, temporal volume over time.
4. **Subsampling** — recent window → iterative k-core → 5M cap. Print counts before/after each step. *Justify k=5 and the year cut.*
5. **Time split** (`src.data.split.time_split`) — plot the train/val/test time boundaries. Run `warn_leakage`. State the positive threshold (rating ≥ 4) and the parent_asin collapse.
6. Freeze train/val/test + items to Drive as parquet.

## 02_baselines.ipynb  (after S1)  — owner: Person A
- Random + Popularity (`src.baselines`). Build the eval harness around `src.eval.metrics.evaluate`.
- Report Recall@20, NDCG@10, Coverage. **Interpret**: popularity's recall vs its near-zero coverage.

## 03_retrieval_mfbpr.ipynb  (after S2)  — owner: Person B
- Train MF-BPR (`src.models.mf_bpr`). Tune dim / lr / reg on **val** only.
- FAISS index item embeddings → top-100 → evaluate. Compare vs popularity.
- **Interpret** the loss curve and where BPR wins/loses.

## 04_retrieval_twotower.ipynb  (after S3)  — owner: Person C
- Train two-tower with in-batch negatives (`src.models.two_tower`). Tune temperature, dim, max_hist.
- FAISS retrieval eval. **Full retrieval comparison table**: Popularity vs MF-BPR vs Two-tower (Recall@20/50, NDCG@10, Coverage).
- Export item embeddings + precompute similar-items for the webapp.

## 05_ranking_lightgbm.ipynb  (after S4)  — owner: Person A (+ all on features)
- Build candidates from the best retriever for train users; label pos/neg.
- 12 features (`src.ranking.features`). Train LightGBM LambdaRank (`src.ranking.ranker`).
- **Ablation**: retrieval-only vs +ranking (NDCG@10, Coverage). Feature importance plot.
- Export `lgbm.pkl`. Then run `scripts/export_artifacts.py` to ship everything to the API.

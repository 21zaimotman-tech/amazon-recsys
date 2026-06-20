# ANALYSIS — results, tables, discussion (no code)

## 1. Final comparison table
_All methods × all metrics, side by side._

| Method | Recall@20 | Recall@50 | NDCG@10 | Coverage |
|--------|-----------|-----------|---------|----------|
| Random | | | | |
| Popularity | | | | |
| MF-BPR | | | | |
| Two-tower | | | | |
| Two-tower + LightGBM | | | | |

## 2. Ablation: retrieval-only vs + ranking
| Setup | NDCG@10 | Coverage |
|-------|---------|----------|
| Best retriever only | | |
| + LightGBM re-rank | | |

_What did re-ranking buy you, and at what cost to coverage?_

## 3. Cold-start analysis
Bucket test users by training activity (e.g. 1–4 / 5–19 / 20+ interactions) and
report metrics per bucket. _Expect retrieval to help most for active users and
popularity to dominate the cold bucket — quantify it._

| Activity bucket | Users | Recall@20 (pop) | Recall@20 (two-tower) |
|-----------------|-------|-----------------|-----------------------|
| 1–4 | | | |
| 5–19 | | | |
| 20+ | | | |

## 4. Feature importance
_Top LightGBM features (gain). Plot or table. Which cross-features mattered?_

## 5. Latency breakdown
_Average ms per component, from the API's `latency_ms` logging._

| Component | Avg ms |
|-----------|--------|
| DB query | |
| User tower | |
| FAISS | |
| LightGBM | |
| **Total** | |

## 6. Limitations
_One concrete limitation + how you'd fix it with more time (e.g. mean-pooling the
user history loses sequence order → a SASRec/GRU4Rec user tower; or popularity
bias in negatives → popularity-corrected sampling)._

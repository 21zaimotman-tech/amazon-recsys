# ANALYSIS — results, tables, discussion

All numbers are from the **full-scale end-to-end run** on Amazon Reviews 2023 Electronics:
22.6M raw reviews (2019→) → iterative 5-core → **4.73M interactions · 551,911 train users ·
148,177 train items**, split 4,019,978 / 236,469 / 472,939 (train/val/test) by global
timestamp quantiles (Notebook 01). Evaluation: test-period positives (rating ≥ 4), train-seen
items excluded, catalog = train item vocabulary.

## 1. Final comparison table (all methods, all metrics)

| Method | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 | Coverage@50 |
|---|---|---|---|---|---|---|---|
| Random | 0.0001 | 0.0001 | 0.0002 | 0.0001 | 0.0003 | 0.0001 | 1.0000 |
| Popularity | 0.0035 | 0.0021 | 0.0078 | 0.0034 | 0.0191 | 0.0062 | 0.0005 |
| MF-BPR | 0.0035 | 0.0020 | 0.0077 | 0.0033 | 0.0190 | 0.0059 | 0.0005 |
| Two-tower | 0.0018 | 0.0011 | 0.0026 | 0.0013 | 0.0042 | 0.0017 | 0.9540 |
| **Two-tower + LightGBM** | **0.0037** | **0.0026** | 0.0046 | 0.0029 | 0.0059 | 0.0032 | **0.8535** |

Three findings, in the order they surprised us:

- **MF-BPR converged to a popularity clone.** With retrieval made faithful to the training
  objective (the item-bias term folded into the FAISS index so served scores ≡ trained
  scores), MF-BPR's metrics land on Popularity's to the third decimal — including the
  collapsed coverage. On a head-heavy catalog (top 1% of items ≈ half of interactions), the
  per-item bias absorbs popularity and recommending the head is the accuracy-optimal shortcut
  for a pairwise objective. An honest negative result, not a bug.
- **The two-tower trades recall for genuine personalization.** 95.4% catalog coverage
  against Popularity's 0.05% — different users genuinely see different items — priced at
  ~4.5× lower Recall@50. It has no bias term to absorb popularity, and in-batch softmax
  penalizes popular items as frequent negatives.
- **The ranking layer buys accuracy back: the full pipeline is the only personalized method
  that beats Popularity on NDCG@10** (0.0026 vs 0.0021, +25%) while still covering **85% of
  the catalog** (1,700× Popularity's coverage). Read Recall and Coverage together, always: a
  recommender can score well by showing everyone the same ~70 bestsellers.

## 2. Ablation: retrieval-only vs + ranking (test period)

| Setup | NDCG@10 | Coverage@50 |
|---|---|---|
| Two-tower only (retrieval) | 0.0011 | 0.9540 |
| Two-tower + LightGBM | **0.0026** (2.5×) | 0.8535 |

Re-ranking the same top-100 candidates with 12 behavioral/content features raises NDCG@10
**2.5×** at a ~10-point coverage cost. Coverage staying at 85% (rather than collapsing)
means the ranker is *not* just reinventing popularity — `item_pop` is only its second
feature by gain (see §4).

## 3. Cold-start analysis (test users bucketed by train-period activity)

| Bucket (train interactions) | Users | Recall@20 pop | NDCG@20 pop | Recall@20 two-tower | NDCG@20 two-tower |
|---|---|---|---|---|---|
| 1–4 | 57,923 | 0.0082 | 0.0036 | 0.0027 | 0.0014 |
| 5–19 | 63,984 | 0.0076 | 0.0031 | 0.0024 | 0.0012 |
| 20+ | 6,001 | 0.0045 | 0.0019 | 0.0033 | **0.0017** |

Popularity dominates the cold buckets — a near-empty history gives the user tower almost
nothing to pool. But the gap **closes monotonically with activity**: in the 20+ bucket the
two-tower reaches ~90% of Popularity's NDCG@20 while personalizing, and Popularity itself
*degrades* for heavy users (their tastes drift furthest from the global head). This is the
empirical basis for the serving design: cold/unknown users get popularity, active users get
the tower.

Cold *items* are structural: **31.1% of test items never appear in train** — no
collaborative model can retrieve them. Product-layer mitigations: popularity fallback,
category onboarding for new accounts, category-guarded similar-items rails.

## 4. Feature importance (LightGBM gain)

| Rank | Feature | Gain | Group |
|---|---|---|---|
| 1 | `item_age_days` | 27,705 | temporal |
| 2 | `item_pop` | 21,174 | item |
| 3 | `retriever_score` | 12,468 | retrieval |
| 4 | `item_price` | 8,300 | item |
| 5 | `brand_match` | 7,909 | cross |
| 6 | `rating_vs_user_mean` | 7,318 | cross |
| 7 | `item_avg_rating` | 7,138 | item |
| 8 | `days_since_user_last` | 6,767 | temporal |
| 9 | `price_gap` | 6,728 | cross |
| 10 | `user_activity` | 4,467 | user |
| 11 | `user_avg_rating` | 3,956 | user |
| 12 | `cat_match` | 2,727 | cross |

**Freshness beats popularity**: `item_age_days` is the top feature — in electronics, recent
items dominate future engagement (catalog turnover), something pure embedding similarity
can't express. All four required feature groups contribute; cross-features (`brand_match`,
`price_gap`, `rating_vs_user_mean`) collectively rival the top single features, which is
where the personalization gain over the raw retriever comes from.

## 5. Latency breakdown (live API, full pipeline, 60 calls)

| Component | Mean (ms) | p95 (ms) |
|---|---|---|
| Postgres history fetch | 1.1 | 1.7 |
| User tower (TorchScript) | 4.3 | 13.5 |
| FAISS top-100 (148K × 64) | 1.7 | 2.0 |
| LightGBM featurize + re-rank | 19.0 | 11.6 |
| **Total end-to-end** | **27.5** | **35.7** |

One production lesson worth its own line: the first full-scale deployment measured
**~1,020ms** in the "LightGBM" step. Profiling showed the model predict was 1ms — the cost
was `featurize()` calling `Series.map(dict)`, which rebuilds a hash index over the *entire*
551K-entry feature-store dict on every request. Converting the store's dicts to pandas
Series once at API startup cut the step to ~19ms — **a 28× end-to-end speedup** from a
one-line data-structure change. Small-sample benchmarks hide this class of bug entirely.

## 6. Bonus — MMR diversification (relevance–diversity trade-off)

Maximal Marginal Relevance re-ranking is implemented in the serving API (`mmr_lambda`
parameter; "Feed variety" slider in the store). Offline sweep on 2,000 test users
(`scripts/mmr_tradeoff.py`, top-10 lists):

| λ | NDCG@10 | Recall@10 | Intra-list similarity |
|---|---|---|---|
| 1.0 (off) | 0.00213 | 0.00339 | 0.518 |
| 0.9 | 0.00211 | 0.00339 | 0.508 |
| 0.8 | 0.00204 | 0.00331 | 0.494 |
| 0.7 | 0.00195 | 0.00331 | 0.477 |
| 0.6 | 0.00205 | 0.00331 | 0.454 |
| 0.5 | 0.00168 | 0.00258 | 0.424 |

The sweet spot is **λ ≈ 0.6–0.7**: intra-list similarity drops ~12% (visibly fewer
near-duplicate products) while NDCG@10 stays within ~4–8% of the undiversified list.
Below λ = 0.5 relevance degrades sharply. The store defaults to λ = 1 and exposes the
trade-off to the user.

## 7. Limitations

1. **Cold items are unreachable by design** — 31% of test items have no training signal.
   With more time: content-based item embeddings (title/category text encoder) blended into
   the FAISS index, so new items are retrievable from day one.
2. **Absolute recall is small everywhere** — the median user has a handful of interactions
   and test positives concentrate on head items, so even the best personalized numbers look
   small in absolute terms. The pairwise story (vs baselines, vs retrieval-only) is the
   meaningful signal; a no-bias MF-BPR ablation is the natural next experiment to isolate
   the popularity-collapse mechanism.
3. **One global ranker** — trained on all users; per-segment rankers (by activity bucket)
   would likely recover more of the cold-bucket gap identified in §3.

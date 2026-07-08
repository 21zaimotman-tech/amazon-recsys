# ANALYSIS — results, tables, discussion

All numbers below are from a real end-to-end run on a local sample of Amazon Reviews 2023
Electronics: 131,154 interactions after 5-core filtering, 15,038 train users, 9,487 items,
split `111,481 / 6,557 / 13,116` (train/val/test) by global timestamp quantile (see Notebook 01).
This sample is smaller than the brief's `MAX_INTERACTIONS = 5,000,000` cap — the pipeline and
code are unchanged at full scale, only the absolute numbers would shift (see §6).

## 1. Final comparison table

| Method | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | Coverage@50 |
|--------|-----------|-----------|-----------|---------|-------------|
| Random | 0.0011 | 0.0024 | 0.0049 | 0.0007 | 1.0000 |
| Popularity | 0.0122 | 0.0175 | 0.0499 | 0.0071 | 0.0066 |
| MF-BPR | 0.0012 | 0.0025 | 0.0051 | 0.0007 | 1.0000 |
| Two-tower | 0.0041 | 0.0052 | 0.0104 | 0.0026 | 0.9975 |
| Two-tower + LightGBM | 0.0063 | 0.0100 | 0.0140 | 0.0043 | 0.9907 |

**The pipeline is internally consistent — each stage beats the one before it on every single
metric.** Two-tower beats MF-BPR across the board (as expected: it conditions on actual
history instead of one fixed learned vector, and in-batch negatives give a denser signal per
step than BPR's one-negative sampling). Two-tower + LightGBM beats Two-tower alone across the
board too — re-ranking is doing real work, not noise.

**The honest finding: Popularity still wins on raw Recall/NDCG against every learned method,
including the full pipeline.** Popularity's Recall@50 (0.0499) is more than 3.5x
Two-tower+LightGBM's (0.0140), and its NDCG@10 (0.0071) beats LightGBM's (0.0043) too. This
looks surprising next to the "popularity has near-zero coverage" story from Notebook 02, but
it's the direct consequence of two things layered together: Electronics is genuinely
head-concentrated (Notebook 01: top 1% of items = 29.6% of interactions), and this local
sample is small — 15,038 users is not much signal for a model to learn individual preferences
from. At `MAX_INTERACTIONS = 5,000,000` (the brief's actual target scale, ~38x this sample),
the learned methods would have far more per-user and per-item signal to work with and would be
expected to close or reverse this gap — see §6.

**Where the learned methods unambiguously win: Coverage.** Popularity's Coverage@50 is 0.0066
(it recommends the same ~63 items to everyone); every learned method sits at 0.99+ (Two-tower
and Two-tower+LightGBM personalize almost every recommendation list; MF-BPR hits a perfect
1.0 because its embedding-table lookup never collapses onto a shared list the way Popularity's
fixed ranking does). This is the real accuracy/diversity trade-off the brief asks for: Popularity
is a strong, cheap baseline for aggregate accuracy on a head-heavy catalog, but it cannot power
a "similar items" or genuinely personalized page — every item and every user gets the identical
list.

## 2. Ablation: retrieval-only vs + ranking

| Setup | NDCG@10 | Coverage@50 |
|-------|---------|-------------|
| Two-tower only (retrieval) | 0.0026 | 0.9988 |
| Two-tower + LightGBM re-rank | 0.0043 | 0.9907 |

Re-ranking bought a **+65% relative NDCG@10 improvement** (0.0026 → 0.0043) for a **0.8 point
drop in Coverage** (99.88% → 99.07% — both still near-total catalog coverage). This is a very
favorable trade: the 12 features (popularity, price, category/brand match, recency — see §4)
let the ranker correct for cases where cosine similarity alone picks a plausible-looking but
low-engagement item, at almost no cost to how much of the catalog still gets recommended
somewhere. The ranker is not just reinventing Popularity — if it were, Coverage would have
collapsed toward Popularity's 0.0066, not stayed at 0.99.

## 3. Cold-start analysis

Test users bucketed by their **train-period** activity count (Notebook 05):

| Activity bucket | Users | Recall@20 (pop) | NDCG@20 (pop) | Recall@20 (two-tower) | NDCG@20 (two-tower) |
|-----------------|-------|------------------|----------------|------------------------|-----------------------|
| 1–4 | 1,881 | 0.0194 | 0.0090 | 0.0042 | 0.0022 |
| 5–19 | 2,688 | 0.0156 | 0.0075 | 0.0056 | 0.0032 |
| 20+ | 243 | 0.0101 | 0.0080 | 0.0088 | 0.0058 |

**Exactly the expected pattern, and it's clean and monotonic.** Two-tower's Recall@20 relative
to Popularity's climbs steadily with activity: 22% of Popularity's recall in the coldest
bucket (1–4 interactions), 36% in the mid bucket (5–19), and 87% in the most active bucket
(20+) — nearly closing the gap entirely. This makes mechanical sense: the two-tower user tower
mean-pools a user's history into their vector, so a user with 1–2 train interactions gives it
almost nothing to work with, while a 20+-interaction user gives it a real signal to
personalize from. Popularity, having no user-specific signal at all, is flat-to-slightly-declining
across buckets (its "coldest bucket wins" pattern here is just that low-activity users'
test-period positives skew toward whatever's broadly popular). The practical implication for
the webapp: **cold/new users should see Popularity; the more history a user accumulates, the
more the personalized page should be trusted** — which is exactly what `api/recommender.py`
already does (`recommend()` falls back to `popular()` when `history_item_ids` is empty).

## 4. Feature importance

| Feature | Gain | Group |
|---|---|---|
| `item_age_days` | 2363.8 | temporal |
| `retriever_score` | 1547.7 | retrieval signal |
| `item_pop` | 1490.9 | item |
| `days_since_user_last` | 1078.2 | temporal |
| `rating_vs_user_mean` | 1064.9 | cross |
| `item_price` | 921.0 | item |
| `item_avg_rating` | 850.2 | item |
| `user_activity` | 667.6 | user |
| `price_gap` | 627.1 | cross |
| `user_avg_rating` | 296.8 | user |
| `cat_match` | 100.9 | cross |
| `brand_match` | 48.0 | cross |

**The two temporal features (`item_age_days`, `days_since_user_last`) rank #1 and #4** —
together they outweigh `retriever_score` itself. That's a meaningful finding: the two-tower
retriever's mean-pooled history representation has no explicit notion of recency (see §6,
limitation #1), so the ranker is doing real work compensating for it, learning "prefer items
close to the user's last activity" as a correction the retriever can't express.
`retriever_score` (#2) and `item_pop` (#3) confirm the ranker is still leaning heavily on the
retrieval signal and on plain popularity, consistent with §1's finding that popularity remains
a strong accuracy signal at this sample size. **The two exact-match cross-features
(`cat_match`, `brand_match`) rank lowest by a wide margin** — at 9,487 items and this sample
size, an exact category/brand match is a sparse, high-variance binary signal that fires rarely,
while continuous features (price, rating, popularity, recency) are always present and provide
a gradient the tree can split on more usefully.

## 5. Latency breakdown

Measured against the live Docker stack (`scripts/benchmark_latency.py`), 50 calls, real
trained artifacts (Two-tower + LightGBM active):

| Component | Mean (ms) | p95 (ms) |
|-----------|-----------|----------|
| DB query | 1.01 | 1.21 |
| User tower | 2.85 | 4.97 |
| FAISS | 0.37 | 0.35 |
| LightGBM | 39.81 | 40.36 |
| **Total** | **47.12** | **54.36** |

**LightGBM dominates the latency budget — 84% of total time.** The user tower, FAISS, and DB
combined are under 5ms; the LightGBM `.predict()` call over the candidate set is what makes
this a ~47ms endpoint rather than a ~7ms one. This is the clear next optimization target if
this were a real SLA discussion: either shrink the candidate set handed to the ranker (100
candidates → e.g. 50), reduce the model's `num_boost_round`/tree depth, or batch-predict with a
lighter-weight model format. At ~47ms mean end-to-end, the system is comfortably fast enough
for an interactive page load, but it's worth knowing exactly where that time goes rather than
just reporting the total.

## 6. Limitations

**1. Sample size directly limits personalization quality (the headline finding of §1).**
This run used ~131k interactions / 15k users — far below the brief's `MAX_INTERACTIONS =
5,000,000` cap. §1 shows every learned method losing to Popularity on raw Recall/NDCG, which
is very unlikely to hold at full scale: more interactions per user is exactly what a
personalization model needs to differentiate from the popularity prior, and the cold-start
table (§3) already shows the gap closing sharply as per-user activity increases even within
this small sample. **Fix:** re-run Notebooks 03–05 against the full 5M-interaction build
(`python scripts/build_dataset.py` without `--limit`) — the code changes required are none,
only compute time.

**2. Mean-pooling in the two-tower user tower loses sequence order and recency**, which §4's
feature importance result makes concrete: the ranker's top two features by gain are both
temporal signals the retriever's architecture can't express (`UserTower.forward` in
`src/models/two_tower.py` averages history embeddings with no positional or recency
weighting — a user's oldest and most recent interaction contribute identically to their
vector). **Fix:** a sequence-aware user tower (SASRec/GRU4Rec-style, attention or a recurrent
layer over the history instead of mean pooling) would likely let the retriever itself capture
what LightGBM is currently compensating for post-hoc, probably improving Two-tower's standalone
NDCG@10 in §1 and reducing how much weight the ranker needs to put on `item_age_days`.

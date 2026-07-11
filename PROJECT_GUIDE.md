# Amazon Electronics RecSys — Complete Project Guide

Repo: `21zaimotman-tech/amazon-recsys`. Dataset: Amazon Reviews 2023, **Electronics** category.
This file is the single source of truth for the team: what's already built, what's missing,
and the exact order of steps to finish the project and pass the defense.

---

## ⚡ STATUS UPDATE — 2026-07-11 (read this first)

**The plan below is essentially complete.** This guide remains as the team's reference
for *why* each step exists; the current state of the system is documented in
[`PROJECT_EXPLAINED.md`](PROJECT_EXPLAINED.md) (the full technical explainer) and
[`README.md`](README.md) (results + how to run). Defense materials:
[`presentation.html`](presentation.html) + [`presentation_script.md`](presentation_script.md).

| Step (below) | Status |
|---|---|
| 1 — `src/data/load.py` + `split.py` | ✅ done (incl. `encode_ids`) |
| 2 — Notebook 01 data prep/EDA | ✅ **executed on the FULL dataset** (22.6M raw → 4.02M train / 236K val / 473K test; 551,911 users / 148,177 items) |
| 3 — Notebook 02 baselines | ✅ full run (Popularity R@50 0.0191, Coverage 0.0005) |
| 4 — Notebook 03 MF-BPR | ✅ full run + **item-bias retrieval fix** (FAISS score ≡ training score). Finding: converges to a popularity clone — see `PROJECT_EXPLAINED.md` §5.2 |
| 5 — Notebook 04 two-tower | ✅ full run (R@50 0.0042, **Coverage 0.9503**) — the deployed retriever |
| 6 — Notebook 05 LightGBM | ✅ code complete + memory-tuned for 8GB machines + macOS OpenMP fix; final full-run numbers land in `data/*.csv` |
| 7 — Webapp | ✅ done and far beyond the brief: realtime feedback loop, type-ahead search + filters, category browse, carts w/ quantities, orders, wishlist, share links, onboarding, reasons, analytics, `?debug=1` mode, production HTTPS overlay (`deploy/`) |
| 8 — README/ANALYSIS | ✅ README updated with full-run results + real task split; ANALYSIS.md refresh pending final 05 numbers |

**Remaining (small):** fill notebook 05's final numbers into README/ANALYSIS/deck ·
re-run `scripts/benchmark_latency.py` against the full artifacts · optional no-bias
MF-BPR ablation for the defense Q&A.

**Notes that supersede sections below:** heavy `data/` files are now git-ignored
(151MB parquet > GitHub's 100MB cap) — the frozen split lives locally / on the team
Drive (`MyDrive/amazon-recsys-data/data`). Colab reruns should use
`notebooks/run_all_colab.ipynb`, which pins the fixed `src/` files and sanity-checks
that the *full* dataset (not the repo's old sample) is restored before training.

---

## 1. What's already in the repo

Your teammate committed a full scaffold (1 commit, ~1,250 lines). It is well-written and
matches the course material exactly — no tech swaps needed:

| Course session | Taught | Scaffold file |
|---|---|---|
| S1 Foundations | similarity, k-NN, baselines | `src/baselines/popularity.py` |
| S2 Matrix Factorization | Funk SVD, BPR, neg. sampling | `src/models/mf_bpr.py` |
| S3 Retrieval at Scale | Two-tower, FAISS/ANN | `src/models/two_tower.py`, `src/retrieval/faiss_index.py` |
| S4 Ranking Models | features, LambdaRank, GBDT | `src/ranking/features.py`, `src/ranking/ranker.py` |

Also done: `src/eval/metrics.py` (Recall@k, NDCG@k, Coverage — shared everywhere), `src/eval/harness.py`
(batch eval glue), `src/eval/cold_start.py` (activity-bucket analysis), FastAPI service (`api/`),
Streamlit frontend (`frontend/`), Postgres schema (`db/init.sql`), Docker Compose, and scripts
(`build_dataset.py`, `run_baselines.py`, `export_artifacts.py`, `populate_db.py`, `benchmark_latency.py`).

## 2. What's missing (blocking everything) — ✅ RESOLVED

*(Historical: kept for context. `src/data/load.py` and `src/data/split.py` exist and
the whole pipeline has run on the full dataset — see the status table above.)*

`src/data/load.py` and `src/data/split.py` **did not exist**, even though `scripts/build_dataset.py`
and `scripts/run_baselines.py` already imported from them (`subsample`, `_clean_price`, `time_split`,
`warn_leakage`, `ground_truth_from`). Nothing in the pipeline could run until these two files existed.

### Smaller issues to fix along the way
- `EmbeddingIndex.similar_items()` in `src/retrieval/faiss_index.py` throws an uncaught error for an
  unknown `item_id` — wrap it so `/similar/{item_id}` degrades to popularity instead of crashing.
- `src/config.py` has two dead lines (`REVIEW_CONFIG`, `META_CONFIG`) left over from an approach the
  team abandoned (streaming `datasets` library) in favor of direct JSONL streaming — delete them.
- `.gitignore` excludes `artifacts/*.npy|.pt|.pkl` but not `artifacts/*.json` — add `artifacts/*.json`
  so `similar_items.json` (one entry per catalog item) doesn't get committed.

## 3. Step-by-step: how to finish the project

Work through these in order. Each step names the exact file(s) to create/edit and which
grading criterion it satisfies.

### Step 1 — Write `src/data/load.py` and `src/data/split.py` (unblocks everything)

Create `src/data/__init__.py` (empty) plus:

**`src/data/load.py`** needs:
- `_clean_price(price)` — the raw metadata price field is a messy string (e.g. `"$49.99"`, `None`,
  ranges); parse to `float` or `None`.
- `subsample(df)` — implements the brief's mandatory 5M cap: recent-window filter (already applied in
  `build_dataset.py` before this is called) → iterative k-core filtering using
  `C.KCORE_USER` / `C.KCORE_ITEM` (repeatedly drop users/items below the threshold until stable,
  since removing items can drop users below threshold and vice versa) → truncate to
  `C.MAX_INTERACTIONS` if still over. Print counts at each stage — the notebook must show this.
- `encode_ids(train_df, val_df, test_df)` (or similar) — the models never see raw string
  `user_id`/`item_id`. Fit an encoder on the **train** IDs only, mapping each to a contiguous
  integer `0..n-1`, add `u`/`i` integer columns to all three splits (unseen val/test ids map to an
  `<UNK>` index — that's your cold-user/cold-item case), and save the mapping (e.g. as arrays or a
  dict) so `scripts/export_artifacts.py`'s `item_encoder.json` and the API's `idx_map` use the exact
  same encoding at serving time. `train_mfbpr`/`train_two_tower` read the `u`/`i` columns directly;
  `make_history_batches` additionally needs `timestamp` (already present) to sort each user's
  sequence before slicing history windows.

**`src/data/split.py`** needs:
- `time_split(df)` — sort by `timestamp`, split by **global** timestamp quantiles using
  `C.VAL_QUANTILE` (0.85) and `C.TEST_QUANTILE` (0.90) from `src/config.py`. This is the
  mandatory time-based split — a random split scores 0 on the "Data preparation & baselines"
  criterion (10% of your grade).
- `warn_leakage(train, test)` — assert `train.timestamp.max() <= test.timestamp.min()`; print a
  warning (don't crash) if violated.
- `ground_truth_from(df, positive_only=True)` — returns `{user_id -> set(item_id)}`, filtering to
  `rating >= C.POSITIVE_RATING_THRESHOLD` when `positive_only=True`. Used by `run_baselines.py` and
  every notebook's evaluation.

Once these exist, `python scripts/build_dataset.py --limit 200000 --meta-scan-cap 1000000` should
run end-to-end as a dry run, and `python scripts/run_baselines.py` should print a metrics table.

### Step 2 — Notebook 01: Data prep & EDA (owner: Person A, matches S1)

Follow `notebooks/README.md`'s outline. Load via `stream_reviews`/`stream_meta`
(from `scripts/build_dataset.py`, or import the logic into `src/data/load.py` directly), then:
sparsity, rating distribution, long-tail plot, interactions per user/item, temporal volume. Run
`subsample` and `time_split`, printing before/after counts and the leakage check. State and justify:
why Electronics, why `KCORE_USER=KCORE_ITEM=5`, why `RECENT_FROM_YEAR=2019`, why the positive
threshold is `rating >= 4`. Freeze `train/val/test/items` to Drive as parquet — every later notebook
starts by loading these, not by re-running this one.

### Step 3 — Notebook 02: Baselines (owner: Person A, matches S1)

Run `PopularityRecommender` and `RandomRecommender` from `src/baselines/popularity.py` through
`src/eval/metrics.evaluate`. Report Recall@20, Recall@50, NDCG@10, Coverage. Interpret: popularity
should have strong recall and near-zero coverage — that contrast is a required finding for
`ANALYSIS.md` section 1.

### Step 4 — Notebook 03: MF-BPR (owner: Person B, matches S2, 25% of grade with two-tower)

Use `src.models.mf_bpr.train_mfbpr`. Tune `dim`, `lr`, `reg` on **val only** (never test). Build a
FAISS `EmbeddingIndex` from `export_item_embeddings(model)`, retrieve top-100, evaluate with
`src.eval.harness.evaluate_index`. Plot the training loss curve. **Before the defense, every team
member must be able to explain** `bpr_loss` line-by-line (it's fully commented in the file already)
and how `sample_triplets` avoids sampling a negative that's secretly a positive.

### Step 5 — Notebook 04: Two-tower (owner: Person C, matches S3, heaviest component)

Use `src.models.two_tower.train_two_tower` with `make_history_batches` (time-ordered histories, no
leakage — only earlier items predict the target). Tune `temperature`, `dim`, `max_hist`. Export item
embeddings + build the retrieval comparison table: **Popularity vs MF-BPR vs Two-tower**
(Recall@20/50, NDCG@10, Coverage) — this table is required in both the README and `ANALYSIS.md`.
Precompute `similar_items` for every item (via `EmbeddingIndex.similar_items`) — this powers the
"Similar items" and "Because you liked X" webapp sections. **Be ready to explain**: why in-batch
negatives, why softmax cross-entropy over the batch (both called out explicitly in the file's
docstring as defense questions).

### Pretrained models and LLM APIs (e.g. Gemini) — what's allowed and what isn't

**Not allowed, for anything that produces a recommendation.** MF-BPR, two-tower, and LightGBM must
be trained by your own code on your own interaction data — no downloading a trained
MF/two-tower/recommender checkpoint, no RecBole/Surprise/LensKit, and **no calling an LLM API
(Gemini or otherwise) to generate or influence which items get recommended.** An LLM can't replace
gradient descent on your interaction data, and doing so would violate the "implement from scratch"
requirement you have to defend live. This applies everywhere a recommendation is produced — training
notebooks, the ranker, and the live API's `/recommend`, `/similar`, `/because-you-liked` endpoints.

**Allowed, optional, and unrelated to the recommendation logic — two narrow uses:**

1. **Pretrained text embeddings as one input feature inside your own item tower** (not a
   replacement for the model). This is the exact pattern Session 3's "Item Tower Inputs" slide
   shows: ID embedding + text encoder + categorical + numerical, concatenated, then passed through
   your own MLP. Concretely: run each item's title through a pretrained sentence encoder to get a
   fixed embedding, concatenate it with `ItemTower`'s existing `nn.Embedding` lookup in
   `src/models/two_tower.py` before the `mlp`, and train the whole thing exactly as before — same
   loss, same loop, still your code. Use `sentence-transformers` (local, free, no API key) for this,
   not the Gemini API — a live external API call has no place in a training loop, and this doesn't
   need Gemini specifically. This helps the cold-start problem: a new/rare item gets a meaningful
   starting representation from its title instead of a near-random one. Document it as a deliberate
   design choice in the notebook and `ANALYSIS.md`'s cold-start section if you do it.
2. **Explainability (a listed bonus item), strictly after recommendations are already produced.**
   Once your models have picked the top-10 items for a user, you could optionally use the Gemini API
   to generate a short natural-language reason for the recommendation, e.g. "similar to your recent
   audio purchases and trending this week." This never touches which items get recommended — it only
   narrates a decision your own models already made. Keep this offline/precomputed, not a live call
   inside `api/recommender.py`'s request path — the demo has to run reliably via `docker compose up`
   without depending on external API uptime during the defense.

If in doubt: your Gemini key can write words about a recommendation, never choose one.

### Step 6 — Notebook 05: LightGBM ranking (owner: Person A + everyone on features, matches S4)

Retrieve top-100 candidates for train users with your best retriever. Label positive if the item
appears in the **validation-period** ground truth (not train — see the leakage-safe design already
documented in `src/ranking/pipeline.py`). Build features with
`src.ranking.features.build_feature_store` + `featurize` (12 features across user/item/cross/temporal
— exceeds the "at least 10" requirement). Train with `src.ranking.ranker.train_ranker`
(`objective="lambdarank"`). Report the ablation: retrieval-only vs +ranking (NDCG@10, Coverage) and
plot `feature_importance(model)`. Export everything via `scripts/export_artifacts.py`.

### Step 7 — Webapp (owner: Person D, 20% of grade)

The webapp already runs against popularity alone from S2 onward (graceful degradation is built into
`api/recommender.py`) — don't wait for the ranker to start testing it. Once artifacts exist:
```bash
cp .env.example .env
python scripts/export_artifacts.py
docker compose up --build
python scripts/populate_db.py     # after postgres is healthy
```
API docs: http://localhost:8000/docs · Frontend: http://localhost:8501. Run
`python scripts/benchmark_latency.py http://localhost:8000 <user_ids...>` and record the
mean/p95 per component — required for the README and `ANALYSIS.md` latency table.

### Step 8 — `ANALYSIS.md` and `README.md`

`ANALYSIS.md` already has the exact section skeleton (final comparison table, ablation, cold-start,
feature importance, latency, limitations) — fill it in, no code, just results and discussion. Update
the `README.md` results table, add the task-split table with real names, add a GIF of the webapp, and
report the average API response time.

### Step 9 — Fix the smaller issues (Section 2 above) and do a final pass

Wrap `similar_items()` for unknown ids, delete the dead config lines, update `.gitignore`. Then read
back through every notebook top-to-bottom as if you'd never seen the code — the brief explicitly
grades narration quality, not just correctness.

## 4. Training strategy: the dataset is large, here's how you actually train on it

Electronics has ~43.9M ratings across 18.3M users before any filtering — you cannot train
directly on the raw file. The scaffold's approach (all on Google Colab, which your professor
has approved) is already designed around this:

1. **Stream, never load the full file into memory.** `scripts/build_dataset.py` reads the raw
   JSONL line-by-line (`stream_jsonl`/`stream_reviews`) instead of loading it all at once —
   required at this scale even on a Colab high-RAM runtime.
2. **Filter while streaming.** `RECENT_FROM_YEAR = 2019` (in `src/config.py`) drops old rows as
   each line is read, before they ever land in a DataFrame.
3. **k-core, then hard cap.** `subsample()` (Step 1 above) iteratively drops users/items with
   fewer than 5 interactions (`KCORE_USER`/`KCORE_ITEM`), repeating until stable, then truncates
   to the brief's mandatory `MAX_INTERACTIONS = 5_000_000`. Print and document before/after
   counts in the notebook — this is graded, not optional cleanup.
4. **Metadata is fetched lazily too.** `stream_meta()` only keeps metadata for items that survived
   subsampling and stops scanning once they're all found.
5. **Training on the resulting ~5M rows is cheap.** MF-BPR and two-tower both train via
   mini-batch SGD/Adam (`batch_size=4096` / `1024`) — memory footprint is the embedding tables
   plus one batch, not the whole dataset. A free Colab **T4 GPU runtime** handles this fine for
   both PyTorch models. FAISS indexing and LightGBM training don't need a GPU at all — CPU runtime
   is fine and saves your GPU quota for the two heavy models.
6. **Checkpoint to Google Drive.** Colab free-tier sessions disconnect; save model checkpoints and
   the frozen train/val/test parquet split to Drive periodically so a disconnect doesn't cost you
   a full re-run.
7. **Iterate small, then commit to the full run.** Use `python scripts/build_dataset.py --limit
   200000 --meta-scan-cap 1000000` to validate your code on a small slice in minutes before
   running the full pass.

**Which model is "best"?** Both MF-BPR and two-tower are mandatory — you don't get to pick one.
"Best" is the empirical result you report (Recall@20/50, NDCG@10, Coverage on the test split), not
a decision made upfront. Expect two-tower to edge out MF-BPR (it can use history/side features and
in-batch negatives give a stronger signal), but build and validate MF-BPR first — it's simpler and
lets you prove the FAISS + eval pipeline works before sinking 10-15h into two-tower. Whichever wins
is the one you feed into the LightGBM ranker (Step 6).

### What actually goes into each model, per batch

- **MF-BPR**: triplets `(u, i, j)` — encoded user index, one item they interacted with, one random
  item they didn't. `sample_triplets` builds these on the fly from the encoded `train_df`; just
  three integers per row, no sequence.
- **Two-tower**: sequences — `history_ids` (a user's earlier item indices, time-ordered, truncated
  to `max_hist=20`, padded to a common length), a `history_mask` (1 = real item, 0 = padding), and
  the `pos_item` target. `make_history_batches` sorts by `timestamp` to build these so only earlier
  items ever predict the next one (no leakage).
- **LightGBM**: no sequences at all — a flat feature vector per `(user, candidate_item)` row from
  `featurize()` (the 12 columns), plus a 0/1 label.

### Where the data actually lives, at each stage

| Stage | Storage | Notes |
|---|---|---|
| Raw JSONL (reviews + metadata) | Streamed from HuggingFace URLs, never saved | `stream_jsonl` reads line-by-line; pass `--reviews-file`/`--meta-file` to `build_dataset.py` to read a local download instead |
| Frozen split (`train/val/test/items.parquet`) | Colab VM disk (`./data`) during the session → copy to Google Drive right after | Ephemeral otherwise — a disconnect loses it. Every notebook after 01 loads from Drive, doesn't regenerate |
| Trained artifacts (embeddings, FAISS inputs, `lgbm.pkl`, etc.) | Same: `./artifacts` on the VM → copy to Drive | Written by `scripts/export_artifacts.py` |
| Webapp demo | Your local machine, not Colab | Download `artifacts/` from Drive, mount it into Docker (`docker-compose.yml` already mounts `./artifacts:/artifacts:ro`); Postgres is a local Docker volume populated by `scripts/populate_db.py` |
| GitHub repo | Neither raw nor processed data | `.gitignore` already excludes `data/` and the large artifact file types — only code is committed |

## 5. Grading weights (so you know where to spend time)

| Criterion | Weight |
|---|---|
| Retrieval (MF-BPR + two-tower) | 25% |
| Demo webapp | 20% |
| Ranking (features + LightGBM) | 15% |
| Analysis & insight | 15% |
| Data prep & baselines | 10% |
| Presentation & defense | 10% |
| Notebook quality | 5% |
| Bonus | up to +10% |

## 6. Timeline reminder

| After | Do |
|---|---|
| S1 (now) | Step 1 (src/data), Notebook 01, Notebook 02 |
| S2 | Notebook 03 (MF-BPR), start feature engineering, webapp shows popularity |
| S3 | Notebook 04 (two-tower), FAISS eval, webapp shows two-tower similar-items |
| S4 | Notebook 05 (LightGBM), final analysis, webapp fully wired |
| S5 | Polish, ANALYSIS.md, README, presentation, **defense** |

## 7. Build it like a product, not a class notebook

Your professor's point: the notebooks prove you understand the models, but the **webapp is what
makes this a portfolio piece** — something you'd demo in a job interview, not just a grade. The
brief says this directly: "a portfolio-ready project you can demo in interviews and show on your
resume." Concretely, what separates "notebook exercise" from "product":

- **Different pages use different models, like a real product does.** This is already the design
  in `notebooks/README.md`'s table and `frontend/app.py`: logged-out homepage = popularity,
  logged-in homepage = two-tower + ranking, item page = embedding similarity. Don't collapse these
  into one generic "recommendations" list — the whole point is showing you understand *when* to use
  which model, which is exactly what a recsys engineer does at a real company.
- **Show your work in the UI, not just the results.** `api/schemas.py`'s `model_label` field and
  `frontend/app.py`'s `st.caption(f"model: {label}")` already surface which model produced each row
  ("Popular right now" vs "Two-tower + LightGBM") — keep this visible. A real product wouldn't
  expose it to end users, but for a defense it's the difference between "here are some
  recommendations" and "here's why the system chose these."
- **Real service architecture, not a script.** Postgres + FastAPI + Streamlit as three independent
  Docker services (not one monolithic notebook cell) mirrors how this would actually be deployed —
  swap any one piece without touching the others. This is worth pointing out explicitly in your
  presentation, not just something that happens to be true.
- **Handle failure gracefully, like production has to.** Cold users fall back to popularity, missing
  artifacts degrade gracefully (`api/recommender.py`'s `_maybe`), unknown items shouldn't 500 (the
  `similar_items()` bug in Section 2 above is exactly the kind of thing that breaks the "product"
  illusion during a live demo — fix it before the defense, not after).
- **Measure and report latency, like a product team would.** You're already required to log and
  report average response time — treat that number as a real SLA discussion point in your
  presentation ("here's where the 80ms goes, here's what we'd optimize next"), not just a number to
  fill into a table.
- **Polish the demo path, not the code you'll never show.** Before the defense: seed a few
  realistic-looking users in Postgres, make sure images actually render, make sure the "Because you
  liked X" row has a sensible seed item, and rehearse the live `docker compose up` once end-to-end
  so it isn't the first time it's run outside your own laptop.

## 8. Defense reminders

Every team member must be able to explain any part of the code and justify design choices — the
brief is explicit that this will be checked live. The two things examiners will probe hardest, per
the code's own comments: the BPR loss derivation (`mf_bpr.py`) and why in-batch negatives +
softmax cross-entropy work for two-tower (`two_tower.py`). Rehearse both out loud before the defense.
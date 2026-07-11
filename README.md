# ElectroPicks — Real-Time Recommender on Amazon Electronics

End-to-end recommender system: **22.6M raw reviews → trained retrieval + ranking
models → a live web store** where every click reshapes the next page. Dataset:
Amazon Reviews 2023, **Electronics** (McAuley-Lab/Amazon-Reviews-2023, HuggingFace).

Stack: PyTorch (models from scratch) · FAISS · LightGBM · FastAPI · Postgres ·
Streamlit · Docker Compose (+ Caddy production overlay).

**Docs:** [`PROJECT_EXPLAINED.md`](PROJECT_EXPLAINED.md) — the complete technical
explainer (models, math, serving, everything) ·
[`presentation.html`](presentation.html) + [`presentation_script.md`](presentation_script.md)
— the defense deck and per-slide script · [`ANALYSIS.md`](ANALYSIS.md) — findings.

## Results — full dataset (4.02M train interactions · 148K items · 552K users)

Test-period evaluation, train-seen items excluded, positives = rating ≥ 4.0.

| Method | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | Coverage@50 |
|--------|-----------|---------|-----------|---------|-------------|
| Random | 0.0001 | 0.0001 | 0.0003 | 0.0001 | 1.0000 |
| Popularity | 0.0035 | 0.0021 | 0.0191 | 0.0062 | 0.0005 |
| MF-BPR | 0.0035 | 0.0020 | 0.0190 | 0.0059 | 0.0005 |
| Two-tower | 0.0018 | 0.0011 | 0.0042 | 0.0017 | **0.9503** |
| Two-tower + LightGBM | *(final full run in progress — see `data/comparison_results.csv`)* | | | | |

Two findings worth reading together (full discussion in `PROJECT_EXPLAINED.md` §5–6):

- **MF-BPR converged to a popularity clone** — with retrieval made faithful to the
  training objective (item bias folded into the FAISS index), its per-item bias term
  dominates on this head-heavy catalog and recall/coverage land exactly on
  Popularity's. An honest negative result, not a bug.
- **The two-tower trades recall for genuine personalization** — 95% catalog coverage
  vs Popularity's 0.05%. Never read Recall without Coverage: a recommender can score
  well by showing everyone the same ~70 bestsellers.

## Architecture

**Offline:** `01 data prep → 02 baselines → 03 MF-BPR → 04 two-tower → 05 LightGBM`
— each notebook feeds the next; 05 exports `artifacts/` (towers, FAISS inputs,
LightGBM model, popularity, feature store).

**Online, per request:** Postgres history (last 20 distinct items) → TorchScript user
tower → FAISS top-100 → LightGBM re-rank → seeded temperature sampling (fresh feed
per visit, stable within a session) → brand cap → "Because you viewed …" reasons.
Cold/unknown users fall back to popularity. Per-component latency is returned with
every response (visible in debug mode).

```
src/         config, data pipeline, eval, baselines, models, retrieval, ranking
notebooks/   the executed pipeline (full-dataset outputs saved inside)
api/         FastAPI service — Swagger at /docs
frontend/    the ElectroPicks store (Streamlit)
db/          Postgres schema (pg_trgm search indexes, carts w/ qty, orders, sessions)
scripts/     build_dataset, export_artifacts, populate_db, benchmark_latency
deploy/      Caddy config + production walkthrough
artifacts/   what the API serves (exported by notebook 05)
data/        frozen split + result CSVs (heavy binaries git-ignored — regenerate or
             pull from the team Drive)
```

## The store

Realtime personalization (every view/save/cart/buy changes the next page — no
retraining), type-ahead search with filters & sorting, category browsing, carts with
quantities, order history, wishlist, shareable product links (`?item=…`), cold-start
onboarding, category-guarded "Similar items", star ratings & badges, and a built-in
analytics dashboard.

**Client mode by default** — model labels, latency pills, the diversity slider, and
analytics are hidden unless you append **`?debug=1`** to the URL.

## Run it

```bash
cp .env.example .env
docker compose up -d --build          # postgres + api + frontend
python -m scripts.populate_db         # once: items + test-period interactions
```

- Store: http://localhost:8501 (debug: http://localhost:8501/?debug=1)
- API: http://localhost:8000/docs

The stack boots even before models exist — the API degrades gracefully to
popularity-only.

**Production:** one command behind Caddy with automatic HTTPS —
see [`deploy/README.md`](deploy/README.md).

```bash
DOMAIN=shop.example.com docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

## Reproduce the training

- **Colab (recommended):** upload `notebooks/run_all_colab.ipynb`, put the prepared
  `data/` folder in Drive at `MyDrive/amazon-recsys-data/data/`, pick a T4 runtime,
  Run all. It pins the fixed `src/` files, refuses to run on stale/sample data, and
  syncs results back to Drive.
- **Locally:** run notebooks 01→05 in order. Note for 8 GB machines: notebook 05 is
  memory-tuned (retrieves candidates only for users that can form LambdaRank groups)
  and LightGBM runs single-threaded to avoid a macOS OpenMP clash with torch/FAISS.

## Who did what

| Member | Owned |
|--------|-------|
| **Arun Kumar Aluru** | Two-tower retrieval (Notebook 04 — the deployed core), system architecture, full-dataset production runs, integration/GitHub, deployment |
| **Nimisha Busaniwar** | MF-BPR from scratch (Notebook 03): BPR loss, negative sampling, item-bias retrieval fix |
| **Otmane Zaim** | Project scaffold & repo, data pipeline + time split (Notebook 01), baselines (Notebook 02), LightGBM ranking lead (Notebook 05) |
| **Ram Navlani** | Webapp & serving stack (FastAPI · Postgres · Streamlit), realtime feedback loop, deployment stack |

Ranking features were a shared effort across all four.

## Engineering notes (the war stories)

Real issues found and fixed during the full-scale run — details in
`PROJECT_EXPLAINED.md` §9: MF-BPR retrieval made faithful to the training score
(bias folding); an 8 GB OOM fixed by retrieving only group-eligible users (~7× memory,
identical training data); a macOS LightGBM/OpenMP segfault (`num_threads=1`); browser
auto-translate crashing the UI (`notranslate` guard); GitHub's 100 MB cap vs a 151 MB
parquet (heavy data untracked); single-item Buy-now vs whole-cart checkout; multi-word
search + pg_trgm indexes.

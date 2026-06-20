# Real-Time Recommendation System — Amazon Electronics (2023)

Retrieval + ranking recommender served through a FastAPI + Postgres + Streamlit
stack, all running with `docker compose up`. Dataset: Amazon Reviews 2023,
**Electronics** category (McAuley-Lab/Amazon-Reviews-2023 on HuggingFace).

> _Add a GIF of the webapp in action here._

## Results (fill in after training)

| Method | Recall@20 | Recall@50 | NDCG@10 | Coverage |
|--------|-----------|-----------|---------|----------|
| Random | | | | |
| Popularity | | | | |
| MF-BPR | | | | |
| Two-tower | | | | |
| Two-tower + LightGBM | | | | |

**Average API response time:** _XX ms_ (see ANALYSIS.md for the per-component breakdown).

## Architecture

**Offline (notebook):** subsample → time split → train MF-BPR & two-tower → FAISS →
LightGBM ranker → export embeddings/index/model to `artifacts/`, populate Postgres.

**Online (API):** request → fetch user's *test-period* history from Postgres →
user-tower forward pass → FAISS top-100 → LightGBM re-rank → top-10. Cold/unknown
users fall back to popularity. Different pages use different models (see the table
in `notebooks/README.md`).

```
src/        data, eval (shared metrics), baselines, models, retrieval, ranking
api/        FastAPI service (/docs for Swagger)
frontend/   Streamlit UI
db/         Postgres schema
scripts/    export_artifacts.py, populate_db.py
notebooks/  the report (5 narrated notebooks)
artifacts/  exported models + index inputs (git-ignored)
```

## Run it

```bash
cp .env.example .env
# 1) train in the notebooks, then export:
python scripts/export_artifacts.py        # writes artifacts/
# 2) bring up the stack:
docker compose up --build
# 3) load data into Postgres (after postgres is healthy):
python scripts/populate_db.py
```

- API: http://localhost:8000/docs
- Frontend: http://localhost:8501

The stack also boots **before** models exist (S2): the API serves popularity and
degrades gracefully, so you can demo early and add models as they land.

## Reproduce the notebook
Open `notebooks/` in Colab (GPU runtime for PyTorch). Each notebook is
self-contained and imports from `src/`. Save the frozen split + checkpoints to
Google Drive to survive session timeouts.

## Who did what
| Member | Responsibility |
|--------|----------------|
| A | Data prep, EDA, time split, shared eval pipeline, baselines, LightGBM ranking |
| B | MF-BPR (from scratch), FAISS retrieval |
| C | Two-tower (from scratch), embedding export, similar-items |
| D | Webapp: Postgres, FastAPI, Streamlit, Docker, latency logging |

_(Adjust to reality — the brief grades an accurate task split.)_

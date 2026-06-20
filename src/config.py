"""Central config. Edit these once; everyone imports from here so the whole
team trains/evaluates on identical settings."""
from pathlib import Path

# ---- Dataset ----
CATEGORY = "Electronics"                      # Amazon Reviews 2023 category
HF_DATASET = "McAuley-Lab/Amazon-Reviews-2023"
REVIEW_CONFIG = f"raw_review_{CATEGORY}"
META_CONFIG = f"raw_meta_{CATEGORY}"

# ---- Subsampling (DOCUMENT every choice in the notebook) ----
RECENT_FROM_YEAR = 2019      # keep interactions on/after this year before k-core
KCORE_USER = 5               # min interactions per user  (keep light to preserve cold-start tail)
KCORE_ITEM = 5               # min interactions per item
MAX_INTERACTIONS = 5_000_000 # hard cap required by the brief

# ---- Positive interaction ----
POSITIVE_RATING_THRESHOLD = 4.0   # rating >= this counts as a "liked" positive for eval ground truth
USE_ITEM_LEVEL = "parent_asin"    # collapse colour/size variants onto the parent item

# ---- Time-based split (MANDATORY: random split => 0 on grading) ----
# Fractions of the global timeline. Train | Val | Test by timestamp quantiles.
VAL_QUANTILE = 0.85    # last 15% of *train period* used for tuning
TEST_QUANTILE = 0.90   # final 10% of the whole timeline is the test period

# ---- Evaluation ----
K_VALUES = (10, 20, 50)
TOP_N_CANDIDATES = 100   # retriever returns this many before ranking

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"     # embeddings, faiss index, lgbm model, encoders
DATA = ROOT / "data"               # frozen split parquet files live here (git-ignored)
ARTIFACTS.mkdir(exist_ok=True)
SEED = 42

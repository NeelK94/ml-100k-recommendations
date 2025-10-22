# config.py
from pathlib import Path

# -----------------------------
# Data source
# -----------------------------
DATA_SOURCE = "full"  # "small" or "full"

# Resolve repo root from this file (src/config.py -> repo root is parent of src)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOTS = {
    "small": PROJECT_ROOT / "data" / "raw" / "ml-latest-small",
    "full": PROJECT_ROOT / "data" / "raw" / "ml-latest",
}

RATINGS_CSV = "ratings.csv"
MOVIES_CSV = "movies.csv"
TAGS_CSV = "tags.csv"

# -----------------------------
# Output / artifacts
# -----------------------------
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
ENCODERS_DIR = OUTPUT_DIR / "encoders"

# -----------------------------
# Model / training parameters
# -----------------------------
EMBEDDING_SIZE = 64
HIDDEN_DIM = 128
DROPOUT = 0.1
BATCH_SIZE = 16
NUM_WORKERS = 0      
LR = 5e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 10
TFIDF_MAX_FEATURES = 1000  
USE_TAGS = True
USE_GENRES = True
SEED = 42

# -----------------------------
# Safety / memory settings
# -----------------------------
MOVIE_FEATURES_NP_THRESHOLD = 500 * 1024 * 1024  # 500 MB

# -----------------------------
# Make directories on import if needed
# -----------------------------
for d in (OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, ENCODERS_DIR):
    d.mkdir(parents=True, exist_ok=True)

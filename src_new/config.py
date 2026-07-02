import torch

# ============================================================
# Base model
# ============================================================

MODEL_NAMES = [
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]

# Backward-compatible default for one-off imports.
MODEL_NAME = MODEL_NAMES[0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 128

# Qwen 3B on 24GB GPU
BATCH_SIZE = 8

LR = 2e-4
MAX_EPOCHS = 25
TRAIN_STEPS = MAX_EPOCHS
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 0.001
SEED = 42

USE_SHORT_ANSWER_IN_TEXT = False
VALIDATION_RATIO = 0.2


# ============================================================
# Experiment grid
# ============================================================

CSV_PATH = "inputs/processed_qa_hallucination_dataset.csv"
DATASET_NAMES = [
    "hotpotqa",
    "triviaqa",
    "truthfulqa",
]
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "outputs/checkpoints"
HISTORY_DIR = "outputs/histories"
PLOT_DIR = "outputs/plots"


# ============================================================
# Semantic neighbour settings
# ============================================================

K_NEIGHBOURS = 3
NEIGHBOUR_BACKEND = "sentence"
NEIGHBOUR_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# Loss weights
# ============================================================

LAMBDA_BCE = 1.0

# Reduce pair rank slightly to avoid overfitting matched synthetic pairs.
LAMBDA_PAIR_RANK = 0.3
LAMBDA_INBATCH_RANK = 0.4
LAMBDA_NEIGHBOUR_RANK = 0.05


# ============================================================
# Margins
# ============================================================

RANK_MARGIN = 1.0
NEIGHBOUR_MARGIN = 1.0

# ============================================================
# Energy model head
# ============================================================

PROJ_DIM = 64
DROPOUT = 0.4
WEIGHT_DECAY = 3e-3

NORMALIZE_PROJECTED_STATES = False
USE_FEATURE_STANDARDIZATION = False


# ============================================================
# Tuning grid
# ============================================================

# These configs are intentionally small and method-driven:
#   - direct supervised losses should dominate retrieved-neighbour losses
#   - same-question PairRank should be at least as strong as InBatchRank
#   - neighbour margins should not exceed direct pair margins
#   - no-neighbour is the required ablation for the proposed method
TUNING_CONFIGS = [
    {
        "name": "no_neighbour_pair_inbatch",
        "neighbour_backend": "none",
        "k_neighbours": 0,
        "lambda_bce": 1.0,
        "lambda_pair_rank": 0.5,
        "lambda_inbatch_rank": 0.3,
        "lambda_neighbour_rank": 0.0,
        "rank_margin": 1.0,
        "neighbour_margin": 0.0,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "name": "tfidf_current",
        "neighbour_backend": "tfidf",
        "k_neighbours": 3,
        "lambda_bce": 1.0,
        "lambda_pair_rank": 0.3,
        "lambda_inbatch_rank": 0.4,
        "lambda_neighbour_rank": 0.05,
        "rank_margin": 1.0,
        "neighbour_margin": 1.0,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "name": "tfidf_conservative",
        "neighbour_backend": "tfidf",
        "k_neighbours": 3,
        "lambda_bce": 1.0,
        "lambda_pair_rank": 0.5,
        "lambda_inbatch_rank": 0.3,
        "lambda_neighbour_rank": 0.05,
        "rank_margin": 1.0,
        "neighbour_margin": 0.75,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "name": "dense_conservative",
        "neighbour_backend": "sentence",
        "k_neighbours": 3,
        "lambda_bce": 1.0,
        "lambda_pair_rank": 0.5,
        "lambda_inbatch_rank": 0.3,
        "lambda_neighbour_rank": 0.05,
        "rank_margin": 1.0,
        "neighbour_margin": 0.75,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
    },
]

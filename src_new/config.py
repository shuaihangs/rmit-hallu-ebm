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

# Reproduction setting for the HotpotQA -> TruthfulQA LLM-hidden K=5 result
# (AUC ~= 0.6725).
BATCH_SIZE = 8

LR = 2e-4
MAX_EPOCHS = 25
TRAIN_STEPS = MAX_EPOCHS
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 0.001
EVAL_EVERY_EPOCH = False
SEED = 42

USE_SHORT_ANSWER_IN_TEXT = False
VALIDATION_RATIO = 0.2

# Precompute frozen LLM answer-token pooled hidden states once, then train the
# projection and energy heads from cached raw selected-layer representations.
CACHE_FROZEN_LLM_FEATURES = True
FEATURE_CACHE_DIR = "outputs_qwen_k_sweep_rawloss_weighted_llmknn/feature_cache"
FEATURE_CACHE_BATCH_SIZE = 16


# ============================================================
# Experiment grid
# ============================================================

CSV_PATH = "inputs/processed_qa_hallucination_dataset.csv"
DATASET_NAMES = [
    "hotpotqa",
    "triviaqa",
    "truthfulqa",
    "squadqa",
]
OUTPUT_DIR = "outputs_qwen_k_sweep_rawloss_weighted_llmknn"
CHECKPOINT_DIR = "outputs_qwen_k_sweep_rawloss_weighted_llmknn/checkpoints"
HISTORY_DIR = "outputs_qwen_k_sweep_rawloss_weighted_llmknn/histories"
PLOT_DIR = "outputs_qwen_k_sweep_rawloss_weighted_llmknn/plots"


# ============================================================
# Semantic neighbour settings
# ============================================================

NEIGHBOUR_K_SWEEP = [3, 5, 10, 20]


# ============================================================
# Fixed raw-loss coefficients
# ============================================================

# The objective is the weighted sum of the raw loss terms.
LAMBDA_BCE = 1.0
LAMBDA_PAIR_RANK = 0.9
LAMBDA_INBATCH_RANK = 0.9
LAMBDA_NEIGHBOUR_RANK = 0.7


# ============================================================
# Optional automatic loss weighting
# ============================================================

AUTO_LOSS_WEIGHTING = False
AUTO_LOSS_REFERENCE = "bce_loss"
AUTO_LOSS_SCALE_BATCHES = 100
AUTO_LOSS_SCALE_STATISTIC = "median"


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

def make_llm_hidden_config(k):
    return {
        "name": f"llm_hidden_k{k}_rawloss",
        "neighbour_backend": "llm_hidden",
        "k_neighbours": k,
        "lambda_bce": LAMBDA_BCE,
        "lambda_pair_rank": LAMBDA_PAIR_RANK,
        "lambda_inbatch_rank": LAMBDA_INBATCH_RANK,
        "lambda_neighbour_rank": LAMBDA_NEIGHBOUR_RANK,
        "rank_margin": 1.0,
        "neighbour_margin": 1.0,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
    }


# Compare the no-neighbour control with frozen-LLM hidden-state neighbours
# while holding every loss coefficient and model hyperparameter fixed.
TUNING_CONFIGS = [
    {
        "name": "no_neighbour_pair_inbatch",
        "neighbour_backend": "none",
        "k_neighbours": 0,
        "lambda_bce": LAMBDA_BCE,
        "lambda_pair_rank": LAMBDA_PAIR_RANK,
        "lambda_inbatch_rank": LAMBDA_INBATCH_RANK,
        "lambda_neighbour_rank": 0.0,
        "rank_margin": 1.0,
        "neighbour_margin": 0.0,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
    },
    *[
        make_llm_hidden_config(k)
        for k in NEIGHBOUR_K_SWEEP
    ],
]

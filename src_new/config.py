import torch

# ============================================================
# Base model
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
#MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
#MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 128

# Qwen 3B on 24GB GPU
BATCH_SIZE = 8

LR = 2e-4
TRAIN_STEPS = 15
SEED = 42

USE_SHORT_ANSWER_IN_TEXT = False


# ============================================================
# Semantic neighbour settings
# ============================================================

K_NEIGHBOURS = 3


# ============================================================
# Loss weights
# ============================================================

LAMBDA_BCE = 1.0

# Reduce pair rank slightly to avoid overfitting matched synthetic pairs.
LAMBDA_PAIR_RANK = 0.3
LAMBDA_INBATCH_RANK = 0.4
LAMBDA_NEIGHBOUR_RANK = 0.05
# Keep cluster off.
LAMBDA_CLUSTER = 0.0


# ============================================================
# Margins
# ============================================================

RANK_MARGIN = 1.0
NEIGHBOUR_MARGIN = 1.0

DETACH_NEIGHBOUR_ANCHORS = True


# ============================================================
# Checkpoint
# ============================================================

BEST_CKPT_PATH = "best_qwen_answer_pool_pair03_inbatch04_no_neighbour.pt"


# ============================================================
# Energy model head
# ============================================================

PROJ_DIM = 64
DROPOUT = 0.4
WEIGHT_DECAY = 3e-3

NORMALIZE_PROJECTED_STATES = False
USE_FEATURE_STANDARDIZATION = False
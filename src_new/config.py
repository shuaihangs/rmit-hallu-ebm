import torch

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 64
LR = 5e-4
TRAIN_STEPS = 30
BATCH_SIZE = 16
SEED = 42

# Existing BCE + energy-bound objective
ALPHA = 0.5
M_IN = -1.0
M_OUT = 1.0
USE_SHORT_ANSWER_IN_TEXT = False

# ============================================================
# Neighbour-regularised energy training
# ============================================================
# For each HotpotQA training row, find K semantically similar HotpotQA
# questions using TF-IDF cosine nearest neighbours. Neighbour positives
# act as low-energy anchors and neighbour negatives act as high-energy anchors.
K_NEIGHBOURS = 5

# Loss weights
LAMBDA_PAIR_RANK = 0.5
LAMBDA_NEIGHBOUR_RANK = 0.5
LAMBDA_CLUSTER = 0.1

# Ranking margins. Since label 0 = truthful and label 1 = hallucinated,
# we want: E_positive + margin < E_negative.
RANK_MARGIN = 0.5
NEIGHBOUR_MARGIN = 0.5

# If True, cluster loss treats neighbour energy means as fixed anchors.
# This is usually more stable at the beginning.
DETACH_NEIGHBOUR_ANCHORS = True

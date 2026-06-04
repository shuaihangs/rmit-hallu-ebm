"""
Run training from terminal/tmux instead of a notebook.

Place this file in your project root, i.e. the folder that contains:

    src/
      config.py
      data.py
      model.py
      training.py
      evaluation.py
      utils.py

Then run:

    python run_train_rank_neighbour.py

Recommended tmux command:

    python run_train_rank_neighbour.py 2>&1 | tee train_rank_neighbour.log
"""

import os
import sys
import torch

# ---------------------------------------------------------------------
# Make sure Python can import src/
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src_new.config import (
    MODEL_NAME,
    DEVICE,
    MAX_LENGTH,
    LR,
    TRAIN_STEPS,
    BATCH_SIZE,
    SEED,
    ALPHA,
    M_IN,
    M_OUT,
    USE_SHORT_ANSWER_IN_TEXT,
    K_NEIGHBOURS,
    LAMBDA_PAIR_RANK,
    LAMBDA_NEIGHBOUR_RANK,
    LAMBDA_CLUSTER,
    RANK_MARGIN,
    NEIGHBOUR_MARGIN,
)

from src_new.utils import set_seed

from src_new.data import (
    load_rows_from_csv,
    split_examples_by_dataset,
    print_dataset_counts,
    validate_required_splits,
    build_dataloaders,
)

from src_new.model import (
    load_frozen_lm,
    build_energy_model,
)

from src_new.training import train_model
from src_new.evaluation import evaluate_loader, print_metrics


# ---------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------
CSV_PATH = "inputs/processed_qa_hallucination_dataset.csv"
# CSV_PATH = "processed_qa_hallucination_dataset.csv"

TRAIN_STEPS = 30

# Neighbour settings
K_NEIGHBOURS = 5

# L = L_rank + lambda * L_neighbour_rank + beta * L_cluster
LAMBDA_NEIGHBOUR_RANK = 1   # lambda
LAMBDA_CLUSTER = 0.5       # beta

RANK_MARGIN = 1.0
NEIGHBOUR_MARGIN = 2.0

DETACH_NEIGHBOUR_ANCHORS = True

BEST_CKPT_PATH = "best_hotpot_rank_neighbour_cluster_less_overfit.pt"

# Model head settings
PROJ_DIM = 128
DROPOUT = 0.2

# Reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main():
    print("==============================================")
    print("Rank + Neighbour-rank + Cluster training")
    print("==============================================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"CSV path: {CSV_PATH}")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"DEVICE: {DEVICE}")
    print(f"TRAIN_STEPS: {TRAIN_STEPS}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"K_NEIGHBOURS: {K_NEIGHBOURS}")
    print(f"LAMBDA_NEIGHBOUR_RANK lambda: {LAMBDA_NEIGHBOUR_RANK}")
    print(f"LAMBDA_CLUSTER beta: {LAMBDA_CLUSTER}")
    print(f"RANK_MARGIN gamma: {RANK_MARGIN}")
    print(f"NEIGHBOUR_MARGIN gamma_n: {NEIGHBOUR_MARGIN}")
    print(f"DETACH_NEIGHBOUR_ANCHORS: {DETACH_NEIGHBOUR_ANCHORS}")
    print("==============================================")

    set_seed(SEED)

    # Optional deterministic mode.
    # warn_only=True avoids crashing if some transformer CUDA ops are nondeterministic.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    rows = load_rows_from_csv(CSV_PATH)
    splits = split_examples_by_dataset(rows)

    print_dataset_counts(
        rows=splits["rows"],
        examples=splits["examples"],
    )

    validate_required_splits(splits)

    # -----------------------------------------------------------------
    # Load frozen base LM and energy model
    # -----------------------------------------------------------------
    print("\nLoading frozen base LM...")
    tokenizer, base_model = load_frozen_lm(MODEL_NAME, DEVICE)
    base_model.eval()

    print("\nBuilding energy model...")
    set_seed(SEED)
    energy_model = build_energy_model(
        base_model=base_model,
        device=DEVICE,
        proj_dim=PROJ_DIM,
        dropout=DROPOUT,
    )

    optimizer = torch.optim.AdamW(
        energy_model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    # -----------------------------------------------------------------
    # Build dataloaders
    # -----------------------------------------------------------------
    print("\nBuilding dataloaders and neighbours...")

    # Supports both versions of your build_dataloaders:
    # 1. with base_model/device for Qwen-based neighbours
    # 2. without base_model/device for sentence-transformer or TF-IDF neighbours
    try:
        loaders = build_dataloaders(
            splits=splits,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE,
            use_short_answer=USE_SHORT_ANSWER_IN_TEXT,
            num_workers=0,
            k_neighbours=K_NEIGHBOURS,
            )
    except TypeError:
        loaders = build_dataloaders(
            splits=splits,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE,
            use_short_answer=USE_SHORT_ANSWER_IN_TEXT,
            num_workers=0,
            k_neighbours=K_NEIGHBOURS,
        )

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    print("\nStarting training...")

    history = train_model(
        loaders=loaders,
        base_model=base_model,
        energy_model=energy_model,
        optimizer=optimizer,
        device=DEVICE,
        train_steps=TRAIN_STEPS,
        lambda_neighbour_rank=LAMBDA_NEIGHBOUR_RANK,
        lambda_cluster=LAMBDA_CLUSTER,
        rank_margin=RANK_MARGIN,
        neighbour_margin=NEIGHBOUR_MARGIN,
        detach_neighbour_anchors=DETACH_NEIGHBOUR_ANCHORS,
        best_ckpt_path=BEST_CKPT_PATH,
    )

    # -----------------------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------------------
    print("\nFinal evaluation using current model state:")

    hotpot_metrics = evaluate_loader(
        loaders["hotpot_eval"],
        base_model,
        energy_model,
        DEVICE,
    )

    trivia_metrics = evaluate_loader(
        loaders["trivia"],
        base_model,
        energy_model,
        DEVICE,
    )

    truthfulqa_metrics = evaluate_loader(
        loaders["truthfulqa"],
        base_model,
        energy_model,
        DEVICE,
    )

    print_metrics("HotpotQA", hotpot_metrics)
    print_metrics("TriviaQA", trivia_metrics)
    print_metrics("TruthfulQA", truthfulqa_metrics)

    # -----------------------------------------------------------------
    # Save training history
    # -----------------------------------------------------------------
    try:
        import pandas as pd

        hist_path = "history_rank_neighbour_cluster.csv"
        pd.DataFrame(history).to_csv(hist_path, index=False)
        print(f"\nSaved history to: {hist_path}")
    except Exception as e:
        print(f"\nCould not save history CSV: {e}")

    print(f"Best checkpoint path: {BEST_CKPT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()

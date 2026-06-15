import os
import sys

# Better to set this before importing torch / CUDA.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src_new.config import (
    MODEL_NAME,
    DEVICE,
    MAX_LENGTH,
    LR,
    BATCH_SIZE,
    SEED,
    USE_SHORT_ANSWER_IN_TEXT,
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

# Your previous run peaked around epoch 19, so start with 20.
TRAIN_STEPS = 20

K_NEIGHBOURS = 5

# Loss:
#   L = lambda_bce * BCE
#     + lambda_pair_rank * pair rank
#     + lambda_inbatch_rank * in-batch rank
#     + lambda_neighbour_rank * neighbour rank
#     + lambda_cluster * cluster
LAMBDA_PAIR_RANK = 0.5
LAMBDA_BCE = 1.0
LAMBDA_INBATCH_RANK = 0.2
LAMBDA_NEIGHBOUR_RANK = 0.1
LAMBDA_CLUSTER = 0.0

RANK_MARGIN = 1.0
NEIGHBOUR_MARGIN = 1.0
DETACH_NEIGHBOUR_ANCHORS = True

# Rename so you know this checkpoint uses contextual answer-token pooling.
BEST_CKPT_PATH = "best_answer_pool_projection_bce_pair_inbatch_neighbour.pt"

# Model head settings
PROJ_DIM = 64
DROPOUT = 0.3
WEIGHT_DECAY = 1e-3

# For this ablation, keep this False first.
NORMALIZE_PROJECTED_STATES = False
USE_FEATURE_STANDARDIZATION = False


def main():
    print("==============================================")
    print("Projection EBM training")
    print("Contextual answer-token pooling")
    print("BCE + PairRank + InBatchRank + NeighbourRank")
    print("==============================================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"CSV path: {CSV_PATH}")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"DEVICE: {DEVICE}")
    print(f"TRAIN_STEPS: {TRAIN_STEPS}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"MAX_LENGTH: {MAX_LENGTH}")
    print(f"K_NEIGHBOURS: {K_NEIGHBOURS}")
    print(f"LAMBDA_BCE: {LAMBDA_BCE}")
    print(f"LAMBDA_PAIR_RANK: {LAMBDA_PAIR_RANK}")
    print(f"LAMBDA_INBATCH_RANK: {LAMBDA_INBATCH_RANK}")
    print(f"LAMBDA_NEIGHBOUR_RANK: {LAMBDA_NEIGHBOUR_RANK}")
    print(f"LAMBDA_CLUSTER: {LAMBDA_CLUSTER}")
    print(f"RANK_MARGIN: {RANK_MARGIN}")
    print(f"NEIGHBOUR_MARGIN: {NEIGHBOUR_MARGIN}")
    print(f"DETACH_NEIGHBOUR_ANCHORS: {DETACH_NEIGHBOUR_ANCHORS}")
    print(f"PROJ_DIM: {PROJ_DIM}")
    print(f"DROPOUT: {DROPOUT}")
    print(f"WEIGHT_DECAY: {WEIGHT_DECAY}")
    print(f"NORMALIZE_PROJECTED_STATES: {NORMALIZE_PROJECTED_STATES}")
    print(f"USE_FEATURE_STANDARDIZATION: {USE_FEATURE_STANDARDIZATION}")
    print(f"BEST_CKPT_PATH: {BEST_CKPT_PATH}")
    print("==============================================")

    set_seed(SEED)

    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    print("\nLoading rows from CSV...")
    rows = load_rows_from_csv(CSV_PATH)
    splits = split_examples_by_dataset(rows)

    print_dataset_counts(
        rows=splits["rows"],
        examples=splits["examples"],
    )

    validate_required_splits(splits)

    # -----------------------------------------------------------------
    # Load frozen base LM
    # -----------------------------------------------------------------
    print("\nLoading frozen base LM...")
    tokenizer, base_model = load_frozen_lm(MODEL_NAME, DEVICE)
    base_model.eval()

    # -----------------------------------------------------------------
    # Build contextual-answer-pooling EBM
    # -----------------------------------------------------------------
    print("\nBuilding projection EBM...")
    set_seed(SEED)

    energy_model = build_energy_model(
        base_model=base_model,
        device=DEVICE,
        proj_dim=PROJ_DIM,
        dropout=DROPOUT,
        normalize_projected_states=NORMALIZE_PROJECTED_STATES,
        use_feature_standardization=USE_FEATURE_STANDARDIZATION,
    )

    print("\nEnergy model feature information:")
    feature_names = energy_model.get_feature_names()

    print(f"Number of features: {len(feature_names)}")

    if hasattr(energy_model, "num_selected_layers"):
        print(f"Number of selected layers: {energy_model.num_selected_layers}")

    if hasattr(energy_model, "proj_dim"):
        print(f"Projection dimension per layer: {energy_model.proj_dim}")

    if hasattr(energy_model, "selected_layer_names"):
        print("Selected layer groups:")
        for name in energy_model.selected_layer_names:
            print(f"  - {name}")

    if hasattr(energy_model, "selected_layer_indices"):
        print("Selected transformer layer indices:")
        print(f"  {energy_model.selected_layer_indices}")

    print("First 10 feature names:")
    for name in feature_names[:10]:
        print(f"  - {name}")

    if len(feature_names) > 10:
        print("  ...")

    # -----------------------------------------------------------------
    # Build dataloaders
    # -----------------------------------------------------------------
    print("\nBuilding dataloaders with answer masks and neighbours...")

    loaders = build_dataloaders(
        splits=splits,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        use_short_answer=USE_SHORT_ANSWER_IN_TEXT,
        num_workers=0,
        k_neighbours=K_NEIGHBOURS,
    )

    # Quick sanity check: make sure answer_mask exists.
    sanity_batch = next(iter(loaders["train"]))
    required_keys = [
        "pos_answer_mask",
        "neg_answer_mask",
    ]

    for key in required_keys:
        if key not in sanity_batch:
            raise KeyError(
                f"Missing {key} in train batch. "
                "Your data.py is not returning answer masks correctly."
            )

    print("Answer-mask sanity check passed.")
    print(f"pos_answer_mask shape: {sanity_batch['pos_answer_mask'].shape}")
    print(f"neg_answer_mask shape: {sanity_batch['neg_answer_mask'].shape}")
    print(
        "Mean positive answer tokens per sample:",
        sanity_batch["pos_answer_mask"].sum(dim=1).float().mean().item(),
    )
    print(
        "Mean negative answer tokens per sample:",
        sanity_batch["neg_answer_mask"].sum(dim=1).float().mean().item(),
    )

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        energy_model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
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
        lambda_pair_rank=LAMBDA_PAIR_RANK,
        lambda_bce=LAMBDA_BCE,
        lambda_inbatch_rank=LAMBDA_INBATCH_RANK,
        lambda_neighbour_rank=LAMBDA_NEIGHBOUR_RANK,
        lambda_cluster=LAMBDA_CLUSTER,
        rank_margin=RANK_MARGIN,
        neighbour_margin=NEIGHBOUR_MARGIN,
        detach_neighbour_anchors=DETACH_NEIGHBOUR_ANCHORS,
        best_ckpt_path=BEST_CKPT_PATH,
    )

    # -----------------------------------------------------------------
    # Reload best checkpoint before final evaluation
    # -----------------------------------------------------------------
    print("\nLoading best checkpoint before final evaluation...")

    ckpt = torch.load(
        BEST_CKPT_PATH,
        map_location=DEVICE,
    )

    energy_model.load_state_dict(ckpt["model_state_dict"])
    energy_model.eval()

    print(f"Loaded best checkpoint from epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Best checkpoint mean_eval_auc: {ckpt.get('mean_eval_auc', 'unknown')}")

    # -----------------------------------------------------------------
    # Final evaluation using BEST model state
    # -----------------------------------------------------------------
    print("\nFinal evaluation using BEST checkpoint:")

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

        hist_path = "history_answer_pool_projection_bce_pair_inbatch_neighbour.csv"
        pd.DataFrame(history).to_csv(hist_path, index=False)
        print(f"\nSaved history to: {hist_path}")

    except Exception as e:
        print(f"\nCould not save history CSV: {e}")

    print(f"Best checkpoint path: {BEST_CKPT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
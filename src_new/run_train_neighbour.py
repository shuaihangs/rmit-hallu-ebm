"""
Example training entry point.

Put this file one level above the src/ package or run it with the correct
PYTHONPATH. Example:

    python run_train_neighbour.py

Assumes the CSV has columns:
    dataset, question, short_answer, positive, negative
"""
import torch
import torch.optim as optim

from src.config import (
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
    DETACH_NEIGHBOUR_ANCHORS,
)
from src.utils import set_seed
from src.data import (
    load_rows_from_csv,
    split_examples_by_dataset,
    print_dataset_counts,
    validate_required_splits,
    build_dataloaders,
)
from src.model import load_frozen_lm, build_energy_model
from src.training import train_model


CSV_PATH = "processed_qa_hallucination_dataset_checkpoint.csv"


def main():
    set_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Base model: {MODEL_NAME}")

    rows = load_rows_from_csv(CSV_PATH)
    splits = split_examples_by_dataset(rows)
    print_dataset_counts(rows, splits["examples"])
    validate_required_splits(splits)

    tokenizer, base_model = load_frozen_lm(MODEL_NAME, DEVICE)
    energy_model = build_energy_model(base_model, DEVICE)

    loaders = build_dataloaders(
        splits,
        tokenizer,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        use_short_answer=USE_SHORT_ANSWER_IN_TEXT,
        k_neighbours=K_NEIGHBOURS,
    )

    optimizer = optim.AdamW(energy_model.parameters(), lr=LR)

    history = train_model(
        loaders=loaders,
        base_model=base_model,
        energy_model=energy_model,
        optimizer=optimizer,
        device=DEVICE,
        train_steps=TRAIN_STEPS,
        alpha=ALPHA,
        m_in=M_IN,
        m_out=M_OUT,
        lambda_pair_rank=LAMBDA_PAIR_RANK,
        lambda_neighbour_rank=LAMBDA_NEIGHBOUR_RANK,
        lambda_cluster=LAMBDA_CLUSTER,
        rank_margin=RANK_MARGIN,
        neighbour_margin=NEIGHBOUR_MARGIN,
        detach_neighbour_anchors=DETACH_NEIGHBOUR_ANCHORS,
    )

    torch.save({
        "model_state_dict": energy_model.state_dict(),
        "history": history,
    }, "final_neighbour_energy_model.pt")


if __name__ == "__main__":
    main()

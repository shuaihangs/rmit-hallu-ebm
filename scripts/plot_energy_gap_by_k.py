#!/usr/bin/env python3
"""Plot in-domain validation energy gaps by model, dataset, and neighbour K."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODEL_LABELS = {
    "Qwen/Qwen2.5-3B-Instruct": "Qwen 2.5-3B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2-3B",
    "microsoft/Phi-3.5-mini-instruct": "Phi 3.5-mini",
}
DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "triviaqa": "TriviaQA",
    "truthfulqa": "TruthfulQA",
    "squadqa": "SQuAD",
}
K_ORDER = [0, 3, 5, 10, 20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument(
        "--output-pdf",
        type=Path,
        help="Optional publication-quality PDF destination.",
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        help="Optional CSV containing the 60 plotted values.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    selected = []
    for row in rows:
        train_dataset = row["train_dataset"]
        if row["eval_dataset"] != f"{train_dataset}_val":
            continue
        selected.append(
            {
                "model_name": row["model_name"],
                "model_label": MODEL_LABELS.get(row["model_name"], row["model_name"]),
                "dataset": train_dataset,
                "dataset_label": DATASET_LABELS.get(train_dataset, train_dataset),
                "k_neighbours": int(float(row["k_neighbours"])),
                "mean_truthful_energy": float(row["mean_pos_energy"]),
                "mean_hallucinated_energy": float(row["mean_neg_energy"]),
                "energy_gap": float(row["energy_gap"]),
            }
        )

    expected = len(MODEL_LABELS) * len(DATASET_LABELS) * len(K_ORDER)
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} in-domain validation rows, found {len(selected)}."
        )

    keys = {
        (row["model_name"], row["dataset"], row["k_neighbours"])
        for row in selected
    }
    if len(keys) != expected:
        raise ValueError("Duplicate or missing model/dataset/K result rows.")
    return selected


def save_plot_data(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "model_label",
        "dataset",
        "dataset_label",
        "k_neighbours",
        "mean_truthful_energy",
        "mean_hallucinated_energy",
        "energy_gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary_csv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.86,
        bottom=0.12,
        hspace=0.40,
        wspace=0.20,
    )
    model_order = list(MODEL_LABELS)
    dataset_order = list(DATASET_LABELS)
    markers = ["o", "s", "^"]

    for axis, dataset in zip(axes.flat, dataset_order):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for model, marker in zip(model_order, markers):
            model_rows = sorted(
                (row for row in dataset_rows if row["model_name"] == model),
                key=lambda row: K_ORDER.index(row["k_neighbours"]),
            )
            axis.plot(
                [row["k_neighbours"] for row in model_rows],
                [row["energy_gap"] for row in model_rows],
                marker=marker,
                linewidth=2.0,
                markersize=6,
                label=MODEL_LABELS[model],
            )

        values = [row["energy_gap"] for row in dataset_rows]
        span = max(values) - min(values)
        padding = max(0.6, span * 0.15)
        axis.set_ylim(min(values) - padding, max(values) + padding)
        axis.set_xticks(K_ORDER)
        axis.set_title(DATASET_LABELS[dataset], fontweight="bold")
        axis.set_xlabel("Neighbour count K (0 = no neighbours)")
        axis.set_ylabel("Energy gap: hallucinated − truthful")
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "V7 in-domain validation energy separation by neighbour count",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.025,
        "Panels use independent y-axis ranges. Higher positive gaps mean stronger "
        "separation of hallucinated from truthful claims.",
        ha="center",
        fontsize=9,
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_png, dpi=220, bbox_inches="tight")
    if args.output_pdf:
        args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output_pdf, bbox_inches="tight")
    plt.close(figure)

    if args.output_data:
        save_plot_data(rows, args.output_data)

    print(f"Saved PNG: {args.output_png}")
    if args.output_pdf:
        print(f"Saved PDF: {args.output_pdf}")
    if args.output_data:
        print(f"Saved plot data: {args.output_data}")


if __name__ == "__main__":
    main()

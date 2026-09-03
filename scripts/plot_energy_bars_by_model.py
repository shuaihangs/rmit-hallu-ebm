#!/usr/bin/env python3
"""Plot truthful and hallucinated mean energies by model and dataset at one K."""

from __future__ import annotations

import argparse
import csv
import re
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
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--output-data",
        type=Path,
        help="Optional CSV containing the plotted in-domain validation rows.",
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=K_ORDER,
        default=10,
        help="Neighbour count to plot (default: 10).",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    rows: list[dict[str, object]] = []
    for row in source_rows:
        train_dataset = row["train_dataset"]
        if row["eval_dataset"] != f"{train_dataset}_val":
            continue
        rows.append(
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
    if len(rows) != expected:
        raise ValueError(
            f"Expected {expected} in-domain validation rows, found {len(rows)}."
        )

    keys = {
        (row["model_name"], row["dataset"], row["k_neighbours"])
        for row in rows
    }
    if len(keys) != expected:
        raise ValueError("Duplicate or missing model/dataset/K result rows.")
    return rows


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


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def plot_model(
    rows: list[dict[str, object]],
    model: str,
    k_neighbours: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    dataset_order = list(DATASET_LABELS)
    truthful: list[float] = []
    hallucinated: list[float] = []
    for dataset in dataset_order:
        match = next(
            row
            for row in rows
            if row["model_name"] == model
            and row["dataset"] == dataset
            and row["k_neighbours"] == k_neighbours
        )
        truthful.append(float(match["mean_truthful_energy"]))
        hallucinated.append(float(match["mean_hallucinated_energy"]))

    x = np.arange(len(dataset_order))
    width = 0.30
    figure, axis = plt.subplots(figsize=(6.2, 3.9))
    figure.subplots_adjust(left=0.11, right=0.98, top=0.75, bottom=0.15)

    truthful_bars = axis.bar(
        x - width / 2,
        truthful,
        width,
        label="Truthful",
        color="#3B82F6",
        linewidth=0,
        zorder=3,
    )
    hallucinated_bars = axis.bar(
        x + width / 2,
        hallucinated,
        width,
        label="Hallucinated",
        color="#F97316",
        linewidth=0,
        zorder=3,
    )
    axis.axhline(0.0, color="#555555", linewidth=0.9, zorder=2)
    axis.set_xticks(x, [DATASET_LABELS[dataset] for dataset in dataset_order])
    axis.set_ylabel("Mean energy", color="#344054")
    axis.tick_params(axis="both", labelsize=9, length=0)
    for label in axis.get_xticklabels():
        label.set_fontweight("bold")

    all_values = truthful + hallucinated
    value_span = max(all_values) - min(all_values)
    axis.set_ylim(
        min(all_values) - value_span * 0.12,
        max(all_values) + value_span * 0.14,
    )
    axis.bar_label(truthful_bars, fmt="%.1f", padding=2, fontsize=8, color="#1D4ED8")
    axis.bar_label(
        hallucinated_bars,
        fmt="%.1f",
        padding=2,
        fontsize=8,
        color="#C2410C",
    )

    figure.suptitle(
        f"Average claim energy by {MODEL_LABELS[model]} (k = {k_neighbours})",
        x=0.50,
        y=0.96,
        ha="center",
        fontsize=12.5,
        fontweight="bold",
    )
    figure.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 0.89),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=1.6,
        columnspacing=1.2,
    )
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.65, zorder=0)
    axis.grid(axis="x", visible=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    stem = f"energy_by_claim_type_{slugify(MODEL_LABELS[model])}_k{k_neighbours}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary_csv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in MODEL_LABELS:
        png_path, pdf_path = plot_model(rows, model, args.k, args.output_dir)
        print(f"Saved PNG: {png_path}")
        print(f"Saved PDF: {pdf_path}")

    if args.output_data:
        plotted_rows = [row for row in rows if row["k_neighbours"] == args.k]
        save_plot_data(plotted_rows, args.output_data)
        print(f"Saved plot data: {args.output_data}")


if __name__ == "__main__":
    main()

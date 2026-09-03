#!/usr/bin/env python3
"""Compare EBM and probe baselines on in-domain validation and OOD transfers."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections.abc import Iterable
from pathlib import Path


DATASETS = ["hotpotqa", "triviaqa", "truthfulqa", "squadqa"]
DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "triviaqa": "TriviaQA",
    "truthfulqa": "TruthfulQA",
    "squadqa": "SQuAD",
}
MODEL_LABELS = {
    "Qwen/Qwen2.5-3B-Instruct": "Qwen 2.5-3B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2-3B",
    "microsoft/Phi-3.5-mini-instruct": "Phi 3.5-mini",
}
METHODS = ["EBM", "ICR", "SEP", "SAPLMA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ebm_csv", type=Path)
    parser.add_argument("icr_csv", type=Path)
    parser.add_argument("sep_csv", type=Path)
    parser.add_argument("saplama_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--model-name",
        choices=list(MODEL_LABELS),
        default="microsoft/Phi-3.5-mini-instruct",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_ood(train_dataset: str, eval_dataset: str) -> bool:
    return (
        train_dataset in DATASETS
        and eval_dataset in DATASETS
        and train_dataset != eval_dataset
    )


def classify_evaluation(
    train_dataset: str, raw_eval_dataset: str
) -> tuple[str, str] | None:
    if train_dataset not in DATASETS:
        return None
    if raw_eval_dataset == f"{train_dataset}_val":
        return train_dataset, "in_domain_validation"
    if is_ood(train_dataset, raw_eval_dataset):
        return raw_eval_dataset, "out_of_domain"
    return None


def standard_row(
    method: str,
    model_name: str,
    train_dataset: str,
    eval_dataset: str,
    auc: float,
    evaluation_type: str,
    *,
    saplama_layer: int | None = None,
    saplama_selection_auc: float | None = None,
) -> dict[str, object]:
    return {
        "method": method,
        "model_name": model_name,
        "model_label": MODEL_LABELS.get(model_name, model_name),
        "train_dataset": train_dataset,
        "train_label": DATASET_LABELS[train_dataset],
        "eval_dataset": eval_dataset,
        "eval_label": DATASET_LABELS[eval_dataset],
        "evaluation_type": evaluation_type,
        "auc": auc,
        "saplama_layer_from_end": saplama_layer,
        "saplama_selection_val_auc": saplama_selection_auc,
    }


def load_ebm(path: Path, k_neighbours: int) -> list[dict[str, object]]:
    rows = []
    for row in read_csv(path):
        train_dataset = row["train_dataset"]
        classified = classify_evaluation(train_dataset, row["eval_dataset"])
        if int(float(row["k_neighbours"])) != k_neighbours:
            continue
        if classified is not None:
            eval_dataset, evaluation_type = classified
            rows.append(
                standard_row(
                    "EBM",
                    row["model_name"],
                    train_dataset,
                    eval_dataset,
                    float(row["energy_auc"]),
                    evaluation_type,
                )
            )
    return rows


def load_icr(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in read_csv(path):
        train_dataset = row["train_dataset"]
        classified = classify_evaluation(train_dataset, row["eval_dataset"])
        if classified is not None:
            eval_dataset, evaluation_type = classified
            rows.append(
                standard_row(
                    "ICR",
                    row["model_name"],
                    train_dataset,
                    eval_dataset,
                    float(row["ROC-AUC"]),
                    evaluation_type,
                )
            )
    return rows


def load_sep(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in read_csv(path):
        train_dataset = row["train_dataset"]
        classified = classify_evaluation(train_dataset, row["dataset"])
        if classified is not None:
            eval_dataset, evaluation_type = classified
            rows.append(
                standard_row(
                    "SEP",
                    row["model_name"],
                    train_dataset,
                    eval_dataset,
                    float(row["hallucination_auroc"]),
                    evaluation_type,
                )
            )
    return rows


def load_saplama(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_rows = read_csv(path)
    selections: list[dict[str, object]] = []
    selected_layers: dict[tuple[str, str], tuple[int, float]] = {}

    for model_name in MODEL_LABELS:
        for train_dataset in DATASETS:
            validation_rows = [
                row
                for row in source_rows
                if row["model_name"] == model_name
                and row["train_dataset"] == train_dataset
                and row["eval_dataset"] == f"{train_dataset}_val"
            ]
            if not validation_rows:
                raise ValueError(
                    f"No SAPLMA validation rows for {model_name} / {train_dataset}."
                )
            best = max(
                validation_rows,
                key=lambda row: (
                    float(row["auc_hallucinated_mean"]),
                    int(row["layer_from_end"]),
                ),
            )
            layer = int(best["layer_from_end"])
            val_auc = float(best["auc_hallucinated_mean"])
            selected_layers[(model_name, train_dataset)] = (layer, val_auc)
            selections.append(
                {
                    "model_name": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "train_dataset": train_dataset,
                    "train_label": DATASET_LABELS[train_dataset],
                    "selected_layer_from_end": layer,
                    "selection_val_auc": val_auc,
                }
            )

    rows = []
    for row in source_rows:
        model_name = row["model_name"]
        train_dataset = row["train_dataset"]
        classified = classify_evaluation(train_dataset, row["eval_dataset"])
        if classified is None:
            continue
        eval_dataset, evaluation_type = classified
        layer, val_auc = selected_layers[(model_name, train_dataset)]
        if int(row["layer_from_end"]) != layer:
            continue
        rows.append(
            standard_row(
                "SAPLMA",
                model_name,
                train_dataset,
                eval_dataset,
                float(row["auc_hallucinated_mean"]),
                evaluation_type,
                saplama_layer=layer,
                saplama_selection_auc=val_auc,
            )
        )
    return rows, selections


def dataset_pairs() -> list[tuple[str, str]]:
    return [
        (train_dataset, eval_dataset)
        for train_dataset in DATASETS
        for eval_dataset in DATASETS
    ]


def validate(rows: list[dict[str, object]]) -> None:
    expected_keys = {
        (method, model_name, train_dataset, eval_dataset)
        for method in METHODS
        for model_name in MODEL_LABELS
        for train_dataset, eval_dataset in dataset_pairs()
    }
    actual_keys = {
        (
            str(row["method"]),
            str(row["model_name"]),
            str(row["train_dataset"]),
            str(row["eval_dataset"]),
        )
        for row in rows
    }
    if len(rows) != len(actual_keys):
        raise ValueError("Duplicate method/model/train/eval rows were found.")
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        raise ValueError(
            f"Expected {len(expected_keys)} comparison rows, found {len(actual_keys)}; "
            f"missing={sorted(missing)}; extra={sorted(extra)}"
        )
    for row in rows:
        auc = float(row["auc"])
        if not math.isfinite(auc) or not 0.0 <= auc <= 1.0:
            raise ValueError(f"Invalid AUROC in row: {row}")


def ordered_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    model_index = {name: index for index, name in enumerate(MODEL_LABELS)}
    method_index = {name: index for index, name in enumerate(METHODS)}
    transfer_index = {pair: index for index, pair in enumerate(dataset_pairs())}
    return sorted(
        rows,
        key=lambda row: (
            model_index[str(row["model_name"])],
            method_index[str(row["method"])],
            transfer_index[(str(row["train_dataset"]), str(row["eval_dataset"]))],
        ),
    )


def save_rows(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "method",
        "model_name",
        "model_label",
        "train_dataset",
        "train_label",
        "eval_dataset",
        "eval_label",
        "evaluation_type",
        "auc",
        "is_best_for_transfer",
        "saplama_layer_from_end",
        "saplama_selection_val_auc",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_selections(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    args = parse_args()
    saplama_rows, saplama_selections = load_saplama(args.saplama_csv)
    rows = (
        load_ebm(args.ebm_csv, args.k)
        + load_icr(args.icr_csv)
        + load_sep(args.sep_csv)
        + saplama_rows
    )
    validate(rows)
    rows = ordered_rows(rows)

    for model_name in MODEL_LABELS:
        for train_dataset, eval_dataset in dataset_pairs():
            matching = [
                row
                for row in rows
                if row["model_name"] == model_name
                and row["train_dataset"] == train_dataset
                and row["eval_dataset"] == eval_dataset
            ]
            best_auc = max(float(row["auc"]) for row in matching)
            for row in matching:
                row["is_best_for_transfer"] = math.isclose(
                    float(row["auc"]), best_auc, abs_tol=1e-12
                )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    model_name = args.model_name
    model_rows = [row for row in rows if row["model_name"] == model_name]
    vmin = 0.6
    vmax = 0.9
    cmap = plt.colormaps["Blues"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = slugify(MODEL_LABELS[model_name])

    matrices: dict[str, np.ndarray] = {}
    for method in METHODS:
        matrix = np.full((len(DATASETS), len(DATASETS)), np.nan, dtype=float)
        for train_index, train_dataset in enumerate(DATASETS):
            for eval_index, eval_dataset in enumerate(DATASETS):
                row = next(
                    row
                    for row in model_rows
                    if row["method"] == method
                    and row["train_dataset"] == train_dataset
                    and row["eval_dataset"] == eval_dataset
                )
                matrix[train_index, eval_index] = float(row["auc"])
        matrices[method] = matrix

    figure, axes = plt.subplots(
        1,
        len(METHODS),
        figsize=(14.4, 4.0),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(left=0.085, right=0.90, top=0.84, bottom=0.22, wspace=0.10)
    labels = [DATASET_LABELS[dataset] for dataset in DATASETS]
    image = None

    for method_index, (method, axis) in enumerate(zip(METHODS, axes, strict=True)):
        matrix = matrices[method]
        image = axis.imshow(
            matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        method_label = f"SCONE (k = {args.k})" if method == "EBM" else method
        axis.set_title(
            method_label,
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        axis.set_xticks(range(len(DATASETS)), labels, rotation=25, ha="right")
        axis.set_yticks(range(len(DATASETS)), labels)
        axis.tick_params(
            axis="both",
            labelsize=8.5,
            length=0,
            labelleft=method_index == 0,
        )
        axis.set_xticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=2)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)

        for train_index in range(len(DATASETS)):
            for eval_index in range(len(DATASETS)):
                value = matrix[train_index, eval_index]
                axis.text(
                    eval_index,
                    train_index,
                    f"{value:.4f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if value >= 0.75 else "#17202A",
                )

    figure.supxlabel("Evaluation dataset", fontsize=10, y=0.04)
    figure.supylabel("Training dataset", fontsize=10, x=0.015)
    colorbar_axis = figure.add_axes((0.925, 0.22, 0.015, 0.62))
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="vertical")
    colorbar.set_label("AUROC", fontsize=9)
    colorbar.ax.tick_params(labelsize=8.5, length=2)

    png_path = args.output_dir / f"ood_auc_{model_slug}_k{args.k}_combined.png"
    pdf_path = args.output_dir / f"ood_auc_{model_slug}_k{args.k}_combined.pdf"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")

    data_path = args.output_dir / f"ood_auc_{model_slug}_k{args.k}_comparison.csv"
    selection_path = args.output_dir / "saplama_layer_selection.csv"
    save_rows(model_rows, data_path)
    save_selections(
        [row for row in saplama_selections if row["model_name"] == model_name],
        selection_path,
    )

    print(f"Validated comparison rows: {len(rows)}")
    print(f"Plotted {len(model_rows)} rows for {MODEL_LABELS[model_name]}")
    print(f"Saved comparison data: {data_path}")
    print(f"Saved SAPLMA layer selections: {selection_path}")


if __name__ == "__main__":
    main()

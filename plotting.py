import argparse
import glob
import os

import pandas as pd

try:
    from src_new.config import DATASET_NAMES, OUTPUT_DIR, HISTORY_DIR, PLOT_DIR
except Exception:
    DATASET_NAMES = ["hotpotqa", "triviaqa", "truthfulqa"]
    OUTPUT_DIR = "outputs"
    HISTORY_DIR = os.path.join(OUTPUT_DIR, "histories")
    PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create plots and aggregate tables from EBM experiment outputs."
    )
    parser.add_argument(
        "--summary",
        default=os.path.join(OUTPUT_DIR, "experiment_summary.csv"),
        help="Path to experiment_summary.csv.",
    )
    parser.add_argument(
        "--history-dir",
        default=HISTORY_DIR,
        help="Directory containing per-run history CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        default=PLOT_DIR,
        help="Directory for plots and aggregate CSVs.",
    )
    return parser.parse_args()


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def is_source_train(row):
    return row["eval_dataset"] == f"{row['train_dataset']}_train"


def is_source_val(row):
    return row["eval_dataset"] == f"{row['train_dataset']}_val"


def add_split_type(summary):
    summary = summary.copy()
    split_types = []

    for _, row in summary.iterrows():
        if is_source_train(row):
            split_types.append("source_train")
        elif is_source_val(row):
            split_types.append("source_val")
        else:
            split_types.append("ood")

    summary["split_type"] = split_types
    return summary


def load_summary(path):
    if not os.path.exists(path):
        print(f"Summary file not found: {path}")
        return pd.DataFrame()

    summary = pd.read_csv(path)

    if summary.empty:
        print(f"Summary file is empty: {path}")
        return summary

    required = {
        "config_name",
        "model_name",
        "train_dataset",
        "eval_dataset",
        "energy_auc",
    }
    missing = required - set(summary.columns)

    if missing:
        raise ValueError(
            f"Summary file is missing required columns: {sorted(missing)}"
        )

    return add_split_type(summary)


def save_aggregate_tables(summary, out_dir):
    if summary.empty:
        return {}

    tables = {}

    ood = summary[summary["split_type"] == "ood"].copy()
    source_val = summary[summary["split_type"] == "source_val"].copy()

    if not ood.empty:
        ood_mean = (
            ood.groupby(["config_name", "model_name", "train_dataset"], as_index=False)
            .agg(
                ood_mean_auc=("energy_auc", "mean"),
                ood_min_auc=("energy_auc", "min"),
                ood_max_auc=("energy_auc", "max"),
            )
        )
        path = os.path.join(out_dir, "ood_mean_auc.csv")
        ood_mean.to_csv(path, index=False)
        tables["ood_mean"] = ood_mean
        print(f"Wrote {path}")

    if not source_val.empty:
        source_val_ranked = source_val.sort_values(
            ["model_name", "train_dataset", "energy_auc"],
            ascending=[True, True, False],
        )
        best_by_source = source_val_ranked.groupby(
            ["model_name", "train_dataset"],
            as_index=False,
        ).first()
        path = os.path.join(out_dir, "best_config_by_source_validation.csv")
        best_by_source.to_csv(path, index=False)
        tables["best_by_source"] = best_by_source
        print(f"Wrote {path}")

    return tables


def plot_bar_mean(summary, out_dir, split_type, filename, title):
    data = summary[summary["split_type"] == split_type]

    if data.empty:
        return

    plot_data = (
        data.groupby("config_name", as_index=False)
        .agg(mean_auc=("energy_auc", "mean"), std_auc=("energy_auc", "std"))
        .sort_values("mean_auc", ascending=False)
    )

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_data["config_name"], plot_data["mean_auc"])
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_truthfulqa_transfer(summary, out_dir):
    data = summary[
        (summary["eval_dataset"] == "truthfulqa")
        & (summary["train_dataset"] != "truthfulqa")
    ].copy()

    if data.empty:
        return

    plot_data = (
        data.groupby("config_name", as_index=False)
        .agg(mean_auc=("energy_auc", "mean"))
        .sort_values("mean_auc", ascending=False)
    )

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_data["config_name"], plot_data["mean_auc"])
    ax.set_ylabel("TruthfulQA transfer AUC")
    ax.set_title("TruthfulQA Transfer From Other Source Datasets")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "truthfulqa_transfer_auc_by_config.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_heatmaps(summary, out_dir):
    if summary.empty:
        return

    plt = setup_matplotlib()

    for (model_name, config_name), group in summary.groupby(
        ["model_name", "config_name"]
    ):
        pivot = group.pivot_table(
            index="train_dataset",
            columns="eval_dataset",
            values="energy_auc",
            aggfunc="mean",
        )

        if pivot.empty:
            continue

        fig_width = max(7, 1.2 * len(pivot.columns))
        fig_height = max(4, 0.9 * len(pivot.index))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(pivot.values, vmin=0.0, vmax=1.0, cmap="viridis")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"AUC Heatmap | {config_name} | {model_name}")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if pd.notna(value):
                    ax.text(
                        j,
                        i,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="white" if value < 0.75 else "black",
                        fontsize=8,
                    )

        fig.colorbar(image, ax=ax, label="AUC")
        fig.tight_layout()
        path = os.path.join(
            out_dir,
            f"heatmap_{slugify(model_name)}_{slugify(config_name)}.png",
        )
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Wrote {path}")


def slugify(value):
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()


def load_histories(history_dir):
    paths = sorted(glob.glob(os.path.join(history_dir, "history_*.csv")))

    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        if not frame.empty:
            frame["history_path"] = path
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def add_history_ood_mean(history):
    history = history.copy()
    ood_means = []

    for _, row in history.iterrows():
        train_dataset = row.get("train_dataset")
        aucs = []

        for dataset_name in DATASET_NAMES:
            if dataset_name == train_dataset:
                continue

            column = f"{dataset_name}_auc"
            if column in history.columns and pd.notna(row.get(column)):
                aucs.append(float(row[column]))

        ood_means.append(sum(aucs) / len(aucs) if aucs else float("nan"))

    history["ood_mean_auc"] = ood_means
    return history


def plot_history_curves(history, out_dir):
    if history.empty:
        return

    history = add_history_ood_mean(history)
    plt = setup_matplotlib()

    for metric, filename, title in [
        ("monitor_auc", "monitor_auc_curve_by_config.png", "Source Validation AUC"),
        ("ood_mean_auc", "ood_mean_auc_curve_by_config.png", "OOD Mean AUC"),
    ]:
        if metric not in history.columns:
            continue

        data = history.dropna(subset=[metric])

        if data.empty:
            continue

        curve = (
            data.groupby(["config_name", "epoch"], as_index=False)[metric]
            .mean()
            .sort_values(["config_name", "epoch"])
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        for config_name, group in curve.groupby("config_name"):
            ax.plot(group["epoch"], group[metric], marker="o", label=config_name)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Wrote {path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    summary = load_summary(args.summary)
    save_aggregate_tables(summary, args.out_dir)

    if not summary.empty:
        summary.to_csv(
            os.path.join(args.out_dir, "experiment_summary_with_split_type.csv"),
            index=False,
        )
        plot_bar_mean(
            summary,
            args.out_dir,
            "source_val",
            "source_validation_auc_by_config.png",
            "Source Validation AUC by Config",
        )
        plot_bar_mean(
            summary,
            args.out_dir,
            "ood",
            "ood_auc_by_config.png",
            "OOD Transfer AUC by Config",
        )
        plot_truthfulqa_transfer(summary, args.out_dir)
        plot_heatmaps(summary, args.out_dir)

    history = load_histories(args.history_dir)

    if not history.empty:
        plot_history_curves(history, args.out_dir)

    print("Plotting complete.")


if __name__ == "__main__":
    main()

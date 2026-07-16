import argparse
import os
import re
import sys

# Better to set this before importing torch / CUDA.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src_new.config import (
    MODEL_NAMES,
    DEVICE,
    MAX_LENGTH,
    LR,
    TRAIN_STEPS,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    EVAL_EVERY_EPOCH,
    BATCH_SIZE,
    SEED,
    USE_SHORT_ANSWER_IN_TEXT,
    CACHE_FROZEN_LLM_FEATURES,
    FEATURE_CACHE_DIR,
    FEATURE_CACHE_BATCH_SIZE,
    VALIDATION_RATIO,
    CSV_PATH,
    DATASET_NAMES,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    HISTORY_DIR,
    AUTO_LOSS_WEIGHTING,
    AUTO_LOSS_REFERENCE,
    AUTO_LOSS_SCALE_BATCHES,
    AUTO_LOSS_SCALE_STATISTIC,
    LOSS_NORMALIZATION,
    LOSS_SCALE_EMA_DECAY,
    LOSS_SCALE_EPS,

    NEIGHBOUR_EMBEDDING_MODEL,
    TUNING_CONFIGS,

    PROJ_DIM,
    NORMALIZE_PROJECTED_STATES,
    USE_FEATURE_STANDARDIZATION,
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


def slugify(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return value.lower()


def config_value(config, key):
    if key not in config:
        raise KeyError(f"Missing tuning config key: {key}")
    return config[key]


def config_slug(config):
    return slugify(config_value(config, "name"))


def checkpoint_path(config, model_name, train_dataset):
    return os.path.join(
        CHECKPOINT_DIR,
        f"best_{config_slug(config)}_{slugify(model_name)}_train_{slugify(train_dataset)}.pt",
    )


def history_path(config, model_name, train_dataset):
    return os.path.join(
        HISTORY_DIR,
        f"history_{config_slug(config)}_{slugify(model_name)}_train_{slugify(train_dataset)}.csv",
    )


def ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)


def configure_determinism():
    set_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run projection-EBM tuning experiments."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model names or slugs to run.",
    )
    parser.add_argument(
        "--train-datasets",
        nargs="*",
        default=None,
        help="Optional train dataset names to run.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional tuning config names/slugs to run.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=TRAIN_STEPS,
        help="Maximum epochs per run.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Early stopping patience measured on source validation AUC.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=EARLY_STOPPING_MIN_DELTA,
        help="Minimum source validation AUC gain that resets patience.",
    )
    return parser.parse_args()


def matches_requested(value, requested_values):
    if requested_values is None:
        return True

    requested = {str(x) for x in requested_values}
    requested_slugs = {slugify(x) for x in requested_values}
    return value in requested or slugify(value) in requested_slugs


def select_models(requested):
    return [
        model_name
        for model_name in MODEL_NAMES
        if matches_requested(model_name, requested)
    ]


def select_datasets(requested):
    return [
        dataset_name
        for dataset_name in DATASET_NAMES
        if matches_requested(dataset_name, requested)
    ]


def select_configs(requested):
    return [
        config
        for config in TUNING_CONFIGS
        if matches_requested(config_value(config, "name"), requested)
    ]


def print_run_settings(models, train_datasets, configs, args):
    print("==============================================")
    print("Projection EBM tuning experiments")
    print("Contextual answer-token pooling")
    print("BCE + PairRank + InBatchRank + NeighbourRank")
    print("==============================================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"CSV path: {CSV_PATH}")
    print(f"MODEL_NAMES: {models}")
    print(f"DATASET_NAMES: {DATASET_NAMES}")
    print(f"TRAIN_DATASETS: {train_datasets}")
    print(f"TUNING_CONFIGS: {[c['name'] for c in configs]}")
    print(f"DEVICE: {DEVICE}")
    print(f"MAX_EPOCHS: {args.max_epochs}")
    print(f"EARLY_STOPPING_PATIENCE: {args.patience}")
    print(f"EARLY_STOPPING_MIN_DELTA: {args.min_delta}")
    print(f"EVAL_EVERY_EPOCH: {EVAL_EVERY_EPOCH}")
    print(f"AUTO_LOSS_WEIGHTING: {AUTO_LOSS_WEIGHTING}")
    print(f"AUTO_LOSS_REFERENCE: {AUTO_LOSS_REFERENCE}")
    print(f"AUTO_LOSS_SCALE_BATCHES: {AUTO_LOSS_SCALE_BATCHES}")
    print(f"AUTO_LOSS_SCALE_STATISTIC: {AUTO_LOSS_SCALE_STATISTIC}")
    print(f"LOSS_NORMALIZATION: {LOSS_NORMALIZATION}")
    print(f"LOSS_SCALE_EMA_DECAY: {LOSS_SCALE_EMA_DECAY}")
    print(f"LOSS_SCALE_EPS: {LOSS_SCALE_EPS}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"MAX_LENGTH: {MAX_LENGTH}")
    print(f"VALIDATION_RATIO: {VALIDATION_RATIO}")
    print(f"CACHE_FROZEN_LLM_FEATURES: {CACHE_FROZEN_LLM_FEATURES}")
    print(f"FEATURE_CACHE_DIR: {FEATURE_CACHE_DIR}")
    print(f"FEATURE_CACHE_BATCH_SIZE: {FEATURE_CACHE_BATCH_SIZE}")
    print(f"NEIGHBOUR_EMBEDDING_MODEL: {NEIGHBOUR_EMBEDDING_MODEL}")
    print(f"PROJ_DIM: {PROJ_DIM}")
    print(f"NORMALIZE_PROJECTED_STATES: {NORMALIZE_PROJECTED_STATES}")
    print(f"USE_FEATURE_STANDARDIZATION: {USE_FEATURE_STANDARDIZATION}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    print(f"HISTORY_DIR: {HISTORY_DIR}")
    print("==============================================")

    print("\nTuning configs")
    print("--------------")
    for config in configs:
        print(
            f"{config['name']}: "
            f"backend={config['neighbour_backend']}, "
            f"k={config['k_neighbours']}, "
            f"bce_gate={config['lambda_bce']}, "
            f"pair_gate={config['lambda_pair_rank']}, "
            f"inbatch_gate={config['lambda_inbatch_rank']}, "
            f"neighbour_gate={config['lambda_neighbour_rank']}, "
            f"rank_margin={config['rank_margin']}, "
            f"neighbour_margin={config['neighbour_margin']}, "
            f"dropout={config['dropout']}, "
            f"weight_decay={config['weight_decay']}"
        )


def print_split_counts(splits):
    print("\nConfigured dataset groups")
    print("-------------------------")

    for dataset_name in splits["dataset_names"]:
        dataset = splits["datasets"][dataset_name]
        print(
            f"{dataset_name}: "
            f"rows={len(dataset['rows'])}, "
            f"train_rows={len(dataset['train_rows'])}, "
            f"validation_rows={len(dataset['validation_rows'])}, "
            f"examples={len(dataset['examples'])}"
        )


def print_feature_info(energy_model):
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


def sanity_check_train_loader(loaders):
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


def add_config_columns(rows, config):
    config_columns = {
        "config_name": config_value(config, "name"),
        "neighbour_backend": config_value(config, "neighbour_backend"),
        "k_neighbours": config_value(config, "k_neighbours"),
        "lambda_bce": config_value(config, "lambda_bce"),
        "lambda_pair_rank": config_value(config, "lambda_pair_rank"),
        "lambda_inbatch_rank": config_value(config, "lambda_inbatch_rank"),
        "lambda_neighbour_rank": config_value(config, "lambda_neighbour_rank"),
        "rank_margin": config_value(config, "rank_margin"),
        "neighbour_margin": config_value(config, "neighbour_margin"),
        "auto_loss_weighting": AUTO_LOSS_WEIGHTING,
        "auto_loss_reference": AUTO_LOSS_REFERENCE,
        "auto_loss_scale_batches": AUTO_LOSS_SCALE_BATCHES,
        "auto_loss_scale_statistic": AUTO_LOSS_SCALE_STATISTIC,
        "loss_normalization": LOSS_NORMALIZATION,
        "loss_scale_ema_decay": LOSS_SCALE_EMA_DECAY,
        "loss_scale_eps": LOSS_SCALE_EPS,
        "dropout": config_value(config, "dropout"),
        "weight_decay": config_value(config, "weight_decay"),
    }

    return [
        {
            **config_columns,
            **row,
        }
        for row in rows
    ]


def save_history(history, config, model_name, train_dataset):
    try:
        import pandas as pd

        hist_path = history_path(config, model_name, train_dataset)
        pd.DataFrame(add_config_columns(history, config)).to_csv(
            hist_path,
            index=False,
        )
        print(f"\nSaved history to: {hist_path}")

    except Exception as e:
        print(f"\nCould not save history CSV: {e}")


def run_dataset_experiment(
    config,
    model_name,
    train_dataset,
    tokenizer,
    base_model,
    splits,
    args,
):
    ood_datasets = [
        dataset_name
        for dataset_name in DATASET_NAMES
        if dataset_name != train_dataset
    ]

    print("\n==============================================")
    print(f"Config: {config['name']}")
    print(f"Base model: {model_name}")
    print(f"Train dataset: {train_dataset}")
    print(f"OOD eval datasets: {ood_datasets}")
    print("==============================================")

    set_seed(SEED)

    print("\nBuilding projection EBM...")
    energy_model = build_energy_model(
        base_model=base_model,
        device=DEVICE,
        proj_dim=PROJ_DIM,
        dropout=config_value(config, "dropout"),
        normalize_projected_states=NORMALIZE_PROJECTED_STATES,
        use_feature_standardization=USE_FEATURE_STANDARDIZATION,
    )

    print_feature_info(energy_model)

    print("\nBuilding dataloaders with answer masks and neighbours...")
    feature_cache_path = None

    if CACHE_FROZEN_LLM_FEATURES:
        feature_cache_path = os.path.join(
            FEATURE_CACHE_DIR,
            (
                "raw_layer_reprs_"
                f"{slugify(model_name)}"
                f"_max{MAX_LENGTH}"
                f"_short{int(bool(USE_SHORT_ANSWER_IN_TEXT))}.pt"
            ),
        )
        print(f"Frozen feature cache path: {feature_cache_path}")

    loaders = build_dataloaders(
        splits=splits,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_datasets=ood_datasets,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        use_short_answer=USE_SHORT_ANSWER_IN_TEXT,
        num_workers=0,
        k_neighbours=config_value(config, "k_neighbours"),
        neighbour_backend=config_value(config, "neighbour_backend"),
        neighbour_embedding_model=NEIGHBOUR_EMBEDDING_MODEL,
        neighbour_llm_base_model=base_model,
        neighbour_llm_device=DEVICE,
        neighbour_llm_batch_size=BATCH_SIZE,
        cache_frozen_features=CACHE_FROZEN_LLM_FEATURES,
        feature_cache_base_model=base_model,
        feature_cache_energy_model=energy_model,
        feature_cache_device=DEVICE,
        feature_cache_path=feature_cache_path,
        feature_cache_batch_size=FEATURE_CACHE_BATCH_SIZE,
    )

    sanity_check_train_loader(loaders)
    print(f"Epoch eval datasets: {loaders['eval_datasets']}")
    print(f"Checkpoint monitor datasets: {loaders['monitor_datasets']}")

    optimizer = torch.optim.AdamW(
        energy_model.parameters(),
        lr=LR,
        weight_decay=config_value(config, "weight_decay"),
    )

    best_ckpt_path = checkpoint_path(config, model_name, train_dataset)

    print("\nStarting training...")
    history = train_model(
        loaders=loaders,
        base_model=base_model,
        energy_model=energy_model,
        optimizer=optimizer,
        device=DEVICE,
        train_steps=args.max_epochs,
        lambda_pair_rank=config_value(config, "lambda_pair_rank"),
        lambda_bce=config_value(config, "lambda_bce"),
        lambda_inbatch_rank=config_value(config, "lambda_inbatch_rank"),
        lambda_neighbour_rank=config_value(config, "lambda_neighbour_rank"),
        rank_margin=config_value(config, "rank_margin"),
        neighbour_margin=config_value(config, "neighbour_margin"),
        best_ckpt_path=best_ckpt_path,
        model_name=model_name,
        train_dataset=train_dataset,
        config_name=config_value(config, "name"),
        experiment_config=dict(config),
        eval_datasets=loaders["eval_datasets"],
        monitor_datasets=loaders["monitor_datasets"],
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        eval_every_epoch=EVAL_EVERY_EPOCH,
        loss_normalization=LOSS_NORMALIZATION,
        loss_scale_ema_decay=LOSS_SCALE_EMA_DECAY,
        loss_scale_eps=LOSS_SCALE_EPS,
        auto_loss_weighting=AUTO_LOSS_WEIGHTING,
        auto_loss_reference=AUTO_LOSS_REFERENCE,
        auto_loss_scale_batches=AUTO_LOSS_SCALE_BATCHES,
        auto_loss_scale_statistic=AUTO_LOSS_SCALE_STATISTIC,
    )

    save_history(history, config, model_name, train_dataset)

    print("\nLoading best checkpoint before final evaluation...")
    ckpt = torch.load(
        best_ckpt_path,
        map_location=DEVICE,
    )

    energy_model.load_state_dict(ckpt["model_state_dict"])
    energy_model.eval()

    best_epoch = ckpt.get("epoch", "unknown")
    best_monitor_auc = ckpt.get("monitor_auc", "unknown")
    best_mean_eval_auc = ckpt.get("mean_eval_auc", "unknown")

    print(f"Loaded best checkpoint from epoch: {best_epoch}")
    print(f"Best checkpoint monitor_auc: {best_monitor_auc}")
    print(f"Best checkpoint mean_eval_auc: {best_mean_eval_auc}")

    print("\nFinal evaluation using BEST checkpoint:")

    result_rows = []
    stopped_epoch = history[-1]["epoch"] if history else None
    early_stopped = bool(
        history
        and args.patience is not None
        and args.patience > 0
        and stopped_epoch is not None
        and stopped_epoch + 1 < args.max_epochs
    )

    common_result = {
        "config_name": config_value(config, "name"),
        "model_name": model_name,
        "train_dataset": train_dataset,
        "checkpoint_path": best_ckpt_path,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "early_stopped": early_stopped,
        "best_monitor_auc": best_monitor_auc,
        "best_mean_eval_auc": best_mean_eval_auc,
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.patience,
        "early_stopping_min_delta": args.min_delta,
    }
    common_result.update(add_config_columns([{}], config)[0])

    for eval_dataset in loaders["eval_datasets"]:
        metrics = evaluate_loader(
            loaders["eval"][eval_dataset],
            base_model,
            energy_model,
            DEVICE,
        )

        print_metrics(eval_dataset, metrics)

        result_rows.append(
            {
                **common_result,
                "eval_dataset": eval_dataset,
                **metrics,
            }
        )

    print(f"Best checkpoint path: {best_ckpt_path}")

    del energy_model
    del optimizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_rows


def save_summary(result_rows):
    if not result_rows:
        return

    try:
        import pandas as pd

        summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
        pd.DataFrame(result_rows).to_csv(summary_path, index=False)
        print(f"\nSaved experiment summary to: {summary_path}")

    except Exception as e:
        print(f"\nCould not save experiment summary CSV: {e}")


def main():
    args = parse_args()
    selected_models = select_models(args.models)
    selected_train_datasets = select_datasets(args.train_datasets)
    selected_configs = select_configs(args.configs)

    if not selected_models:
        raise ValueError("No matching models selected.")
    if not selected_train_datasets:
        raise ValueError("No matching train datasets selected.")
    if not selected_configs:
        raise ValueError("No matching tuning configs selected.")

    print_run_settings(
        models=selected_models,
        train_datasets=selected_train_datasets,
        configs=selected_configs,
        args=args,
    )
    configure_determinism()
    ensure_output_dirs()

    print("\nLoading rows from CSV...")
    rows = load_rows_from_csv(CSV_PATH)
    splits = split_examples_by_dataset(
        rows,
        dataset_names=DATASET_NAMES,
        validation_ratio=VALIDATION_RATIO,
        seed=SEED,
    )

    print_dataset_counts(
        rows=splits["rows"],
        examples=splits["examples"],
    )
    print_split_counts(splits)
    validate_required_splits(splits, DATASET_NAMES)

    all_result_rows = []

    for model_name in selected_models:
        print("\n==============================================")
        print(f"Loading frozen base LM: {model_name}")
        print("==============================================")

        tokenizer, base_model = load_frozen_lm(model_name, DEVICE)
        base_model.eval()

        try:
            for config in selected_configs:
                for train_dataset in selected_train_datasets:
                    result_rows = run_dataset_experiment(
                        config=config,
                        model_name=model_name,
                        train_dataset=train_dataset,
                        tokenizer=tokenizer,
                        base_model=base_model,
                        splits=splits,
                        args=args,
                    )
                    all_result_rows.extend(result_rows)
                    save_summary(all_result_rows)

        finally:
            del base_model
            del tokenizer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_summary(all_result_rows)
    print("Done.")


if __name__ == "__main__":
    main()

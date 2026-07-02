import numpy as np
import torch
import torch.nn.functional as F

from .model import forward_energy
from .evaluation import evaluate_loader


def get_hard_neighbour_anchors(
    flat_neigh_pos_logits,
    flat_neigh_neg_logits,
    k_list,
):
    """
    Build hard neighbour anchors for each item.

    For each training example i:

        hard_pos_i = max_j E(neighbour_positive_j)

    This is the hardest truthful neighbour, because it has the highest energy.

        hard_neg_i = min_j E(neighbour_negative_j)

    This is the hardest hallucinated neighbour, because it has the lowest energy.

    Returns:
        hard_pos_logits: [B]
        hard_neg_logits: [B]
        valid_mask:      [B] bool
    """

    hard_pos = []
    hard_neg = []
    valid = []

    start = 0

    for k in k_list.detach().cpu().tolist():
        k = int(k)
        end = start + k

        if k <= 0:
            hard_pos.append(
                torch.zeros(
                    (),
                    device=flat_neigh_pos_logits.device,
                    dtype=flat_neigh_pos_logits.dtype,
                )
            )

            hard_neg.append(
                torch.zeros(
                    (),
                    device=flat_neigh_neg_logits.device,
                    dtype=flat_neigh_neg_logits.dtype,
                )
            )

            valid.append(False)

        else:
            pos_chunk = flat_neigh_pos_logits[start:end]
            neg_chunk = flat_neigh_neg_logits[start:end]

            hard_pos.append(pos_chunk.max())
            hard_neg.append(neg_chunk.min())
            valid.append(True)

        start = end

    hard_pos_logits = torch.stack(hard_pos, dim=0)
    hard_neg_logits = torch.stack(hard_neg, dim=0)

    valid_mask = torch.tensor(
        valid,
        device=flat_neigh_pos_logits.device,
        dtype=torch.bool,
    )

    return hard_pos_logits, hard_neg_logits, valid_mask


# ============================================================
# Loss
# ============================================================

def compute_energy_losses(
    pos_logits,
    neg_logits,
    neigh_pos_logits=None,
    neigh_neg_logits=None,
    k_list=None,
    lambda_pair_rank=0.5,
    lambda_bce=1.0,
    lambda_inbatch_rank=0.2,
    lambda_neighbour_rank=0.1,
    rank_margin=1.0,
    neighbour_margin=1.0,
):
    """
    Projection EBM objective with hard cross-neighbour ranking.

    Convention:
        truthful / positive      -> label 0, lower energy
        hallucinated / negative  -> label 1, higher energy

    Loss:

        L_total =
            lambda_bce * L_bce
          + lambda_pair_rank * L_pair
          + lambda_inbatch_rank * L_inbatch
          + lambda_neighbour_rank * L_hard_neighbour

    Hard cross-neighbour ranking:

        L_hard_neighbour =
            0.5 * [
                max(0, margin + E(pos_i) - min_j E(neigh_neg_ij))
              + max(0, margin + max_j E(neigh_pos_ij) - E(neg_i))
            ]

    The 0.5 here is only averaging the two cross-neighbour terms.
    The real weight is lambda_neighbour_rank in the total loss.
    """

    # ---------------------------------------------------------
    # 1. BCE calibration
    # ---------------------------------------------------------
    pos_labels = torch.zeros_like(pos_logits)
    neg_labels = torch.ones_like(neg_logits)

    bce_pos = F.binary_cross_entropy_with_logits(
        pos_logits,
        pos_labels,
    )

    bce_neg = F.binary_cross_entropy_with_logits(
        neg_logits,
        neg_labels,
    )

    bce_loss = 0.5 * (bce_pos + bce_neg)

    # ---------------------------------------------------------
    # 2. Pair ranking
    # E_pos + margin < E_neg
    # ---------------------------------------------------------
    pair_rank_loss = F.relu(
        rank_margin + pos_logits - neg_logits
    ).mean()

    # ---------------------------------------------------------
    # 3. In-batch ranking
    # Every positive should be lower than every negative in the batch.
    # ---------------------------------------------------------
    pos_matrix = pos_logits.unsqueeze(1)  # [B, 1]
    neg_matrix = neg_logits.unsqueeze(0)  # [1, B]

    inbatch_rank_loss = F.relu(
        rank_margin + pos_matrix - neg_matrix
    ).mean()

    # ---------------------------------------------------------
    # 4. Hard cross-neighbour ranking
    # ---------------------------------------------------------
    use_neighbours = (
        neigh_pos_logits is not None
        and neigh_neg_logits is not None
        and k_list is not None
    )

    if not use_neighbours:
        neighbour_rank_loss = torch.zeros(
            (),
            device=pos_logits.device,
            dtype=pos_logits.dtype,
        )

    else:
        hard_neigh_pos_logits, hard_neigh_neg_logits, valid_mask = (
            get_hard_neighbour_anchors(
                flat_neigh_pos_logits=neigh_pos_logits,
                flat_neigh_neg_logits=neigh_neg_logits,
                k_list=k_list,
            )
        )

        if valid_mask.any():
            valid_pos_logits = pos_logits[valid_mask]
            valid_neg_logits = neg_logits[valid_mask]

            hard_neigh_pos_logits = hard_neigh_pos_logits[valid_mask]
            hard_neigh_neg_logits = hard_neigh_neg_logits[valid_mask]

            # Current truthful should be lower than hardest neighbour hallucination.
            pos_vs_neigh_neg_loss = F.relu(
                neighbour_margin
                + valid_pos_logits
                - hard_neigh_neg_logits
            ).mean()

            # Hardest neighbour truthful should be lower than current hallucination.
            neigh_pos_vs_neg_loss = F.relu(
                neighbour_margin
                + hard_neigh_pos_logits
                - valid_neg_logits
            ).mean()

            # Clean hard cross-neighbour loss.
            # No extra internal weighting here.
            # lambda_neighbour_rank is applied in total_loss below.
            neighbour_rank_loss = 0.5 * (
                pos_vs_neigh_neg_loss
                + neigh_pos_vs_neg_loss
            )

        else:
            neighbour_rank_loss = torch.zeros(
                (),
                device=pos_logits.device,
                dtype=pos_logits.dtype,
            )

    # ---------------------------------------------------------
    # 5. Total objective
    # ---------------------------------------------------------
    total_loss = (
        lambda_bce * bce_loss
        + lambda_pair_rank * pair_rank_loss
        + lambda_inbatch_rank * inbatch_rank_loss
        + lambda_neighbour_rank * neighbour_rank_loss
    )

    return {
        "total_loss": total_loss,
        "bce_loss": bce_loss,
        "pair_rank_loss": pair_rank_loss,
        "inbatch_rank_loss": inbatch_rank_loss,
        "neighbour_rank_loss": neighbour_rank_loss,
    }


# ============================================================
# One epoch
# ============================================================

def train_one_epoch(
    train_loader,
    base_model,
    energy_model,
    optimizer,
    device,
    grad_clip=1.0,
    lambda_pair_rank=0.5,
    lambda_bce=1.0,
    lambda_inbatch_rank=0.2,
    lambda_neighbour_rank=0.1,
    rank_margin=1.0,
    neighbour_margin=1.0,
):
    """
    One epoch of projection-EBM training.

    Uses:
        BCE
        PairRank
        InBatchRank
        HardCrossNeighbourRank
    """

    base_model.eval()
    energy_model.train()

    running = {
        "total_loss": 0.0,
        "bce_loss": 0.0,
        "pair_rank_loss": 0.0,
        "inbatch_rank_loss": 0.0,
        "neighbour_rank_loss": 0.0,
    }

    seen = 0

    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)

        # -----------------------------------------------------
        # Positive / truthful examples
        # -----------------------------------------------------
        pos_out = forward_energy(
            base_model,
            energy_model,
            batch["pos_input_ids"],
            batch["pos_attention_mask"],
            device,
            answer_mask=batch.get("pos_answer_mask", None),
        )

        # -----------------------------------------------------
        # Negative / hallucinated examples
        # -----------------------------------------------------
        neg_out = forward_energy(
            base_model,
            energy_model,
            batch["neg_input_ids"],
            batch["neg_attention_mask"],
            device,
            answer_mask=batch.get("neg_answer_mask", None),
        )

        pos_logits = pos_out["energy_logit"]
        neg_logits = neg_out["energy_logit"]

        neigh_pos_logits = None
        neigh_neg_logits = None
        k_list = None

        # -----------------------------------------------------
        # Neighbours
        # Only compute neighbour forward passes if they are actually used.
        # -----------------------------------------------------
        has_neighbours = batch.get("has_neighbours", False)

        if torch.is_tensor(has_neighbours):
            has_neighbours = bool(has_neighbours.any().item())
        else:
            has_neighbours = bool(has_neighbours)

        need_neighbours = (
            has_neighbours
            and lambda_neighbour_rank > 0.0
        )

        if need_neighbours:
            neigh_pos_out = forward_energy(
                base_model,
                energy_model,
                batch["neigh_pos_input_ids"],
                batch["neigh_pos_attention_mask"],
                device,
                answer_mask=batch.get("neigh_pos_answer_mask", None),
            )

            neigh_neg_out = forward_energy(
                base_model,
                energy_model,
                batch["neigh_neg_input_ids"],
                batch["neigh_neg_attention_mask"],
                device,
                answer_mask=batch.get("neigh_neg_answer_mask", None),
            )

            # IMPORTANT:
            # Keep neighbour logits flat.
            # Hard cross-neighbour ranking needs individual neighbour scores.
            neigh_pos_logits = neigh_pos_out["energy_logit"]
            neigh_neg_logits = neigh_neg_out["energy_logit"]
            k_list = batch["k_list"]

        # -----------------------------------------------------
        # Compute loss
        # -----------------------------------------------------
        loss_dict = compute_energy_losses(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            neigh_pos_logits=neigh_pos_logits,
            neigh_neg_logits=neigh_neg_logits,
            k_list=k_list,
            lambda_pair_rank=lambda_pair_rank,
            lambda_bce=lambda_bce,
            lambda_inbatch_rank=lambda_inbatch_rank,
            lambda_neighbour_rank=lambda_neighbour_rank,
            rank_margin=rank_margin,
            neighbour_margin=neighbour_margin,
        )

        loss = loss_dict["total_loss"]

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                energy_model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()

        bs = pos_logits.size(0)

        for key in running:
            running[key] += float(loss_dict[key].detach().cpu().item()) * bs

        seen += bs

    return {
        key: value / max(seen, 1)
        for key, value in running.items()
    }


# ============================================================
# Full training loop
# ============================================================

def train_model(
    loaders,
    base_model,
    energy_model,
    optimizer,
    device,
    train_steps=30,
    lambda_pair_rank=0.5,
    lambda_bce=1.0,
    lambda_inbatch_rank=0.2,
    lambda_neighbour_rank=0.1,
    rank_margin=1.0,
    neighbour_margin=1.0,
    best_ckpt_path="best_answer_pool_hard_cross_neighbour.pt",
    model_name=None,
    train_dataset=None,
    config_name=None,
    experiment_config=None,
    eval_datasets=None,
    monitor_datasets=None,
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
):
    """
    Full training loop.

    Saves best checkpoint based on:

        mean evaluation AUC across the provided evaluation datasets.
    """

    best_monitor_auc = -1.0
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    eval_loaders = loaders.get("eval", {})

    if eval_datasets is None:
        eval_datasets = list(eval_loaders.keys())

    if monitor_datasets is None:
        monitor_datasets = list(eval_datasets)

    for epoch in range(train_steps):
        train_losses = train_one_epoch(
            train_loader=loaders["train"],
            base_model=base_model,
            energy_model=energy_model,
            optimizer=optimizer,
            device=device,
            lambda_pair_rank=lambda_pair_rank,
            lambda_bce=lambda_bce,
            lambda_inbatch_rank=lambda_inbatch_rank,
            lambda_neighbour_rank=lambda_neighbour_rank,
            rank_margin=rank_margin,
            neighbour_margin=neighbour_margin,
        )

        eval_metrics = {}

        for dataset_name in eval_datasets:
            print(f"Evaluating {dataset_name}...", flush=True)
            eval_metrics[dataset_name] = evaluate_loader(
                eval_loaders[dataset_name],
                base_model,
                energy_model,
                device,
            )
            print(
                f"{dataset_name}: "
                f"loss={eval_metrics[dataset_name]['loss']:.4f}, "
                f"acc={eval_metrics[dataset_name]['accuracy']:.4f}, "
                f"auc={eval_metrics[dataset_name]['energy_auc']:.4f}, "
                f"logit_auc={eval_metrics[dataset_name]['logit_auc']:.4f}",
                flush=True,
            )

        eval_aucs = [
            metrics["energy_auc"]
            for metrics in eval_metrics.values()
        ]
        valid_eval_aucs = [
            auc
            for auc in eval_aucs
            if not np.isnan(auc)
        ]
        monitor_aucs = [
            eval_metrics[name]["energy_auc"]
            for name in monitor_datasets
            if name in eval_metrics
            and not np.isnan(eval_metrics[name]["energy_auc"])
        ]

        mean_eval_auc = (
            float(np.mean(valid_eval_aucs))
            if valid_eval_aucs
            else float("nan")
        )
        monitor_auc = (
            float(np.mean(monitor_aucs))
            if monitor_aucs
            else float("nan")
        )
        comparable_auc = monitor_auc

        if np.isnan(comparable_auc):
            comparable_auc = -float("inf")

        saved_best = False
        improved = (
            best_epoch is None
            or comparable_auc > best_monitor_auc + early_stopping_min_delta
        )

        if improved:
            best_monitor_auc = comparable_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            saved_best = True

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": energy_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),

                    "loss_type": "answer_pool_bce_pair_inbatch_hard_cross_neighbour",
                    "lambda_pair_rank": lambda_pair_rank,
                    "lambda_bce": lambda_bce,
                    "lambda_inbatch_rank": lambda_inbatch_rank,
                    "lambda_neighbour_rank": lambda_neighbour_rank,
                    "rank_margin": rank_margin,
                    "neighbour_margin": neighbour_margin,

                    "model_name": model_name,
                    "train_dataset": train_dataset,
                    "config_name": config_name,
                    "experiment_config": experiment_config,
                    "eval_datasets": list(eval_datasets),
                    "monitor_datasets": list(monitor_datasets),
                    "eval_metrics": eval_metrics,

                    "mean_eval_auc": mean_eval_auc,
                    "monitor_auc": monitor_auc,
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_min_delta": early_stopping_min_delta,
                },
                best_ckpt_path,
            )
            print(f"Saved best checkpoint to {best_ckpt_path}", flush=True)

        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "model_name": model_name,
            "train_dataset": train_dataset,
            "config_name": config_name,
            **train_losses,

            "mean_eval_auc": mean_eval_auc,
            "monitor_auc": monitor_auc,
            "best_monitor_auc": best_monitor_auc,
            "epochs_without_improvement": epochs_without_improvement,
            "saved_best": saved_best,
        }

        for dataset_name, metrics in eval_metrics.items():
            row[f"{dataset_name}_loss"] = metrics["loss"]
            row[f"{dataset_name}_acc"] = metrics["accuracy"]
            row[f"{dataset_name}_auc"] = metrics["energy_auc"]
            row[f"{dataset_name}_logit_auc"] = metrics["logit_auc"]

        history.append(row)

        eval_text = " | ".join(
            f"{dataset_name}AUC={row[f'{dataset_name}_auc']:.4f}"
            for dataset_name in eval_metrics
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Total={row['total_loss']:.4f} | "
            f"BCE={row['bce_loss']:.4f} | "
            f"PairRank={row['pair_rank_loss']:.4f} | "
            f"InBatchRank={row['inbatch_rank_loss']:.4f} | "
            f"HardCrossNeighRank={row['neighbour_rank_loss']:.4f} | "
            f"{eval_text} | "
            f"MeanEvalAUC={row['mean_eval_auc']:.4f} | "
            f"MonitorAUC={row['monitor_auc']:.4f}"
            f"{' | saved best' if saved_best else ''}",
            flush=True,
        )

        should_stop = (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        )

        if should_stop:
            print(
                "Early stopping: "
                f"no monitor AUC improvement > {early_stopping_min_delta} "
                f"for {early_stopping_patience} epochs. "
                f"Best epoch={best_epoch}, best monitor AUC={best_monitor_auc:.4f}.",
                flush=True,
            )
            break

    return history

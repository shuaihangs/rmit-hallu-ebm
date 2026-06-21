import numpy as np
import torch
import torch.nn.functional as F

from .model import forward_energy
from .evaluation import evaluate_loader


# ============================================================
# Neighbour helpers
# ============================================================

def reshape_neighbour_logits(flat_logits, k_list):
    """
    Convert flat neighbour logits [sum(k_i)] into mean neighbour energy per item [B].

    This is kept for optional debugging / cluster variants, but hard cross-neighbour
    ranking mainly uses individual neighbour logits.
    """

    means = []
    start = 0

    for k in k_list.detach().cpu().tolist():
        k = int(k)
        end = start + k

        if k <= 0:
            means.append(
                torch.zeros(
                    (),
                    device=flat_logits.device,
                    dtype=flat_logits.dtype,
                )
            )
        else:
            means.append(flat_logits[start:end].mean())

        start = end

    return torch.stack(means, dim=0)


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
    lambda_cluster=0.0,
    rank_margin=1.0,
    neighbour_margin=1.0,
    detach_neighbour_anchors=True,
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
          + lambda_cluster * L_cluster

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

        cluster_loss = torch.zeros(
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

            # Optional cluster compactness.
            # Usually keep lambda_cluster = 0.0 first.
            if detach_neighbour_anchors:
                pos_anchor = hard_neigh_pos_logits.detach()
                neg_anchor = hard_neigh_neg_logits.detach()
            else:
                pos_anchor = hard_neigh_pos_logits
                neg_anchor = hard_neigh_neg_logits

            cluster_loss = (
                F.smooth_l1_loss(valid_pos_logits, pos_anchor)
                +
                F.smooth_l1_loss(valid_neg_logits, neg_anchor)
            )

        else:
            neighbour_rank_loss = torch.zeros(
                (),
                device=pos_logits.device,
                dtype=pos_logits.dtype,
            )

            cluster_loss = torch.zeros(
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
        + lambda_cluster * cluster_loss
    )

    return {
        "total_loss": total_loss,
        "bce_loss": bce_loss,
        "pair_rank_loss": pair_rank_loss,
        "inbatch_rank_loss": inbatch_rank_loss,
        "neighbour_rank_loss": neighbour_rank_loss,
        "cluster_loss": cluster_loss,
    }


# Backward-compatible name.
def compute_rank_neighbour_cluster_loss(
    pos_logits,
    neg_logits,
    neigh_pos_logits=None,
    neigh_neg_logits=None,
    k_list=None,
    lambda_pair_rank=0.5,
    lambda_bce=1.0,
    lambda_inbatch_rank=0.2,
    lambda_neighbour_rank=0.1,
    lambda_cluster=0.0,
    rank_margin=1.0,
    neighbour_margin=1.0,
    detach_neighbour_anchors=True,
):
    return compute_energy_losses(
        pos_logits=pos_logits,
        neg_logits=neg_logits,
        neigh_pos_logits=neigh_pos_logits,
        neigh_neg_logits=neigh_neg_logits,
        k_list=k_list,
        lambda_pair_rank=lambda_pair_rank,
        lambda_bce=lambda_bce,
        lambda_inbatch_rank=lambda_inbatch_rank,
        lambda_neighbour_rank=lambda_neighbour_rank,
        lambda_cluster=lambda_cluster,
        rank_margin=rank_margin,
        neighbour_margin=neighbour_margin,
        detach_neighbour_anchors=detach_neighbour_anchors,
    )


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
    lambda_cluster=0.0,
    rank_margin=1.0,
    neighbour_margin=1.0,
    detach_neighbour_anchors=True,
):
    """
    One epoch of projection-EBM training.

    Uses:
        BCE
        PairRank
        InBatchRank
        HardCrossNeighbourRank
        optional Cluster
    """

    base_model.eval()
    energy_model.train()

    running = {
        "total_loss": 0.0,
        "bce_loss": 0.0,
        "pair_rank_loss": 0.0,
        "inbatch_rank_loss": 0.0,
        "neighbour_rank_loss": 0.0,
        "cluster_loss": 0.0,
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
            and (
                lambda_neighbour_rank > 0.0
                or lambda_cluster > 0.0
            )
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
            lambda_cluster=lambda_cluster,
            rank_margin=rank_margin,
            neighbour_margin=neighbour_margin,
            detach_neighbour_anchors=detach_neighbour_anchors,
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
    lambda_cluster=0.0,
    rank_margin=1.0,
    neighbour_margin=1.0,
    detach_neighbour_anchors=True,
    best_ckpt_path="best_answer_pool_hard_cross_neighbour.pt",
):
    """
    Full training loop.

    Saves best checkpoint based on:

        mean(TriviaQA AUC, TruthfulQA AUC)
    """

    best_mean_eval_auc = -1.0
    history = []

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
            lambda_cluster=lambda_cluster,
            rank_margin=rank_margin,
            neighbour_margin=neighbour_margin,
            detach_neighbour_anchors=detach_neighbour_anchors,
        )

        hotpot_metrics = evaluate_loader(
            loaders["hotpot_eval"],
            base_model,
            energy_model,
            device,
        )

        trivia_metrics = evaluate_loader(
            loaders["trivia"],
            base_model,
            energy_model,
            device,
        )

        truthfulqa_metrics = evaluate_loader(
            loaders["truthfulqa"],
            base_model,
            energy_model,
            device,
        )

        mean_eval_auc = np.nanmean(
            [
                trivia_metrics["energy_auc"],
                truthfulqa_metrics["energy_auc"],
            ]
        )

        saved_best = False

        if mean_eval_auc > best_mean_eval_auc:
            best_mean_eval_auc = mean_eval_auc
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
                    "lambda_cluster": lambda_cluster,
                    "rank_margin": rank_margin,
                    "neighbour_margin": neighbour_margin,
                    "detach_neighbour_anchors": detach_neighbour_anchors,

                    "train_energy_auc": hotpot_metrics["energy_auc"],
                    "train_accuracy": hotpot_metrics["accuracy"],

                    "trivia_energy_auc": trivia_metrics["energy_auc"],
                    "trivia_accuracy": trivia_metrics["accuracy"],

                    "truthfulqa_energy_auc": truthfulqa_metrics["energy_auc"],
                    "truthfulqa_accuracy": truthfulqa_metrics["accuracy"],

                    "mean_eval_auc": mean_eval_auc,
                },
                best_ckpt_path,
            )

        row = {
            "epoch": epoch,
            **train_losses,

            "hotpot_acc": hotpot_metrics["accuracy"],
            "hotpot_auc": hotpot_metrics["energy_auc"],

            "trivia_acc": trivia_metrics["accuracy"],
            "trivia_auc": trivia_metrics["energy_auc"],

            "truthfulqa_acc": truthfulqa_metrics["accuracy"],
            "truthfulqa_auc": truthfulqa_metrics["energy_auc"],

            "mean_eval_auc": mean_eval_auc,
            "saved_best": saved_best,
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"Total={row['total_loss']:.4f} | "
            f"BCE={row['bce_loss']:.4f} | "
            f"PairRank={row['pair_rank_loss']:.4f} | "
            f"InBatchRank={row['inbatch_rank_loss']:.4f} | "
            f"HardCrossNeighRank={row['neighbour_rank_loss']:.4f} | "
            f"Cluster={row['cluster_loss']:.4f} | "
            f"HotpotAUC={row['hotpot_auc']:.4f} | "
            f"TriviaAUC={row['trivia_auc']:.4f} | "
            f"TruthfulQAAUC={row['truthfulqa_auc']:.4f} | "
            f"MeanEvalAUC={row['mean_eval_auc']:.4f}"
            f"{' | saved best' if saved_best else ''}"
        )

    return history
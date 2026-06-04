import numpy as np
import torch
import torch.nn.functional as F

from .model import forward_energy
from .evaluation import evaluate_loader


def reshape_neighbour_logits(flat_logits, k_list):
    """
    Convert flat neighbour logits [sum(k_i)] into mean neighbour energy per item [batch].

    Example:
      flat_logits = energies for all neighbours in the batch flattened together
      k_list      = number of neighbours for each original item

    Returns:
      mean_logits = [batch]
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


def compute_rank_neighbour_cluster_loss(
    pos_logits,
    neg_logits,
    neigh_pos_logits=None,
    neigh_neg_logits=None,
    lambda_neighbour_rank=0.5,
    lambda_cluster=0.5,
    rank_margin=1,
    neighbour_margin=0.5,
    detach_neighbour_anchors=True,
):
    """
    Clean proposed objective:

        L = L_rank
            + lambda * L_neighbour_rank
            + beta * L_cluster

    Convention:
        positive/truthful answer      -> lower energy
        negative/hallucinated answer  -> higher energy

    Pair ranking loss:
        L_rank = max(0, gamma + E_pos - E_neg)

    Neighbour ranking loss:
        L_neighbour_rank = max(0, gamma_n + mean(E_neigh_pos) - mean(E_neigh_neg))

    Cluster compactness loss:
        L_cluster = (E_pos - mean(E_neigh_pos))^2
                  + (E_neg - mean(E_neigh_neg))^2
    """

    pair_rank_loss = F.relu(
        rank_margin + pos_logits - neg_logits
    ).mean()

    if neigh_pos_logits is None or neigh_neg_logits is None:
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
        neighbour_rank_loss = F.relu(
            neighbour_margin + neigh_pos_logits - neigh_neg_logits
        ).mean()

        if detach_neighbour_anchors:
            pos_anchor = neigh_pos_logits.detach()
            neg_anchor = neigh_neg_logits.detach()
        else:
            pos_anchor = neigh_pos_logits
            neg_anchor = neigh_neg_logits

        cluster_loss = (
                F.smooth_l1_loss(pos_logits, pos_anchor)
                +
                F.smooth_l1_loss(neg_logits, neg_anchor)
            )

    total_loss = (
        pair_rank_loss
        + lambda_neighbour_rank * neighbour_rank_loss
        + lambda_cluster * cluster_loss
    )

    return {
        "total_loss": total_loss,
        "pair_rank_loss": pair_rank_loss,
        "neighbour_rank_loss": neighbour_rank_loss,
        "cluster_loss": cluster_loss,
    }


def train_one_epoch(
    train_loader,
    base_model,
    energy_model,
    optimizer,
    device,
    grad_clip=1.0,
    lambda_neighbour_rank=0.5,
    lambda_cluster=0.1,
    rank_margin=0.5,
    neighbour_margin=0.5,
    detach_neighbour_anchors=True,
):
    """
    One epoch of rank + neighbour-rank + cluster training.

    This version removes:
      - BCE loss
      - energy-bound loss
      - alpha, m_in, m_out

    It only uses:

        L = L_rank + lambda L_neighbour_rank + beta L_cluster
    """
    energy_model.train()

    running = {
        "total_loss": 0.0,
        "pair_rank_loss": 0.0,
        "neighbour_rank_loss": 0.0,
        "cluster_loss": 0.0,
    }

    seen = 0

    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)

        pos_out = forward_energy(
            base_model,
            energy_model,
            batch["pos_input_ids"],
            batch["pos_attention_mask"],
            device,
        )

        neg_out = forward_energy(
            base_model,
            energy_model,
            batch["neg_input_ids"],
            batch["neg_attention_mask"],
            device,
        )

        pos_logits = pos_out["energy_logit"]
        neg_logits = neg_out["energy_logit"]

        neigh_pos_logits = None
        neigh_neg_logits = None

        if batch.get("has_neighbours", False):
            neigh_pos_out = forward_energy(
                base_model,
                energy_model,
                batch["neigh_pos_input_ids"],
                batch["neigh_pos_attention_mask"],
                device,
            )

            neigh_neg_out = forward_energy(
                base_model,
                energy_model,
                batch["neigh_neg_input_ids"],
                batch["neigh_neg_attention_mask"],
                device,
            )

            neigh_pos_logits = reshape_neighbour_logits(
                neigh_pos_out["energy_logit"],
                batch["k_list"],
            )

            neigh_neg_logits = reshape_neighbour_logits(
                neigh_neg_out["energy_logit"],
                batch["k_list"],
            )

        loss_dict = compute_rank_neighbour_cluster_loss(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            neigh_pos_logits=neigh_pos_logits,
            neigh_neg_logits=neigh_neg_logits,
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


def train_model(
    loaders,
    base_model,
    energy_model,
    optimizer,
    device,
    train_steps=30,
    lambda_neighbour_rank=0.5,
    lambda_cluster=0.1,
    rank_margin=0.5,
    neighbour_margin=0.5,
    detach_neighbour_anchors=True,
    best_ckpt_path="best_hotpot_rank_neighbour_cluster.pt",
):
    """
    Full training loop for the clean proposed loss.
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

        mean_eval_auc = np.nanmean([
            trivia_metrics["energy_auc"],
            truthfulqa_metrics["energy_auc"],
        ])

        saved_best = False

        if mean_eval_auc > best_mean_eval_auc:
            best_mean_eval_auc = mean_eval_auc
            saved_best = True

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": energy_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),

                    "loss_type": "rank_neighbour_cluster",
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
            f"PairRank={row['pair_rank_loss']:.4f} | "
            f"NeighRank={row['neighbour_rank_loss']:.4f} | "
            f"Cluster={row['cluster_loss']:.4f} | "
            f"HotpotAUC={row['hotpot_auc']:.4f} | "
            f"TriviaAUC={row['trivia_auc']:.4f} | "
            f"TruthfulQAAUC={row['truthfulqa_auc']:.4f} | "
            f"MeanEvalAUC={row['mean_eval_auc']:.4f}"
            f"{' | saved best' if saved_best else ''}"
        )

    return history

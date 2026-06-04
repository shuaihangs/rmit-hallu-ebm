import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from .utils import safe_auc, safe_mean
from .model import forward_energy


@torch.no_grad()
def evaluate_loader(loader, base_model, energy_model, device):
    energy_model.eval()
    all_labels, all_logits, all_probs = [], [], []

    # Removed layer disagreement.
    # all_layer_mean, all_layer_std = [], []

    all_update_mean, all_update_std, all_update_max = [], [], []
    all_update_last, all_update_range, all_update_trend = [], [], []
    all_early_late_update_gap = []
    total_loss, total_count = 0.0, 0

    for batch in loader:
        labels = batch["labels"].to(device)
        out = forward_energy(base_model, energy_model, batch["input_ids"], batch["attention_mask"], device)
        logits = out["energy_logit"]
        probs = out["hallucination_prob"]
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_count += bs

        all_labels.append(labels.cpu())
        all_logits.append(logits.cpu())
        all_probs.append(probs.cpu())

        # Removed layer disagreement.
        # all_layer_mean.append(out["layer_disagreement"].cpu())
        # all_layer_std.append(out["layer_disagreement_std"].cpu())

        all_update_mean.append(out["update_mean"].cpu())
        all_update_std.append(out["update_std"].cpu())
        all_update_max.append(out["update_max"].cpu())
        all_update_last.append(out["update_last"].cpu())
        all_update_range.append(out["update_range"].cpu())
        all_update_trend.append(out["update_trend"].cpu())
        all_early_late_update_gap.append(out["early_late_update_gap"].cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_logits = torch.cat(all_logits).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # Removed layer disagreement.
    # all_layer_mean = torch.cat(all_layer_mean).numpy()
    # all_layer_std = torch.cat(all_layer_std).numpy()

    all_update_mean = torch.cat(all_update_mean).numpy()
    all_update_std = torch.cat(all_update_std).numpy()
    all_update_max = torch.cat(all_update_max).numpy()
    all_update_last = torch.cat(all_update_last).numpy()
    all_update_range = torch.cat(all_update_range).numpy()
    all_update_trend = torch.cat(all_update_trend).numpy()
    all_early_late_update_gap = torch.cat(all_early_late_update_gap).numpy()

    pred_labels = (all_probs >= 0.5).astype(int)
    pos_mask = all_labels == 0
    neg_mask = all_labels == 1

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": accuracy_score(all_labels, pred_labels),
        "energy_auc": safe_auc(all_labels, all_probs),
        "logit_auc": safe_auc(all_labels, all_logits),

        # Removed layer disagreement.
        # "layer_mean_auc": safe_auc(all_labels, all_layer_mean),
        # "layer_std_auc": safe_auc(all_labels, all_layer_std),

        "update_mean_auc": safe_auc(all_labels, all_update_mean),
        "update_std_auc": safe_auc(all_labels, all_update_std),
        "update_max_auc": safe_auc(all_labels, all_update_max),
        "update_last_auc": safe_auc(all_labels, all_update_last),
        "update_range_auc": safe_auc(all_labels, all_update_range),
        "update_trend_auc": safe_auc(all_labels, all_update_trend),
        "early_late_update_gap_auc": safe_auc(all_labels, all_early_late_update_gap),

        "mean_prob_positive": safe_mean(all_probs[pos_mask]),
        "mean_prob_negative": safe_mean(all_probs[neg_mask]),
        "mean_logit_positive": safe_mean(all_logits[pos_mask]),
        "mean_logit_negative": safe_mean(all_logits[neg_mask]),

        # Removed layer disagreement.
        # "mean_layer_positive": safe_mean(all_layer_mean[pos_mask]),
        # "mean_layer_negative": safe_mean(all_layer_mean[neg_mask]),
        # "mean_layer_std_positive": safe_mean(all_layer_std[pos_mask]),
        # "mean_layer_std_negative": safe_mean(all_layer_std[neg_mask]),

        "mean_update_mean_positive": safe_mean(all_update_mean[pos_mask]),
        "mean_update_mean_negative": safe_mean(all_update_mean[neg_mask]),
        "mean_update_std_positive": safe_mean(all_update_std[pos_mask]),
        "mean_update_std_negative": safe_mean(all_update_std[neg_mask]),
        "mean_update_max_positive": safe_mean(all_update_max[pos_mask]),
        "mean_update_max_negative": safe_mean(all_update_max[neg_mask]),
        "mean_update_last_positive": safe_mean(all_update_last[pos_mask]),
        "mean_update_last_negative": safe_mean(all_update_last[neg_mask]),
        "mean_update_range_positive": safe_mean(all_update_range[pos_mask]),
        "mean_update_range_negative": safe_mean(all_update_range[neg_mask]),
        "mean_update_trend_positive": safe_mean(all_update_trend[pos_mask]),
        "mean_update_trend_negative": safe_mean(all_update_trend[neg_mask]),
        "mean_early_late_update_gap_positive": safe_mean(all_early_late_update_gap[pos_mask]),
        "mean_early_late_update_gap_negative": safe_mean(all_early_late_update_gap[neg_mask]),
    }


def print_metrics(name, metrics):
    print(f"\n=== {name} ===")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy at threshold 0.5: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC using P(hallucinated): {metrics['energy_auc']:.4f}")
    print(f"ROC-AUC using raw energy logit: {metrics['logit_auc']:.4f}")

    # Removed layer disagreement.
    # print(f"ROC-AUC using layer disagreement mean: {metrics['layer_mean_auc']:.4f}")
    # print(f"ROC-AUC using layer disagreement std: {metrics['layer_std_auc']:.4f}")

    print(f"ROC-AUC using update mean: {metrics['update_mean_auc']:.4f}")
    print(f"ROC-AUC using update std: {metrics['update_std_auc']:.4f}")
    print(f"ROC-AUC using update max: {metrics['update_max_auc']:.4f}")
    print(f"ROC-AUC using update last: {metrics['update_last_auc']:.4f}")
    print(f"ROC-AUC using update range: {metrics['update_range_auc']:.4f}")
    print(f"ROC-AUC using update trend: {metrics['update_trend_auc']:.4f}")
    print(f"ROC-AUC using early-late update gap: {metrics['early_late_update_gap_auc']:.4f}")
    print(f"Mean P(hallucinated) positive: {metrics['mean_prob_positive']:.4f}")
    print(f"Mean P(hallucinated) negative: {metrics['mean_prob_negative']:.4f}")
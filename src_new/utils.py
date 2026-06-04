import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_text(x):
    if x is None:
        return ""
    return str(x).strip()


def normalize_dataset_name(x):
    x = clean_text(x).lower()
    x = x.replace("-", "_")
    x = x.replace(" ", "_")
    return x


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    summed = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-8)
    return summed / denom


def l2_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.sqrt(torch.sum(x ** 2, dim=dim) + 1e-8)


def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return float("nan")


def safe_mean(x):
    if len(x) == 0:
        return float("nan")
    return float(np.mean(x))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import masked_mean, l2_norm


class LearnedClaimEnergy(nn.Module):
    def __init__(self, hidden_size: int, proj_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )
        self.num_features = 7

        self.energy_head = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def get_raw_layer_reprs(self, hidden_states, attention_mask):
        """
        Convert token-level hidden states into one vector per layer.

        hidden_states: tuple of [B, T, H]
        attention_mask: [B, T]

        Returns:
            raw_layer_reprs: [B, num_layers, H]

        We skip hidden_states[0], which is the embedding output.
        """
        layer_reprs = [
            masked_mean(h, attention_mask)
            for h in hidden_states[1:]
        ]
        return torch.stack(layer_reprs, dim=1)

    def project_layer_reprs(self, raw_layer_reprs):
        """
        raw_layer_reprs: [B, L, H]
        returns: [B, L, proj_dim]
        """
        bsz, num_layers, hidden_size = raw_layer_reprs.shape
        proj_dtype = next(self.proj.parameters()).dtype

        flat = raw_layer_reprs.reshape(bsz * num_layers, hidden_size)
        flat = flat.to(dtype=proj_dtype)

        projected = self.proj(flat)
        projected = projected.reshape(bsz, num_layers, -1)

        return projected

    # def layer_disagreement_features(self, layer_reprs):
    #     final_repr = layer_reprs[:, -1:, :]
    #     diffs = layer_reprs[:, :-1, :] - final_repr
    #     dists = l2_norm(diffs, dim=-1)
    #     return dists.mean(dim=1), dists.std(dim=1, unbiased=False)

    def layer_update_features(self, layer_reprs):
        updates = layer_reprs[:, 1:, :] - layer_reprs[:, :-1, :]
        update_norms = l2_norm(updates, dim=-1)

        update_mean = update_norms.mean(dim=1)
        update_std = update_norms.std(dim=1, unbiased=False)
        update_max = update_norms.max(dim=1).values
        update_min = update_norms.min(dim=1).values
        update_range = update_max - update_min
        update_first = update_norms[:, 0]
        update_last = update_norms[:, -1]
        update_trend = update_last - update_first

        num_updates = update_norms.size(1)
        split = max(num_updates // 3, 1)

        early_update_mean = update_norms[:, :split].mean(dim=1)
        late_update_mean = update_norms[:, -split:].mean(dim=1)
        early_late_update_gap = late_update_mean - early_update_mean

        return {
            "update_mean": update_mean,
            "update_std": update_std,
            "update_max": update_max,
            "update_last": update_last,
            "update_range": update_range,
            "update_trend": update_trend,
            "early_late_update_gap": early_late_update_gap,
        }

    def energy_from_raw_layer_reprs(self, raw_layer_reprs):
        """
        Fast cached path.

        raw_layer_reprs: [B, L, H]
        """
        layer_reprs = self.project_layer_reprs(raw_layer_reprs)
        update_feats = self.layer_update_features(layer_reprs)

        features = torch.stack([
            update_feats["update_mean"],
            update_feats["update_std"],
            update_feats["update_max"],
            update_feats["update_last"],
            update_feats["update_range"],
            update_feats["update_trend"],
            update_feats["early_late_update_gap"],
        ], dim=1)

        energy_logit = self.energy_head(features).squeeze(-1)
        hallucination_prob = torch.sigmoid(energy_logit)

        return {
            "update_mean": update_feats["update_mean"],
            "update_std": update_feats["update_std"],
            "update_max": update_feats["update_max"],
            "update_last": update_feats["update_last"],
            "update_range": update_feats["update_range"],
            "update_trend": update_feats["update_trend"],
            "early_late_update_gap": update_feats["early_late_update_gap"],
            "features": features,
            "energy_logit": energy_logit,
            "hallucination_prob": hallucination_prob,
        }

    def forward(self, hidden_states, attention_mask):
        raw_layer_reprs = self.get_raw_layer_reprs(hidden_states, attention_mask)
        return self.energy_from_raw_layer_reprs(raw_layer_reprs)


def load_frozen_lm(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)

    base_model.eval()

    for p in base_model.parameters():
        p.requires_grad = False

    return tokenizer, base_model


def get_hidden_size(base_model):
    hidden_size = getattr(base_model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = base_model.config.n_embd
    return hidden_size


def build_energy_model(base_model, device, proj_dim=128, dropout=0.1):
    hidden_size = get_hidden_size(base_model)
    return LearnedClaimEnergy(
        hidden_size=hidden_size,
        proj_dim=proj_dim,
        dropout=dropout,
    ).to(device)


def build_energy_model_from_hidden_size(hidden_size, device, proj_dim=128, dropout=0.1):
    return LearnedClaimEnergy(
        hidden_size=hidden_size,
        proj_dim=proj_dim,
        dropout=dropout,
    ).to(device)


def get_hidden_states(base_model, input_ids, attention_mask, device):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = tuple(h.detach() for h in outputs.hidden_states)
    return attention_mask, hidden_states


def forward_energy(base_model, energy_model, input_ids, attention_mask, device):
    attention_mask, hidden_states = get_hidden_states(
        base_model,
        input_ids,
        attention_mask,
        device,
    )
    return energy_model(hidden_states, attention_mask)


@torch.no_grad()
def cached_raw_layer_reprs(base_model, energy_model, input_ids, attention_mask, device):
    """
    Used by precompute script.

    Returns:
        raw_layer_reprs: [B, L, H], moved to CPU float16
    """
    attention_mask, hidden_states = get_hidden_states(
        base_model,
        input_ids,
        attention_mask,
        device,
    )

    raw_layer_reprs = energy_model.get_raw_layer_reprs(
        hidden_states,
        attention_mask,
    )

    return raw_layer_reprs.detach().cpu().to(torch.float16)
import torch
from .model import forward_energy


def format_claim_text(question, claim, short_answer=None, use_short_answer=False):
    if use_short_answer and short_answer:
        return f"Question: {question}\nKnown answer: {short_answer}\nClaim: {claim}"
    return f"Question: {question}\nClaim: {claim}"


@torch.no_grad()
def score_claim(question, claim, tokenizer, base_model, energy_model, device, max_length=128, short_answer=None, use_short_answer=False):
    energy_model.eval()
    text = format_claim_text(question, claim, short_answer, use_short_answer)
    enc = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    out = forward_energy(base_model, energy_model, enc["input_ids"], enc["attention_mask"], device)
    return {
        "text": text,
        "energy_logit": float(out["energy_logit"].detach().cpu().item()),
        "hallucination_prob": float(out["hallucination_prob"].detach().cpu().item()),
        #"layer_disagreement": float(out["layer_disagreement"].detach().cpu().item()),
        "update_mean": float(out["update_mean"].detach().cpu().item()),
        "update_std": float(out["update_std"].detach().cpu().item()),
    }

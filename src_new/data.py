import csv
import random
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------

def load_rows_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            dataset = str(row.get("dataset", "")).strip()
            question = str(row.get("question", "")).strip()
            short_answer = str(row.get("short_answer", "")).strip()
            positive = str(row.get("positive", "")).strip()
            negative = str(row.get("negative", "")).strip()

            if not dataset or not question or not positive or not negative:
                continue

            rows.append(
                {
                    "dataset": dataset,
                    "question": question,
                    "short_answer": short_answer,
                    "positive": positive,
                    "negative": negative,
                }
            )

    return rows


def _is_hotpot(dataset_name: str) -> bool:
    return "hotpot" in dataset_name.lower()


def _is_trivia(dataset_name: str) -> bool:
    return "trivia" in dataset_name.lower()


def _is_truthfulqa(dataset_name: str) -> bool:
    return "truthful" in dataset_name.lower()


def rows_to_claim_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples = []

    for row in rows:
        base = {
            "dataset": row["dataset"],
            "question": row["question"],
            "short_answer": row.get("short_answer", ""),
        }

        examples.append(
            {
                **base,
                "claim": row["positive"],
                "label": 0.0,
                "label_name": "truthful",
            }
        )

        examples.append(
            {
                **base,
                "claim": row["negative"],
                "label": 1.0,
                "label_name": "hallucinated",
            }
        )

    return examples


def split_examples_by_dataset(
    rows: List[Dict[str, Any]],
    hotpot_eval_ratio: float = 0.2,
    seed: int = 42,
):
    hotpot_rows = [r for r in rows if _is_hotpot(r["dataset"])]
    trivia_rows = [r for r in rows if _is_trivia(r["dataset"])]
    truthfulqa_rows = [r for r in rows if _is_truthfulqa(r["dataset"])]

    rng = random.Random(seed)
    hotpot_rows = list(hotpot_rows)
    rng.shuffle(hotpot_rows)

    n_eval = max(1, int(len(hotpot_rows) * hotpot_eval_ratio))
    hotpot_eval_rows = hotpot_rows[:n_eval]
    hotpot_train_rows = hotpot_rows[n_eval:]

    if len(hotpot_train_rows) == 0:
        hotpot_train_rows = hotpot_rows
        hotpot_eval_rows = hotpot_rows

    hotpot_eval_examples = rows_to_claim_examples(hotpot_eval_rows)
    trivia_examples = rows_to_claim_examples(trivia_rows)
    truthfulqa_examples = rows_to_claim_examples(truthfulqa_rows)

    all_examples = (
        rows_to_claim_examples(hotpot_train_rows)
        + hotpot_eval_examples
        + trivia_examples
        + truthfulqa_examples
    )

    return {
        "rows": rows,
        "examples": all_examples,

        "hotpot_rows": hotpot_train_rows,
        "hotpot_eval_rows": hotpot_eval_rows,
        "trivia_rows": trivia_rows,
        "truthfulqa_rows": truthfulqa_rows,

        "hotpot_examples": hotpot_eval_examples,
        "trivia_examples": trivia_examples,
        "truthfulqa_examples": truthfulqa_examples,
    }


def print_dataset_counts(rows, examples):
    print("\nDataset row counts")
    print("------------------")

    counts = {}

    for row in rows:
        name = row["dataset"]
        counts[name] = counts.get(name, 0) + 1

    for name, count in sorted(counts.items()):
        print(f"{name}: rows={count}")

    print(f"Total rows: {len(rows)}")
    print(f"Total individual examples: {len(examples)}")


def validate_required_splits(splits):
    if len(splits["hotpot_rows"]) == 0:
        raise ValueError(
            "No HotpotQA rows found for training. "
            "Check CSV dataset column contains 'hotpotqa'."
        )

    if len(splits["trivia_examples"]) == 0:
        raise ValueError(
            "No TriviaQA examples found for evaluation. "
            "Check CSV dataset column contains 'triviaqa'."
        )

    if len(splits["truthfulqa_examples"]) == 0:
        raise ValueError(
            "No TruthfulQA examples found for evaluation. "
            "Check CSV dataset column contains 'truthfulqa'."
        )


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------

def format_question_claim(
    question: str,
    claim: str,
    short_answer: str = "",
    use_short_answer: bool = False,
) -> str:
    if use_short_answer and short_answer:
        return (
            f"Question: {question}\n"
            f"Known answer: {short_answer}\n"
            f"Claim: {claim}"
        )

    return f"Question: {question}\nClaim: {claim}"


# ---------------------------------------------------------------------
# Answer-mask tokenisation
# ---------------------------------------------------------------------

def _split_prompt_and_answer_from_text(text: str):
    """
    Splits formatted text into:

        prompt part: question / known answer / marker
        answer part: claim text

    We support both:
        Question: ...\nClaim: ...
        Question: ...\nAnswer: ...
    """

    markers = ["\nClaim:", "\nAnswer:"]

    for marker in markers:
        idx = text.rfind(marker)

        if idx >= 0:
            marker_end = idx + len(marker)
            prompt = text[:marker_end]
            answer = text[marker_end:]
            return prompt, answer

    # Fallback:
    # if no marker is found, treat the whole text as answer.
    return "", text


def encode_text_with_answer_mask(
    tokenizer,
    text: str,
    max_length: int,
):
    """
    Tokenise a formatted question-claim text and build answer_mask.

    answer_mask:
        0 = question / prompt / padding
        1 = answer / claim tokens

    Important:
        The model still sees the whole text.
        We only use answer_mask to pool hidden states over answer tokens.
    """

    prompt, answer = _split_prompt_and_answer_from_text(text)

    # Preserve leading space before the claim if needed.
    if answer and not answer.startswith(" "):
        answer = " " + answer

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    answer_ids = tokenizer(
        answer,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    # If answer is empty for any reason, fall back to masking the full text later.
    if len(answer_ids) == 0:
        full = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        input_ids = full[:max_length]
        attention_mask = [1] * len(input_ids)
        answer_mask = [1] * len(input_ids)

    else:
        # Keep answer tokens as much as possible.
        if len(answer_ids) >= max_length:
            answer_ids = answer_ids[:max_length]
            prompt_ids = []
        else:
            max_prompt_len = max_length - len(answer_ids)
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + answer_ids
        attention_mask = [1] * len(input_ids)
        answer_mask = [0] * len(prompt_ids) + [1] * len(answer_ids)

    pad_len = max_length - len(input_ids)

    if pad_len > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        answer_mask = answer_mask + [0] * pad_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "answer_mask": torch.tensor(answer_mask, dtype=torch.long),
    }


def tokenize_texts_with_answer_masks(
    tokenizer,
    texts: List[str],
    max_length: int,
):
    encoded = [
        encode_text_with_answer_mask(
            tokenizer=tokenizer,
            text=text,
            max_length=max_length,
        )
        for text in texts
    ]

    return {
        "input_ids": torch.stack([x["input_ids"] for x in encoded], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in encoded], dim=0),
        "answer_mask": torch.stack([x["answer_mask"] for x in encoded], dim=0),
    }


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------

class ClaimDataset(Dataset):
    """
    Individual examples for evaluation.
    """

    def __init__(
        self,
        examples,
        tokenizer=None,
        max_length=128,
        use_short_answer=False,
    ):
        self.examples = examples
        self.use_short_answer = use_short_answer

    def __len__(self):
        return len(self.examples)

    def format_text(self, question, claim, short_answer=None):
        return format_question_claim(
            question=question,
            claim=claim,
            short_answer=short_answer,
            use_short_answer=self.use_short_answer,
        )

    def __getitem__(self, idx):
        ex = self.examples[idx]

        text = self.format_text(
            ex["question"],
            ex["claim"],
            ex.get("short_answer", ""),
        )

        return {
            "text": text,
            "dataset": ex["dataset"],
            "question": ex["question"],
            "short_answer": ex.get("short_answer", ""),
            "claim": ex["claim"],
            "label": float(ex["label"]),
            "label_name": ex["label_name"],
        }


class PairClaimDataset(Dataset):
    """
    Paired truthful/hallucinated examples for training.

    Each item contains:
        pos_text
        neg_text
        neighbour positive texts
        neighbour negative texts
    """

    def __init__(
        self,
        rows,
        tokenizer=None,
        max_length=128,
        use_short_answer=False,
        k_neighbours=5,
    ):
        self.rows = rows
        self.use_short_answer = use_short_answer
        self.k_neighbours = k_neighbours

        self.neighbour_indices = self._build_neighbour_indices(k_neighbours)

    def __len__(self):
        return len(self.rows)

    def format_text(self, question, claim, short_answer=None):
        return format_question_claim(
            question=question,
            claim=claim,
            short_answer=short_answer,
            use_short_answer=self.use_short_answer,
        )

    def _build_neighbour_indices(self, k):
        if k <= 0 or len(self.rows) <= 1:
            return [[] for _ in self.rows]

        questions = [r["question"] for r in self.rows]

        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=50000,
            )
            x = vectorizer.fit_transform(questions)
            sim = cosine_similarity(x)
        except Exception:
            return [[] for _ in self.rows]

        neighbours = []

        for i in range(len(self.rows)):
            order = sim[i].argsort()[::-1].tolist()
            order = [j for j in order if j != i]
            neighbours.append(order[:k])

        return neighbours

    def __getitem__(self, idx):
        row = self.rows[idx]

        question = row["question"]
        short_answer = row.get("short_answer", "")

        pos_text = self.format_text(
            question=question,
            claim=row["positive"],
            short_answer=short_answer,
        )

        neg_text = self.format_text(
            question=question,
            claim=row["negative"],
            short_answer=short_answer,
        )

        neigh_pos_texts = []
        neigh_neg_texts = []

        for j in self.neighbour_indices[idx]:
            nrow = self.rows[j]

            neigh_pos_texts.append(
                self.format_text(
                    question=nrow["question"],
                    claim=nrow["positive"],
                    short_answer=nrow.get("short_answer", ""),
                )
            )

            neigh_neg_texts.append(
                self.format_text(
                    question=nrow["question"],
                    claim=nrow["negative"],
                    short_answer=nrow.get("short_answer", ""),
                )
            )

        return {
            "dataset": row["dataset"],
            "question": question,
            "short_answer": short_answer,

            "positive": row["positive"],
            "negative": row["negative"],

            "pos_text": pos_text,
            "neg_text": neg_text,

            "neigh_pos_texts": neigh_pos_texts,
            "neigh_neg_texts": neigh_neg_texts,
        }


# ---------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------

def make_pair_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        pos_texts = [x["pos_text"] for x in batch]
        neg_texts = [x["neg_text"] for x in batch]

        neigh_pos_texts = []
        neigh_neg_texts = []
        k_list = []

        for x in batch:
            k = len(x["neigh_pos_texts"])
            k_list.append(k)
            neigh_pos_texts.extend(x["neigh_pos_texts"])
            neigh_neg_texts.extend(x["neigh_neg_texts"])

        pos_enc = tokenize_texts_with_answer_masks(
            tokenizer,
            pos_texts,
            max_length,
        )

        neg_enc = tokenize_texts_with_answer_masks(
            tokenizer,
            neg_texts,
            max_length,
        )

        out = {
            "pos_input_ids": pos_enc["input_ids"],
            "pos_attention_mask": pos_enc["attention_mask"],
            "pos_answer_mask": pos_enc["answer_mask"],

            "neg_input_ids": neg_enc["input_ids"],
            "neg_attention_mask": neg_enc["attention_mask"],
            "neg_answer_mask": neg_enc["answer_mask"],

            "k_list": torch.tensor(k_list, dtype=torch.long),
            "has_neighbours": sum(k_list) > 0,
            "raw_batch": batch,
        }

        if sum(k_list) > 0:
            neigh_pos_enc = tokenize_texts_with_answer_masks(
                tokenizer,
                neigh_pos_texts,
                max_length,
            )

            neigh_neg_enc = tokenize_texts_with_answer_masks(
                tokenizer,
                neigh_neg_texts,
                max_length,
            )

            out.update(
                {
                    "neigh_pos_input_ids": neigh_pos_enc["input_ids"],
                    "neigh_pos_attention_mask": neigh_pos_enc["attention_mask"],
                    "neigh_pos_answer_mask": neigh_pos_enc["answer_mask"],

                    "neigh_neg_input_ids": neigh_neg_enc["input_ids"],
                    "neigh_neg_attention_mask": neigh_neg_enc["attention_mask"],
                    "neigh_neg_answer_mask": neigh_neg_enc["answer_mask"],
                }
            )

        return out

    return collate_fn


def make_claim_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.float)

        enc = tokenize_texts_with_answer_masks(
            tokenizer,
            texts,
            max_length,
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "answer_mask": enc["answer_mask"],
            "labels": labels,
            "raw_batch": batch,
        }

    return collate_fn


# ---------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------

def build_dataloaders(
    splits,
    tokenizer,
    max_length=128,
    batch_size=16,
    use_short_answer=False,
    num_workers=0,
    k_neighbours=5,
):
    pair_collate_fn = make_pair_collate_fn(tokenizer, max_length)
    claim_collate_fn = make_claim_collate_fn(tokenizer, max_length)

    train_ds = PairClaimDataset(
        splits["hotpot_rows"],
        tokenizer,
        max_length,
        use_short_answer=use_short_answer,
        k_neighbours=k_neighbours,
    )

    hotpot_eval_ds = ClaimDataset(
        splits["hotpot_examples"],
        tokenizer,
        max_length,
        use_short_answer,
    )

    trivia_ds = ClaimDataset(
        splits["trivia_examples"],
        tokenizer,
        max_length,
        use_short_answer,
    )

    truthfulqa_ds = ClaimDataset(
        splits["truthfulqa_examples"],
        tokenizer,
        max_length,
        use_short_answer,
    )

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pair_collate_fn,
            num_workers=num_workers,
        ),
        "hotpot_eval": DataLoader(
            hotpot_eval_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=claim_collate_fn,
            num_workers=num_workers,
        ),
        "trivia": DataLoader(
            trivia_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=claim_collate_fn,
            num_workers=num_workers,
        ),
        "truthfulqa": DataLoader(
            truthfulqa_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=claim_collate_fn,
            num_workers=num_workers,
        ),
    }
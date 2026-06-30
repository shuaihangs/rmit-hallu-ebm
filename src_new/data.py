import csv
import random
from typing import Dict, List, Any, Tuple

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


# ---------------------------------------------------------------------
# Dataset name matching
# ---------------------------------------------------------------------

def _is_hotpot(dataset_name: str) -> bool:
    return "hotpot" in dataset_name.lower()


def _is_trivia(dataset_name: str) -> bool:
    return "trivia" in dataset_name.lower()


def _is_truthfulqa(dataset_name: str) -> bool:
    return "truthful" in dataset_name.lower()


DATASET_MATCHERS = {
    "hotpot": _is_hotpot,
    "trivia": _is_trivia,
    "truthfulqa": _is_truthfulqa,
}


def canonical_dataset_name(dataset_name: str):
    """
    Convert raw dataset name from CSV into one of:
        hotpot, trivia, truthfulqa

    Returns None if not recognised.
    """

    for canonical_name, matcher in DATASET_MATCHERS.items():
        if matcher(dataset_name):
            return canonical_name

    return None


def group_rows_by_dataset(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {name: [] for name in DATASET_MATCHERS.keys()}

    for row in rows:
        name = canonical_dataset_name(row["dataset"])

        if name is not None:
            grouped[name].append(row)

    return grouped


def _loader_key(dataset_name: str) -> str:
    """
    Keeps your existing naming style.

    hotpot -> hotpot_eval
    trivia -> trivia
    truthfulqa -> truthfulqa
    """

    dataset_name = dataset_name.lower()

    if dataset_name == "hotpot":
        return "hotpot_eval"

    return dataset_name


# ---------------------------------------------------------------------
# Convert paired rows into individual examples
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Flexible splitting
# ---------------------------------------------------------------------

def split_examples_by_dataset(
    rows: List[Dict[str, Any]],
    train_dataset: str = "trivia",
    eval_datasets: Tuple[str, ...] = ("hotpot", "truthfulqa"),
    train_eval_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Flexible splitter.

    Default behaviour:
        Train on TriviaQA
        Evaluate on HotpotQA and TruthfulQA

    Example 1:
        Train TriviaQA, evaluate HotpotQA + TruthfulQA

        split_examples_by_dataset(
            rows,
            train_dataset="trivia",
            eval_datasets=("hotpot", "truthfulqa"),
        )

    Example 2:
        Train HotpotQA, evaluate TriviaQA + TruthfulQA

        split_examples_by_dataset(
            rows,
            train_dataset="hotpot",
            eval_datasets=("trivia", "truthfulqa"),
        )

    Example 3:
        Train HotpotQA, evaluate held-out HotpotQA + TriviaQA + TruthfulQA

        split_examples_by_dataset(
            rows,
            train_dataset="hotpot",
            eval_datasets=("hotpot", "trivia", "truthfulqa"),
            train_eval_ratio=0.2,
        )
    """

    train_dataset = train_dataset.lower()
    eval_datasets = tuple(x.lower() for x in eval_datasets)

    if train_dataset not in DATASET_MATCHERS:
        raise ValueError(
            f"Unknown train_dataset='{train_dataset}'. "
            f"Choose from {list(DATASET_MATCHERS.keys())}."
        )

    for name in eval_datasets:
        if name not in DATASET_MATCHERS:
            raise ValueError(
                f"Unknown eval dataset='{name}'. "
                f"Choose from {list(DATASET_MATCHERS.keys())}."
            )

    grouped = group_rows_by_dataset(rows)

    rng = random.Random(seed)

    train_rows_all = list(grouped[train_dataset])
    rng.shuffle(train_rows_all)

    if len(train_rows_all) == 0:
        raise ValueError(
            f"No rows found for train_dataset='{train_dataset}'. "
            "Check the dataset column in your CSV."
        )

    # If the training dataset is also an eval dataset, create held-out eval split.
    # If not, use all training rows for training.
    if train_dataset in eval_datasets and train_eval_ratio > 0:
        n_eval = max(1, int(len(train_rows_all) * train_eval_ratio))

        train_eval_rows = train_rows_all[:n_eval]
        train_rows = train_rows_all[n_eval:]

        if len(train_rows) == 0:
            train_rows = train_rows_all
            train_eval_rows = train_rows_all
    else:
        train_rows = train_rows_all
        train_eval_rows = []

    eval_examples = {}
    eval_rows_by_name = {}

    for name in eval_datasets:
        if name == train_dataset:
            eval_rows = train_eval_rows
        else:
            eval_rows = grouped[name]

        loader_name = _loader_key(name)
        eval_rows_by_name[loader_name] = eval_rows
        eval_examples[loader_name] = rows_to_claim_examples(eval_rows)

    all_examples = rows_to_claim_examples(train_rows)

    for examples in eval_examples.values():
        all_examples.extend(examples)

    splits = {
        "rows": rows,
        "examples": all_examples,

        "train_dataset": train_dataset,
        "train_rows": train_rows,
        "train_eval_rows": train_eval_rows,

        "eval_datasets": eval_datasets,
        "eval_examples": eval_examples,
        "eval_rows_by_name": eval_rows_by_name,

        # Backward-compatible fields
        "hotpot_rows": grouped["hotpot"],
        "trivia_rows": grouped["trivia"],
        "truthfulqa_rows": grouped["truthfulqa"],

        "hotpot_examples": rows_to_claim_examples(grouped["hotpot"]),
        "trivia_examples": rows_to_claim_examples(grouped["trivia"]),
        "truthfulqa_examples": rows_to_claim_examples(grouped["truthfulqa"]),
    }

    return splits


def print_dataset_counts(rows, examples=None, splits=None):
    print("\nDataset row counts")
    print("------------------")

    counts = {}

    for row in rows:
        name = row["dataset"]
        counts[name] = counts.get(name, 0) + 1

    for name, count in sorted(counts.items()):
        print(f"{name}: rows={count}")

    print(f"Total rows: {len(rows)}")

    if examples is not None:
        print(f"Total individual examples: {len(examples)}")

    if splits is not None:
        print("\nTraining split")
        print("--------------")
        print(f"Train dataset: {splits['train_dataset']}")
        print(f"Train rows: {len(splits['train_rows'])}")
        print(f"Train individual examples: {len(rows_to_claim_examples(splits['train_rows']))}")

        print("\nEvaluation splits")
        print("-----------------")
        for loader_name, eval_examples in splits["eval_examples"].items():
            eval_rows = splits["eval_rows_by_name"][loader_name]
            print(
                f"{loader_name}: rows={len(eval_rows)}, "
                f"individual examples={len(eval_examples)}"
            )


def validate_required_splits(splits):
    if len(splits["train_rows"]) == 0:
        raise ValueError(
            f"No training rows found for train_dataset='{splits['train_dataset']}'."
        )

    for loader_name, examples in splits["eval_examples"].items():
        if len(examples) == 0:
            raise ValueError(
                f"No evaluation examples found for loader='{loader_name}'. "
                "Check your CSV dataset column."
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

    Supports:
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
    """

    prompt, answer = _split_prompt_and_answer_from_text(text)

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
# Flexible dataloaders
# ---------------------------------------------------------------------

def build_dataloaders(
    splits,
    tokenizer,
    max_length=128,
    batch_size=16,
    eval_batch_size=None,
    use_short_answer=False,
    num_workers=0,
    k_neighbours=5,
):
    if eval_batch_size is None:
        eval_batch_size = batch_size

    pair_collate_fn = make_pair_collate_fn(tokenizer, max_length)
    claim_collate_fn = make_claim_collate_fn(tokenizer, max_length)

    train_ds = PairClaimDataset(
        splits["train_rows"],
        tokenizer,
        max_length,
        use_short_answer=use_short_answer,
        k_neighbours=k_neighbours,
    )

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pair_collate_fn,
            num_workers=num_workers,
        )
    }

    for loader_name, examples in splits["eval_examples"].items():
        eval_ds = ClaimDataset(
            examples,
            tokenizer,
            max_length,
            use_short_answer=use_short_answer,
        )

        loaders[loader_name] = DataLoader(
            eval_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=claim_collate_fn,
            num_workers=num_workers,
        )

    return loaders

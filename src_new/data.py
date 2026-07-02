import csv
import hashlib
import random
from typing import Dict, List, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors


_NEIGHBOUR_INDEX_CACHE = {}


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


def normalize_dataset_key(dataset_name: str) -> str:
    name = str(dataset_name).strip().lower()
    name = name.replace("-", "_")
    name = name.replace(" ", "_")

    if "hotpot" in name:
        return "hotpotqa"

    if "truthful" in name:
        return "truthfulqa"

    if "halueval" in name or "halu_eval" in name or "hallu_eval" in name:
        return "halueval"

    if "trivia" in name:
        return "triviaqa"

    return name


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
    dataset_names=None,
    validation_ratio: float = 0.2,
    seed: int = 42,
):
    if dataset_names is None:
        dataset_names = sorted({
            normalize_dataset_key(row["dataset"])
            for row in rows
        })
    else:
        dataset_names = [
            normalize_dataset_key(name)
            for name in dataset_names
        ]

    datasets = {
        name: {
            "rows": [],
            "examples": [],
        }
        for name in dataset_names
    }

    for row in rows:
        dataset_key = normalize_dataset_key(row["dataset"])

        if dataset_key not in datasets:
            continue

        datasets[dataset_key]["rows"].append(row)

    all_examples = []

    for dataset_idx, dataset in enumerate(datasets.values()):
        dataset_rows = list(dataset["rows"])
        rng = random.Random(seed + dataset_idx)
        rng.shuffle(dataset_rows)

        if validation_ratio > 0.0 and len(dataset_rows) > 1:
            n_validation = max(1, int(len(dataset_rows) * validation_ratio))
            n_validation = min(n_validation, len(dataset_rows) - 1)
        else:
            n_validation = 0

        validation_rows = dataset_rows[:n_validation]
        train_rows = dataset_rows[n_validation:]

        if len(train_rows) == 0:
            train_rows = dataset_rows
            validation_rows = dataset_rows

        dataset["train_rows"] = train_rows
        dataset["validation_rows"] = validation_rows
        dataset["examples"] = rows_to_claim_examples(dataset["rows"])
        dataset["train_examples"] = rows_to_claim_examples(train_rows)
        dataset["validation_examples"] = rows_to_claim_examples(validation_rows)
        all_examples.extend(dataset["examples"])

    return {
        "rows": rows,
        "examples": all_examples,
        "dataset_names": dataset_names,
        "datasets": datasets,
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


def validate_required_splits(splits, dataset_names=None):
    if dataset_names is None:
        dataset_names = splits["dataset_names"]

    dataset_names = [
        normalize_dataset_key(name)
        for name in dataset_names
    ]

    missing = [
        name
        for name in dataset_names
        if len(splits["datasets"].get(name, {}).get("rows", [])) == 0
    ]

    if missing:
        available = sorted({
            normalize_dataset_key(row["dataset"])
            for row in splits["rows"]
        })
        raise ValueError(
            "Missing required dataset rows for: "
            f"{', '.join(missing)}. "
            f"Available dataset keys in CSV: {', '.join(available)}."
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
        use_short_answer=False,
        k_neighbours=5,
        neighbour_backend="sentence",
        neighbour_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.rows = rows
        self.use_short_answer = use_short_answer
        self.k_neighbours = k_neighbours
        self.neighbour_backend = neighbour_backend
        self.neighbour_embedding_model = neighbour_embedding_model

        self.neighbour_indices = self._build_neighbour_indices(
            k=k_neighbours,
            backend=neighbour_backend,
            embedding_model_name=neighbour_embedding_model,
        )

    def __len__(self):
        return len(self.rows)

    def format_text(self, question, claim, short_answer=None):
        return format_question_claim(
            question=question,
            claim=claim,
            short_answer=short_answer,
            use_short_answer=self.use_short_answer,
        )

    def _build_neighbour_indices(
        self,
        k,
        backend,
        embedding_model_name,
    ):
        """
        Build K nearest question neighbours among training questions.

        Supported backends:
            none     -> no neighbours
            tfidf    -> sparse lexical nearest neighbours
            sentence -> dense sentence-embedding nearest neighbours

        The returned indices point into self.rows.
        """
        backend = str(backend or "none").strip().lower()

        if backend in {"none", "off", "disabled"} or k <= 0 or len(self.rows) <= 1:
            return [[] for _ in self.rows]

        questions = [r["question"] for r in self.rows]
        question_digest = hashlib.sha1(
            "\n".join(questions).encode("utf-8")
        ).hexdigest()
        cache_key = (
            backend,
            int(k),
            str(embedding_model_name),
            question_digest,
        )

        if cache_key in _NEIGHBOUR_INDEX_CACHE:
            print(
                f"Using cached {backend} question neighbours "
                f"(k={k}, rows={len(self.rows)})."
            )
            return _NEIGHBOUR_INDEX_CACHE[cache_key]

        if backend == "tfidf":
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                print(
                    f"Building TF-IDF question neighbours "
                    f"(k={k}, rows={len(self.rows)})..."
                )
                vectorizer = TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=50000,
                )
                features = vectorizer.fit_transform(questions)
                n_neighbours = min(k + 1, len(self.rows))
                index = NearestNeighbors(
                    n_neighbors=n_neighbours,
                    metric="cosine",
                    algorithm="brute",
                )
                index.fit(features)
                _, nearest_indices = index.kneighbors(features)

            except Exception as e:
                raise RuntimeError(
                    "Could not build TF-IDF neighbours. "
                    "Check scikit-learn and the input questions."
                ) from e

        elif backend in {"sentence", "embedding", "dense"}:
            try:
                from sentence_transformers import SentenceTransformer

                print(
                    "Building sentence-embedding question neighbours with "
                    f"{embedding_model_name} (k={k}, rows={len(self.rows)})..."
                )

                embedder = SentenceTransformer(embedding_model_name)
                embeddings = embedder.encode(
                    questions,
                    batch_size=64,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                )

                embeddings = np.asarray(embeddings)
                n_neighbours = min(k + 1, len(self.rows))

                index = NearestNeighbors(
                    n_neighbors=n_neighbours,
                    metric="cosine",
                )
                index.fit(embeddings)
                _, nearest_indices = index.kneighbors(embeddings)

            except Exception as e:
                raise RuntimeError(
                    "Could not build sentence-embedding neighbours. "
                    "Install sentence-transformers in the active environment or "
                    "check the embedding model name."
                ) from e

        else:
            raise ValueError(
                "Unknown neighbour backend: "
                f"{backend}. Use one of: none, tfidf, sentence."
            )

        neighbours = []

        for i, row_indices in enumerate(nearest_indices):
            row_indices = [
                int(j)
                for j in row_indices
                if int(j) != i
            ]
            neighbours.append(row_indices[:k])

        _NEIGHBOUR_INDEX_CACHE[cache_key] = neighbours
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
    train_dataset,
    eval_datasets=None,
    max_length=128,
    batch_size=16,
    use_short_answer=False,
    num_workers=0,
    k_neighbours=5,
    neighbour_backend="sentence",
    neighbour_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
):
    train_dataset = normalize_dataset_key(train_dataset)

    if eval_datasets is None:
        eval_datasets = [
            name
            for name in splits["dataset_names"]
            if name != train_dataset
        ]
    else:
        eval_datasets = [
            normalize_dataset_key(name)
            for name in eval_datasets
        ]

    pair_collate_fn = make_pair_collate_fn(tokenizer, max_length)
    claim_collate_fn = make_claim_collate_fn(tokenizer, max_length)

    if train_dataset not in splits["datasets"]:
        raise KeyError(f"Unknown train dataset: {train_dataset}")

    train_rows = splits["datasets"][train_dataset]["rows"]
    train_rows = splits["datasets"][train_dataset].get("train_rows", train_rows)

    if len(train_rows) == 0:
        raise ValueError(f"No rows available for train dataset: {train_dataset}")

    train_ds = PairClaimDataset(
        train_rows,
        use_short_answer=use_short_answer,
        k_neighbours=k_neighbours,
        neighbour_backend=neighbour_backend,
        neighbour_embedding_model=neighbour_embedding_model,
    )

    eval_loaders = {}
    train_eval_name = f"{train_dataset}_train"
    validation_eval_name = f"{train_dataset}_val"

    train_examples = splits["datasets"][train_dataset].get(
        "train_examples",
        rows_to_claim_examples(train_rows),
    )
    validation_examples = splits["datasets"][train_dataset].get(
        "validation_examples",
        [],
    )

    if len(validation_examples) == 0:
        validation_examples = train_examples

    for eval_name, eval_examples in [
        (train_eval_name, train_examples),
        (validation_eval_name, validation_examples),
    ]:
        eval_ds = ClaimDataset(
            eval_examples,
            use_short_answer=use_short_answer,
        )

        eval_loaders[eval_name] = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=claim_collate_fn,
            num_workers=num_workers,
        )

    for dataset_name in eval_datasets:
        if dataset_name not in splits["datasets"]:
            raise KeyError(f"Unknown eval dataset: {dataset_name}")

        eval_examples = splits["datasets"][dataset_name]["examples"]

        if len(eval_examples) == 0:
            raise ValueError(f"No examples available for eval dataset: {dataset_name}")

        eval_ds = ClaimDataset(
            eval_examples,
            use_short_answer=use_short_answer,
        )

        eval_loaders[dataset_name] = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=claim_collate_fn,
            num_workers=num_workers,
        )

    return {
        "train_dataset": train_dataset,
        "eval_datasets": list(eval_loaders.keys()),
        "monitor_datasets": [validation_eval_name],
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pair_collate_fn,
            num_workers=num_workers,
        ),
        "eval": eval_loaders,
    }

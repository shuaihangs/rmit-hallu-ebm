import csv
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from .utils import clean_text, normalize_dataset_name

HOTPOT_NAMES = {"hotpotqa", "hotpot_qa"}
TRIVIA_NAMES = {"triviaqa", "trivia_qa", "triviaqa_wiki", "trivia_qa_wiki", "triviaqa_unfiltered"}
TRUTHFULQA_NAMES = {"truthfulqa", "truthful_qa", "truthfulqa_generation", "truthful_qa_generation", "truthfulqa_mc", "truthful_qa_mc"}


def load_rows_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"dataset", "question", "short_answer", "positive", "negative"}
        found_cols = set(reader.fieldnames or [])
        missing = required_cols - found_cols
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}\nFound columns: {reader.fieldnames}")

        for row in reader:
            dataset = normalize_dataset_name(row.get("dataset"))
            question = clean_text(row.get("question"))
            short_answer = clean_text(row.get("short_answer"))
            positive = clean_text(row.get("positive"))
            negative = clean_text(row.get("negative"))
            if not question or not positive or not negative:
                continue
            rows.append({
                "dataset": dataset,
                "question": question,
                "short_answer": short_answer,
                "positive": positive,
                "negative": negative,
            })

    if len(rows) == 0:
        raise ValueError("No valid rows were loaded from the CSV.")
    return rows


def make_individual_examples(rows):
    examples = []
    for row in rows:
        examples.append({
            "dataset": row["dataset"],
            "question": row["question"],
            "short_answer": row["short_answer"],
            "claim": row["positive"],
            "label": 0,
            "label_name": "positive",
        })
        examples.append({
            "dataset": row["dataset"],
            "question": row["question"],
            "short_answer": row["short_answer"],
            "claim": row["negative"],
            "label": 1,
            "label_name": "negative",
        })
    return examples


def split_examples_by_dataset(rows):
    examples = make_individual_examples(rows)
    hotpot_rows = [row for row in rows if row["dataset"] in HOTPOT_NAMES]
    hotpot_examples = [ex for ex in examples if ex["dataset"] in HOTPOT_NAMES]
    trivia_examples = [ex for ex in examples if ex["dataset"] in TRIVIA_NAMES]
    truthfulqa_examples = [ex for ex in examples if ex["dataset"] in TRUTHFULQA_NAMES]
    return {
        "rows": rows,
        "examples": examples,
        "hotpot_rows": hotpot_rows,
        "hotpot_examples": hotpot_examples,
        "trivia_examples": trivia_examples,
        "truthfulqa_examples": truthfulqa_examples,
    }


def print_dataset_counts(rows, examples):
    print("\nDataset counts:")
    for d in sorted(set(row["dataset"] for row in rows)):
        row_n = sum(1 for row in rows if row["dataset"] == d)
        ex_n = sum(1 for ex in examples if ex["dataset"] == d)
        pos_n = sum(1 for ex in examples if ex["dataset"] == d and ex["label"] == 0)
        neg_n = sum(1 for ex in examples if ex["dataset"] == d and ex["label"] == 1)
        print(f"  {d}: rows={row_n}, individual={ex_n}, positive={pos_n}, negative={neg_n}")

def build_question_neighbours(rows, k=5, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Build K nearest semantic neighbours among training questions using sentence embeddings.

    This is training-only. It does not affect inference. The returned indices
    are row indices into `rows`, and each neighbour row has its own positive
    and negative answer.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors

    n = len(rows)

    if n <= 1 or k <= 0:
        return [[] for _ in range(n)]

    questions = [row["question"] for row in rows]

    print(f"Building semantic question neighbours with {embedding_model_name}...")
    embedder = SentenceTransformer(embedding_model_name)

    q_emb = embedder.encode(
        questions,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    q_emb = np.asarray(q_emb)

    n_neighbors = min(k + 1, n)

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
    )

    nn.fit(q_emb)

    _, indices = nn.kneighbors(q_emb)

    neighbour_indices = []

    for i, neigh in enumerate(indices):
        neigh = [int(j) for j in neigh if int(j) != i]
        neighbour_indices.append(neigh[:k])

    return neighbour_indices

class PairClaimDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length, use_short_answer=False, k_neighbours=5):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_short_answer = use_short_answer
        self.k_neighbours = k_neighbours
        self.neighbour_indices = build_question_neighbours(rows, k=k_neighbours)

    def __len__(self):
        return len(self.rows)

    def format_text(self, question, claim, short_answer=None):
        if self.use_short_answer and short_answer:
            return f"Question: {question}\nKnown answer: {short_answer}\nClaim: {claim}"
        return f"Question: {question}\nClaim: {claim}"

    def __getitem__(self, idx):
        ex = self.rows[idx]
        pos_text = self.format_text(ex["question"], ex["positive"], ex.get("short_answer", ""))
        neg_text = self.format_text(ex["question"], ex["negative"], ex.get("short_answer", ""))

        neigh_pos_texts = []
        neigh_neg_texts = []
        for j in self.neighbour_indices[idx]:
            n_ex = self.rows[j]
            neigh_pos_texts.append(self.format_text(n_ex["question"], n_ex["positive"], n_ex.get("short_answer", "")))
            neigh_neg_texts.append(self.format_text(n_ex["question"], n_ex["negative"], n_ex.get("short_answer", "")))

        return {
            "dataset": ex["dataset"],
            "question": ex["question"],
            "short_answer": ex.get("short_answer", ""),
            "positive": ex["positive"],
            "negative": ex["negative"],
            "pos_text": pos_text,
            "neg_text": neg_text,
            "neigh_pos_texts": neigh_pos_texts,
            "neigh_neg_texts": neigh_neg_texts,
        }


class ClaimDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length, use_short_answer=False):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_short_answer = use_short_answer

    def __len__(self):
        return len(self.examples)

    def format_text(self, question, claim, short_answer=None):
        if self.use_short_answer and short_answer:
            return f"Question: {question}\nKnown answer: {short_answer}\nClaim: {claim}"
        return f"Question: {question}\nClaim: {claim}"

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = self.format_text(ex["question"], ex["claim"], ex.get("short_answer", ""))
        return {
            "text": text,
            "dataset": ex["dataset"],
            "question": ex["question"],
            "short_answer": ex.get("short_answer", ""),
            "claim": ex["claim"],
            "label": float(ex["label"]),
            "label_name": ex["label_name"],
        }


def _tokenize_texts(tokenizer, texts, max_length):
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)


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

        pos_enc = _tokenize_texts(tokenizer, pos_texts, max_length)
        neg_enc = _tokenize_texts(tokenizer, neg_texts, max_length)

        out = {
            "pos_input_ids": pos_enc["input_ids"],
            "pos_attention_mask": pos_enc["attention_mask"],
            "neg_input_ids": neg_enc["input_ids"],
            "neg_attention_mask": neg_enc["attention_mask"],
            "k_list": torch.tensor(k_list, dtype=torch.long),
            "has_neighbours": sum(k_list) > 0,
            "raw_batch": batch,
        }

        if sum(k_list) > 0:
            neigh_pos_enc = _tokenize_texts(tokenizer, neigh_pos_texts, max_length)
            neigh_neg_enc = _tokenize_texts(tokenizer, neigh_neg_texts, max_length)
            out.update({
                "neigh_pos_input_ids": neigh_pos_enc["input_ids"],
                "neigh_pos_attention_mask": neigh_pos_enc["attention_mask"],
                "neigh_neg_input_ids": neigh_neg_enc["input_ids"],
                "neigh_neg_attention_mask": neigh_neg_enc["attention_mask"],
            })

        return out
    return collate_fn


def make_claim_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.float)
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "raw_batch": batch,
        }
    return collate_fn


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
        use_short_answer,
        k_neighbours=k_neighbours,
    )
    hotpot_eval_ds = ClaimDataset(splits["hotpot_examples"], tokenizer, max_length, use_short_answer)
    trivia_ds = ClaimDataset(splits["trivia_examples"], tokenizer, max_length, use_short_answer)
    truthfulqa_ds = ClaimDataset(splits["truthfulqa_examples"], tokenizer, max_length, use_short_answer)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pair_collate_fn, num_workers=num_workers),
        "hotpot_eval": DataLoader(hotpot_eval_ds, batch_size=batch_size, shuffle=False, collate_fn=claim_collate_fn, num_workers=num_workers),
        "trivia": DataLoader(trivia_ds, batch_size=batch_size, shuffle=False, collate_fn=claim_collate_fn, num_workers=num_workers),
        "truthfulqa": DataLoader(truthfulqa_ds, batch_size=batch_size, shuffle=False, collate_fn=claim_collate_fn, num_workers=num_workers),
    }


def validate_required_splits(splits):
    if len(splits["hotpot_rows"]) == 0:
        raise ValueError("No HotpotQA rows found for training. Check CSV dataset column contains 'hotpotqa'.")
    if len(splits["trivia_examples"]) == 0:
        raise ValueError("No TriviaQA examples found for evaluation. Check CSV dataset column contains 'triviaqa'.")
    if len(splits["truthfulqa_examples"]) == 0:
        raise ValueError("No TruthfulQA examples found for evaluation. Check CSV dataset column contains 'truthfulqa'.")

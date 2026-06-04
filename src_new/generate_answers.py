import os
import re
import random
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Config
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

MAX_SAMPLES_PER_DATASET = 10000
MAX_NEW_TOKENS_FACTUAL = 48
MAX_NEW_TOKENS_HALLUCINATED = 64

OUTPUT_PATH = "processed_qa_10000hallucination_dataset.csv"
CHECKPOINT_PATH = "processed_qa_hallucination_dataset_checkpoint.csv"

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Load Qwen
# ============================================================

print(f"Loading model: {MODEL_NAME}")
print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    trust_remote_code=True,
).to(DEVICE)

model.eval()


# ============================================================
# Text utilities
# ============================================================

def clean_text(text):
    if text is None:
        return None

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    if text == "":
        return None

    return text


def ensure_sentence(text):
    text = clean_text(text)

    if text is None:
        return None

    if text[-1] not in [".", "!", "?"]:
        text += "."

    return text


def remove_prompt_echo(text):
    """
    Removes prompt echo if the model repeats the instruction.
    """
    text = clean_text(text)

    if text is None:
        return None

    markers = [
        "Factual sentence:",
        "Full sentence answer:",
        "Hallucinated answer:",
        "Answer:",
    ]

    for marker in markers:
        if marker in text:
            text = text.split(marker)[-1].strip()

    return clean_text(text)


def keep_first_sentence(text):
    """
    Keep only the first sentence to avoid explanations, notes, citations,
    repeated outputs, or multiple hallucinated answers.
    """
    text = clean_text(text)

    if text is None:
        return None

    bad_starts = [
        "To determine",
        "The correct answer",
        "Therefore",
        "Note:",
        "This hallucinated answer",
        "Explanation:",
        "Here is",
        "Sure",
    ]

    for bad in bad_starts:
        if text.lower().startswith(bad.lower()):
            return None

    # Remove text after common unwanted continuations.
    stop_markers = [
        " Note:",
        " Explanation:",
        " Source:",
        " [source]",
        " [cite]",
        " Hallucinated answer:",
        " Factual sentence:",
        "\n",
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    match = re.search(r"(.+?[.!?])(\s|$)", text)

    if match:
        return clean_text(match.group(1))

    return ensure_sentence(text)


def clean_generated_answer(text):
    text = remove_prompt_echo(text)
    text = keep_first_sentence(text)
    text = ensure_sentence(text)
    return text


def is_valid_answer(text):
    text = clean_text(text)

    if text is None:
        return False

    lowered = text.lower()

    bad_phrases = [
        "you are converting",
        "you are generating",
        "requirements:",
        "question:",
        "correct answer:",
        "factual sentence:",
        "hallucinated answer:",
        "do not explain",
        "training data",
        "dataset",
        "note:",
        "source",
        "cite",
        "to determine",
        "therefore",
        "explanation",
    ]

    if any(p in lowered for p in bad_phrases):
        return False

    bad_exact = {
        "",
        "none",
        "unknown",
        "i don't know",
        "i do not know",
        "cannot answer",
        "not enough information",
    }

    if lowered in bad_exact:
        return False

    word_count = len(text.split())

    if word_count < 3:
        return False

    if word_count > 45:
        return False

    return True


# ============================================================
# Qwen generation
# ============================================================

def run_llm(
    prompt,
    max_new_tokens=64,
    do_sample=False,
    temperature=0.8,
    top_p=0.9,
):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful data generation assistant. "
                "Return only one requested sentence. "
                "Do not repeat the prompt. "
                "Do not explain anything."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(DEVICE)

    input_length = inputs["input_ids"].shape[1]

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generation_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })
    else:
        generation_kwargs.update({
            "do_sample": False,
        })

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    # CRITICAL FIX:
    # Qwen is decoder-only, so output_ids contains prompt + generation.
    # Slice off the prompt and decode only the new generated tokens.
    generated_ids = output_ids[0, input_length:]

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return clean_generated_answer(raw_output)


# ============================================================
# Prompt 1: factual full sentence
# ============================================================

def build_full_sentence_prompt(question, short_answer):
    return f"""
You are converting short-answer QA data into full factual claim sentences for hallucination detection research.

Given a question and the correct short answer, write one complete factual sentence that directly answers the question.

Requirements:
1. The sentence must be factually consistent with the given correct answer.
2. The sentence must be fluent and natural.
3. The sentence must be a complete sentence, not just a noun phrase.
4. The sentence should preserve the meaning of the original question.
5. Do not add extra facts that are not required by the question.
6. Do not explain anything.
7. Do not mention that this is a dataset or training example.
8. Return only the factual sentence.

Question: {question}
Correct answer: {short_answer}

Factual sentence:
""".strip()


def convert_answer_to_full_sentence(question, short_answer):
    prompt = build_full_sentence_prompt(
        question=question,
        short_answer=short_answer,
    )

    return run_llm(
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS_FACTUAL,
        do_sample=False,
    )


# ============================================================
# Prompt 2: hallucinated answer
# ============================================================

def build_hallucinated_answer_prompt(question, correct_answer):
    return f"""
You are generating training data for hallucination detection.

Given a question and its correct answer, write one hallucinated answer.

Requirements:
1. The hallucinated answer must be factually incorrect.
2. It must be fluent and plausible.
3. It must be a complete sentence.
4. It should have similar length and style to the truthful answer.
5. It should not say "I don't know" or express uncertainty.
6. It should not be obviously absurd.
7. It should change only the key factual entity or fact, while keeping the rest of the sentence structure similar.
8. Do not explain why it is wrong.
9. Return only the hallucinated answer.

Question: {question}
Correct answer: {correct_answer}

Hallucinated answer:
""".strip()


def generate_negative_answer(question, correct_answer):
    prompt = build_hallucinated_answer_prompt(
        question=question,
        correct_answer=correct_answer,
    )

    return run_llm(
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS_HALLUCINATED,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )


# ============================================================
# Dataset helpers
# ============================================================

def safe_select_dataset(ds, max_samples):
    n = min(max_samples, len(ds))
    return ds.select(range(n))


def make_example(dataset_name, question, positive, negatives, short_answer=None):
    return {
        "dataset": dataset_name,
        "question": clean_text(question),
        "short_answer": clean_text(short_answer) if short_answer is not None else None,
        "positive": clean_text(positive),
        "negatives": negatives,
    }


def save_checkpoint(examples):
    if len(examples) == 0:
        return

    df = pd.DataFrame(examples)
    df.to_csv(CHECKPOINT_PATH, index=False)


# ============================================================
# Process HotpotQA
# ============================================================

def process_hotpotqa(split="train", max_samples=1000):
    print("\nLoading HotpotQA...")

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    ds = safe_select_dataset(ds, max_samples)

    examples = []

    for ex in tqdm(ds, desc="Processing HotpotQA"):
        question = clean_text(ex.get("question"))
        short_answer = clean_text(ex.get("answer"))

        if question is None or short_answer is None:
            continue

        positive = convert_answer_to_full_sentence(
            question=question,
            short_answer=short_answer,
        )

        if not is_valid_answer(positive):
            print("Bad positive:", positive)
            continue

        negative = generate_negative_answer(
            question=question,
            correct_answer=positive,
        )

        if not is_valid_answer(negative):
            print("Bad negative:", negative)
            continue

        if positive.lower() == negative.lower():
            print("Skipped identical positive/negative:", positive)
            continue

        examples.append(
            make_example(
                dataset_name="hotpotqa",
                question=question,
                short_answer=short_answer,
                positive=positive,
                negatives=[negative],
            )
        )

        if len(examples) % 100 == 0:
            save_checkpoint(examples)

    return examples


# ============================================================
# Process TriviaQA Wiki
# ============================================================

def extract_triviaqa_answer(ex):
    answer = ex.get("answer")

    if isinstance(answer, dict):
        if "value" in answer:
            return clean_text(answer["value"])

        if "normalized_value" in answer:
            return clean_text(answer["normalized_value"])

        if "aliases" in answer and len(answer["aliases"]) > 0:
            return clean_text(answer["aliases"][0])

    if isinstance(answer, str):
        return clean_text(answer)

    return None


def process_triviaqa_wiki(split="train", max_samples=1000):
    print("\nLoading TriviaQA Wiki...")

    ds = load_dataset("trivia_qa", "rc.wikipedia", split=split)
    ds = safe_select_dataset(ds, max_samples)

    examples = []

    for ex in tqdm(ds, desc="Processing TriviaQA Wiki"):
        question = clean_text(ex.get("question"))
        short_answer = extract_triviaqa_answer(ex)

        if question is None or short_answer is None:
            continue

        positive = convert_answer_to_full_sentence(
            question=question,
            short_answer=short_answer,
        )

        if not is_valid_answer(positive):
            print("Bad positive:", positive)
            continue

        negative = generate_negative_answer(
            question=question,
            correct_answer=positive,
        )

        if not is_valid_answer(negative):
            print("Bad negative:", negative)
            continue

        if positive.lower() == negative.lower():
            print("Skipped identical positive/negative:", positive)
            continue

        examples.append(
            make_example(
                dataset_name="triviaqa_wiki",
                question=question,
                short_answer=short_answer,
                positive=positive,
                negatives=[negative],
            )
        )

        if len(examples) % 100 == 0:
            save_checkpoint(examples)

    return examples


# ============================================================
# Process TruthfulQA
# No modification.
# Uses original Question, Best Answer, and Incorrect Answers.
# ============================================================

def process_truthfulqa(split="train", max_samples=1000):
    print("\nLoading TruthfulQA...")

    ds = load_dataset("TruthfulQA", split=split)
    ds = safe_select_dataset(ds, max_samples)

    examples = []

    for ex in tqdm(ds, desc="Processing TruthfulQA"):
        question = ex.get("Question")
        positive = ex.get("Best Answer")
        incorrect_str = ex.get("Incorrect Answers", "")

        if not question or not positive or not incorrect_str:
            continue

        negatives = [
            x.strip()
            for x in incorrect_str.split(";")
            if x.strip()
        ]

        if negatives:
            examples.append({
                "dataset": "truthfulqa",
                "question": question.strip(),
                "short_answer": positive.strip(),
                "positive": positive.strip(),
                "negatives": negatives,
            })

        if len(examples) % 100 == 0:
            save_checkpoint(examples)

    return examples


# ============================================================
# Flatten negatives for pairwise EBM training
# ============================================================

def flatten_examples(examples):
    rows = []

    for ex in examples:
        question = ex["question"]
        positive = ex["positive"]
        negatives = ex["negatives"]

        for negative in negatives:
            rows.append({
                "dataset": ex["dataset"],
                "question": question,
                "short_answer": ex.get("short_answer"),
                "positive": positive,
                "negative": negative,
            })

    return rows


# ============================================================
# Main
# ============================================================

def build_dataset(save_flattened=True):
    all_examples = []

    hotpot_examples = process_hotpotqa(
        split="train",
        max_samples=MAX_SAMPLES_PER_DATASET,
    )
    all_examples.extend(hotpot_examples)
    save_checkpoint(all_examples)

    trivia_examples = process_triviaqa_wiki(
        split="train",
        max_samples=MAX_SAMPLES_PER_DATASET,
    )
    all_examples.extend(trivia_examples)
    save_checkpoint(all_examples)

    truthfulqa_examples = process_truthfulqa(
        split="train",
        max_samples=MAX_SAMPLES_PER_DATASET,
    )
    all_examples.extend(truthfulqa_examples)
    save_checkpoint(all_examples)

    if save_flattened:
        rows = flatten_examples(all_examples)
        df = pd.DataFrame(rows)

        df = df.dropna(subset=["question", "positive", "negative"])
        df = df.drop_duplicates(
            subset=["dataset", "question", "positive", "negative"]
        )
        df = df.reset_index(drop=True)

        df.to_csv(OUTPUT_PATH, index=False)

        print("\nDone.")
        print(f"Saved flattened pairwise dataset to: {OUTPUT_PATH}")
        print(f"Number of pairwise examples: {len(df)}")
        print(df.head())

        return df

    df = pd.DataFrame(all_examples)

    df = df.dropna(subset=["question", "positive", "negatives"])
    df = df.drop_duplicates(
        subset=["dataset", "question", "positive"]
    )
    df = df.reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nDone.")
    print(f"Saved grouped dataset to: {OUTPUT_PATH}")
    print(f"Number of grouped examples: {len(df)}")
    print(df.head())

    return df


if __name__ == "__main__":
    df = build_dataset(save_flattened=True)
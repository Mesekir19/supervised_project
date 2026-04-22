import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME     = "Qwen/Qwen3-0.6B"
ADAPTER_PATH   = "./results/final_model"
TEST_FILE      = "test.json"
OUTPUT_DIR     = "./results"
MAX_NEW_TOKENS = 60

SYSTEM_PROMPT = (
    "You are a geospatial assistant specialized in spatial reasoning about "
    "points of interest. When asked about the relationship between two POIs, "
    "respond with a factual sentence that includes the distance in meters and "
    "the cardinal direction."
)

DIRECTION_LABELS = [
    "north", "east", "south", "west",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# LOAD MODEL
def load_model(adapter_path: str):
    print(f"[Model] Loading tokenizer from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Model] Loading base model: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[Model] Attaching LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    device = next(model.parameters()).device
    print(f"[Model] Ready on: {device}\n")
    return model, tokenizer, str(device)


def generate_completion(model, tokenizer, user_question: str, device: str) -> str:

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [tokenizer.eos_token_id]
    if im_end_id and im_end_id != tokenizer.unk_token_id:
        stop_ids.append(im_end_id)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# EXTRACT DIRECTION FROM GENERATED TEXT

def extract_direction(text: str) -> str:
    """
    Direction is the last token of the sentence.
    e.g. "Place Stanislas is approximately 312.5 meters from Café du Commerce,
          to the north" -> "north"
    """
    words = text.strip().lower().split()
    for word in reversed(words):
        clean = word.strip(".,;:!?")
        if clean in DIRECTION_LABELS:
            return clean
    return ""


# EVALUATION LOOP
def evaluate(model, tokenizer, device, test_dataset) -> pd.DataFrame:
    results = []

    for sample in tqdm(test_dataset, desc="Evaluating test set"):
        generated = generate_completion(model, tokenizer, sample["input"], device)

        pred_direction = extract_direction(generated)
        true_direction = sample["direction"].lower()

        
        dist_str     = str(int(float(sample["distance_m"])))
        has_distance = dist_str in generated

        dir_ok  = (pred_direction == true_direction)
        full_ok = generated.strip().lower() == sample["output"].strip().lower()

        results.append({
            "input":             sample["input"],
            "expected":          sample["output"],
            "generated":         generated,
            "true_direction":    true_direction,
            "pred_direction":    pred_direction,
            "distance_m":        sample["distance_m"],
            "has_distance":      has_distance,
            "direction_correct": dir_ok,
            "full_match":        full_ok,
        })

    return pd.DataFrame(results)


# PLOT

def plot_per_direction(results_df: pd.DataFrame, save_dir: str):
    per_dir = (
        results_df.groupby("true_direction")
        .agg(total=("direction_correct", "count"),
             correct=("direction_correct", "sum"))
    )
    per_dir["accuracy"] = per_dir["correct"] / per_dir["total"]
    per_dir = per_dir.reindex(DIRECTION_LABELS, fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["seagreen" if a >= 0.5 else "tomato" for a in per_dir["accuracy"]]
    bars = ax.bar(per_dir.index, per_dir["accuracy"], color=colors, edgecolor="white")

    for bar, (_, row) in zip(bars, per_dir.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{int(row['correct'])}/{int(row['total'])}",
                ha="center", va="bottom", fontsize=9)

    ax.axhline(0.125, color="gray", linestyle="--", linewidth=1, label="Random (1/8)")
    ax.axhline(0.5,   color="navy", linestyle=":",  linewidth=1, label="50% line")
    ax.set_xlabel("True direction")
    ax.set_ylabel("Accuracy")
    ax.set_title("Test direction prediction accuracy per class")
    ax.set_ylim([0, 1.12])
    ax.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, "per_direction_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved -> {path}")
    return per_dir


# REPORT

def save_report(results_df: pd.DataFrame, per_dir: pd.DataFrame, save_dir: str):
    n            = len(results_df)
    dir_acc      = results_df["direction_correct"].sum() / n
    full_acc     = results_df["full_match"].sum() / n
    dist_mention = results_df["has_distance"].sum() / n
    no_direction = (results_df["pred_direction"] == "").sum()

    lines = [
        "=" * 54,
        "  EVALUATION REPORT — Geospatial LLM (Qwen3-0.6B)",
        "=" * 54,
        "",
        f"  Test samples        : {n}",
        f"  Direction accuracy  : {dir_acc:.4f}  ({results_df['direction_correct'].sum()}/{n})",
        f"  Exact match         : {full_acc:.4f}  ({results_df['full_match'].sum()}/{n})",
        f"  Distance mentioned  : {dist_mention:.4f}  ({results_df['has_distance'].sum()}/{n})",
        f"  No direction found  : {no_direction}  (model failed to end with a direction)",
        "",
        "  Per-direction breakdown:",
    ]

    for direction, row in per_dir.iterrows():
        if row["total"] > 0:
            bar = "#" * int(row["accuracy"] * 20)
            lines.append(
                f"    {direction:12s}: {row['accuracy']:.3f}  [{bar:<20}]"
                f"  ({int(row['correct'])}/{int(row['total'])})"
            )

    lines += ["", "  Sample correct predictions:"]
    for _, row in results_df[results_df["direction_correct"]].head(3).iterrows():
        lines += [f"    Q: {row['input']}", f"    A: {row['generated']}", ""]

    lines += ["  Sample incorrect predictions:"]
    for _, row in results_df[~results_df["direction_correct"]].head(3).iterrows():
        lines += [
            f"    Q: {row['input']}",
            f"    Expected : {row['expected']}",
            f"    Got      : {row['generated']}",
            "",
        ]

    lines.append("=" * 54)
    text = "\n".join(lines)

    path = os.path.join(save_dir, "evaluation_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(); print(text)
    print(f"\n[Report] Saved -> {path}")


# MAIN
def main():
    print("=" * 60)
    print("  Geospatial LLM -- Test Set Evaluation")
    print("=" * 60)

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"'{TEST_FILE}' not found -- run step1 first.")
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"'{ADAPTER_PATH}' not found -- run step2 first.")

    model, tokenizer, device = load_model(ADAPTER_PATH)

    test_dataset = json.load(open(TEST_FILE, encoding="utf-8"))
    print(f"[Data] {len(test_dataset)} test samples loaded from '{TEST_FILE}'")

    # Sanity check: direction is last word in every expected output
    bad = [s for s in test_dataset[:20]
           if s["output"].strip().split()[-1].lower() not in DIRECTION_LABELS]
    if bad:
        print(f"  WARNING: {len(bad)} samples don't end with a direction word — check data")
    else:
        print("  Sanity check OK: direction is last word in all checked samples")

    # Quick single-example test before full loop
    print("\n[Test] Single example preview:")
    sample = test_dataset[0]
    preview = generate_completion(model, tokenizer, sample["input"], device)
    print(f"  Input    : {sample['input']}")
    print(f"  Expected : {sample['output']}")
    print(f"  Generated: {preview}")
    print(f"  Predicted direction: '{extract_direction(preview)}'  "
          f"(true: '{sample['direction']}')")

    # Full evaluation
    print()
    results_df = evaluate(model, tokenizer, device, test_dataset)

    csv_path = os.path.join(OUTPUT_DIR, "test_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[Save] Results -> {csv_path}")

    per_dir = plot_per_direction(results_df, OUTPUT_DIR)
    save_report(results_df, per_dir, OUTPUT_DIR)

    n = len(results_df)
    print(f"\nDone.")
    print(f"  Direction accuracy : {results_df['direction_correct'].sum()/n:.4f}")
    print(f"  Exact match        : {results_df['full_match'].sum()/n:.4f}")


if __name__ == "__main__":
    main()

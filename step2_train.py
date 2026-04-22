import os
import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# CONFIG

MODEL_NAME       = "Qwen/Qwen3-0.6B"
OUTPUT_DIR       = "./results"
LOG_DIR          = "./logs"
MAX_LENGTH       = 384
BATCH_SIZE       = 1
GRAD_ACCUM_STEPS = 16
NUM_EPOCHS       = 2
LEARNING_RATE    = 1e-4

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
os.makedirs(LOG_DIR, exist_ok=True)


# CARBON TRACKER
class CarbonTracker:
    def __init__(self):
        self.tracker = None
        self.available = False
        self.start_ts = None
        try:
            from codecarbon import EmissionsTracker
            self.tracker = EmissionsTracker(
                project_name="geospatial_llm_finetuning",
                output_dir=OUTPUT_DIR, log_level="error", save_to_file=True)
            self.available = True
            print("[Carbon] CodeCarbon tracker initialised.")
        except ImportError:
            print("[Carbon] Using wall-clock estimate.")

    def start(self):
        self.start_ts = time.time()
        if self.available:
            self.tracker.start()

    def stop(self) -> dict:
        duration_h = (time.time() - self.start_ts) / 3600.0
        report = {"duration_hours": round(duration_h, 4)}
        if self.available:
            emissions = self.tracker.stop()
            report.update({"kg_co2eq": round(emissions, 6),
                           "g_co2eq":  round(emissions * 1000, 3),
                           "source":   "codecarbon"})
        else:
            kwh  = 0.40 * 1.5 * duration_h
            kg   = kwh * 0.276
            report.update({"kg_co2eq": round(kg, 6), "g_co2eq": round(kg*1000, 3),
                           "kwh_estimated": round(kwh, 4),
                           "source": "wall_clock_estimate"})
        return report


# MODEL + TOKENIZER

def load_model_and_tokenizer():
    print(f"[Model] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"[Model] Loading in 4-bit qLoRA mode...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


# DIRECTION TOKEN IDs

def build_direction_token_ids(tokenizer):
    direction_token_ids = {}
    all_direction_ids   = {}

    print("[Metric] Direction -> token ID mapping:")
    for d in DIRECTION_LABELS:
        tids = tokenizer.encode(" " + d, add_special_tokens=False)
        rep  = tids[0]
        direction_token_ids[d] = rep
        for tid in tids:
            all_direction_ids[tid] = d
        print(f"  {d:12s}: rep={rep:6d} '{tokenizer.decode([rep])}' [{len(tids)} tok]")

    seen = {}
    collisions = []
    for d, tid in direction_token_ids.items():
        if tid in seen:
            collisions.append((d, seen[tid], tid))
        else:
            seen[tid] = d

    has_collisions = len(collisions) > 0
    if has_collisions:
        print("  WARNING: collisions detected, using string decode fallback")
    else:
        print("  OK: all IDs unique\n")

    return direction_token_ids, all_direction_ids, has_collisions


# TOKENIZATION

def make_tokenize_fn(tokenizer):
    def tokenize_example(example):
        full = tokenizer(example["text"], truncation=True, max_length=MAX_LENGTH,
                         padding="max_length", return_tensors=None)
        prompt_part = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{example['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompt_toks = tokenizer(prompt_part, truncation=True,
                                max_length=MAX_LENGTH, return_tensors=None)

        labels = full["input_ids"].copy()
        plen   = len(prompt_toks["input_ids"])
        labels[:plen] = [-100] * plen
        for k, mask in enumerate(full["attention_mask"]):
            if mask == 0:
                labels[k] = -100
        full["labels"] = labels
        return full
    return tokenize_example


def preprocess_logits_for_metrics(logits, labels):
    """Reduce (batch, seq, vocab) -> (batch, seq) argmax to prevent GPU OOM."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)



def build_compute_metrics(tokenizer, direction_token_ids, all_direction_ids, has_collisions):
    all_dir_set  = set(all_direction_ids.keys())

    def compute_metrics(eval_pred: EvalPrediction):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        correct = 0
        total   = 0
        skipped = 0

        for i in range(labels.shape[0]):
            valid_pos = np.where(labels[i] != -100)[0]
            if len(valid_pos) == 0:
                continue

            # Find the direction token by scanning backwards
            direction_pos = None
            for pos in reversed(valid_pos):
                if int(labels[i][pos]) in all_dir_set:
                    direction_pos = pos
                    break

            if direction_pos is None or direction_pos == 0:
                skipped += 1
                continue

            true_token_id  = int(labels[i][direction_pos])
            true_direction = all_direction_ids[true_token_id]

            pred_token_id = int(predictions[i, direction_pos - 1])

            if has_collisions:
                pred_word = tokenizer.decode([pred_token_id]).strip().lower()
                match = (pred_word == true_direction) or (pred_word in true_direction)
            else:
                match = (pred_token_id == true_token_id)

            if match:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {
            "direction_accuracy":    round(accuracy, 4),
            "eval_samples_scored":   total,
            "eval_samples_skipped":  skipped,
        }

    return compute_metrics


# PLOT
def plot_curves(log_history, save_dir):
    train_steps, train_losses         = [], []
    eval_steps, eval_losses, eval_acc = [], [], []

    for e in log_history:
        if "loss" in e and "eval_loss" not in e:
            train_steps.append(e["step"]); train_losses.append(e["loss"])
        if "eval_loss" in e:
            eval_steps.append(e["step"]); eval_losses.append(e["eval_loss"])
            if "eval_direction_accuracy" in e:
                eval_acc.append(e["eval_direction_accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_steps, train_losses, label="Train Loss",
                 color="royalblue", linewidth=1.5)
    if eval_losses:
        axes[0].plot(eval_steps, eval_losses, label="Val Loss",
                     color="tomato", linewidth=2, marker="o")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    if eval_acc:
        axes[1].plot(eval_steps[:len(eval_acc)], eval_acc,
                     color="seagreen", linewidth=2, marker="s", label="Direction Accuracy")
        axes[1].axhline(0.125, color="gray", linestyle="--", label="Random (1/8)")
        axes[1].set_xlabel("Step"); axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Direction Prediction Accuracy (Validation)")
        axes[1].set_ylim([0, 1]); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved -> {path}")


# CARBON REPORT

def save_carbon_report(report, train_result, save_dir):
    lines = [
        "=" * 52,
        "  CARBON EMISSION REPORT",
        f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 52,
        f"  Model          : {MODEL_NAME}",
        f"  Training steps : {train_result.global_step}",
        f"  Training loss  : {train_result.training_loss:.4f}",
        "",
        f"  Duration       : {report['duration_hours']:.4f} h ({report['duration_hours']*60:.1f} min)",
        f"  CO2 equivalent : {report['g_co2eq']:.3f} g CO2eq",
        f"  Source         : {report['source']}",
        "",
        f"  ~= {report['g_co2eq']/120:.3f} km driven by an average car",
        "=" * 52,
    ]
    text = "\n".join(lines)
    path = os.path.join(save_dir, "carbon_report.txt")
    with open(path, "w") as f:
        f.write(text)
    print(); print(text)
    print(f"\n[Carbon] Saved -> {path}")


# MAIN

def main():
    print("=" * 60)
    print("  Geospatial LLM — Training (v4, off-by-one fix)")
    print("=" * 60)

    for fname in ["train.json", "val.json"]:
        if not os.path.exists(fname):
            raise FileNotFoundError(f"'{fname}' not found — run step1 first.")

    model, tokenizer = load_model_and_tokenizer()
    dir_ids, all_dir_ids, has_collisions = build_direction_token_ids(tokenizer)

    print("[Data] Loading and tokenizing...")
    raw = load_dataset("json", data_files={"train": "train.json", "validation": "val.json"})
    tokenized = raw.map(make_tokenize_fn(tokenizer),
                        remove_columns=raw["train"].column_names)
    print(f"  Train: {len(tokenized['train']):,} | Val: {len(tokenized['validation']):,}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=50,
        logging_dir=LOG_DIR,
        eval_strategy="steps",
        eval_steps=200,
        eval_accumulation_steps=1,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="direction_accuracy",
        greater_is_better=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    compute_metrics = build_compute_metrics(tokenizer, dir_ids, all_dir_ids, has_collisions)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    carbon = CarbonTracker()
    carbon.start()

    print(f"\n[Train] {NUM_EPOCHS} epochs | batch={BATCH_SIZE} | "
          f"grad_accum={GRAD_ACCUM_STEPS} | effective_batch={BATCH_SIZE*GRAD_ACCUM_STEPS}")
    train_result = trainer.train()

    carbon_report = carbon.stop()

    model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    plot_curves(trainer.state.log_history, OUTPUT_DIR)
    save_carbon_report(carbon_report, train_result, OUTPUT_DIR)

    print(f"\nDone. Loss: {train_result.training_loss:.4f} | "
          f"CO2: {carbon_report['g_co2eq']:.3f}g")


if __name__ == "__main__":
    main()

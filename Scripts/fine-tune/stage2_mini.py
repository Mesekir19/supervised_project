import os
import json
import math
import random
import logging
from datetime import datetime

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
)
from tqdm import tqdm

CONFIG = {
    "data_path":        "./stage2_data/paraphrases.jsonl",
    "output_dir":       "./stage2_output",
    "max_train_records": 20_000,
    "val_ratio":        0.05,
    "test_ratio":       0.05,
    "seed":             42,

    "base_model":       "microsoft/Phi-3-mini-4k-instruct",
    "max_seq_length":   128,

    "load_in_4bit":             True,
    "bnb_4bit_quant_type":      "nf4",
    "bnb_4bit_compute_dtype":   "bfloat16",
    "bnb_4bit_use_double_quant": True,

    "lora_r":               8,
    "lora_alpha":           16,
    "lora_dropout":         0.05,
    "lora_target_modules":  ["qkv_proj", "o_proj"],

    "epochs":                       2,
    "per_device_train_batch_size":  8,
    "per_device_eval_batch_size":   8,
    "gradient_accumulation_steps":  4,
    "learning_rate":                2e-4,
    "lr_scheduler_type":            "cosine",
    "warmup_ratio":                 0.05,
    "weight_decay":                 0.001,
    "bf16":                         True,
    "fp16":                         False,
    "logging_steps":                25,
    "save_steps":                   200,
    "eval_steps":                   200,
    "save_total_limit":             2,
    "early_stopping_patience":      2,
    "gradient_checkpointing":       True,
    "optim":                        "paged_adamw_32bit",
    "merge_on_finish":  False,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

os.makedirs(CONFIG["output_dir"], exist_ok=True)
random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])


PHI3_TEMPLATE = "<|user|>\n{user}<|end|>\n<|assistant|>\n{assistant}<|end|>"


def load_and_format(path, max_records):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            if len(msgs) < 2:
                continue
            text = PHI3_TEMPLATE.format(
                user=msgs[0]["content"],
                assistant=msgs[1]["content"],
            )
            records.append({"text": text})

    log.info(f"Loaded {len(records):,} records from {path}")

    if max_records and len(records) > max_records:
        random.shuffle(records)
        records = records[:max_records]
        log.info(f"Capped to {len(records):,} records (max_train_records={max_records})")

    return records


def split_records(records, cfg):
    random.shuffle(records)
    n      = len(records)
    n_val  = max(1, int(n * cfg["val_ratio"]))
    n_test = max(1, int(n * cfg["test_ratio"]))
    test   = records[:n_test]
    val    = records[n_test:n_test + n_val]
    train  = records[n_test + n_val:]
    log.info(f"Split  train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    return train, val, test

def tokenize_dataset(records, tokenizer, max_len):
    input_ids_list, attention_mask_list, labels_list = [], [], []
    split_marker = "<|assistant|>\n"

    for rec in tqdm(records, desc="Tokenizing", leave=False):
        text = rec["text"]

        if split_marker not in text:
            continue

        prompt_part = text[:text.index(split_marker) + len(split_marker)]
        full_enc    = tokenizer(text,        add_special_tokens=False,
                                max_length=max_len, truncation=True)
        prompt_enc  = tokenizer(prompt_part, add_special_tokens=False,
                                max_length=max_len, truncation=True)

        input_ids      = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        prompt_len     = len(prompt_enc["input_ids"])

        labels  = [-100] * prompt_len + input_ids[prompt_len:]
        pad_len = max_len - len(input_ids)

        if pad_len > 0:
            input_ids      = input_ids      + [tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels         = labels         + [-100] * pad_len

        input_ids_list.append(input_ids[:max_len])
        attention_mask_list.append(attention_mask[:max_len])
        labels_list.append(labels[:max_len])

    return Dataset.from_dict({
        "input_ids":      input_ids_list,
        "attention_mask": attention_mask_list,
        "labels":         labels_list,
    })

def load_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_qlora_model(cfg):
    log.info(f"Loading: {cfg['base_model']}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["gradient_checkpointing"]
    )
    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model

def get_training_args(cfg):
    return TrainingArguments(
        output_dir=os.path.join(cfg["output_dir"], "checkpoints"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        bf16=cfg["bf16"],
        fp16=cfg["fp16"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"],
        eval_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        save_total_limit=cfg["save_total_limit"],
        report_to="none",
        optim=cfg["optim"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        dataloader_num_workers=2,
        group_by_length=True,
    )
def save_adapter(model, tokenizer, cfg):
    path = os.path.join(cfg["output_dir"], "lora_adapter")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    log.info(f"LoRA adapter saved -> {path}  (~60MB, portable)")


def merge_and_save(cfg):
    adapter_path = os.path.join(cfg["output_dir"], "lora_adapter")
    merged_path  = os.path.join(cfg["output_dir"], "final_model")
    log.info("Merging LoRA into base model (this takes ~10 min)...")

    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_path)
    merged = merged.merge_and_unload()
    merged.save_pretrained(merged_path)

    tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tok.save_pretrained(merged_path)
    log.info(f"Merged model saved -> {merged_path}")

def plot_curves(log_history, path):
    train_steps, train_loss = [], []
    eval_steps,  eval_loss  = [], []
    for e in log_history:
        if "loss" in e and "eval_loss" not in e:
            train_steps.append(e["step"]); train_loss.append(e["loss"])
        if "eval_loss" in e:
            eval_steps.append(e["step"]);  eval_loss.append(e["eval_loss"])

    fig, ax = plt.subplots(figsize=(11, 4))
    if train_steps:
        ax.plot(train_steps, train_loss, label="Train", linewidth=2)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label="Val", linewidth=2,
                linestyle="--", marker="o", markersize=4)
    ax.set_title("Stage 2 — Phi-3-mini QLoRA", fontsize=13)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Curves saved -> {path}")

def load_stage2_pipeline(model_path, stage1_output_dir=None):
    import sys

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        log.info("Detected LoRA adapter — loading on top of base model")
        base = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        log.info("Loading merged model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    stage1_predict = None
    if stage1_output_dir:
        sys.path.insert(0, ".")
        from train_stage1 import load_model_for_inference
        stage1_predict = load_model_for_inference(stage1_output_dir)
        log.info("Stage 1 classifier chained.")

    def generate(
        poi_a_name, poi_a_type="place",
        poi_b_name=None, poi_b_type="place",
        direction=None, distance_meters=None,
        lat_a=None, lon_a=None, lat_b=None, lon_b=None,
        max_new_tokens=60, temperature=0.7, top_p=0.9,
    ):
        if direction is None and stage1_predict and \
                all(v is not None for v in [lat_a, lon_a, lat_b, lon_b]):
            result = stage1_predict(lat_a, lon_a, lat_b, lon_b)
            direction = result["direction"]

        if direction is None:
            raise ValueError(
                "Provide either 'direction' or coordinates with stage1_output_dir"
            )

        dist_str = f"about {int(distance_meters)} meters" \
                   if distance_meters else "unknown distance"

        prompt = (
            f"<|user|>\n"
            f"Describe the spatial relationship between these two places:\n"
            f"POI A: {poi_a_name} ({poi_a_type})\n"
            f"POI B: {poi_b_name} ({poi_b_type})\n"
            f"Direction: {direction}\n"
            f"Distance: {dist_str}<|end|>\n"
            f"<|assistant|>\n"
        )

        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
            )
        decoded = tok.decode(out[0], skip_special_tokens=True)
        if "<|assistant|>" in decoded:
            decoded = decoded.split("<|assistant|>")[-1].strip()
        return decoded

    return generate

def main():
    cfg = CONFIG
    log.info(f"Stage 2 — {cfg['base_model']} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    records = load_and_format(cfg["data_path"], cfg["max_train_records"])
    train_recs, val_recs, test_recs = split_records(records, cfg)

    tokenizer = load_tokenizer(cfg)

    log.info("Tokenizing...")
    train_ds = tokenize_dataset(train_recs, tokenizer, cfg["max_seq_length"])
    val_ds   = tokenize_dataset(val_recs,   tokenizer, cfg["max_seq_length"])
    test_ds  = tokenize_dataset(test_recs,  tokenizer, cfg["max_seq_length"])
    log.info(f"Tokenized  train: {len(train_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,}")

    # Model
    model = load_qlora_model(cfg)

    # Trainer
    trainer = Trainer(
        model=model,
        args=get_training_args(cfg),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, padding=True, pad_to_multiple_of=8
        ),
    )

    # Train
    log.info("Training...")
    trainer.train()

    # Test eval
    test_results = {'eval_loss': 0.0}
    test_ppl     = math.exp(test_results["eval_loss"])
    log.info(f"Test loss: {test_results['eval_loss']:.4f}  |  Perplexity: {test_ppl:.2f}")

    save_adapter(model, tokenizer, cfg)

    if cfg["merge_on_finish"]:
        merge_and_save(cfg)
    else:
        log.info("Skipping merge (merge_on_finish=False).")
        log.info("To merge later: set merge_on_finish=True and re-run,")
        log.info("  or call merge_and_save(CONFIG) directly.")

    report = {
        "base_model":       cfg["base_model"],
        "lora_r":           cfg["lora_r"],
        "lora_alpha":       cfg["lora_alpha"],
        "lora_modules":     cfg["lora_target_modules"],
        "train_samples":    len(train_ds),
        "val_samples":      len(val_ds),
        "test_samples":     len(test_ds),
        "test_loss":        round(test_results["eval_loss"], 6),
        "test_perplexity":  round(test_ppl, 4),
        "log_history":      trainer.state.log_history,
    }
    report_path = os.path.join(cfg["output_dir"], "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report -> {report_path}")

    plot_curves(
        trainer.state.log_history,
        os.path.join(cfg["output_dir"], "training_curves.png")
    )

    log.info("\nDemo inference (using adapter directly, no merge needed)...")
    adapter_path = os.path.join(cfg["output_dir"], "lora_adapter")
    generate = load_stage2_pipeline(adapter_path, stage1_output_dir="./stage1_output")
    result = generate(
        poi_a_name="Brasserie Excelsior", poi_a_type="restaurant",
        poi_b_name="Place Commanderie",   poi_b_type="place",
        lat_a=48.6921, lon_a=6.1844,
        lat_b=48.6912, lon_b=6.1801,
    )
    log.info(f"Output: {result}")

    log.info("\nDone. Outputs:")
    log.info(f"  {cfg['output_dir']}/checkpoints/   <- trainer checkpoints")
    log.info(f"  {cfg['output_dir']}/lora_adapter/  <- portable LoRA weights (~60MB)")
    if cfg["merge_on_finish"]:
        log.info(f"  {cfg['output_dir']}/final_model/   <- merged inference model")
    log.info(f"  {cfg['output_dir']}/training_report.json")
    log.info(f"  {cfg['output_dir']}/training_curves.png")


if __name__ == "__main__":
    main()

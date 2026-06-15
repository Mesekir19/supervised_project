import os
import re
import sys
import json
import math
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

CONFIG = {
    "gold_standard_path":   "./eval_data/gold_standard.jsonl",
    "pairs_path":           "./nancy_data/nancy_poi_pairs.jsonl",
    "stage1_dir":           "./stage1_output",
    "base_model":           "microsoft/Phi-3-mini-4k-instruct",
    "adapter_path":         "./stage2_output/lora_adapter",
    "output_dir":           "./eval_results",
    "stage1_eval_samples":  1000,
    "max_new_tokens":       60,
    "temperature":          0.7,
    "top_p":                0.9,
    "factual_weights": {
        "direction": 0.50,
        "poi_recall": 0.30,
        "distance":  0.20,
    },
    "seed": 42,
}

DIRECTIONS    = ["NORTH", "SOUTH", "EAST", "WEST"]
LABEL2IDX     = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3}
IDX2LABEL     = {v: k for k, v in LABEL2IDX.items()}
DIRECTION_WORDS = {
    "NORTH": ["north", "northern", "northward", "northwards"],
    "SOUTH": ["south", "southern", "southward", "southwards"],
    "EAST":  ["east",  "eastern",  "eastward",  "eastwards"],
    "WEST":  ["west",  "western",  "westward",  "westwards"],
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
os.makedirs(CONFIG["output_dir"], exist_ok=True)

def mean(vals):
    return round(sum(vals) / len(vals), 4) if vals else 0.0

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def format_distance(meters):
    if meters < 100:    return f"{int(meters)} meters"
    elif meters < 1000: return f"about {int(round(meters/50)*50)} meters"
    elif meters < 2000: return f"about {meters/1000:.1f} km"
    else:               return f"{meters/1000:.1f} km"

def osm_label(t):
    return {"amenity":"amenity","shop":"shop","tourism":"tourist attraction",
            "leisure":"leisure facility","historic":"historic site",
            "office":"office","public_transport":"transit stop"}.get(t,"place")

def load_gold_standard(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def load_pairs(path, min_conf=0.75, min_dist=10):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("confidence_score", 0) >= min_conf and r.get("distance_meters", 0) >= min_dist:
                records.append(r)
    return records

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_stage1_model(stage1_dir):
    sys.path.insert(0, ".")
    for fname in ["train_stage1", "stage1", "prompt_pipeline"]:
        try:
            mod = __import__(fname)
            predict = mod.load_model_for_inference(stage1_dir)
            return predict
        except (ImportError, ModuleNotFoundError, Exception):
            continue
    raise ImportError("Could not find Stage 1 script (train_stage1.py / stage1.py)")

def evaluate_stage1(cfg):
    predict  = load_stage1_model(cfg["stage1_dir"])
    all_pairs = load_pairs(cfg["pairs_path"])

    by_dir = defaultdict(list)
    for r in all_pairs:
        by_dir[r["cardinal_direction"]].append(r)

    n_per_dir = cfg["stage1_eval_samples"]
    sampled   = []
    for d in DIRECTIONS:
        pool  = by_dir[d]
        chunk = random.sample(pool, min(n_per_dir, len(pool)))
        sampled.extend(chunk)

    y_true, y_pred   = [], []
    conf_scores      = []
    results_by_conf  = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in tqdm(sampled):
        true_dir = r["cardinal_direction"]
        result   = predict(r["poi_a"]["lat"], r["poi_a"]["lon"], r["poi_b"]["lat"], r["poi_b"]["lon"])
        pred_dir   = result["direction"]
        confidence = result["confidence"]

        y_true.append(LABEL2IDX[true_dir])
        y_pred.append(LABEL2IDX[pred_dir])
        conf_scores.append(confidence)

        bucket = f"{int(confidence*10)*10}-{int(confidence*10)*10+10}%"
        results_by_conf[bucket]["total"]   += 1
        results_by_conf[bucket]["correct"] += int(pred_dir == true_dir)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class = {}
    for i, d in IDX2LABEL.items():
        per_class[d] = {
            "precision": round(precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, labels=[i], average="macro", zero_division=0), 4),
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])

    fig, ax = plt.subplots(figsize=(7, 6))
    labels  = [IDX2LABEL[i] for i in range(4)]
    cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Stage 1 — Confusion Matrix (%)")
    plt.tight_layout()
    cm_path = os.path.join(cfg["output_dir"], "stage1_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    conf_accuracy = {}
    for bucket, vals in sorted(results_by_conf.items()):
        if vals["total"] > 0:
            conf_accuracy[bucket] = {
                "accuracy": round(vals["correct"] / vals["total"], 4),
                "samples":  vals["total"]
            }

    clf_report = classification_report(y_true, y_pred, target_names=[IDX2LABEL[i] for i in range(4)], output_dict=True)

    report = {
        "timestamp":        datetime.now().isoformat(),
        "eval_samples":     len(sampled),
        "samples_per_dir":  n_per_dir,
        "overall": {
            "accuracy":          round(acc, 4),
            "precision_macro":   round(prec, 4),
            "recall_macro":      round(rec, 4),
            "f1_macro":          round(f1, 4),
            "f1_weighted":       round(f1_w, 4),
            "mean_confidence":   round(mean(conf_scores), 4),
        },
        "per_class":        per_class,
        "confusion_matrix": {
            "labels":  [IDX2LABEL[i] for i in range(4)],
            "matrix":  cm.tolist(),
            "matrix_pct": cm_pct.tolist(),
        },
        "per_confidence_bucket": conf_accuracy,
        "classification_report": clf_report,
    }

    return report, predict

def load_stage2_model(cfg):
    tok  = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, cfg["adapter_path"])
    model.eval()
    return model, tok

def clean_output(decoded):
    for marker in ["<|assistant|>", "[/INST]", "<|start_header_id|>assistant<|end_header_id|>"]:
        if marker in decoded:
            decoded = decoded.split(marker)[-1]
    for token in ["<|end|>", "<|eot_id|>", "</s>", "<|endoftext|>", "<|end_of_text|>", "<|user|>"]:
        decoded = decoded.replace(token, "")
    skip = ("poi a:", "poi b:", "direction:", "distance:", "describe the", "<|")
    lines  = decoded.strip().split("\n")
    clean  = [l.strip() for l in lines if l.strip() and not any(l.lower().strip().startswith(s) for s in skip)]
    result = " ".join(clean).strip()
    if any(p in result.lower() for p in ("poi a:", "describe the spatial")):
        sentences = re.split(r"(?<=[.!?])\s+", result)
        result    = sentences[-1] if sentences else result
    words = result.split()
    if words and re.match(r"^\d", words[0]) and len(words) > 3:
        result = " ".join(words[1:]).strip()
    return result

def generate_sentence(model, tok, record, direction, cfg):
    ta       = osm_label(record["poi_a"].get("type", "place"))
    tb       = osm_label(record["poi_b"].get("type", "place"))
    dist_str = format_distance(record["distance_meters"])
    prompt   = (
        f"<|user|>\n"
        f"Describe the spatial relationship between these two places:\n"
        f"POI A: {record['poi_a']['name']} ({ta})\n"
        f"POI B: {record['poi_b']['name']} ({tb})\n"
        f"Direction: {direction}\n"
        f"Distance: {dist_str}"
        f"<|end|>\n<|assistant|>\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            do_sample=True,
            use_cache=False,
            pad_token_id=tok.eos_token_id,
        )
    decoded = tok.decode(out[0], skip_special_tokens=False)
    return clean_output(decoded)

def s2_direction(sentence, expected):
    s       = sentence.lower()
    correct = DIRECTION_WORDS[expected]
    wrong   = [w for d, ws in DIRECTION_WORDS.items() if d != expected for w in ws]
    has_c   = any(w in s for w in correct)
    has_w   = any(w in s for w in wrong)
    if has_c and not has_w: return 1.0
    elif has_c and has_w:   return 0.5
    else:                   return 0.0

def s2_poi_recall(sentence, a, b):
    s = sentence.lower()
    return sum([a.lower() in s, b.lower() in s]) / 2.0

def s2_distance_recall(sentence, dist_m):
    nums = re.findall(r"\d+(?:\.\d+)?", sentence)
    for n in nums:
        v = float(n)
        if dist_m > 0 and abs(v - dist_m) / dist_m < 0.20: return 1.0
        if dist_m >= 1000 and abs(v - dist_m/1000) < 0.3:  return 1.0
    return 0.0

def s2_prompt_leak(sentence):
    return float(any(p in sentence.lower() for p in ["describe the spatial", "poi a:", "poi b:", "direction:", "<|user|>", "<|assistant|>"]))

def s2_bleu(sentence, references):
    smoothie = SmoothingFunction().method1
    hyp = nltk.word_tokenize(sentence.lower())
    if not hyp: return 0.0
    best = 0.0
    for ref in references:
        r = nltk.word_tokenize(ref.lower())
        try:
            best = max(best, sentence_bleu([r], hyp, smoothing_function=smoothie))
        except Exception:
            pass
    return round(best, 4)

def s2_semantic(sentence, references, embedder):
    if not sentence or not references: return 0.0
    try:
        ge   = embedder.encode(sentence,    convert_to_tensor=True)
        re_  = embedder.encode(references,  convert_to_tensor=True)
        sims = util.cos_sim(ge, re_)
        return round(float(sims.max()), 4)
    except Exception:
        return 0.0

def s2_diversity(sentences):
    all_bigrams = []
    for s in sentences:
        toks    = nltk.word_tokenize(s.lower())
        bigrams = list(zip(toks, toks[1:]))
        all_bigrams.extend(bigrams)
    if not all_bigrams: return 0.0
    return round(len(set(all_bigrams)) / len(all_bigrams), 4)

def run_stage2_eval(records, model, tok, embedder, cfg, label="stage2", direction_source="gold"):
    results = []

    for rec in tqdm(records):
        if callable(direction_source):
            direction = direction_source(rec)
        else:
            direction = rec["cardinal_direction"]

        sentence  = generate_sentence(model, tok, rec, direction, cfg)
        refs      = rec.get("reference_sentences", [])
        dist_m    = rec["distance_meters"]

        results.append({
            "id":                  rec.get("id", ""),
            "tier":                rec.get("tier", "unknown"),
            "poi_a":               rec["poi_a"]["name"],
            "poi_b":               rec["poi_b"]["name"],
            "true_direction":      rec["cardinal_direction"],
            "used_direction":      direction,
            "distance_meters":     dist_m,
            "reference":           refs[0] if refs else "",
            "generated":           sentence,
            "direction_accuracy":  s2_direction(sentence, rec["cardinal_direction"]),
            "poi_recall":          s2_poi_recall(sentence, rec["poi_a"]["name"], rec["poi_b"]["name"]),
            "distance_recall":     s2_distance_recall(sentence, dist_m),
            "prompt_leak":         s2_prompt_leak(sentence),
            "bleu_score":          s2_bleu(sentence, refs),
            "semantic_similarity": s2_semantic(sentence, refs, embedder),
            "output_length":       len(sentence.split()),
        })

    by_dir    = defaultdict(list)
    by_tier   = defaultdict(list)
    for r in results:
        by_dir[r["true_direction"]].append(r)
        by_tier[r["tier"]].append(r)

    def agg(recs):
        return {
            "n":                    len(recs),
            "direction_accuracy":   mean([r["direction_accuracy"]  for r in recs]),
            "poi_recall":           mean([r["poi_recall"]          for r in recs]),
            "distance_recall":      mean([r["distance_recall"]     for r in recs]),
            "prompt_leak_rate":     mean([r["prompt_leak"]         for r in recs]),
            "bleu_score":           mean([r["bleu_score"]          for r in recs]),
            "semantic_similarity":  mean([r["semantic_similarity"] for r in recs]),
            "factual_score":        mean([
                r["direction_accuracy"] * cfg["factual_weights"]["direction"] +
                r["poi_recall"]         * cfg["factual_weights"]["poi_recall"] +
                r["distance_recall"]    * cfg["factual_weights"]["distance"]
                for r in recs
            ]),
            "diversity_score":      s2_diversity([r["generated"] for r in recs]),
            "avg_output_length":    round(mean([r["output_length"] for r in recs]), 1),
            "min_output_length":    min(r["output_length"] for r in recs),
            "max_output_length":    max(r["output_length"] for r in recs),
        }

    report = {
        "label":          label,
        "timestamp":      datetime.now().isoformat(),
        "eval_samples":   len(results),
        "overall":        agg(results),
        "by_direction":   {d: agg(v) for d, v in by_dir.items()},
        "by_tier":        {t: agg(v) for t, v in by_tier.items()},
        "samples":        results,
    }

    return report

def main():
    cfg = CONFIG
    gold = load_gold_standard(cfg["gold_standard_path"])
    model, tok = load_stage2_model(cfg)
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")

    stage1_report, stage1_predict = evaluate_stage1(cfg)
    save_json(stage1_report, os.path.join(cfg["output_dir"], "stage1_eval.json"))

    stage2_report = run_stage2_eval(gold, model, tok, embedder, cfg, label="STAGE_2_MODEL", direction_source="gold")
    stage2_save = {k: v for k, v in stage2_report.items() if k != "samples"}
    save_json(stage2_save, os.path.join(cfg["output_dir"], "stage2_eval.json"))

    samples_path = os.path.join(cfg["output_dir"], "stage2_eval_samples.jsonl")
    with open(samples_path, "w", encoding="utf-8") as f:
        for s in stage2_report["samples"]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    def pipeline_direction(record):
        result = stage1_predict(record["poi_a"]["lat"], record["poi_a"]["lon"], record["poi_b"]["lat"], record["poi_b"]["lon"])
        return result["direction"]

    pipeline_report = run_stage2_eval(gold, model, tok, embedder, cfg, label="PIPELINE_STAGE1+STAGE2", direction_source=pipeline_direction)

    s1_correct_in_pipeline = sum(1 for r in pipeline_report["samples"] if r["used_direction"] == r["true_direction"])
    pipeline_report["stage1_direction_accuracy_in_pipeline"] = round(s1_correct_in_pipeline / len(pipeline_report["samples"]), 4)

    pipeline_save = {k: v for k, v in pipeline_report.items() if k != "samples"}
    save_json(pipeline_save, os.path.join(cfg["output_dir"], "pipeline_eval.json"))

    full_report = {
        "timestamp":  datetime.now().isoformat(),
        "stage1":     stage1_report,
        "stage2":     stage2_save,
        "pipeline":   pipeline_save,
    }
    save_json(full_report, os.path.join(cfg["output_dir"], "full_eval_report.json"))

if __name__ == "__main__":
    main()
import os
import re
import json
import random
from datetime import datetime

import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

CONFIG = {
    "gold_standard_path": "./eval_data/gold_standard.jsonl",
    "base_model":         "microsoft/Phi-3-mini-4k-instruct",
    "adapter_path":       "./stage2_output/lora_adapter",
    "output_dir":         "./eval_results",
    "max_new_tokens":     60,
    "temperature":        0.7,
    "top_p":              0.9,
    "seed":               42,
}

DIRECTIONS = ["NORTH", "SOUTH", "EAST", "WEST"]
DIRECTION_WORDS = {
    "NORTH": ["north", "northern", "northward"],
    "SOUTH": ["south", "southern", "southward"],
    "EAST":  ["east",  "eastern",  "eastward"],
    "WEST":  ["west",  "western",  "westward"],
}

random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
os.makedirs(CONFIG["output_dir"], exist_ok=True)

def mean(vals):
    return round(sum(vals) / len(vals), 4) if vals else 0.0

def format_distance(m):
    if m < 100:    return f"{int(m)} meters"
    elif m < 1000: return f"about {int(round(m/50)*50)} meters"
    elif m < 2000: return f"about {m/1000:.1f} km"
    else:          return f"{m/1000:.1f} km"

def osm_label(t):
    return {"amenity":"amenity","shop":"shop","tourism":"tourist attraction",
            "leisure":"leisure facility","historic":"historic site",
            "office":"office","public_transport":"transit stop"}.get(t,"place")

def load_gold(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: records.append(json.loads(line))
    return records

def score_direction(sentence, expected):
    s = sentence.lower()
    correct = DIRECTION_WORDS[expected]
    wrong   = [w for d, ws in DIRECTION_WORDS.items() if d != expected for w in ws]
    has_c = any(w in s for w in correct)
    has_w = any(w in s for w in wrong)
    if has_c and not has_w: return 1.0
    elif has_c and has_w:   return 0.5
    else:                   return 0.0

def score_poi_recall(sentence, a, b):
    s = sentence.lower()
    return sum([a.lower() in s, b.lower() in s]) / 2.0

def score_distance_recall(sentence, dist_m):
    nums = re.findall(r"\d+(?:\.\d+)?", sentence)
    for n in nums:
        v = float(n)
        if dist_m > 0 and abs(v - dist_m) / dist_m < 0.20: return 1.0
        if dist_m >= 1000 and abs(v - dist_m/1000) < 0.3:  return 1.0
    return 0.0

def score_prompt_leak(sentence):
    return float(any(p in sentence.lower() for p in
                     ["describe the spatial", "poi a:", "poi b:",
                      "direction:", "<|user|>", "<|assistant|>"]))

def score_bleu(sentence, references):
    smoothie = SmoothingFunction().method1
    hyp = nltk.word_tokenize(sentence.lower())
    if not hyp: return 0.0
    best = 0.0
    for ref in references:
        r = nltk.word_tokenize(ref.lower())
        try: best = max(best, sentence_bleu([r], hyp, smoothing_function=smoothie))
        except: pass
    return round(best, 4)

def score_semantic(sentence, references, embedder):
    if not sentence or not references: return 0.0
    try:
        ge  = embedder.encode(sentence,   convert_to_tensor=True)
        re_ = embedder.encode(references, convert_to_tensor=True)
        return round(float(util.cos_sim(ge, re_).max()), 4)
    except: return 0.0

def diversity_score(sentences):
    all_bigrams = []
    for s in sentences:
        toks = nltk.word_tokenize(s.lower())
        all_bigrams.extend(list(zip(toks, toks[1:])))
    if not all_bigrams: return 0.0
    return round(len(set(all_bigrams)) / len(all_bigrams), 4)

def clean_output(decoded):
    for marker in ["<|assistant|>", "[/INST]", "<|start_header_id|>assistant<|end_header_id|>"]:
        if marker in decoded:
            decoded = decoded.split(marker)[-1]
    for token in ["<|end|>","<|eot_id|>","</s>","<|endoftext|>","<|user|>"]:
        decoded = decoded.replace(token, "")
    skip = ("poi a:","poi b:","direction:","distance:","describe the","<|")
    lines = decoded.strip().split("\n")
    clean = [l.strip() for l in lines if l.strip() and not any(l.lower().strip().startswith(s) for s in skip)]
    result = " ".join(clean).strip()
    words = result.split()
    if words and re.match(r"^\d", words[0]) and len(words) > 3:
        result = " ".join(words[1:]).strip()
    return result

def build_zero_shot_prompt(record):
    poi_a = record["poi_a"]
    poi_b = record["poi_b"]
    ta    = osm_label(poi_a.get("type", "place"))
    tb    = osm_label(poi_b.get("type", "place"))
    dist  = format_distance(record["distance_meters"])
    return (
        f"<|user|>\n"
        f"Describe the spatial relationship between these two places in one natural English sentence:\n"
        f"Place A: {poi_a['name']} ({ta})\n"
        f"Place B: {poi_b['name']} ({tb})\n"
        f"Direction from B to A: {record['cardinal_direction']}\n"
        f"Distance: {dist}"
        f"<|end|>\n<|assistant|>\n"
    )

FEW_SHOT_EXAMPLES = [
    {
        "a": "Brasserie Excelsior", "ta": "restaurant",
        "b": "Place Commanderie",   "tb": "place",
        "dir": "EAST", "dist": "about 300 meters",
        "sentence": "Brasserie Excelsior is located to the east of Place Commanderie, about 300 meters away."
    },
    {
        "a": "Pharmacie des Rives", "ta": "amenity",
        "b": "Place Stanislas",     "tb": "place",
        "dir": "NORTH", "dist": "about 800 meters",
        "sentence": "Heading north from Place Stanislas, you will reach Pharmacie des Rives after about 800 meters."
    },
    {
        "a": "Le Georges",     "ta": "amenity",
        "b": "Gare de Nancy", "tb": "transit stop",
        "dir": "SOUTH", "dist": "about 450 meters",
        "sentence": "Le Georges is positioned south of Gare de Nancy at a distance of about 450 meters."
    },
]

def build_few_shot_prompt(record):
    poi_a = record["poi_a"]
    poi_b = record["poi_b"]
    ta    = osm_label(poi_a.get("type", "place"))
    tb    = osm_label(poi_b.get("type", "place"))
    dist  = format_distance(record["distance_meters"])

    prompt = "<|user|>\nYou will be given the names of two places, their spatial relationship, and the distance between them. Generate one natural English sentence describing the relationship. Here are three examples:\n\n"

    for ex in FEW_SHOT_EXAMPLES:
        prompt += (f"Place A: {ex['a']} ({ex['ta']})\n"
                   f"Place B: {ex['b']} ({ex['tb']})\n"
                   f"Direction from B to A: {ex['dir']}\n"
                   f"Distance: {ex['dist']}\n"
                   f"Sentence: {ex['sentence']}\n\n")

    prompt += (f"Now generate a sentence for:\n"
               f"Place A: {poi_a['name']} ({ta})\n"
               f"Place B: {poi_b['name']} ({tb})\n"
               f"Direction from B to A: {record['cardinal_direction']}\n"
               f"Distance: {dist}"
               f"<|end|>\n<|assistant|>\n")
    return prompt

def build_finetuned_prompt(record):
    poi_a = record["poi_a"]
    poi_b = record["poi_b"]
    ta    = osm_label(poi_a.get("type", "place"))
    tb    = osm_label(poi_b.get("type", "place"))
    dist  = format_distance(record["distance_meters"])
    return (
        f"<|user|>\n"
        f"Describe the spatial relationship between these two places:\n"
        f"POI A: {poi_a['name']} ({ta})\n"
        f"POI B: {poi_b['name']} ({tb})\n"
        f"Direction: {record['cardinal_direction']}\n"
        f"Distance: {dist}"
        f"<|end|>\n<|assistant|>\n"
    )

def generate(model, tok, prompt, cfg):
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

def evaluate_system(name, records, prompt_fn, model, tok, embedder, cfg):
    results = []
    for rec in tqdm(records):
        prompt   = prompt_fn(rec)
        sentence = generate(model, tok, prompt, cfg)
        refs     = rec.get("reference_sentences", [])
        dist_m   = rec["distance_meters"]
        direction = rec["cardinal_direction"]

        results.append({
            "id":                  rec.get("id", ""),
            "direction":           direction,
            "tier":                rec.get("tier", "unknown"),
            "poi_a":               rec["poi_a"]["name"],
            "poi_b":               rec["poi_b"]["name"],
            "generated":           sentence,
            "reference":           refs[0] if refs else "",
            "direction_accuracy":  score_direction(sentence, direction),
            "poi_recall":          score_poi_recall(sentence, rec["poi_a"]["name"], rec["poi_b"]["name"]),
            "distance_recall":     score_distance_recall(sentence, dist_m),
            "prompt_leak":         score_prompt_leak(sentence),
            "bleu_score":          score_bleu(sentence, refs),
            "semantic_similarity": score_semantic(sentence, refs, embedder),
            "output_length":       len(sentence.split()),
        })

    all_sents = [r["generated"] for r in results]
    report = {
        "system":               name,
        "n":                    len(results),
        "direction_accuracy":   mean([r["direction_accuracy"]  for r in results]),
        "poi_recall":           mean([r["poi_recall"]          for r in results]),
        "distance_recall":      mean([r["distance_recall"]     for r in results]),
        "prompt_leak_rate":     mean([r["prompt_leak"]         for r in results]),
        "bleu_score":           mean([r["bleu_score"]          for r in results]),
        "semantic_similarity":  mean([r["semantic_similarity"] for r in results]),
        "diversity_score":      diversity_score(all_sents),
        "avg_output_length":    round(mean([r["output_length"] for r in results]), 1),
        "samples":              results,
    }
    return report

def main():
    cfg   = CONFIG
    gold  = load_gold(cfg["gold_standard_path"])
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    base_model.eval()

    zs_report = evaluate_system("base_zero_shot", gold, build_zero_shot_prompt, base_model, tok, embedder, cfg)
    fs_report = evaluate_system("base_few_shot", gold, build_few_shot_prompt, base_model, tok, embedder, cfg)

    del base_model
    torch.cuda.empty_cache()

    base2 = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    ft_model = PeftModel.from_pretrained(base2, cfg["adapter_path"])
    ft_model.eval()

    ft_report = evaluate_system("finetuned_qlora", gold, build_finetuned_prompt, ft_model, tok, embedder, cfg)

    systems = [
        ("Base — Zero-Shot",  zs_report),
        ("Base — Few-Shot",   fs_report),
        ("Fine-tuned QLoRA",  ft_report),
    ]

    metrics = [
        ("Direction Accuracy",   "direction_accuracy"),
        ("POI Name Recall",      "poi_recall"),
        ("Distance Recall",      "distance_recall"),
        ("Prompt Leak Rate",     "prompt_leak_rate"),
        ("BLEU Score",           "bleu_score"),
        ("Semantic Similarity",  "semantic_similarity"),
        ("Diversity Score",      "diversity_score"),
        ("Avg Output Length",    "avg_output_length"),
    ]

    full = {
        "timestamp":   datetime.now().isoformat(),
        "base_zero_shot": {k: v for k, v in zs_report.items() if k != "samples"},
        "base_few_shot":  {k: v for k, v in fs_report.items() if k != "samples"},
        "finetuned":      {k: v for k, v in ft_report.items() if k != "samples"},
    }
    json_path = os.path.join(cfg["output_dir"], "base_vs_finetuned.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("BASE MODEL vs FINE-TUNED MODEL — COMPARISON REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Gold standard pairs: {len(gold)}")
    lines.append("")
    lines.append(f"{'Metric':<26} {'Zero-Shot':>12} {'Few-Shot':>12} {'Fine-tuned':>12}")
    lines.append("-"*64)
    for label, key in metrics:
        vals = [r[key] for _, r in systems]
        if key == "avg_output_length":
            row = "  ".join(f"{v:>11.1f}" for v in vals)
        else:
            row = "  ".join(f"{v*100:>11.1f}%" for v in vals)
        lines.append(f"{label:<26} {row}")

    txt_path = os.path.join(cfg["output_dir"], "base_vs_finetuned_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
import os
import json
import math
import random
import argparse
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

CONFIG = {
    "input_path":       "./nancy_data/nancy_poi_pairs.jsonl",
    "output_dir":       "./eval_data",
    "total_pairs":      300,
    "min_confidence":   0.85,
    "seed":             42,
    "tiers": {
        "easy":     {"min_conf": 0.95, "max_dist": 500,  "count": 25},
        "medium":   {"min_conf": 0.85, "max_dist": 2000, "count": 35},
        "hard":     {"min_conf": 0.85, "max_dist": 5000, "count": 15},
    },
}

DIRECTIONS = ["NORTH", "SOUTH", "EAST", "WEST"]

os.makedirs(CONFIG["output_dir"], exist_ok=True)
random.seed(CONFIG["seed"])

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def format_distance(meters):
    if meters < 100:
        return f"{int(meters)} meters"
    elif meters < 1000:
        return f"about {int(round(meters/50)*50)} meters"
    elif meters < 2000:
        return f"about {meters/1000:.1f} km"
    else:
        return f"{meters/1000:.1f} km"

def osm_label(osm_type):
    return {
        "amenity": "amenity", "shop": "shop",
        "tourism": "tourist attraction", "leisure": "leisure facility",
        "historic": "historic site", "office": "office",
        "public_transport": "transit stop",
    }.get(osm_type, "place")

def is_long_name(name, threshold=25):
    return len(name) > threshold

def tier_of(record):
    conf = record["confidence_score"]
    dist = record["distance_meters"]
    name_a = record["poi_a"]["name"]
    name_b = record["poi_b"]["name"]

    if conf >= 0.95 and dist <= 500 and not is_long_name(name_a) and not is_long_name(name_b):
        return "easy"
    elif is_long_name(name_a) or is_long_name(name_b) or dist > 2000:
        return "hard"
    else:
        return "medium"

SENTENCE_TEMPLATES = {
    "NORTH": [
        lambda a, b, d, ta, tb: f"{a} is located to the north of {b}.",
        lambda a, b, d, ta, tb: f"Heading north from {b}, you will reach {a} after {d}.",
        lambda a, b, d, ta, tb: f"{b} has {a} to its north, about {d} away.",
    ],
    "SOUTH": [
        lambda a, b, d, ta, tb: f"{a} is located to the south of {b}.",
        lambda a, b, d, ta, tb: f"Heading south from {b}, you will reach {a} after {d}.",
        lambda a, b, d, ta, tb: f"{b} has {a} to its south, about {d} away.",
    ],
    "EAST": [
        lambda a, b, d, ta, tb: f"{a} is located to the east of {b}.",
        lambda a, b, d, ta, tb: f"Heading east from {b}, you will reach {a} after {d}.",
        lambda a, b, d, ta, tb: f"{b} has {a} to its east, about {d} away.",
    ],
    "WEST": [
        lambda a, b, d, ta, tb: f"{a} is located to the west of {b}.",
        lambda a, b, d, ta, tb: f"Heading west from {b}, you will reach {a} after {d}.",
        lambda a, b, d, ta, tb: f"{b} has {a} to its west, about {d} away.",
    ],
}

def generate_sentences(record):
    a   = record["poi_a"]["name"]
    b   = record["poi_b"]["name"]
    d   = format_distance(record["distance_meters"])
    ta  = osm_label(record["poi_a"].get("osm_type", "place"))
    tb  = osm_label(record["poi_b"].get("osm_type", "place"))
    DIR = record["cardinal_direction"]

    sentences = []
    for tmpl in SENTENCE_TEMPLATES[DIR]:
        try:
            sentences.append(tmpl(a, b, d, ta, tb))
        except Exception:
            sentences.append("")
    return sentences

def load_pairs(cfg):
    records = []
    with open(cfg["input_path"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("confidence_score", 0) >= cfg["min_confidence"]:
                if r.get("distance_meters", 0) > 10:
                    records.append(r)
    return records

def sample_tiered(records, cfg):
    by_dir_tier = defaultdict(lambda: defaultdict(list))
    for r in records:
        d = r["cardinal_direction"]
        t = tier_of(r)
        by_dir_tier[d][t].append(r)

    sampled = []
    for direction in DIRECTIONS:
        dir_sample = []
        for tier, tier_cfg in cfg["tiers"].items():
            pool  = by_dir_tier[direction][tier]
            count = tier_cfg["count"]
            random.shuffle(pool)
            chunk = pool[:count]
            for r in chunk:
                r["_tier"] = tier
            dir_sample.extend(chunk)
        sampled.extend(dir_sample)

    return sampled

def generate_csv(cfg):
    records = load_pairs(cfg)
    sampled = sample_tiered(records, cfg)

    rows = []
    for i, r in enumerate(tqdm(sampled)):
        sentences = generate_sentences(r)
        while len(sentences) < 3:
            sentences.append("")

        rows.append({
            "id":                   f"nancy_{i+1:04d}",
            "tier":                 r.get("_tier", "medium"),
            "cardinal_direction":   r["cardinal_direction"],
            "confidence_score":     round(r["confidence_score"], 4),
            "distance_meters":      round(r["distance_meters"], 1),
            "poi_a_name":           r["poi_a"]["name"],
            "poi_a_type":           r["poi_a"].get("osm_type", "place"),
            "poi_a_lat":            r["poi_a"]["lat"],
            "poi_a_lon":            r["poi_a"]["lon"],
            "poi_b_name":           r["poi_b"]["name"],
            "poi_b_type":           r["poi_b"].get("osm_type", "place"),
            "poi_b_lat":            r["poi_b"]["lat"],
            "poi_b_lon":            r["poi_b"]["lon"],
            "sentence_1":           sentences[0],
            "sentence_2":           sentences[1],
            "sentence_3":           sentences[2],
            "verified":             "",
            "final_sentence_1":     "",
            "final_sentence_2":     "",
            "final_sentence_3":     "",
            "notes":                "",
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(cfg["output_dir"], "gold_standard_TO_VERIFY.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

def finalize_jsonl(cfg):
    csv_path = os.path.join(cfg["output_dir"], "gold_standard_TO_VERIFY.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    verified = df[df["verified"].str.strip().str.lower().isin(["ok", "edit"])]
    rejected = df[df["verified"].str.strip().str.lower() == "reject"]
    pending  = df[~df["verified"].str.strip().str.lower().isin(["ok", "edit", "reject"])]

    records = []
    stats   = defaultdict(lambda: defaultdict(int))

    for _, row in verified.iterrows():
        direction = str(row["cardinal_direction"]).strip()
        tier      = str(row.get("tier", "medium")).strip()
        status    = str(row["verified"]).strip().lower()

        def resolve(gen_col, final_col):
            final = str(row.get(final_col, "")).strip()
            gen   = str(row.get(gen_col, "")).strip()
            return final if final and final.lower() not in ("nan", "") else gen

        s1 = resolve("sentence_1", "final_sentence_1")
        s2 = resolve("sentence_2", "final_sentence_2")
        s3 = resolve("sentence_3", "final_sentence_3")

        sentences = [s for s in [s1, s2, s3] if s and s.lower() != "nan"]
        if not sentences:
            continue

        record = {
            "id":                   str(row["id"]),
            "tier":                 tier,
            "city":                 "Nancy",
            "verification_status":  status,
            "poi_a": {
                "name":     str(row["poi_a_name"]),
                "type":     str(row.get("poi_a_type", "place")),
                "lat":      float(row["poi_a_lat"]),
                "lon":      float(row["poi_a_lon"]),
            },
            "poi_b": {
                "name":     str(row["poi_b_name"]),
                "type":     str(row.get("poi_b_type", "place")),
                "lat":      float(row["poi_b_lat"]),
                "lon":      float(row["poi_b_lon"]),
            },
            "cardinal_direction":   direction,
            "distance_meters":      float(row["distance_meters"]),
            "confidence_score":     float(row["confidence_score"]),
            "reference_sentences":  sentences,
            "notes":                str(row.get("notes", "")).strip(),
        }
        records.append(record)
        stats[direction][tier] += 1

    out_path = os.path.join(cfg["output_dir"], "gold_standard.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats_path = os.path.join(cfg["output_dir"], "gold_standard_stats.json")
    stats_out  = {
        "total":            len(records),
        "by_direction":     {d: dict(t) for d, t in stats.items()},
        "rejected":         len(rejected),
        "pending":          len(pending),
    }
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.generate:
        generate_csv(CONFIG)
    elif args.finalize:
        finalize_jsonl(CONFIG)
    elif args.status:
        csv_path = os.path.join(CONFIG["output_dir"], "gold_standard_TO_VERIFY.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            verified = df["verified"].str.strip().str.lower().isin(["ok", "edit"]).sum()
            rejected = (df["verified"].str.strip().str.lower() == "reject").sum()
            pending  = len(df) - verified - rejected
            print(f"Total: {len(df)}, Done: {verified}, Rejected: {rejected}, Pending: {pending}")
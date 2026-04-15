import os
import json
import random
import logging
from collections import Counter
from tqdm import tqdm

CONFIG = {
    "input_path":           "./nancy_data/nancy_poi_pairs.jsonl",
    "output_dir":           "./stage2_data",
    "max_pairs":            50_000,
    "n_variants":           10,
    "balance_directions":   True,
    "min_confidence":       0.75,
    "seed":                 42,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

os.makedirs(CONFIG["output_dir"], exist_ok=True)
random.seed(CONFIG["seed"])

def _opposite(DIR):
    return {"NORTH": "south", "SOUTH": "north", "EAST": "west", "WEST": "east"}[DIR]


def _format_distance(meters):
    if meters < 100:
        return f"{int(meters)} meters"
    elif meters < 1000:
        rounded = round(meters / 50) * 50
        return f"about {int(rounded)} meters"
    elif meters < 2000:
        return f"about {meters / 1000:.1f} km"
    else:
        return f"{meters / 1000:.1f} km"


def _osm_label(osm_type):
    return {
        "amenity":          "amenity",
        "shop":             "shop",
        "tourism":          "tourist attraction",
        "leisure":          "leisure facility",
        "historic":         "historic site",
        "office":           "office",
        "public_transport": "transit stop",
    }.get(osm_type, "place")



TEMPLATES = [
    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"{a} is located to the {dir} of {b}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"{a} lies {dir} of {b}, about {d} away.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"Heading {dir} from {b}, you will reach {a}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"{b} has {a} to its {dir}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"{Dir} of {b} stands {a}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"{a}, a {ta}, sits to the {dir} of {b}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"From {b}, {a} is just {d} to the {dir}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"{a} is a {dir}ward neighbor of {b}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"If you walk {dir} from {b}, you will find {a}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"To the {dir} of {b}, {d} out, lies {a}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"{a} is {d} to the {dir} of {b}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"{Dir} of {b}, you will find {a}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"The {ta} {a} is situated {dir} of the {tb} {b}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"{a} can be found {d} {dir} of {b}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"Starting at {b} and heading {dir}, you reach {a} after {d}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"{b} is about {d} to the {_opposite(DIR)} of {a}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"{a} and {b} are {d} apart, with {a} to the {dir}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"The {dir}ern side of {b} is where you find {a}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"{a} is positioned {dir} of {b} at a distance of {d}.",

    lambda a, b, dir, Dir, DIR, d, ta, tb:
        f"Relative to {b}, {a} is to the {dir}, roughly {d} away.",
]


def generate_variants(record, n):
    poi_a = record["poi_a"]
    poi_b = record["poi_b"]
    DIR   = record["cardinal_direction"]

    a   = poi_a["name"]
    b   = poi_b["name"]
    dir = DIR.lower()
    Dir = DIR.capitalize()
    d   = _format_distance(record["distance_meters"])
    ta  = _osm_label(poi_a.get("osm_type", "place"))
    tb  = _osm_label(poi_b.get("osm_type", "place"))

    chosen   = random.sample(TEMPLATES, min(n, len(TEMPLATES)))
    variants = []
    for tmpl in chosen:
        try:
            variants.append(tmpl(a, b, dir, Dir, DIR, d, ta, tb))
        except Exception as e:
            log.warning(f"Template error: {e}")
    return variants


def sample_balanced(records, n, balance):
    if not balance or n is None:
        size = n if n else len(records)
        return random.sample(records, min(size, len(records)))

    by_dir = {"NORTH": [], "SOUTH": [], "EAST": [], "WEST": []}
    for r in records:
        d = r.get("cardinal_direction")
        if d in by_dir:
            by_dir[d].append(r)

    per_dir = n // 4
    sampled = []
    for direction, recs in by_dir.items():
        random.shuffle(recs)
        chunk = recs[:per_dir]
        sampled.extend(chunk)
        log.info(f"  {direction}: {len(chunk):,} pairs  (pool: {len(recs):,})")

    random.shuffle(sampled)
    log.info(f"Total sampled: {len(sampled):,} pairs")
    return sampled

def to_training_record(record, sentence):
    poi_a = record["poi_a"]
    poi_b = record["poi_b"]
    user_content = (
        f"Describe the spatial relationship between these two places:\n"
        f"POI A: {poi_a['name']} ({_osm_label(poi_a.get('osm_type', 'place'))})\n"
        f"POI B: {poi_b['name']} ({_osm_label(poi_b.get('osm_type', 'place'))})\n"
        f"Direction: {record['cardinal_direction']}\n"
        f"Distance: {_format_distance(record['distance_meters'])}"
    )
    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": sentence.strip()},
        ],
        "metadata": {
            "poi_a":              poi_a["name"],
            "poi_b":              poi_b["name"],
            "cardinal_direction": record["cardinal_direction"],
            "distance_meters":    record["distance_meters"],
            "confidence_score":   record["confidence_score"],
        }
    }

def print_and_save_stats(sampled, total_written, output_dir):
    dir_counts   = Counter(r["cardinal_direction"] for r in sampled)
    dist_buckets = {"<100m": 0, "100-500m": 0, "500m-1km": 0, "1km-5km": 0}
    for r in sampled:
        d = r["distance_meters"]
        if d < 100:       dist_buckets["<100m"] += 1
        elif d < 500:     dist_buckets["100-500m"] += 1
        elif d < 1000:    dist_buckets["500m-1km"] += 1
        else:             dist_buckets["1km-5km"] += 1

    stats = {
        "total_pairs":            len(sampled),
        "total_training_records": total_written,
        "variants_per_pair":      CONFIG["n_variants"],
        "direction_distribution": dict(dir_counts),
        "distance_distribution":  dist_buckets,
    }

    print("\n" + "=" * 50)
    print("  DATASET STATS")
    print("=" * 50)
    print(f"  Pairs processed:    {len(sampled):,}")
    print(f"  Training records:   {total_written:,}")
    print(f"  Variants per pair:  {CONFIG['n_variants']}")
    print("\n  Cardinal distribution:")
    for k, v in dir_counts.items():
        pct = v / len(sampled) * 100
        bar = "#" * int(pct / 2)
        print(f"    {k:<6}  {v:>8,}  ({pct:5.1f}%)  {bar}")
    print("\n  Distance distribution:")
    for k, v in dist_buckets.items():
        pct = v / len(sampled) * 100
        print(f"    {k:<12}  {v:>8,}  ({pct:5.1f}%)")
    print("=" * 50 + "\n")

    path = os.path.join(output_dir, "dataset_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats saved -> {path}")

def main():
    cfg = CONFIG

    # Load
    log.info(f"Loading pairs from {cfg['input_path']}...")
    records = []
    with open(cfg["input_path"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("confidence_score", 0) >= cfg["min_confidence"]:
                records.append(r)
    log.info(f"Loaded {len(records):,} records (confidence >= {cfg['min_confidence']})")

    # Sample
    log.info("Sampling pairs...")
    sampled = sample_balanced(records, cfg["max_pairs"], cfg["balance_directions"])

    # Output paths
    train_path = os.path.join(cfg["output_dir"], "paraphrases.jsonl")
    raw_path   = os.path.join(cfg["output_dir"], "paraphrases_raw.jsonl")

    total_written = 0

    log.info(f"Generating {cfg['n_variants']} variants per pair...")

    with open(train_path, "w", encoding="utf-8") as ft, \
         open(raw_path,   "w", encoding="utf-8") as fr:

        for record in tqdm(sampled, desc="Generating"):
            variants = generate_variants(record, cfg["n_variants"])

            # Raw output
            fr.write(json.dumps({
                "poi_a":     record["poi_a"]["name"],
                "poi_b":     record["poi_b"]["name"],
                "direction": record["cardinal_direction"],
                "distance":  record["distance_meters"],
                "variants":  variants,
            }, ensure_ascii=False) + "\n")

            for sentence in variants:
                ft.write(json.dumps(
                    to_training_record(record, sentence),
                    ensure_ascii=False
                ) + "\n")
                total_written += 1

    log.info(f"Written: {total_written:,} training records -> {train_path}")
    log.info(f"Written: {len(sampled):,} raw records      -> {raw_path}")

    print_and_save_stats(sampled, total_written, cfg["output_dir"])

    print("Sample output for first pair:")
    print("-" * 55)
    preview   = sampled[0]
    variants  = generate_variants(preview, cfg["n_variants"])
    print(f"  POI A:     {preview['poi_a']['name']}")
    print(f"  POI B:     {preview['poi_b']['name']}")
    print(f"  Direction: {preview['cardinal_direction']}")
    print(f"  Distance:  {_format_distance(preview['distance_meters'])}")
    print()
    for i, v in enumerate(variants, 1):
        print(f"  {i:>2}. {v}")
    print("-" * 55)


if __name__ == "__main__":
    main()

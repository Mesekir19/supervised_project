import os
import math
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict

# CONFIG

PLACE_NAME  = "Nancy, France"
RANDOM_SEED = 42
DIST_MIN_M  = 50
DIST_MAX_M  = 3_000
OUTPUT_DIR  = "."


DIRECTION_LABELS = ["north", "east", "south", "west"]

# Balanced sizes: 4 dirs x per_dir = total
TRAIN_PER_DIR = 2500   # 4 x 2500 = 10,000
VAL_PER_DIR   = 124    # 4 x 124  =    496
TEST_PER_DIR  = 124    # 4 x 124  =    496

SYSTEM_PROMPT = (
    "You are a geospatial assistant specialized in spatial reasoning about "
    "points of interest. When asked about the relationship between two POIs, "
    "respond with a factual sentence that includes the distance in meters and "
    "the cardinal direction."
)

# Distance buckets: (min_m, max_m, label)
DISTANCE_BUCKETS = [
    (50,   200,  100),
    (200,  400,  300),
    (400,  700,  550),
    (700,  1100, 900),
    (1100, 1700, 1400),
    (1700, 3000, 2300),
]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# GEOMETRY

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def get_direction_4(lat1, lon1, lat2, lon2) -> str:
    """
    4-way cardinal direction using atan2.
    Boundaries at 45°, 135°, 225°, 315° — clean 90° sectors.
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    angle = math.degrees(math.atan2(dlon, dlat)) % 360

    if angle < 45 or angle >= 315:
        return "north"
    elif angle < 135:
        return "east"
    elif angle < 225:
        return "south"
    else:
        return "west"


def get_distance_label(dist_m: float) -> int:
    for lo, hi, label in DISTANCE_BUCKETS:
        if lo <= dist_m < hi:
            return label
    return int(round(dist_m / 100) * 100)


def get_bucket_id(dist_m: float) -> int:
    for i, (lo, hi, _) in enumerate(DISTANCE_BUCKETS):
        if lo <= dist_m < hi:
            return i
    return len(DISTANCE_BUCKETS)


# POI EXTRACTION

def extract_pois(place_name: str) -> pd.DataFrame:
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("Run: pip install osmnx")

    print(f"[1/5] Fetching POIs for '{place_name}'...")
    tags = {"amenity": True, "tourism": True, "leisure": True,
            "shop": True, "historic": True}

    pois = ox.features_from_place(place_name, tags)
    pois = pois[pois.geometry.type == "Point"].copy()
    pois["latitude"]  = pois.geometry.y
    pois["longitude"] = pois.geometry.x

    keep_cols = ["name", "latitude", "longitude"]
    for tag in ["amenity", "tourism", "leisure", "shop", "historic"]:
        if tag in pois.columns:
            keep_cols.append(tag)

    df = pois[keep_cols].dropna(subset=["name"]).copy()

    def get_category(row):
        for tag in ["amenity", "tourism", "leisure", "shop", "historic"]:
            if tag in row.index and pd.notna(row[tag]):
                return str(row[tag])
        return "place"

    df["category"] = df.apply(get_category, axis=1)
    df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "pois.csv"), index=False)
    print(f"    {len(df)} unique POIs extracted")
    return df

# POI SPLIT

def split_pois(df: pd.DataFrame):
    print("[2/5] Splitting POIs (no overlap)...")
    pois = df["name"].unique().copy()
    np.random.shuffle(pois)
    n = len(pois)
    train = set(pois[:int(0.8*n)])
    val   = set(pois[int(0.8*n):int(0.9*n)])
    test  = set(pois[int(0.9*n):])
    assert not (train & val) and not (train & test) and not (val & test)
    print(f"    Train: {len(train)} | Val: {len(val)} | Test: {len(test)} POIs")
    return train, val, test


# RELATION GENERATION

def generate_relations(df, poi_subset, label, add_reverse=False):
    """
    Generate pairwise relations, bucketed by (direction, dist_bucket).
    add_reverse: if True, also add the B->A relation (for train only).
    """
    sub = df[df["name"].isin(poi_subset)].reset_index(drop=True)
    buckets = defaultdict(list)

    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            lat1, lon1 = sub.iloc[i]["latitude"], sub.iloc[i]["longitude"]
            lat2, lon2 = sub.iloc[j]["latitude"], sub.iloc[j]["longitude"]

            dist = haversine(lat1, lon1, lat2, lon2)
            if not (DIST_MIN_M < dist < DIST_MAX_M):
                continue

            direction  = get_direction_4(lat1, lon1, lat2, lon2)
            bucket_id  = get_bucket_id(dist)
            dist_label = get_distance_label(dist)

            rel = {
                "poi_a":      sub.iloc[i]["name"],
                "poi_b":      sub.iloc[j]["name"],
                "distance_m": dist_label,
                "direction":  direction,
                "bucket_id":  bucket_id,
            }
            buckets[(direction, bucket_id)].append(rel)

            # Reverse pair: B->A has the opposite direction
            if add_reverse:
                rev_direction = get_direction_4(lat2, lon2, lat1, lon1)
                rev_rel = {
                    "poi_a":      sub.iloc[j]["name"],
                    "poi_b":      sub.iloc[i]["name"],
                    "distance_m": dist_label,
                    "direction":  rev_direction,
                    "bucket_id":  bucket_id,
                }
                buckets[(rev_direction, bucket_id)].append(rev_rel)

    dir_totals = defaultdict(int)
    for (d, b), items in buckets.items():
        dir_totals[d] += len(items)
    print(f"    [{label}] Relations: "
          + ", ".join(f"{d}:{dir_totals[d]}" for d in DIRECTION_LABELS))
    return dict(buckets)


# BALANCED SAMPLING

def balanced_sample(buckets, per_dir, label):
    n_buckets = len(DISTANCE_BUCKETS)
    per_cell  = max(1, per_dir // n_buckets)
    sampled   = []

    for direction in DIRECTION_LABELS:
        dir_samples = []

        for bucket_id in range(n_buckets):
            available = buckets.get((direction, bucket_id), [])
            if not available:
                continue
            need   = per_cell
            chosen = random.choices(available, k=need) if len(available) < need \
                     else random.sample(available, need)
            dir_samples.extend(chosen)

        # Top-up or trim to exactly per_dir
        if len(dir_samples) < per_dir:
            all_dir = []
            for b in range(n_buckets):
                all_dir.extend(buckets.get((direction, b), []))
            if all_dir:
                extra = random.choices(all_dir, k=per_dir - len(dir_samples))
                dir_samples.extend(extra)
        if len(dir_samples) > per_dir:
            dir_samples = random.sample(dir_samples, per_dir)

        sampled.extend(dir_samples)

    random.shuffle(sampled)
    print(f"    [{label}] {len(sampled)} total ({per_dir} per direction x {len(DIRECTION_LABELS)})")
    return sampled


# FORMATTING

def format_input(row):
    return (f"What is the spatial relationship between {row['poi_a']} "
            f"and {row['poi_b']} in Nancy, France?")

def format_output(row):
    """Direction is always the LAST word"""
    return (f"{row['poi_b']} is approximately {row['distance_m']} meters "
            f"from {row['poi_a']}, to the {row['direction']}")

def to_chat_entry(row):
    user_msg      = format_input(row)
    assistant_msg = format_output(row)
    return {
        "text": (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        ),
        "input":      user_msg,
        "output":     assistant_msg,
        "direction":  row["direction"],
        "distance_m": row["distance_m"],
        "poi_a":      row["poi_a"],
        "poi_b":      row["poi_b"],
    }

def save_split(relations, path):
    data = [to_chat_entry(r) for r in relations]
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return data


# MAIN

def main():
    print("=" * 60)
    print("  Geospatial LLM — Data Generation (v4, 4 directions)")
    print("=" * 60)

    df = extract_pois(PLACE_NAME)
    train_pois, val_pois, test_pois = split_pois(df)

    print("[3/5] Generating relations...")
    # Train gets reverse pairs for extra variety
    train_buckets = generate_relations(df, train_pois, "train", add_reverse=True)
    val_buckets   = generate_relations(df, val_pois,   "val",   add_reverse=False)
    test_buckets  = generate_relations(df, test_pois,  "test",  add_reverse=False)

    print("[4/5] Balanced sampling...")
    train_rel = balanced_sample(train_buckets, TRAIN_PER_DIR, "train")
    val_rel   = balanced_sample(val_buckets,   VAL_PER_DIR,   "val")
    test_rel  = balanced_sample(test_buckets,  TEST_PER_DIR,  "test")

    print("[5/5] Saving...")
    train_data = save_split(train_rel, os.path.join(OUTPUT_DIR, "train.json"))
    val_data   = save_split(val_rel,   os.path.join(OUTPUT_DIR, "val.json"))
    test_data  = save_split(test_rel,  os.path.join(OUTPUT_DIR, "test.json"))

    # Verify
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        bad = [d for d in data if d["output"].split()[-1] not in DIRECTION_LABELS]
        assert len(bad) == 0, f"{name}: {len(bad)} outputs don't end with a direction"

    # Stats
    from collections import Counter
    print("\nDirection distribution (train):")
    dir_counts = Counter(e["direction"] for e in train_data)
    for d in DIRECTION_LABELS:
        bar = "#" * (dir_counts[d] // 100)
        print(f"  {d:8s}: {dir_counts[d]:5d}  {bar}")

    print("\nDistance bucket distribution (train):")
    dist_counts = Counter(e["distance_m"] for e in train_data)
    for lo, hi, label in DISTANCE_BUCKETS:
        bar = "#" * (dist_counts[label] // 100)
        print(f"  {lo:4d}-{hi:4d}m: {dist_counts[label]:5d}  {bar}")

    print(f"\nDone.")
    print(f"  train.json : {len(train_data):,}")
    print(f"  val.json   : {len(val_data):,}")
    print(f"  test.json  : {len(test_data):,}")


if __name__ == "__main__":
    main()
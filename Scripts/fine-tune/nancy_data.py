import os
import json
import math
import itertools
import logging
from tqdm import tqdm
import pandas as pd
import osmnx as ox
import overpy
from shapely.geometry import mapping

CITY         = "Nancy, France"
OUTPUT_DIR   = "./nancy_data"
MIN_CONFIDENCE = 0.75
MAX_DISTANCE_M = 5000

OSM_TAGS = {
    "amenity": True,
    "shop": True,
    "tourism": True,
    "leisure": True,
    "historic": True,
    "office": True,
    "public_transport": True,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2) -> float:
    dlon = math.radians(lon2 - lon1)
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def cardinal_and_confidence(lat1, lon1, lat2, lon2):
    b = bearing(lat1, lon1, lat2, lon2)
    nearest_axis = round(b / 90) * 90
    deviation = abs(b - nearest_axis)
    if deviation > 45:
        deviation = 90 - deviation
    confidence = round(1.0 - (deviation / 45.0), 4)
    directions = {0: "NORTH", 90: "EAST", 180: "SOUTH", 270: "WEST", 360: "NORTH"}
    cardinal = directions[int(nearest_axis) % 360]
    return cardinal, confidence, round(b, 4)


def fetch_pois(city: str) -> pd.DataFrame:
    log.info(f"Fetching POIs for: {city}")
    gdfs = []
    for tag_key, tag_val in OSM_TAGS.items():
        try:
            gdf = ox.features_from_place(city, tags={tag_key: tag_val})
            gdf["primary_tag"] = tag_key
            gdfs.append(gdf)
            log.info(f"  {tag_key}: {len(gdf)} features")
        except Exception as e:
            log.warning(f"  {tag_key}: failed ({e})")

    if not gdfs:
        raise RuntimeError("No POIs fetched")

    combined = pd.concat(gdfs)

    combined = combined[combined.get("name", pd.Series()).notna()]
    combined = combined[combined["name"].astype(str).str.strip() != ""]

    combined["lat"] = combined.geometry.centroid.y
    combined["lon"] = combined.geometry.centroid.x

    keep_cols = ["name", "lat", "lon", "primary_tag", "geometry"]
    for col in ["amenity", "shop", "tourism", "leisure", "historic", "addr:street",
                "addr:housenumber", "opening_hours", "website", "phone"]:
        if col in combined.columns:
            keep_cols.append(col)

    combined = combined[[c for c in keep_cols if c in combined.columns]].copy()
    combined = combined.drop_duplicates(subset=["name", "lat", "lon"])
    combined = combined.reset_index(drop=True)

    log.info(f"Total unique named POIs: {len(combined)}")
    return combined


def export_geojson(df: pd.DataFrame, path: str):
    features = []
    for _, row in df.iterrows():
        props = {k: v for k, v in row.items() if k != "geometry" and pd.notna(v)}
        features.append({
            "type": "Feature",
            "geometry": mapping(row["geometry"]) if hasattr(row.get("geometry", None), "geom_type") else {
                "type": "Point", "coordinates": [row["lon"], row["lat"]]
            },
            "properties": props
        })
    geojson = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    log.info(f"GeoJSON saved → {path} ({len(features)} features)")


def export_poi_jsonl(df: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "name": row["name"],
                "lat": row["lat"],
                "lon": row["lon"],
                "osm_type": row.get("primary_tag", "unknown"),
                "subtype": (
                    row.get("amenity") or row.get("shop") or
                    row.get("tourism") or row.get("leisure") or
                    row.get("historic") or "unknown"
                ),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"POI JSONL saved → {path}")

def generate_pairs(df: pd.DataFrame, path: str):
    pois = df[["name", "lat", "lon", "primary_tag"]].to_dict("records")
    log.info(f"Generating pairs from {len(pois)} POIs...")

    total_pairs = len(pois) * (len(pois) - 1) // 2
    log.info(f"Total candidate pairs: {total_pairs:,}")

    kept = 0
    dropped_distance = 0
    dropped_confidence = 0

    with open(path, "w", encoding="utf-8") as f:
        for poi_a, poi_b in tqdm(itertools.combinations(pois, 2), total=total_pairs):
            dist = haversine_distance(poi_a["lat"], poi_a["lon"], poi_b["lat"], poi_b["lon"])

            if dist > MAX_DISTANCE_M:
                dropped_distance += 1
                continue

            cardinal, confidence, bear = cardinal_and_confidence(
                poi_a["lat"], poi_a["lon"], poi_b["lat"], poi_b["lon"]
            )

            if confidence < MIN_CONFIDENCE:
                dropped_confidence += 1
                continue

            record = {
                "poi_a": {
                    "name": poi_a["name"],
                    "lat": poi_a["lat"],
                    "lon": poi_a["lon"],
                    "osm_type": poi_a.get("primary_tag", "unknown")
                },
                "poi_b": {
                    "name": poi_b["name"],
                    "lat": poi_b["lat"],
                    "lon": poi_b["lon"],
                    "osm_type": poi_b.get("primary_tag", "unknown")
                },
                "cardinal_direction": cardinal,
                "distance_meters": round(dist, 2),
                "bearing_degrees": bear,
                "confidence_score": confidence,
                "sentence": f"{poi_a['name']} is {cardinal} of {poi_b['name']}."
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    log.info(f"Pairs kept:              {kept:,}")
    log.info(f"Dropped (distance):      {dropped_distance:,}")
    log.info(f"Dropped (confidence):    {dropped_confidence:,}")
    log.info(f"Pair JSONL saved → {path}")
    return kept


def print_stats(pairs_path: str):
    counts = {"NORTH": 0, "SOUTH": 0, "EAST": 0, "WEST": 0}
    dist_buckets = {"<100m": 0, "100-500m": 0, "500m-1km": 0, "1km-5km": 0}
    total = 0

    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            counts[r["cardinal_direction"]] += 1
            d = r["distance_meters"]
            if d < 100:       dist_buckets["<100m"] += 1
            elif d < 500:     dist_buckets["100-500m"] += 1
            elif d < 1000:    dist_buckets["500m-1km"] += 1
            else:             dist_buckets["1km-5km"] += 1
            total += 1

    print("\n" + "="*45)
    print("  DATASET STATS")
    print("="*45)
    print(f"  Total pairs:   {total:,}")
    print("\n  Cardinal distribution:")
    for k, v in counts.items():
        pct = v / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"    {k:<6} {v:>8,}  ({pct:5.1f}%)  {bar}")
    print("\n  Distance distribution:")
    for k, v in dist_buckets.items():
        pct = v / total * 100 if total else 0
        print(f"    {k:<12} {v:>8,}  ({pct:5.1f}%)")
    print("="*45 + "\n")


if __name__ == "__main__":
    df = fetch_pois(CITY)

    export_geojson(df, os.path.join(OUTPUT_DIR, "nancy_pois.geojson"))

    export_poi_jsonl(df, os.path.join(OUTPUT_DIR, "nancy_pois.jsonl"))

    pairs_path = os.path.join(OUTPUT_DIR, "nancy_poi_pairs.jsonl")
    generate_pairs(df, pairs_path)

    print_stats(pairs_path)

    print(f"\n All outputs saved to: {OUTPUT_DIR}/")
    print(f"   nancy_pois.geojson       ← Full GeoJSON")
    print(f"   nancy_pois.jsonl         ← One POI per line")
    print(f"   nancy_poi_pairs.jsonl    ← Stage 1 training pairs\n")

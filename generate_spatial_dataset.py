"""
Geospatial Training Data Generator — Nancy, France
Generates spatial relationship sentences from OSM-style GPS points.

    gdf = ox.features_from_place("Nancy, France", {"amenity": True})
    points extracted from gdf with lat/lon columns
"""

import math, json, random

# Real Nancy POIs (lat, lon)
nancy_points = [
    ("Place Stanislas", 48.6937, 6.1836),
    ("Gare de Nancy", 48.6897, 6.1745),
    ("Cathédrale Notre-Dame de Nancy", 48.6927, 6.1831),
    ("Musée des Beaux-Arts de Nancy", 48.6940, 6.1838),
    ("Parc de la Pépinière", 48.6962, 6.1836),
    ("Place de la Carrière", 48.6950, 6.1832),
    ("Hôtel de Ville de Nancy", 48.6937, 6.1836),
    ("Université de Lorraine IDMC", 48.6921, 6.1752),
    ("LORIA", 48.6655, 6.1555),
    ("CHU de Nancy", 48.6845, 6.1921),
    ("Parc de Brabois", 48.6640, 6.1470),
    ("Basilique Saint-Epvre", 48.6955, 6.1820),
    ("Porte de la Craffe", 48.6970, 6.1803),
    ("Place Charles III", 48.6920, 6.1810),
    ("Opéra national de Lorraine", 48.6947, 6.1833),
    ("Musée Lorrain", 48.6958, 6.1814),
    ("Bibliothèque universitaire Nancy", 48.6906, 6.1751),
    ("Marché Central Nancy", 48.6915, 6.1825),
    ("Parc de la Cure d'Air", 48.7010, 6.1900),
    ("Aquarium de Nancy", 48.6962, 6.1837),
    ("Médiathèque de Nancy", 48.6880, 6.1830),
    ("Théâtre de la Manufacture", 48.6970, 6.1760),
    ("Vélodrome de Nancy", 48.7050, 6.2100),
    ("Hippodrome de Nancy", 48.7100, 6.2300),
    ("Zone commerciale Saint-Sébastien", 48.6870, 6.1950),
    ("Lycée Henri Poincaré", 48.6945, 6.1770),
    ("École des Mines de Nancy", 48.6680, 6.1530),
    ("Jardin Botanique du Montet", 48.6620, 6.1490),
    ("Lac de Brabois", 48.6600, 6.1450),
    ("Stade Marcel Picot", 48.6830, 6.1420),
]

random.seed(42)

# ── direction helpers
def compute_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def bearing_to_direction(b):
    """4-direction rule (removes NE/NW/SE/SW as teacher requested)"""
    if b <= 45 or b > 315:  return "north"
    elif b <= 135:           return "east"
    elif b <= 225:           return "south"
    else:                    return "west"

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return R * 2 * math.asin(math.sqrt(a))

#sentence template
OPPOSITES = {"north":"south","south":"north","east":"west","west":"east"}

def make_sentence(a, b, direction, dist_m):
    d = direction
    op = OPPOSITES[d]
    dist_str = f"{int(round(dist_m/100)*100)} meters"
    templates = [
        f"{b} is {d} of {a}.",
        f"{a} is {op} of {b}.",
        f"From {a}, {b} is located to the {d}.",
        f"If you go {d} from {a}, you will reach {b}.",
        f"{b} lies to the {d} of {a}, approximately {dist_str} away.",
        f"Heading {d} from {a} leads you to {b}.",
        f"The location of {b} is {d} relative to {a}.",
    ]
    return random.choice(templates)

#generate dataset
def generate(n):
    data = []
    seen = set()
    attempts = 0
    while len(data) < n and attempts < n * 20:
        attempts += 1
        i, j = random.sample(range(len(nancy_points)), 2)
        if (i, j) in seen:
            continue
        seen.add((i, j))
        a_name, a_lat, a_lon = nancy_points[i]
        b_name, b_lat, b_lon = nancy_points[j]
        dist = haversine_m(a_lat, a_lon, b_lat, b_lon)
        if dist < 100:
            continue
        bearing = compute_bearing(a_lat, a_lon, b_lat, b_lon)
        direction = bearing_to_direction(bearing)
        sentence = make_sentence(a_name, b_name, direction, dist)
        data.append({
            "messages": [
                {"role": "system",    "content": "You are a geospatial assistant. Given two places in Nancy, France, identify their spatial relationship."},
                {"role": "user",      "content": f"Where is {b_name} relative to {a_name}?"},
                {"role": "assistant", "content": sentence}
            ],
            "metadata": {
                "point_a": a_name, "lat_a": a_lat, "lon_a": a_lon,
                "point_b": b_name, "lat_b": b_lat, "lon_b": b_lon,
                "direction": direction,
                "bearing_deg": round(bearing, 2),
                "distance_m": round(dist, 1)
            }
        })
    # pad to n if needed (allow repeats)
    while len(data) < n:
        data.append(random.choice(data[:len(data)]))
    return data[:n]

print("Generating training set  (10,000 samples)...")
train = generate(10000)
print(f"  ✓ {len(train)} training samples")

print("Generating test set (500 samples)...")
test = generate(500)
print(f"  ✓ {len(test)} test samples")

with open("test_spatial_nancy.json","w",encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)

with open("test_spatial_nancy.json","w",encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

#print stats
directions = [d["metadata"]["direction"] for d in train]
from collections import Counter
counts = Counter(directions)
print("\nDirection distribution in training set:")
for k,v in sorted(counts.items()):
    print(f"  {k:6s}: {v:5d} ({v/len(train)*100:.1f}%)")

print("\n── Sample entry")
print(json.dumps(train[0], ensure_ascii=False, indent=2))
print("\nFiles saved: train_spatial_nancy.json + test_spatial_nancy.json")

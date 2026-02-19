import osmnx as ox
from datetime import datetime

# save_path = "../Data/nancy_features.geojson"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"../Data/nancy_features_{timestamp}.geojson"

# tags = {
#     "amenity": ["university", "library"],
#     "highway": "bus_stop"
# }

tags = {
    "amenity": ["university", "student_accommodation", "bicycle_rental"],
    "building": ["university", "dormitory", "hall_of_residence"],
    "residential": "university",
    "highway": "bus_stop",
    "railway": "tram_stop"
}

gdf = ox.features_from_place("Nancy, France", tags=tags)

cols_to_keep = ['name', 'amenity', 'highway', 'geometry']
existing_cols = [c for c in cols_to_keep if c in gdf.columns]
gdf_simplified = gdf[existing_cols]

gdf_simplified.to_file(save_path, driver='GeoJSON')

print(f"Saved {len(gdf_simplified)} objects to {save_path}")
print(gdf_simplified.head())
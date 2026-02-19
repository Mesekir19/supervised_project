import requests
import json
from datetime import datetime

def fetch_nancy_data():
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    bbox = "48.64, 6.12, 48.72, 6.22"
    
    query = f"""
    [out:json];
    (
      nwr["amenity"="university"]({bbox});
      nwr["building"="university"]({bbox});
      
      nwr["amenity"="student_accommodation"]({bbox});
      nwr["building"="dormitory"]({bbox});
      nwr["building"="hall_of_residence"]({bbox});
      nwr["residential"="university"]({bbox});
      
      nwr["highway"="bus_stop"]({bbox});
      nwr["railway"="tram_stop"]({bbox});
      nwr["amenity"="bicycle_rental"]({bbox});
    );
    out center;
    """
    
    response = requests.post(overpass_url, data={'data': query})
    if response:
        print("Data fetched successfully.", response.status_code)
    if response.status_code == 200:
        data = response.json()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"../Data/osm_nancy_data{timestamp}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)    
        print(f"Retrieved {len(data.get('elements', []))} elements.")
        return data
    else:
        print(f"Failed {response.status_code}")
        return None

fetch_nancy_data()
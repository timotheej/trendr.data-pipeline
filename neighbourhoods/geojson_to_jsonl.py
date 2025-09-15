#!/usr/bin/env python3
"""
Convert GeoJSON to JSONL format for SQL generation
"""
import json
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 geojson_to_jsonl.py neighbourhoods_paris.geojson")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = input_file.replace('.geojson', '.jsonl')
    
    with open(input_file, 'r') as f:
        geojson = json.load(f)
    
    with open(output_file, 'w') as f:
        for feature in geojson['features']:
            record = {
                'city_name': 'Paris',
                'name': feature['properties']['name'],
                'type': feature['properties'].get('type', 'admin'),
                'geometry': feature['geometry'],
                'properties': feature['properties']
            }
            f.write(json.dumps(record) + '\n')
    
    print(f"Converted {len(geojson['features'])} features to {output_file}")

if __name__ == "__main__":
    main()
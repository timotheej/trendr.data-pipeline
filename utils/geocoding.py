#!/usr/bin/env python3
"""
Advanced Neighborhood Attribution V2 - Reverse Geocoding & Proximity-based clustering.
Automatically determines neighborhood for POIs based on coordinates.
"""
import sys
import os
import logging
import requests
import time
from typing import List, Dict, Any, Optional, Tuple
from math import radians, sin, cos, sqrt, atan2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.database import SupabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeighborhoodAttributor:
    """Advanced neighborhood attribution using multiple methods."""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.google_api_key = config.GOOGLE_PLACES_API_KEY  # Reuse Places API key
        
        # Montreal neighborhood boundaries (centers + radius for proximity matching)
        self.montreal_neighborhoods = {
            'Mile End': {'center': (45.5230, -73.5960), 'radius_km': 1.2},
            'Plateau Mont-Royal': {'center': (45.5200, -73.5800), 'radius_km': 1.8},
            'Old Montreal': {'center': (45.5040, -73.5540), 'radius_km': 1.0},
            'Downtown': {'center': (45.5020, -73.5675), 'radius_km': 1.5},
            'Little Italy': {'center': (45.5360, -73.6120), 'radius_km': 1.0},
            'Griffintown': {'center': (45.4950, -73.5600), 'radius_km': 0.8},
            'Westmount': {'center': (45.4900, -73.6000), 'radius_km': 1.2},
            'Rosemont': {'center': (45.5500, -73.5700), 'radius_km': 1.5},
            'Outremont': {'center': (45.5250, -73.6100), 'radius_km': 1.0},
            'Verdun': {'center': (45.4600, -73.5700), 'radius_km': 1.2},
            'Saint-Henri': {'center': (45.4760, -73.5890), 'radius_km': 1.0},
            'Notre-Dame-de-Grâce': {'center': (45.4750, -73.6150), 'radius_km': 1.3}
        }
    
    def calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two coordinates in kilometers."""
        R = 6371.0  # Earth's radius in km
        
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def find_neighborhood_by_proximity(self, lat: float, lng: float, city: str = 'Unknown') -> Optional[str]:
        """Find neighborhood by proximity to known centers."""
        city_lower = city.lower()
        
        # For cities we don't have proximity data, return None to use Google geocoding
        if city_lower not in ['montreal', 'montréal']:
            return None
            
        if city_lower in ['montreal', 'montréal']:
            closest_neighborhood = None
            min_distance = float('inf')
            
            for neighborhood, data in self.montreal_neighborhoods.items():
                center_lat, center_lng = data['center']
                radius = data['radius_km']
                
                distance = self.calculate_distance(lat, lng, center_lat, center_lng)
                
                # Check if within radius and closer than previous matches
                if distance <= radius and distance < min_distance:
                    min_distance = distance
                    closest_neighborhood = neighborhood
            
            return closest_neighborhood
        
        return None
    
    def reverse_geocode_google(self, lat: float, lng: float) -> Optional[Dict[str, Any]]:
        """Use Google Geocoding API for reverse geocoding."""
        if not self.google_api_key:
            return None
        
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                'latlng': f"{lat},{lng}",
                'key': self.google_api_key,
                'language': 'en'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'OK' and data['results']:
                return self.parse_google_response(data['results'][0])
            else:
                logger.warning(f"Google Geocoding failed: {data.get('status', 'Unknown')}")
                return None
                
        except Exception as e:
            logger.error(f"Error in reverse geocoding: {e}")
            return None
    
    def parse_google_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Google Geocoding response."""
        parsed = {
            'formatted_address': result.get('formatted_address', ''),
            'neighborhood': None,
            'sublocality': None,
            'locality': None
        }
        
        for component in result.get('address_components', []):
            types = component.get('types', [])
            name = component.get('long_name', '')
            
            if 'neighborhood' in types:
                parsed['neighborhood'] = name
            elif 'sublocality' in types or 'sublocality_level_1' in types:
                parsed['sublocality'] = name  
            elif 'locality' in types:
                parsed['locality'] = name
        
        # Best guess for neighborhood - avoid using city name
        parsed['best_neighborhood'] = (
            parsed['neighborhood'] or 
            parsed['sublocality'] or
            None  # Don't use locality as it's often the city name
        )
        
        return parsed
    
    def normalize_neighborhood_name(self, google_name: str, city: str = 'Montreal') -> str:
        """Normalize Google neighborhood names to our standards."""
        if city.lower() == 'montreal':
            mappings = {
                'Le Plateau-Mont-Royal': 'Plateau Mont-Royal',
                'Plateau-Mont-Royal': 'Plateau Mont-Royal', 
                'The Plateau': 'Plateau Mont-Royal',
                'Vieux-Montréal': 'Old Montreal',
                'Old Port': 'Old Montreal',
                'Centre-Ville': 'Downtown',
                'Centre-ville': 'Downtown',
                'Downtown Montreal': 'Downtown',
                'Petite Italie': 'Little Italy',
                'NDG': 'Notre-Dame-de-Grâce',
                'Notre Dame de Grace': 'Notre-Dame-de-Grâce'
            }
            
            return mappings.get(google_name, google_name)
        
        return google_name
    
    def determine_neighborhood(self, lat: float, lng: float, city: str = 'Montreal') -> Tuple[Optional[str], str]:
        """
        Determine neighborhood using hybrid approach.
        Returns (neighborhood_name, method_used)
        """
        # Method 1: Proximity-based (fast, local knowledge)
        proximity_result = self.find_neighborhood_by_proximity(lat, lng, city)
        if proximity_result:
            return proximity_result, 'proximity'
        
        # Method 2: Google Reverse Geocoding (slower, authoritative)
        geocoding_result = self.reverse_geocode_google(lat, lng)
        if geocoding_result and geocoding_result['best_neighborhood']:
            neighborhood = self.normalize_neighborhood_name(
                geocoding_result['best_neighborhood'], 
                city
            )
            return neighborhood, 'google_geocoding'
        
        return None, 'none'
    
    def attribute_poi_neighborhood(self, poi_id: str, lat: float, lng: float, city: str = 'Montreal') -> bool:
        """Attribute neighborhood to a specific POI."""
        try:
            neighborhood_name, method = self.determine_neighborhood(lat, lng, city)
            
            if neighborhood_name:
                # Update POI with neighborhood name directly (simplified approach)
                result = self.db.client.table('poi')\
                    .update({
                        'neighborhood': neighborhood_name
                    })\
                    .eq('id', poi_id)\
                    .execute()
                
                logger.info(f"✅ POI {poi_id} → {neighborhood_name} (via {method})")
                return True
            
            logger.warning(f"❌ Could not determine neighborhood for POI {poi_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error attributing POI {poi_id}: {e}")
            return False
    
    def test_coordinate_attribution(self) -> None:
        """Test neighborhood attribution with known Montreal coordinates."""
        test_points = [
            (45.5017, -73.5673, "Downtown Montreal"),
            (45.5230, -73.5960, "Mile End"),
            (45.5040, -73.5540, "Old Montreal"),
            (45.5200, -73.5800, "Plateau Mont-Royal"),
            (45.4900, -73.6000, "Westmount")
        ]
        
        print("🧪 Testing Neighborhood Attribution")
        print("=" * 40)
        
        for lat, lng, expected in test_points:
            neighborhood, method = self.determine_neighborhood(lat, lng)
            status = "✅" if neighborhood else "❌"
            print(f"{status} {expected}: {neighborhood} (via {method})")

def main():
    """CLI interface for neighborhood attribution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced neighborhood attribution for POIs')
    parser.add_argument('--test', action='store_true', help='Test coordinate attribution')
    parser.add_argument('--city', default='Montreal', help='City to process')
    
    args = parser.parse_args()
    
    attributor = NeighborhoodAttributor()
    
    if args.test:
        attributor.test_coordinate_attribution()
    else:
        print(f"Neighborhood Attributor V2 ready for {args.city}")
        print("Use --test to validate coordinate attribution")

def enhance_poi_with_location_data(poi_data: dict) -> dict:
    """Enhance POI data with neighborhood and coordinates (legacy compatibility)."""
    try:
        attributor = NeighborhoodAttributor()
        
        # If we have coordinates, determine neighborhood
        if poi_data.get('latitude') and poi_data.get('longitude'):
            neighborhood_name, method = attributor.determine_neighborhood(
                poi_data['latitude'], 
                poi_data['longitude'],
                poi_data.get('city', 'Unknown')
            )
            if neighborhood_name:
                poi_data['neighborhood'] = neighborhood_name
        
        # If we have address but no coordinates, try to get them
        elif poi_data.get('address') and not poi_data.get('latitude'):
            # For now, return as-is since we don't have geocoding from address
            pass
            
        return poi_data
        
    except Exception as e:
        logger.error(f"Error enhancing POI data: {e}")
        return poi_data

if __name__ == "__main__":
    main()
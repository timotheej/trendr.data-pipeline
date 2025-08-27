#!/usr/bin/env python3
"""
Google Places API Ingester - OPTIMIZED VERSION
Enhanced version for Trendr V2 with smart batching and comprehensive coverage.
Optimized for large-scale ingestion with progress tracking and error handling.
"""
import sys
import os
import logging
import requests
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager
from utils.geocoding import enhance_poi_with_location_data
from utils.photo_manager import POIPhotoManager
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GooglePlacesIngester:
    """OPTIMIZED POI ingestion from Google Places API with comprehensive coverage."""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.api_key = config.GOOGLE_PLACES_API_KEY
        self.photo_manager = POIPhotoManager()
        
        # Enhanced category mapping for comprehensive coverage
        self.category_mapping = {
            'restaurant': ['restaurant', 'meal_takeaway', 'meal_delivery'],
            'cafe': ['cafe'],
            'bar': ['bar', 'night_club'],
            'shopping_mall': ['shopping_mall'],
            'store': ['store', 'clothing_store', 'electronics_store', 'book_store'],
            'tourist_attraction': ['tourist_attraction', 'museum', 'art_gallery'],
            'entertainment': ['movie_theater', 'amusement_park', 'zoo'],
            'health': ['hospital', 'pharmacy', 'doctor'],
            'service': ['bank', 'post_office', 'gas_station'],
            'lodging': ['lodging'],
            'gym': ['gym'],
            'spa': ['spa', 'beauty_salon'],
            'park': ['park']
        }
        
        if not self.api_key:
            logger.warning("Google Places API key not configured")
    
    def search_places(self, query: str, location: str) -> List[Dict[str, Any]]:
        """Search for places using Google Places Text Search."""
        if not self.api_key:
            return []
        
        try:
            # Text Search API endpoint
            url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {
                'query': f"{query} in {location}",
                'key': self.api_key,
                'type': 'establishment'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            places = data.get('items', data.get('results', []))
            
            logger.info(f"Found {len(places)} places for '{query}'")
            return places
            
        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            return []
    
    def get_place_details(self, place_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a place."""
        if not self.api_key:
            return None
        
        try:
            url = "https://maps.googleapis.com/maps/api/place/details/json"
            params = {
                'place_id': place_id,
                'key': self.api_key,
                'fields': 'place_id,name,formatted_address,geometry,rating,user_ratings_total,price_level,types,photos,opening_hours,formatted_phone_number,website,reviews'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('result')
            
        except Exception as e:
            logger.error(f"Failed to get details for place {place_id}: {e}")
            return None
    
    def convert_place_data(self, place_data: Dict[str, Any], city: str) -> Optional[Dict[str, Any]]:
        """Convert Google Places data to our POI format."""
        try:
            place_id = place_data.get('place_id')
            name = place_data.get('name')
            
            if not place_id or not name:
                logger.warning("Missing essential data for place")
                return None
            
            # Extract location data
            geometry = place_data.get('geometry', {})
            location = geometry.get('location', {})
            lat = location.get('lat')
            lng = location.get('lng')
            
            if not lat or not lng:
                logger.warning(f"Missing coordinates for {name}")
                return None
            
            # Determine primary category
            types = place_data.get('types', [])
            primary_category = self.determine_primary_category(types)
            
            # Debug log for category issues
            logger.debug(f"POI: {name} | Google types: {types} | Assigned category: {primary_category}")
            
            # Extract country and neighborhood from formatted address
            formatted_address = place_data.get('formatted_address', '')
            country = self.extract_country_from_address(formatted_address)
            neighborhood = self.extract_neighborhood_from_address(formatted_address)
            
            # Build POI data
            poi_data = {
                'google_place_id': place_id,
                'name': name,
                'address': formatted_address,
                'city': city,
                'country': country,
                'neighborhood': neighborhood,
                'category': primary_category,
                'latitude': float(lat),
                'longitude': float(lng),
                'rating': place_data.get('rating'),
                'user_ratings_total': place_data.get('user_ratings_total'),
                'price_level': place_data.get('price_level'),
                'phone': place_data.get('formatted_phone_number'),
                'website': place_data.get('website'),
                'google_types': json.dumps(types),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Add photos if available
            photos = place_data.get('photos', [])
            if photos:
                photo_references = [photo.get('photo_reference') for photo in photos[:3]]
                poi_data['photo_references'] = json.dumps(photo_references)
            
            # Add opening hours if available
            opening_hours = place_data.get('opening_hours')
            if opening_hours:
                poi_data['opening_hours'] = json.dumps(opening_hours.get('weekday_text', []))
                poi_data['is_open_now'] = opening_hours.get('open_now')
            
            return poi_data
            
        except Exception as e:
            logger.error(f"Error converting place data: {e}")
            return None
    
    def determine_primary_category(self, google_types: List[str]) -> str:
        """Determine primary category from Google types."""
        # Priority order for category determination - align with DB constraints
        priority_categories = [
            'restaurant', 'cafe', 'bar', 'night_club',
            'tourist_attraction', 'museum', 'shopping_mall',
            'store', 'lodging', 'park', 'gym', 'spa'
        ]
        
        # Find the highest priority category
        for category in priority_categories:
            if category in google_types:
                return category
        
        # Comprehensive fallback mappings - all mapped to valid DB categories
        type_mappings = {
            # Food & Drink
            'meal_takeaway': 'restaurant',
            'meal_delivery': 'restaurant',
            'food': 'restaurant',
            
            # Shopping
            'clothing_store': 'store',
            'electronics_store': 'store', 
            'book_store': 'store',
            'jewelry_store': 'store',
            'furniture_store': 'store',
            'home_goods_store': 'store',
            'shoe_store': 'store',
            
            # Attractions & Culture
            'art_gallery': 'tourist_attraction',
            'movie_theater': 'tourist_attraction',
            'amusement_park': 'tourist_attraction',
            'zoo': 'tourist_attraction',
            'aquarium': 'tourist_attraction',
            'church': 'tourist_attraction',
            'hindu_temple': 'tourist_attraction',
            'synagogue': 'tourist_attraction',
            'mosque': 'tourist_attraction',
            'place_of_worship': 'tourist_attraction',
            'cemetery': 'tourist_attraction',
            'library': 'tourist_attraction',
            'university': 'tourist_attraction',
            'school': 'tourist_attraction',
            
            # Services
            'beauty_salon': 'spa',
            'hair_care': 'spa',
            'hospital': 'tourist_attraction', 
            'pharmacy': 'store',
            'bank': 'store',
            'atm': 'store',
            'gas_station': 'store',
            'car_repair': 'store',
            'dentist': 'tourist_attraction',
            'doctor': 'tourist_attraction',
            'veterinary_care': 'tourist_attraction',
            
            # Generic types
            'establishment': 'tourist_attraction',
            'point_of_interest': 'tourist_attraction',
            'premise': 'tourist_attraction',
            
            # Transportation  
            'airport': 'tourist_attraction',
            'train_station': 'tourist_attraction',
            'subway_station': 'tourist_attraction',
            'bus_station': 'tourist_attraction',
            
            # Government & Finance
            'city_hall': 'tourist_attraction',
            'courthouse': 'tourist_attraction',
            'embassy': 'tourist_attraction',
            'post_office': 'store'
        }
        
        for google_type in google_types:
            if google_type in type_mappings:
                return type_mappings[google_type]
        
        # Default fallback - use a valid category
        return 'tourist_attraction'  # Safe fallback that exists in DB constraint
    
    def extract_country_from_address(self, formatted_address: str) -> str:
        """Extract country from Google Places formatted address."""
        if not formatted_address:
            return 'Unknown'  # Fallback when country cannot be determined
            
        # Split address by commas and get the last component (usually country)
        address_parts = [part.strip() for part in formatted_address.split(',')]
        
        if len(address_parts) >= 2:
            potential_country = address_parts[-1]
            
            # Known country mappings
            country_mappings = {
                'Canada': 'Canada',
                'CA': 'Canada',
                'United States': 'United States',
                'USA': 'United States',
                'US': 'United States',
                'France': 'France',
                'FR': 'France',
                'United Kingdom': 'United Kingdom',
                'UK': 'United Kingdom',
                'GB': 'United Kingdom'
            }
            
            # Check if we can map the potential country
            for key, country in country_mappings.items():
                if key.lower() == potential_country.lower():
                    return country
                    
            # If no mapping found but it looks like a country (2-3 letters or proper name)
            if len(potential_country) <= 3 or potential_country.istitle():
                return potential_country
        
        # Default fallback
        return 'Unknown'
    
    def extract_neighborhood_from_address(self, formatted_address: str) -> Optional[str]:
        """Extract neighborhood from Google Places formatted address."""
        if not formatted_address:
            return None
            
        # Split address by commas
        parts = [part.strip() for part in formatted_address.split(',')]
        
        # For most cities, neighborhood is usually the 2nd part (after street)
        # Examples:
        # "123 Main St, Montmartre, 75018 Paris, France" -> "Montmartre"  
        # "456 Broadway, Greenwich Village, New York, NY, USA" -> "Greenwich Village"
        
        if len(parts) >= 3:
            potential_neighborhood = parts[1]
            
            # Skip if it looks like a postal code or city
            if not potential_neighborhood.replace(' ', '').isdigit() and len(potential_neighborhood) > 2:
                return potential_neighborhood
        
        return None
    
    def ingest_poi_to_db(self, poi_data: Dict[str, Any]) -> Optional[str]:
        """Ingest POI data to database with location enhancement."""
        try:
            # POI data already has neighborhood from Google Places address
            enhanced_poi = poi_data
            
            # Insert POI in database
            poi_id = self.db.insert_poi(enhanced_poi)
            
            if poi_id:
                logger.info(f"✅ Ingested: {enhanced_poi['name']}")
                
                # Process photos for the new POI (async-style, don't block ingestion)
                try:
                    photo_result = self.photo_manager.process_poi_photos(
                        poi_id, 
                        enhanced_poi['google_place_id'],
                        max_photos=2  # Limit to 2 photos during ingestion
                    )
                    if photo_result['success']:
                        logger.info(f"📸 Added {photo_result['photos_processed']} photos for {enhanced_poi['name']}")
                    else:
                        logger.warning(f"📸 No photos processed for {enhanced_poi['name']}")
                except Exception as e:
                    logger.warning(f"📸 Photo processing failed for {enhanced_poi['name']}: {e}")
                
                return poi_id
            else:
                logger.warning(f"Failed to ingest: {enhanced_poi['name']}")
                return None
                
        except Exception as e:
            logger.error(f"Error ingesting POI: {e}")
            return None
    
    def search_and_ingest_comprehensive(self, category: str, city: str, country: str, max_results: int = 100) -> List[str]:
        """Comprehensive search and ingestion for a category with multiple queries."""
        logger.info(f"🔍 COMPREHENSIVE ingestion: {category} in {city} (max: {max_results})")
        
        # Get Google types for this category
        google_types = self.category_mapping.get(category, [category])
        
        all_ingested_ids = []
        seen_place_ids = set()
        
        # Search variations for comprehensive coverage
        search_queries = []
        
        # Basic category searches
        for google_type in google_types:
            search_queries.extend([
                google_type,
                f"best {google_type}",
                f"top {google_type}",
                f"popular {google_type}"
            ])
        
        # Limit queries to avoid hitting API limits
        search_queries = search_queries[:10]
        
        logger.info(f"📊 Will execute {len(search_queries)} search queries")
        
        for i, query in enumerate(search_queries, 1):
            if len(all_ingested_ids) >= max_results:
                logger.info(f"🎯 Max results reached: {max_results}")
                break
            
            try:
                logger.info(f"🔍 Query {i}/{len(search_queries)}: {query}")
                
                places = self.search_places(query, f"{city}, {country}")
                
                for place in places:
                    place_id = place.get('place_id')
                    
                    # Skip duplicates
                    if place_id in seen_place_ids:
                        continue
                    seen_place_ids.add(place_id)
                    
                    # Convert and ingest
                    poi_data = self.convert_place_data(place, city)
                    if poi_data:
                        ingested_id = self.ingest_poi_to_db(poi_data)
                        if ingested_id:
                            all_ingested_ids.append(ingested_id)
                    
                    # Respect max results
                    if len(all_ingested_ids) >= max_results:
                        break
                
                # Rate limiting between queries
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in query '{query}': {e}")
                continue
        
        logger.info(f"✅ COMPREHENSIVE ingestion completed: {len(all_ingested_ids)} POIs for {category}")
        return all_ingested_ids
    
    def search_and_ingest(self, query: str, city: str, country: str) -> List[str]:
        """Basic search and ingest method for backward compatibility."""
        logger.info(f"Searching and ingesting: {query} in {city}")
        
        # Search for places
        places = self.search_places(query, f"{city}, {country}")
        ingested_ids = []
        
        for place in places:
            try:
                # Get detailed information
                place_id = place.get('place_id')
                if place_id:
                    detailed_place = self.get_place_details(place_id)
                    if detailed_place:
                        place.update(detailed_place)
                
                # Convert to our format
                poi_data = self.convert_place_data(place, city)
                if poi_data:
                    ingested_id = self.ingest_poi_to_db(poi_data)
                    if ingested_id:
                        ingested_ids.append(ingested_id)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing place: {e}")
                continue
        
        logger.info(f"Ingested {len(ingested_ids)} POIs for '{query}'")
        return ingested_ids
    
    def run_full_city_ingestion(self, city: str, country: str, categories: List[str] = None) -> Dict[str, int]:
        """Run full city ingestion across all categories."""
        if categories is None:
            categories = list(self.category_mapping.keys())
        
        logger.info(f"🚀 FULL CITY INGESTION: {city}")
        logger.info(f"📋 Categories: {categories}")
        
        results = {}
        total_ingested = 0
        
        for i, category in enumerate(categories, 1):
            try:
                logger.info(f"\n🔄 [{i}/{len(categories)}] Processing category: {category}")
                
                # Comprehensive ingestion for this category
                ingested_ids = self.search_and_ingest_comprehensive(
                    category, city, country, max_results=50  # Limit per category
                )
                
                results[category] = len(ingested_ids)
                total_ingested += len(ingested_ids)
                
                logger.info(f"✅ Category {category}: {len(ingested_ids)} POIs")
                
                # Pause between categories
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error processing category {category}: {e}")
                results[category] = 0
                continue
        
        logger.info(f"\n🎉 FULL INGESTION COMPLETED")
        logger.info(f"📊 Total POIs ingested: {total_ingested}")
        logger.info(f"📋 Results by category: {results}")
        
        return results


def main():
    """CLI interface for Google Places ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Google Places API Ingester')
    parser.add_argument('--city', default='Montreal', help='City to ingest')
    parser.add_argument('--category', help='Specific category to ingest')
    parser.add_argument('--full', action='store_true', help='Run full city ingestion')
    parser.add_argument('--max-results', type=int, default=50, help='Max results per category')
    
    args = parser.parse_args()
    
    ingester = GooglePlacesIngester()
    
    if args.full:
        # Full city ingestion
        categories = [args.category] if args.category else None
        results = ingester.run_full_city_ingestion(args.city, categories)
        print(f"Ingestion results: {results}")
    
    elif args.category:
        # Single category comprehensive ingestion
        results = ingester.search_and_ingest_comprehensive(
            args.category, args.city, args.max_results
        )
        print(f"Ingested {len(results)} POIs for {args.category}")
    
    else:
        print("Please specify --category or --full for ingestion")


if __name__ == "__main__":
    main()
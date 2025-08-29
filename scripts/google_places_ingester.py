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
from datetime import datetime, date

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager
# from utils.api_limit_manager import APILimitManager  # Not needed
# from utils.geocoding import enhance_poi_with_location_data  # Not needed
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
        # self.api_limit_manager = APILimitManager()  # Not needed
        
        # Store Google types directly - more scalable approach
        # Classification can be done later as a separate step
        
        # Quota gratuit quotidien optimisé
        self.daily_free_quota = {
            'basic_requests': 10000,    # Champs gratuits
            'contact_requests': 5000,   # Website, price_level
            'atmosphere_requests': 1000 # Rating, reviews
        }
        
        if not self.api_key:
            logger.warning("Google Places API key not configured")
    
    def _check_daily_quota(self, api_type: str = 'google_places') -> bool:
        """Vérifier si on peut encore faire des appels API aujourd'hui"""
        try:
            today = date.today().isoformat()
            
            # Récupérer l'usage actuel
            result = self.db.client.table('api_usage')\
                .select('queries_count')\
                .eq('date', today)\
                .eq('api_type', api_type)\
                .execute()
            
            current_usage = result.data[0]['queries_count'] if result.data else 0
            
            # Limite conservative pour maximiser les quotas gratuits
            daily_limit = 950  # 95% de 1000 quota atmosphère (le plus restrictif)
            
            if current_usage >= daily_limit:
                logger.warning(f"Daily quota limit reached: {current_usage}/{daily_limit}")
                return False
                
            return True
            
        except Exception as e:
            logger.debug(f"Could not check quota: {e}")
            return True  # Continue en cas d'erreur de vérification
    
    def _log_api_usage(self, api_type: str = 'google_places'):
        """Logger l'utilisation API pour suivi des quotas"""
        # API usage logging removed - handled by enhanced_proof_scanner
        pass
    
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
            
        # Vérifier les quotas avant l'appel API
        if not self._check_daily_quota('google_places'):
            logger.warning("Daily quota exceeded for Google Places API")
            return None
        
        try:
            url = "https://maps.googleapis.com/maps/api/place/details/json"
            params = {
                'place_id': place_id,
                'key': self.api_key,
                'fields': 'place_id,name,formatted_address,geometry,types,rating,user_ratings_total,website,price_level'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Logger l'utilisation API
            self._log_api_usage('google_places')
            
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
            
            # Use primary Google type directly - more flexible and scalable
            types = place_data.get('types', [])
            primary_category = types[0] if types else 'establishment'
            
            # Debug log for category issues
            logger.debug(f"POI: {name} | Google types: {types} | Assigned category: {primary_category}")
            
            # Extract country and neighborhood 
            formatted_address = place_data.get('formatted_address', '')
            country = self.extract_country_from_address(formatted_address)
            
            # Try geocoding first for more accurate neighborhoods
            neighborhood = self.get_neighborhood_from_coordinates(lat, lng)
            
            # Fallback to address parsing if geocoding fails
            if not neighborhood:
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
    
    # Removed complex category mapping - using Google types directly now
    
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
    
    def get_neighborhood_from_coordinates(self, lat: float, lng: float) -> Optional[str]:
        """Get neighborhood from coordinates using Google Geocoding API."""
        if not self.api_key:
            return None
        
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                'latlng': f"{lat},{lng}",
                'key': self.api_key,
                'language': 'fr',  # French for Paris neighborhoods
                'result_type': 'neighborhood|sublocality'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'OK' and data['results']:
                for result in data['results']:
                    for component in result.get('address_components', []):
                        types = component.get('types', [])
                        if any(t in types for t in ['neighborhood', 'sublocality', 'sublocality_level_1']):
                            neighborhood = component.get('long_name', '')
                            logger.info(f"Found neighborhood via geocoding: {neighborhood}")
                            return neighborhood
            
            return None
            
        except Exception as e:
            logger.warning(f"Geocoding failed for {lat},{lng}: {e}")
            return None
    
    def extract_neighborhood_from_address(self, formatted_address: str) -> Optional[str]:
        """Extract neighborhood from Google Places formatted address - fallback method."""
        if not formatted_address:
            return None
            
        # Split address by commas
        parts = [part.strip() for part in formatted_address.split(',')]
        
        # For Paris specifically, try to extract arrondissement names
        if len(parts) >= 2:
            for part in parts:
                # Look for Paris arrondissement patterns
                if 'arrondissement' in part.lower() or 'ème' in part:
                    return part
                # Look for known Paris neighborhood names
                paris_neighborhoods = [
                    'Marais', 'Montmartre', 'Saint-Germain', 'Latin Quarter', 'Champs-Élysées',
                    'Belleville', 'Pigalle', 'Bastille', 'République', 'Oberkampf', 'Canal Saint-Martin'
                ]
                for neighborhood in paris_neighborhoods:
                    if neighborhood.lower() in part.lower():
                        return neighborhood
        
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
        
        all_ingested_ids = []
        seen_place_ids = set()
        
        # Simple search variations for comprehensive coverage
        search_queries = [
            category,
            f"best {category}",
            f"top {category}",
            f"popular {category}",
            f"{category} near me"
        ]
        
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
    
    def run_neighborhood_rotation_ingestion(self, city: str, country: str, 
                                           target_neighborhood: str = None,
                                           categories: List[str] = None) -> Dict[str, int]:
        """Run rotation-based ingestion for one neighborhood to detect new POIs cost-effectively."""
        if categories is None:
            categories = ['restaurant', 'cafe', 'bar', 'tourist_attraction']
        
        # Paris neighborhoods for rotation
        paris_neighborhoods = [
            "1er arrondissement", "2ème arrondissement", "3ème arrondissement", "4ème arrondissement",
            "5ème arrondissement", "6ème arrondissement", "7ème arrondissement", "8ème arrondissement", 
            "9ème arrondissement", "10ème arrondissement", "11ème arrondissement", "12ème arrondissement",
            "13ème arrondissement", "14ème arrondissement", "15ème arrondissement", "16ème arrondissement",
            "17ème arrondissement", "18ème arrondissement", "19ème arrondissement", "20ème arrondissement"
        ]
        
        # Determine today's neighborhood (rotation based on day of month)
        if target_neighborhood:
            current_neighborhood = target_neighborhood
        else:
            day_of_month = datetime.now().day
            neighborhood_index = (day_of_month - 1) % len(paris_neighborhoods)
            current_neighborhood = paris_neighborhoods[neighborhood_index]
        
        logger.info(f"🎯 NEIGHBORHOOD ROTATION INGESTION: {current_neighborhood}, {city}")
        logger.info(f"📋 Categories: {categories}")
        
        results = {}
        total_ingested = 0
        new_pois_detected = 0
        
        for i, category in enumerate(categories, 1):
            try:
                logger.info(f"\n🔄 [{i}/{len(categories)}] Processing: {category} in {current_neighborhood}")
                
                # Search query for specific neighborhood
                search_query = f"{category} {current_neighborhood} {city}"
                
                # Get current POIs in this neighborhood+category from DB for comparison
                existing_place_ids = self._get_existing_place_ids(city, current_neighborhood, category)
                logger.info(f"📊 Existing POIs in DB: {len(existing_place_ids)} for {category}")
                
                # Search Google Places
                places = self.search_places(search_query, f"{city}, {country}")
                logger.info(f"🔍 Google returned: {len(places)} places")
                
                category_new_pois = 0
                category_total_ingested = 0
                
                for place in places[:3]:  # LIMIT TO 3 PLACES PER CATEGORY FOR COST CONTROL
                    try:
                        place_id = place.get('place_id')
                        if not place_id:
                            continue
                            
                        # Check if this is a new POI
                        is_new_poi = place_id not in existing_place_ids
                        
                        if is_new_poi:
                            # Get detailed information for new POI
                            detailed_place = self.get_place_details(place_id)
                            if detailed_place:
                                place.update(detailed_place)
                            
                            # Convert and ingest
                            poi_data = self.convert_place_data(place, city)
                            if poi_data:
                                # Ensure neighborhood is properly set
                                poi_data['neighborhood'] = current_neighborhood
                                
                                ingested_id = self.ingest_poi_to_db(poi_data)
                                if ingested_id:
                                    category_new_pois += 1
                                    category_total_ingested += 1
                                    logger.info(f"✅ NEW POI: {poi_data['name']} (place_id: {place_id})")
                                    
                                    # Mark as discovered through neighborhood rotation
                                    self._mark_discovery_method(ingested_id, 'neighborhood_rotation')
                        else:
                            logger.debug(f"⏭️ Existing POI: {place.get('name', 'Unknown')}")
                        
                        # Rate limiting
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Error processing place in {category}: {e}")
                        continue
                
                results[category] = category_total_ingested
                total_ingested += category_total_ingested
                new_pois_detected += category_new_pois
                
                logger.info(f"✅ {category}: {category_new_pois} new POIs, {category_total_ingested} total ingested")
                
                # Pause between categories
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error processing category {category}: {e}")
                results[category] = 0
                continue
        
        logger.info(f"\n🎉 ROTATION INGESTION COMPLETED for {current_neighborhood}")
        logger.info(f"📊 Total POIs ingested: {total_ingested}")
        logger.info(f"🆕 New POIs detected: {new_pois_detected}")
        logger.info(f"📋 Results by category: {results}")
        
        return {
            'total_ingested': total_ingested,
            'new_pois_detected': new_pois_detected,
            'neighborhood': current_neighborhood,
            'results_by_category': results
        }
    
    def _get_existing_place_ids(self, city: str, neighborhood: str, category: str) -> set:
        """Get existing Google Place IDs from database for deduplication"""
        try:
            result = self.db.client.table('poi')\
                .select('google_place_id')\
                .eq('city', city)\
                .eq('category', category)\
                .execute()
            
            place_ids = set()
            for poi in result.data:
                if poi.get('google_place_id'):
                    place_ids.add(poi['google_place_id'])
            
            return place_ids
            
        except Exception as e:
            logger.warning(f"Error getting existing place IDs: {e}")
            return set()
    
    def _mark_discovery_method(self, poi_id: str, method: str):
        """Mark how this POI was discovered for analytics"""
        try:
            update_data = {
                'discovery_method': method,
                'discovered_at': datetime.now().isoformat()
            }
            
            self.db.client.table('poi')\
                .update(update_data)\
                .eq('id', poi_id)\
                .execute()
                
        except Exception as e:
            logger.warning(f"Could not mark discovery method: {e}")
    
    def run_full_city_ingestion(self, city: str, country: str, categories: List[str] = None) -> Dict[str, int]:
        """Legacy method - now redirects to neighborhood rotation for cost efficiency"""
        logger.info("🔄 Redirecting to cost-efficient neighborhood rotation ingestion")
        result = self.run_neighborhood_rotation_ingestion(city, country, categories=categories)
        return result['results_by_category']


def main():
    """CLI interface for Google Places ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Google Places API Ingester - Neighborhood Rotation System')
    parser.add_argument('--city', default='Paris', help='City to ingest (default: Paris)')
    parser.add_argument('--neighborhood', help='Specific neighborhood to ingest (optional)')
    parser.add_argument('--category', help='Specific category to ingest')
    parser.add_argument('--rotation', action='store_true', help='Run neighborhood rotation ingestion (default mode)')
    parser.add_argument('--test', action='store_true', help='Test ingestion with 1er arrondissement')
    
    args = parser.parse_args()
    
    ingester = GooglePlacesIngester()
    
    if args.test:
        # Test with first arrondissement
        print("🧪 TESTING NEIGHBORHOOD ROTATION INGESTION")
        results = ingester.run_neighborhood_rotation_ingestion(
            args.city, 'France', target_neighborhood='1er arrondissement'
        )
        print(f"\n🎯 Test Results:")
        print(f"  • Total ingested: {results['total_ingested']}")
        print(f"  • New POIs detected: {results['new_pois_detected']}")
        print(f"  • Neighborhood: {results['neighborhood']}")
        print(f"  • By category: {results['results_by_category']}")
    
    elif args.rotation or not any([args.category]):
        # Default: neighborhood rotation
        results = ingester.run_neighborhood_rotation_ingestion(
            args.city, 'France', target_neighborhood=args.neighborhood
        )
        print(f"\n🎯 Rotation Results:")
        print(f"  • Total ingested: {results['total_ingested']}")
        print(f"  • New POIs detected: {results['new_pois_detected']}")
        print(f"  • Neighborhood: {results['neighborhood']}")
        print(f"  • By category: {results['results_by_category']}")
    
    elif args.category:
        # Single category search
        results = ingester.search_and_ingest(args.category, args.city, 'France')
        print(f"Ingested {len(results)} POIs for {args.category}")
    
    else:
        print("🎯 Use --rotation (default) or --test to try the new neighborhood rotation system")


if __name__ == "__main__":
    main()
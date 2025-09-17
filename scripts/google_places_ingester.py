#!/usr/bin/env python3
"""
KISS Google Places API Ingester
Config-driven H3-based POI ingester following KISS principles
"""
import sys
import os
import logging
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager
from config import get_config

logger = logging.getLogger(__name__)

class GooglePlacesIngester:
    """KISS H3-based Google Places ingester - config-driven, no hardcoded values"""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.config = get_config()
        
        self.db = SupabaseManager()
        if not self.config.google_places_api_key:
            logger.error("Missing GOOGLE_PLACES_API_KEY")
            sys.exit(1)
        self.api_key = self.config.google_places_api_key
        
        # Use config-driven token limits
        self.daily_tokens = self.config.pipeline_config.daily_api_limit
        self.reset_hour_utc = 0  # Standard UTC reset
        self.basic_cost_per_1000 = 17.0  # Google Places pricing
        self._init_token_bucket()
        
        # Use config-driven quality thresholds
        self.rating_min = self.config.google_ingester.quality.rating_min
        self.min_reviews = self.config.google_ingester.quality.min_reviews
    
    def _init_token_bucket(self):
        """Initialize token bucket"""
        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()
        reset_time = datetime.combine(today, datetime.min.time().replace(hour=self.reset_hour_utc)).replace(tzinfo=timezone.utc)
        if now_utc < reset_time:
            reset_time -= timedelta(days=1)
        
        self.last_reset = reset_time
        self.tokens_remaining = self.daily_tokens
        self.basic_calls = 0
        logger.info(f"Token bucket initialized: {self.tokens_remaining} tokens remaining")
    
    def _consume_token(self, call_type: str) -> bool:
        """Consume tokens for API call"""
        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()
        reset_time = datetime.combine(today, datetime.min.time().replace(hour=self.reset_hour_utc)).replace(tzinfo=timezone.utc)
        
        if now_utc >= reset_time and self.last_reset < reset_time:
            logger.info("Daily token reset triggered")
            self._init_token_bucket()
        
        if self.tokens_remaining <= 0:
            logger.warning(f"Token bucket empty! Blocking {call_type} call.")
            return False
        
        self.tokens_remaining -= 1
        if call_type == 'basic':
            self.basic_calls += 1
        return True
    
    def is_type_allowed(self, types: List[str]) -> bool:
        """Check if POI types intersect with allowed types from config"""
        allowed_types = set(self.config.google_ingester.category_map.keys())
        return bool(set(types or []) & allowed_types)
    
    def pass_quality_gate(self, rating: Optional[float], count: Optional[int]) -> bool:
        """Check if POI passes quality thresholds"""
        if rating is None or count is None:
            return False
        return rating >= self.rating_min and count >= self.min_reviews
    
    def get_primary_category(self, types: List[str]) -> Optional[str]:
        """Extract primary category from Google types using config mapping"""
        category_map = self.config.google_ingester.category_map
        for google_type in types:
            if google_type in category_map:
                return category_map[google_type]
        return None
    
    def get_subcategories(self, types: List[str]) -> List[str]:
        """Extract subcategories from Google types using config mapping"""
        subcategory_map = self.config.google_ingester.subcategory_map
        subcategories = []
        for google_type in types:
            if google_type in subcategory_map:
                subcategory = subcategory_map[google_type]
                if subcategory not in subcategories:
                    subcategories.append(subcategory)
        return subcategories
    
    def get_cost_estimate(self) -> Dict[str, Any]:
        """Get current cost estimate and usage stats"""
        basic_cost = (self.basic_calls / 1000.0) * self.basic_cost_per_1000
        return {
            'basic_calls': self.basic_calls,
            'estimate_usd': round(basic_cost, 4),
            'tokens_remaining': self.tokens_remaining
        }
    
    def get_place_details(self, place_id: str) -> Optional[Dict[str, Any]]:
        """Get place details using Places (New) API"""
        if not self._consume_token('basic'):
            return None
        
        try:
            url = f"https://places.googleapis.com/v1/places/{place_id}"
            headers = {
                'X-Goog-Api-Key': self.api_key,
                'X-Goog-FieldMask': 'formattedAddress,internationalPhoneNumber,websiteUri,currentOpeningHours,rating,userRatingCount,priceLevel,photos',
                'Accept-Language': 'en'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Convert new API format to legacy format for compatibility
            converted_data = {
                'place_id': place_id,
                'formatted_address': data.get('formattedAddress'),
                'rating': data.get('rating'),
                'user_ratings_total': data.get('userRatingCount'),
                'price_level': data.get('priceLevel'),
                'website': data.get('websiteUri'),
                'international_phone_number': data.get('internationalPhoneNumber'),
                'formatted_phone_number': data.get('internationalPhoneNumber'),  # Use same field for both
            }
            
            # Convert opening hours format
            current_hours = data.get('currentOpeningHours')
            if current_hours and current_hours.get('periods'):
                converted_data['opening_hours'] = {
                    'periods': current_hours['periods']
                }
            
            # Convert photos format
            if data.get('photos'):
                converted_data['photos'] = [
                    {'photo_reference': photo.get('name')} 
                    for photo in data.get('photos', [])
                ]
            
            return converted_data
                
        except Exception as e:
            logger.error(f"Place details (New) API error: {e}")
            return None
    
    def search_places_nearby(self, location: str, radius: int, place_type: str) -> List[Dict[str, Any]]:
        """Search places using Google Places Nearby Search (New) API"""
        if not self._consume_token('basic'):
            return []
        
        try:
            # Parse lat,lng from location string
            lat, lng = location.split(',')
            lat, lng = float(lat.strip()), float(lng.strip())
            
            url = "https://places.googleapis.com/v1/places:searchNearby"
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': self.api_key,
                'X-Goog-FieldMask': 'places.id,places.displayName,places.location,places.types,places.formattedAddress,places.photos,places.rating,places.userRatingCount',
                'Accept-Language': 'en'
            }
            
            # Request body for Nearby Search (New)
            body = {
                'includedTypes': [place_type],
                'maxResultCount': 20,
                'locationRestriction': {
                    'circle': {
                        'center': {
                            'latitude': lat,
                            'longitude': lng
                        },
                        'radius': float(radius)
                    }
                }
            }
            
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            
            # Convert new API format to legacy format for compatibility
            places = data.get('places', [])
            converted_places = []
            
            for place in places:
                converted_place = {
                    'id': place.get('id'),  # Keep new API format
                    'place_id': place.get('id'),  # Legacy compatibility
                    'name': place.get('displayName', {}).get('text'),
                    'displayName': place.get('displayName'),  # Keep new API format
                    'rating': place.get('rating'),
                    'userRatingCount': place.get('userRatingCount'),
                    'geometry': {
                        'location': {
                            'lat': place.get('location', {}).get('latitude'),
                            'lng': place.get('location', {}).get('longitude')
                        }
                    },
                    'types': place.get('types', []),
                    'formatted_address': place.get('formattedAddress'),
                    'photos': []
                }
                
                # Convert photos format
                if place.get('photos'):
                    converted_place['photos'] = [
                        {'photo_reference': photo.get('name')} 
                        for photo in place.get('photos', [])
                    ]
                
                converted_places.append(converted_place)
            
            return converted_places
            
        except Exception as e:
            logger.error(f"Nearby search (New) failed for '{place_type}' at {location}: {e}")
            return []
    
    def search_places_by_name(self, poi_name: str, city: str) -> List[Dict[str, Any]]:
        """
        Search places by name using Google Places Text Search (New) API
        Used for trending discovery POI validation
        """
        if not self._consume_token('basic'):
            return []
        
        try:
            url = "https://places.googleapis.com/v1/places:searchText"
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': self.api_key,
                'X-Goog-FieldMask': 'places.id,places.displayName,places.location,places.types,places.formattedAddress,places.rating,places.userRatingCount',
                'Accept-Language': 'en'
            }
            
            # Text search query
            query = f"{poi_name} {city}"
            body = {
                'textQuery': query,
                'maxResultCount': 5,  # Just need first few results
                'includedType': 'restaurant'  # Can be adjusted based on discovery context
            }
            
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            
            # Convert new API format to legacy format for compatibility
            places = data.get('places', [])
            converted_places = []
            
            for place in places:
                converted_place = {
                    'id': place.get('id'),
                    'place_id': place.get('id'),
                    'name': place.get('displayName', {}).get('text'),
                    'rating': place.get('rating'),
                    'user_ratings_total': place.get('userRatingCount'),
                    'geometry': {
                        'location': {
                            'lat': place.get('location', {}).get('latitude'),
                            'lng': place.get('location', {}).get('longitude')
                        }
                    },
                    'types': place.get('types', []),
                    'formatted_address': place.get('formattedAddress')
                }
                converted_places.append(converted_place)
            
            logger.debug(f"Text search '{query}' returned {len(converted_places)} results")
            return converted_places
            
        except Exception as e:
            logger.error(f"Text search failed for '{poi_name}' in {city}: {e}")
            return []
    
    def extract_country_from_address(self, formatted_address: str) -> Optional[str]:
        """Extract country from formatted address"""
        if not formatted_address:
            return None
        
        # Country is typically the last component after final comma
        parts = formatted_address.split(', ')
        if parts:
            potential_country = parts[-1].strip()
            # Basic validation - country should be more than 2 chars
            if potential_country and len(potential_country) > 2:
                return potential_country
        return None
    
    def to_poi_row(self, search_result: Dict[str, Any], city_slug: str, h3_cell_id: str = None) -> Optional[Dict[str, Any]]:
        """Convert Google Search result to minimal POI row"""
        try:
            place_id = search_result.get('place_id')
            name = search_result.get('name')
            
            if not place_id or not name:
                return None

            # Extract location (required)
            geometry = search_result.get('geometry', {})
            location = geometry.get('location', {})
            lat = location.get('lat')
            lng = location.get('lng')
            
            if not lat or not lng:
                return None

            # Check allowed types
            types = search_result.get('types', [])
            if not self.is_type_allowed(types):
                return None

            # Get primary category (required)
            primary_category = self.get_primary_category(types)
            if not primary_category:
                return None

            # Get subcategories from types using config mapping
            subcategories = self.get_subcategories(types)

            # Extract country from address
            formatted_address = search_result.get('formatted_address', '')
            country = self.extract_country_from_address(formatted_address)
            
            # Fallback: infer country from city_slug 
            if not country:
                if city_slug == 'paris':
                    country = 'France'
                else:
                    # Could be extended for other cities
                    logger.info(f"Skipping POI {name} - could not determine country from city_slug: {city_slug}")
                    return None
            
            # Extract city from city_slug
            city = city_slug.replace('_', ' ').title()

            # Rating for snapshot
            rating = search_result.get('rating')
            user_ratings_total = search_result.get('user_ratings_total')

            # Extract all available Google Places data
            website = search_result.get('website')
            price_level = search_result.get('price_level')
            phone = search_result.get('formatted_phone_number') or search_result.get('international_phone_number')
            opening_hours = search_result.get('opening_hours')
            
            # Get primary photo reference if available
            primary_photo_ref = None
            photos = search_result.get('photos', [])
            if photos and len(photos) > 0:
                primary_photo_ref = photos[0].get('photo_reference')

            # POI row with all available fields
            poi_data = {
                'google_place_id': place_id,
                'name': name[:200],
                'category': primary_category,
                'city_slug': city_slug,
                'city': city,
                'country': country,
                'lat': lat,
                'lng': lng,
                'last_ingested_from_google_at': datetime.now(timezone.utc).isoformat()
            }

            # Optional fields from Google Places API
            if formatted_address:
                poi_data['address_street'] = formatted_address[:255]
            
            if subcategories:
                poi_data['subcategories'] = subcategories
            
            if website:
                poi_data['website'] = website[:500]
            
            if phone:
                poi_data['phone'] = phone[:50]
            
            if price_level is not None:
                poi_data['price_level'] = str(price_level)
            
            if primary_photo_ref:
                poi_data['primary_photo_ref'] = primary_photo_ref
            
            if opening_hours and opening_hours.get('periods'):
                # Store only the periods, not open_now status
                poi_data['opening_hours'] = {'periods': opening_hours['periods']}
            
            # Add H3 cell tracking
            if h3_cell_id:
                poi_data['h3_cell_id'] = h3_cell_id
            
            # Store rating for snapshot
            poi_data['_rating'] = rating
            poi_data['_user_ratings_total'] = user_ratings_total

            return poi_data

        except Exception as e:
            logger.error(f"Error converting place data: {e}")
            return None
    
    def create_rating_snapshot(self, poi_id: str, details: Dict[str, Any]) -> bool:
        """Create rating snapshot for POI"""
        try:
            rating = details.get('rating')
            reviews_count = details.get('user_ratings_total')
            
            if not rating or not reviews_count:
                return False
                
            return self.db.client.table('rating_snapshot').insert({
                'poi_id': poi_id,
                'source_id': 'google',
                'rating_value': rating,
                'reviews_count': reviews_count,
                'captured_at': datetime.now(timezone.utc).isoformat()
            }).execute()
            
        except Exception as e:
            logger.error(f"Error creating rating snapshot for POI {poi_id}: {e}")
            return False
    
    def upsert_poi(self, row: Dict[str, Any]) -> Optional[str]:
        """Upsert POI with eligibility status and return poi_id"""
        try:
            google_place_id = row.get('google_place_id')
            if not google_place_id:
                return None
            
            # Extract rating and novelty data
            rating = row.pop('_rating', None)
            user_ratings_total = row.pop('_user_ratings_total', None)
            novelty_score = row.pop('_novelty_score', None)
            novelty_classification = row.pop('_novelty_classification', None)
            
            # Determine eligibility status with novelty awareness
            if novelty_score and novelty_score >= 0.8:
                row['eligibility_status'] = 'emerging_priority'
            elif novelty_score and novelty_score >= 0.6:
                row['eligibility_status'] = 'emerging_potential'
            elif rating is not None and user_ratings_total is not None:
                if rating >= self.rating_min and user_ratings_total >= self.min_reviews:
                    row['eligibility_status'] = 'eligible'
                else:
                    row['eligibility_status'] = 'hold'
            else:
                # No rating data available - mark as hold until we get rating info
                row['eligibility_status'] = 'hold'
            
            # Add novelty fields
            if novelty_score is not None:
                row['novelty_score'] = novelty_score
            if novelty_classification:
                row['novelty_classification'] = novelty_classification
            
            # Add urban area mapping timestamp at insertion time
            row['urban_area_mapped_at'] = datetime.now(timezone.utc).isoformat()
            
            # Check if POI already exists
            result = self.db.client.table('poi')\
                .select('id')\
                .eq('google_place_id', google_place_id)\
                .execute()
            
            if result.data:
                # Update existing POI
                poi_id = result.data[0]['id']
                update_data = row.copy()
                del update_data['google_place_id']
                
                self.db.client.table('poi').update(update_data).eq('id', poi_id).execute()
                logger.debug(f"poi_updated: {row.get('name')} (id: {poi_id})")
            else:
                # Insert new POI - Set first_ingested_at for new POIs
                row['first_ingested_at'] = datetime.now(timezone.utc).isoformat()
                
                insert_result = self.db.client.table('poi').insert(row).execute()
                if insert_result.data:
                    poi_id = insert_result.data[0]['id']
                    logger.debug(f"poi_created: {row.get('name')} (id: {poi_id})")
                else:
                    return None
            
            # Create rating snapshot if rating data available
            if rating is not None and user_ratings_total is not None:
                self.write_rating_snapshot(poi_id, rating, user_ratings_total)
            
            return poi_id
        
        except Exception as e:
            logger.error(f"Error upserting POI: {e}")
            return None
    
    def write_rating_snapshot(self, poi_id: str, rating: float, reviews_count: int) -> bool:
        """Write rating snapshot to rating_snapshot table with config-driven interval checking"""
        try:
            # Check if we need to create a new snapshot based on config interval
            days_interval = self.config.google_ingester.rating_snapshot.days_interval
            
            # Get the latest snapshot for this POI from Google
            latest_result = self.db.client.table('rating_snapshot')\
                .select('captured_at')\
                .eq('poi_id', poi_id)\
                .eq('source_id', 'google')\
                .order('captured_at', desc=True)\
                .limit(1)\
                .execute()
            
            if latest_result.data:
                latest_captured = datetime.fromisoformat(latest_result.data[0]['captured_at'].replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                days_since = (now - latest_captured).days
                
                if days_since < days_interval:
                    logger.debug(f"Rating snapshot skipped: poi_id={poi_id}, last captured {days_since} days ago (interval: {days_interval} days)")
                    return True  # Not an error, just skipped
            
            # Create new snapshot
            snapshot_data = {
                'poi_id': poi_id,
                'source_id': 'google',
                'rating_value': rating,
                'reviews_count': reviews_count,
                'captured_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.db.client.table('rating_snapshot').insert(snapshot_data).execute()
            logger.debug(f"Rating snapshot written: poi_id={poi_id}, rating={rating}, reviews={reviews_count}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing rating snapshot: {e}")
            return False

    def run_individual_poi_ingestion(self, poi_name: str = None, place_id: str = None, 
                                    city: str = 'paris', trending_discovery: bool = False,
                                    commit: bool = False, stdout_json: bool = False) -> Dict[str, Any]:
        """
        PHASE 3 SYNERGIE: Ingest individual POI by name or place_id with trending discovery support
        
        This method supports the trending discovery feedback loop:
        - POI discovered via trending search → ingested with high novelty score
        """
        import json
        from datetime import datetime, timezone
        
        logger.info(f"🎯 Individual POI Ingestion: {poi_name or place_id} (trending: {trending_discovery})")
        
        result = {
            'status': 'error',
            'poi': None,
            'error': None,
            'trending_discovery': trending_discovery
        }
        
        try:
            if place_id:
                # Direct place details by ID
                place_data = self.get_place_details(place_id)
                if not place_data:
                    error_msg = f"Place ID {place_id} not found"
                    result['error'] = error_msg
                    if stdout_json:
                        print(json.dumps(result))
                    return result
            elif poi_name:
                # Search by name in city  
                search_results = self.search_places_by_name(poi_name, city)
                if not search_results:
                    error_msg = f"POI '{poi_name}' not found in {city}"
                    result['error'] = error_msg
                    if stdout_json:
                        print(json.dumps(result))
                    return result
                
                # Take first result - use search result data directly since it has the name
                place_data = search_results[0]
                place_id = place_data.get('place_id')
                
                # Get additional details if needed, but keep search result name
                details = self.get_place_details(place_id)
                if details:
                    place_data.update(details)  # Merge details but keep search result name
                
            else:
                error_msg = "Either poi_name or place_id must be provided"
                result['error'] = error_msg
                if stdout_json:
                    print(json.dumps(result))
                return result
            
            # Calculate novelty score with trending discovery boost
            novelty_score = None
            if trending_discovery:
                # High novelty score for trending discoveries
                from scripts.h3_scheduler import H3SchedulerNovelty
                novelty_detector = H3SchedulerNovelty(self.db)
                
                # Base novelty + trending boost
                base_novelty = novelty_detector.calculate_novelty_score(
                    place_data, 
                    place_data.get('rating', 0), 
                    place_data.get('user_ratings_total', 0)
                )
                # Trending discovery gets +0.3 boost (minimum 0.8)
                novelty_score = max(0.8, base_novelty + 0.3)
                logger.info(f"  🔥 TRENDING BOOST: novelty {base_novelty:.2f} → {novelty_score:.2f}")
            
            # Upsert POI with novelty data
            if commit and not self.dry_run:
                # Convert to poi row format and add novelty data
                poi_row_data = self.to_poi_row(place_data, city.lower())
                
                if not poi_row_data:
                    # For trending discoveries, bypass quality checks and create minimal row
                    poi_row_data = {
                        'google_place_id': place_data.get('place_id'),
                        'name': place_data.get('name'),
                        'city_slug': city.lower(),
                        'lat': place_data.get('geometry', {}).get('location', {}).get('lat'),
                        'lng': place_data.get('geometry', {}).get('location', {}).get('lng'),
                        'address_street': place_data.get('formatted_address'),
                        'city': city.title(),
                        'country': 'France',  # Default for Paris
                        'eligibility_status': 'emerging_priority'  # Bypass quality gates
                    }
                    logger.info(f"  🔄 TRENDING BYPASS: Created minimal row for low-quality POI")
                
                if poi_row_data and novelty_score is not None:
                    poi_row_data['novelty_score'] = novelty_score
                    poi_row_data['novelty_classification'] = novelty_detector.classify_novelty(novelty_score)
                    poi_row_data['first_ingested_at'] = datetime.now(timezone.utc).isoformat()
                    # Set high priority eligibility for trending discoveries
                    if novelty_score >= 0.8:
                        poi_row_data['eligibility_status'] = 'emerging_priority'
                
                poi_id = self.upsert_poi(poi_row_data) if poi_row_data else None
                if poi_id:
                    result['status'] = 'upserted'
                    result['poi'] = poi_row_data.copy()
                    result['poi']['id'] = poi_id
                    logger.info(f"  ✅ UPSERTED: {poi_row_data.get('name')} (ID: {poi_id})")
                else:
                    result['error'] = 'Upsert failed'
            else:
                # Dry run - just return preview
                result['status'] = 'dry_run' 
                result['poi_preview'] = {
                    'name': place_data.get('name'),
                    'place_id': place_data.get('place_id'),
                    'city': city.lower(),
                    'novelty_score': novelty_score
                }
                logger.info(f"  📝 DRY RUN: {place_data.get('name')} (novelty: {novelty_score})")
            
            if stdout_json:
                print(json.dumps(result))
            
            return result
            
        except Exception as e:
            error_msg = f"Individual POI ingestion failed: {str(e)}"
            logger.error(error_msg)
            result['error'] = error_msg
            if stdout_json:
                print(json.dumps(result))
            return result

    def run_h3_ingestion(self, city_slug: str = 'paris', limit_cells: int = 300, 
                        update_interval_days: int = 7, debug_cell: str = None) -> Dict[str, Any]:
        """Run H3-based systematic POI ingestion"""
        from scripts.h3_scheduler import H3Scheduler, get_paris_polygon
        
        logger.info(f"🗺️ H3 Ingestion: city={city_slug}, limit_cells={limit_cells}")
        
        try:
            scheduler = H3Scheduler(self.db)
            poi_categories = self.config.pipeline_config.poi_categories
            
            # Metrics
            cells_scanned = 0
            cells_split = 0
            pois_upserted = 0
            api_requests_total = 0
            
            # Step 1: Seed H3 cells if needed
            cells_seeded = scheduler.seed_h3_cells_if_needed(city_slug, res_base=9)
            if cells_seeded > 0:
                logger.info(f"🌱 Seeded {cells_seeded} H3 cells for {city_slug}")
            
            # Step 2: Select due cells
            if debug_cell:
                logger.info(f"🔍 DEBUG MODE: Scanning only cell {debug_cell}")
                from scripts.h3_scheduler import H3Cell
                due_cells = [H3Cell(h3=debug_cell, city_slug=city_slug, res=9, status='pending')]
            else:
                due_cells = scheduler.select_due_cells(city_slug, limit_cells)
                logger.info(f"📋 Selected {len(due_cells)} due cells for processing")
            
            if not due_cells:
                logger.info("No due cells found - all up to date")
                return self._build_result(0, 0, 0, 0, 0, 0)
            
            # Step 3: Process each cell
            for i, cell in enumerate(due_cells):
                logger.info(f"📍 Processing cell {i+1}/{len(due_cells)}: {cell.h3} (res={cell.res})")
                
                try:
                    scan_result = scheduler.scan_cell(cell.h3, poi_categories, self, city_slug)
                    
                    cells_scanned += 1
                    pois_upserted += len(scan_result.poi_ids_touched)
                    api_requests_total += scan_result.api_requests_made
                    
                    logger.info(f"✅ Cell {cell.h3}: {scan_result.total_results} results, {len(scan_result.poi_ids_touched)} POIs touched")
                    
                    # Update cell status (skip in debug mode)
                    if not debug_cell:
                        if scan_result.saturated and cell.res < 11:
                            # Split cell
                            paris_polygon = get_paris_polygon(self.db)
                            children_created = scheduler.split_cell(cell.h3, paris_polygon, cell.res + 1)
                            cells_split += 1
                            logger.info(f"🔀 Split saturated cell {cell.h3}: {children_created} children created")
                        else:
                            # Normal completion
                            scheduler.update_cell_after_scan(cell.h3, scan_result, update_interval_days, False)
                    
                except Exception as e:
                    logger.error(f"Error processing cell {cell.h3}: {e}")
                    continue
            
            # Final summary
            logger.info(f"🎯 H3 Ingestion Complete:")
            logger.info(f"   Cells scanned: {cells_scanned}")
            logger.info(f"   Cells split: {cells_split}")
            logger.info(f"   POIs upserted: {pois_upserted}")
            logger.info(f"   Total API requests: {api_requests_total}")
            logger.info(f"💰 Estimated cost: ${(api_requests_total * 0.008):.4f}")
            
            return self._build_result(cells_scanned, cells_split, pois_upserted, 0, 0, api_requests_total)
            
        except Exception as e:
            logger.error(f"H3 ingestion failed: {e}")
            raise
    
    def _build_result(self, cells_scanned: int, cells_split: int, pois_upserted: int,
                     details_calls: int, rating_snapshots: int, api_requests: int) -> Dict[str, Any]:
        """Build result dictionary"""
        cost_estimate = self.get_cost_estimate()
        
        return {
            'success': True,
            'cells_scanned': cells_scanned,
            'cells_split': cells_split,
            'total_ingested': pois_upserted,
            'total_skipped': 0,
            'details_calls': details_calls,
            'rating_snapshots_written': rating_snapshots,
            'api_requests_total': api_requests,
            'cost_estimate': cost_estimate
        }


def main():
    """CLI interface for minimal H3-based ingester"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Google Places Ingester - Minimal H3 Version')
    # H3 mode arguments
    parser.add_argument('--h3-ingest', action='store_true', help='Run H3-based ingestion')
    parser.add_argument('--limit-cells', type=int, default=300, help='Max H3 cells to process')
    parser.add_argument('--update-interval-days', type=int, default=7, help='TTL for cell rescanning')
    parser.add_argument('--debug-cell', type=str, help='Debug: scan specific H3 cell')
    
    # Individual POI mode arguments
    parser.add_argument('--poi-name', help='Ingest specific POI by name')
    parser.add_argument('--place-id', help='Ingest specific POI by Google Place ID')
    parser.add_argument('--city', default='paris', help='City for POI search')
    parser.add_argument('--trending-discovery', action='store_true', help='Mark as trending discovery (high novelty boost)')
    parser.add_argument('--commit', action='store_true', help='Commit changes to database')
    parser.add_argument('--stdout-json', action='store_true', help='Output JSON to stdout')
    
    # Common arguments
    parser.add_argument('--city-slug', default='paris', help='City slug (for H3 mode)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        ingester = GooglePlacesIngester(dry_run=args.dry_run)
        
        if args.poi_name or args.place_id:
            # Individual POI ingestion mode
            result = ingester.run_individual_poi_ingestion(
                poi_name=args.poi_name,
                place_id=args.place_id,
                city=args.city,
                trending_discovery=args.trending_discovery,
                commit=args.commit,
                stdout_json=args.stdout_json
            )
        elif args.h3_ingest:
            # H3 ingestion mode
            result = ingester.run_h3_ingestion(
                city_slug=args.city_slug,
                limit_cells=args.limit_cells,
                update_interval_days=args.update_interval_days,
                debug_cell=args.debug_cell
            )
        else:
            # Default to H3 mode
            result = ingester.run_h3_ingestion(city_slug=args.city_slug)
        
        # Print results (only for H3 mode)
        if args.h3_ingest or (not args.poi_name and not args.place_id):
            mode = "DRY RUN" if args.dry_run else "LIVE"
            print(f"\n🎯 H3 Results ({mode}):")
            print(f"   Cells scanned: {result['cells_scanned']}")
            print(f"   Cells split: {result['cells_split']}")
            print(f"   Ingested: {result['total_ingested']} POIs")
            print(f"   Skipped: {result['total_skipped']} POIs")
            print(f"💰 Cost: ${result['cost_estimate']['estimate_usd']}")
            print(f"🪣 Tokens: {result['cost_estimate']['tokens_remaining']}")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
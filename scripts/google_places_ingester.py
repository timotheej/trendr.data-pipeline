#!/usr/bin/env python3
"""
Google Places API Ingester - KISS Refactor
Minimal, idempotent Google Places ingester with categories+subcategories, snapshots only, no legacy fields.
"""
import sys
import os
import logging
import requests
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager
import config

# Global logger - level will be set by CLI args
logger = logging.getLogger(__name__)

# KISS V1 Configuration
ALLOWED_GOOGLE_TYPES = {'restaurant', 'bar', 'cafe', 'bakery'}
DEFAULT_RATING_MIN = 4.3
DEFAULT_MIN_REVIEWS = 50
DEFAULT_ENABLE_DETAILS_FOR_HOLD = False

class GooglePlacesIngesterV2:
    """KISS Google Places Ingester with categories+subcategories, rating snapshots only"""
    
    def __init__(self, dry_run: bool = False, mock_mode: bool = False):
        self.dry_run = dry_run
        self.mock_mode = mock_mode
        
        # Initialize database (skip if mock mode)
        if not mock_mode:
            self.db = SupabaseManager()
            
            # Check for required environment variables
            required_vars = ['GOOGLE_PLACES_API_KEY']
            missing_vars = [var for var in required_vars if not getattr(config, var, None)]
            if missing_vars:
                logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
                sys.exit(1)
                
            self.api_key = config.GOOGLE_PLACES_API_KEY
        else:
            self.db = None
            self.api_key = 'mock-api-key'
        
        # Configuration from config.json or env vars
        try:
            import json
            with open('config.json', 'r') as f:
                config_data = json.load(f)
            google_config = config_data.get('google_ingester', {})
        except Exception:
            google_config = {}
        
        self.rating_snapshot_days_interval = google_config.get('rating_snapshot', {}).get('days_interval', 7)
        self.photo_max_width = google_config.get('photo', {}).get('max_width', 1024)
        self.category_map = google_config.get('category_map', {})
        self.subcategory_map = google_config.get('subcategory_map', {})
        
        # Token Bucket Configuration (env vars with defaults)
        self.daily_tokens = int(os.environ.get('PLACES_DAILY_TOKENS', '5000'))
        self.reset_hour_utc = int(os.environ.get('PLACES_RESET_HOUR_UTC', '0'))
        self.basic_cost_per_1000 = float(os.environ.get('PLACES_BASIC_COST_PER_1000', '17.0'))
        self.contact_cost_per_1000 = float(os.environ.get('PLACES_CONTACT_COST_PER_1000', '3.0'))
        
        # Initialize token counters
        self._init_token_bucket()
        
        # Field Masks - optimized for cost but comprehensive
        self.basic_fields = 'place_id,name,geometry,formatted_address,types,rating,user_ratings_total,opening_hours,price_level,address_components,photos,website,international_phone_number'
        self.contact_fields = 'website,international_phone_number'
        
        # Counters for summary
        self.ingested_count = 0
        self.skipped_count = 0
        
        if not self.api_key and not mock_mode:
            logger.error("Google Places API key not configured")
            sys.exit(1)
        
        # KISS V1 Quality thresholds from config
        self.rating_min = google_config.get('quality', {}).get('rating_min', DEFAULT_RATING_MIN)
        self.min_reviews = google_config.get('quality', {}).get('min_reviews', DEFAULT_MIN_REVIEWS)
        self.enable_details_for_hold = google_config.get('quality', {}).get('enable_details_for_hold', DEFAULT_ENABLE_DETAILS_FOR_HOLD)
    
    def _init_token_bucket(self):
        """Initialize token bucket with daily reset logic"""
        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()
        
        # Check if we need to reset (new day or first run)
        reset_time = datetime.combine(today, datetime.min.time().replace(hour=self.reset_hour_utc)).replace(tzinfo=timezone.utc)
        if now_utc < reset_time:
            reset_time -= timedelta(days=1)  # Yesterday's reset
        
        self.last_reset = reset_time
        self.tokens_remaining = self.daily_tokens
        self.basic_calls = 0
        self.contact_calls = 0
        
        logger.info(f"Token bucket initialized: {self.tokens_remaining} tokens remaining")
    
    def _consume_token(self, call_type: str) -> bool:
        """Consume tokens for API call, block if zero remaining"""
        # Check for daily reset
        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()
        reset_time = datetime.combine(today, datetime.min.time().replace(hour=self.reset_hour_utc)).replace(tzinfo=timezone.utc)
        
        if now_utc >= reset_time and self.last_reset < reset_time:
            logger.info("Daily token reset triggered")
            self._init_token_bucket()
        
        if self.tokens_remaining <= 0:
            logger.warning(f"Token bucket empty! Blocking {call_type} call. Reset at {self.reset_hour_utc}:00 UTC.")
            return False
        
        self.tokens_remaining -= 1
        if call_type == 'basic':
            self.basic_calls += 1
        elif call_type == 'contact':
            self.contact_calls += 1
        
        logger.debug(f"Token consumed for {call_type} call. Remaining: {self.tokens_remaining}")
        return True
    
    def is_type_allowed(self, types: List[str]) -> bool:
        """Check if POI types intersect with allowed types"""
        if not types:
            return False
        return bool(set(types) & ALLOWED_GOOGLE_TYPES)
    
    def pass_quality_gate(self, rating: Optional[float], count: Optional[int]) -> bool:
        """Check if POI passes quality thresholds"""
        if rating is None or count is None:
            return False
        return rating >= self.rating_min and count >= self.min_reviews
    
    def get_primary_category(self, types: List[str]) -> Optional[str]:
        """Extract primary category from Google types"""
        for google_type in types:
            if google_type in ALLOWED_GOOGLE_TYPES:
                return google_type
        return None
    
    def filter_subcategories(self, types: List[str], primary_category: str) -> List[str]:
        """Get subcategories as raw Google types minus primary"""
        return [t for t in types if t != primary_category and t not in ['establishment', 'point_of_interest']]
    
    def get_cost_estimate(self) -> Dict[str, Any]:
        """Get current cost estimate and usage stats"""
        basic_cost = (self.basic_calls / 1000.0) * self.basic_cost_per_1000
        contact_cost = (self.contact_calls / 1000.0) * self.contact_cost_per_1000
        total_cost = basic_cost + contact_cost
        
        return {
            'basic_calls': self.basic_calls,
            'contact_calls': self.contact_calls,
            'estimate_usd': round(total_cost, 4),
            'tokens_remaining': self.tokens_remaining
        }
    
    def get_place_details(self, place_id: str, include_contact: bool = False) -> Optional[Dict[str, Any]]:
        """Get place details with optimized field masks"""
        if not self.api_key:
            return None
        
        # Mock mode - return fake data
        if self.mock_mode:
            return {
                'place_id': place_id,
                'name': f'Mock Restaurant {place_id[-3:]}',
                'geometry': {'location': {'lat': 48.8566, 'lng': 2.3522}},
                'types': ['restaurant'],
                'rating': 4.2,
                'user_ratings_total': 100,
                'formatted_address': '123 Mock St, Paris, France',
                'website': 'https://mock-restaurant.com' if include_contact else None,
                'international_phone_number': '+33 1 23 45 67 89' if include_contact else None
            }
        
        # Determine call type and check tokens
        call_type = 'contact' if include_contact else 'basic'
        if not self._consume_token(call_type):
            return None
        
        try:
            url = "https://maps.googleapis.com/maps/api/place/details/json"
            fields = self.basic_fields
            if include_contact:
                fields = f"{self.basic_fields},{self.contact_fields}"
            
            logger.debug(f"Requesting fields: {fields}")
            
            params = {
                'place_id': place_id,
                'key': self.api_key,
                'fields': fields
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            result = data.get('result')
            
            if result:
                returned_fields = list(result.keys())
                logger.debug(f"API returned fields: {returned_fields}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get details for place {place_id}: {e}")
            return None
    
    # Removed _determine_poi_tier and _is_poi_fresh methods - no longer needed with simplified logic
    
    def revalidate_light(self, poi_id: str, include_contact: bool = False) -> Dict[str, Any]:
        """Light revalidation - refresh POI data and create snapshot if needed"""
        try:
            # Get current POI data
            result = self.db.client.table('poi').select('google_place_id,last_ingested_from_google_at').eq('id', poi_id).execute()
            if not result.data:
                return {'refreshed': False, 'error': 'POI not found'}
            
            poi = result.data[0]
            google_place_id = poi.get('google_place_id')
            if not google_place_id:
                return {'refreshed': False, 'error': 'No google_place_id'}
            
            # Check if refresh needed (simple 24h check)
            last_ingested = poi.get('last_ingested_from_google_at')
            if last_ingested:
                try:
                    last_ingested_dt = datetime.fromisoformat(last_ingested.replace('Z', '+00:00'))
                    hours_since = (datetime.now(timezone.utc) - last_ingested_dt).total_seconds() / 3600
                    if hours_since < 24:
                        return {
                            'refreshed': False,
                            'fields_fetched': [],
                            'snapshot_created': False,
                            'reason': 'still_fresh'
                        }
                except Exception:
                    pass  # Proceed with refresh if date parsing fails
            
            # Fetch new data
            new_data = self.get_place_details(google_place_id, include_contact)
            if not new_data:
                return {'refreshed': False, 'error': 'API call failed'}
            
            # Convert to POI format
            poi_data = self.convert_place_data(new_data)
            if not poi_data:
                return {'refreshed': False, 'error': 'Failed to convert place data'}
            
            # Update POI with identity+display fields only
            update_fields = {k: v for k, v in poi_data.items() if k != 'google_place_id'}
            
            # Extract rating for snapshot logic
            current_rating = new_data.get('rating')
            current_reviews = new_data.get('user_ratings_total')
            
            # Check if rating snapshot needed
            snapshot_created = False
            if current_rating is not None or current_reviews is not None:
                snapshot_created = self._needs_rating_snapshot(poi_id, current_rating, current_reviews)
                if snapshot_created:
                    self._create_rating_snapshot(poi_id, current_rating, current_reviews)
            
            # Update POI
            self.db.client.table('poi').update(update_fields).eq('id', poi_id).execute()
            
            # Calculate cost increment
            call_type = 'contact' if include_contact else 'basic'
            cost_per_call = (self.contact_cost_per_1000 if call_type == 'contact' else self.basic_cost_per_1000) / 1000.0
            
            fields_fetched = ['identity', 'display']
            if include_contact:
                fields_fetched.extend(['website', 'phone'])
            
            return {
                'refreshed': True,
                'fields_fetched': fields_fetched,
                'snapshot_created': snapshot_created,
                'cost_estimate_increment': round(cost_per_call, 4)
            }
            
        except Exception as e:
            logger.error(f"Error in revalidate_light for POI {poi_id}: {e}")
            return {'refreshed': False, 'error': str(e)}
    
    def map_urban_areas(self, lat: float, lng: float) -> Tuple[Optional[str], Optional[str]]:
        """Map POI coordinates to urban areas (district + neighbourhood)"""
        try:
            if self.mock_mode:
                return "16e Arrondissement", "Chaillot"
            
            # Fallback to post-processing association
            # The spatial queries are complex and better handled by the SQL function
            logger.debug(f"Deferring spatial association for coordinates ({lat}, {lng}) to post-processing")
            return None, None
            
        except Exception as e:
            logger.error(f"Error mapping urban areas for ({lat}, {lng}): {e}")
            return None, None

    def map_category(self, types: List[str]) -> Tuple[Optional[str], List[str]]:
        """Map Google types to category and subcategories using KISS static mappings"""
        
        # Use configured mappings with fallbacks
        parent_map = self.category_map if self.category_map else {
            'restaurant': 'restaurant',
            'bar': 'bar', 
            'cafe': 'cafe',
            'bakery': 'bakery',
            'night_club': 'bar',
            'food_court': 'restaurant',
            'meal_takeaway': 'restaurant',
            'meal_delivery': 'restaurant'
        }
        
        subcat_map = self.subcategory_map if self.subcategory_map else {
            'italian_restaurant': 'restaurant italien',
            'french_restaurant': 'restaurant français',
            'lebanese_restaurant': 'restaurant libanais',
            'japanese_restaurant': 'restaurant japonais',
            'chinese_restaurant': 'restaurant chinois',
            'indian_restaurant': 'restaurant indien',
            'thai_restaurant': 'restaurant thaï',
            'mexican_restaurant': 'restaurant mexicain',
            'seafood_restaurant': 'restaurant de fruits de mer',
            'pizza_restaurant': 'pizzeria',
            'wine_bar': 'bar à vins',
            'cocktail_bar': 'bar à cocktails',
            'sports_bar': 'bar sportif',
            'coffee_shop': 'café',
            'tea_house': 'salon de thé',
            'bakery': 'boulangerie',
            'pastry_shop': 'pâtisserie',
            'ice_cream_shop': 'glacier',
            'fast_food_restaurant': 'fast food',
            'vegetarian_restaurant': 'restaurant végétarien',
            'vegan_restaurant': 'restaurant végan'
        }
        
        # Find parent category (first match)
        category = None
        for google_type in types:
            if google_type in parent_map:
                category = parent_map[google_type]
                break
        
        # Build subcategories list
        subcategories = []
        for google_type in types:
            if google_type in subcat_map:
                subcategories.append(subcat_map[google_type])
        
        # Deduplicate and limit to 5
        subcategories = list(dict.fromkeys(subcategories))[:5]
        
        return category, subcategories
    
    def extract_address_components(self, place: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Extract city and country from address_components"""
        try:
            address_components = place.get('address_components', [])
            city = None
            country = None
            
            for component in address_components:
                types = component.get('types', [])
                long_name = component.get('long_name')
                
                # Extract city (locality or postal_town)
                if ('locality' in types or 'postal_town' in types) and not city:
                    city = long_name
                
                # Extract country
                if 'country' in types and not country:
                    country = long_name
            
            return city, country
        except Exception as e:
            logger.warning(f"Error extracting address components: {e}")
            return None, None
    
    def build_photo_url(self, photo_reference: str) -> Optional[str]:
        """Build Google Photos API URL with maxwidth"""
        if not photo_reference or not self.api_key:
            return None
        
        try:
            url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth={self.photo_max_width}&photoreference={photo_reference}&key={self.api_key}"
            return url
        except Exception as e:
            logger.warning(f"Error building photo URL: {e}")
            return None
    
    def convert_place_data(self, result: Dict[str, Any], city_slug: str = None) -> Optional[Dict[str, Any]]:
        """Convert Google Places API result to minimal POI format (identity+display only)"""
        try:
            place_id = result.get('place_id')
            name = result.get('name')
            
            if not place_id or not name:
                return None
            
            # Extract location
            geometry = result.get('geometry', {})
            location = geometry.get('location', {})
            lat = location.get('lat')
            lng = location.get('lng')
            
            if not lat or not lng:
                return None
            
            # Map POI to urban areas (district + neighbourhood) - KISS method
            district_name, neighbourhood_name = self.map_urban_areas(lat, lng)
            
            # Map Google types to category and subcategories (KISS V1: store raw types)
            types = result.get('types', [])
            category, mapped_subcategories = self.map_category(types)
            
            # KISS V1: Store raw Google types as subcategories (minus primary and common ones)
            subcategories = [t for t in types if t not in ['establishment', 'point_of_interest'] and t != category]
            
            if not category:
                return None  # Skip if no matching category
            
            # Extract city and country from address_components
            city, country = self.extract_address_components(result)
            
            # Fallback for city using city_slug or formatted_address
            if not city:
                if city_slug:
                    city = city_slug
                else:
                    city = self._extract_city_from_address(result.get('formatted_address', ''))
            
            # Fallback for country if not found in address_components
            if not country:
                formatted_address = result.get('formatted_address', '')
                if 'France' in formatted_address:
                    country = 'France'
                elif 'Paris' in formatted_address:
                    country = 'France'  # Assume Paris = France
                else:
                    country = 'Unknown'  # Default fallback
            
            # Build city_slug (use resolver if present, otherwise slugify)
            if hasattr(self, '_resolve_city_slug'):
                city_slug_resolved = self._resolve_city_slug(city)
            else:
                city_slug_resolved = city.lower().replace(' ', '_').replace('-', '_') if city else None
            
            # Handle price_level mapping (0-4 -> enum)
            price_level = result.get('price_level')
            price_level_mapped = self._convert_price_level(price_level) if price_level is not None else None
            
            # Extract opening hours as jsonb (remove open_now, keep only periods)
            opening_hours = result.get('opening_hours')
            opening_hours_jsonb = None
            if opening_hours:
                # Remove open_now, keep only periods
                clean_hours = {}
                if 'periods' in opening_hours:
                    clean_hours['periods'] = opening_hours['periods']
                
                if clean_hours:  # Only save if we have periods
                    opening_hours_str = json.dumps(clean_hours)
                    # Truncate to 500 chars if needed (database limit)
                    if len(opening_hours_str) > 500:
                        opening_hours_jsonb = opening_hours_str[:497] + '...'
                    else:
                        opening_hours_jsonb = opening_hours_str
            
            # Handle primary photo (KISS V1: only store reference, not URL)
            photos = result.get('photos', [])
            primary_photo_ref = None
            if photos:
                first_photo = photos[0]
                photo_ref = first_photo.get('photo_reference')
                if photo_ref:
                    primary_photo_ref = photo_ref
            
            # Truncate fields that might be too long
            address_street = result.get('formatted_address', '')
            if len(address_street) > 500:
                address_street = address_street[:497] + '...'
            
            website = result.get('website', '')
            if website and len(website) > 500:
                website = website[:497] + '...'
            
            phone = result.get('international_phone_number', '')
            if phone and len(phone) > 50:
                phone = phone[:47] + '...'
            
            # Ensure all fields are properly truncated
            if primary_photo_ref and len(primary_photo_ref) > 500:
                primary_photo_ref = primary_photo_ref[:500]
            
            poi_data = {
                'google_place_id': place_id,
                'name': name[:255] if name else name,  # Truncate name if too long
                'city': city[:100] if city else city,  # Truncate city
                'country': country[:100] if country else country,  # Truncate country
                'city_slug': city_slug_resolved[:100] if city_slug_resolved else city_slug_resolved,  # Truncate city_slug
                'address_street': address_street,
                'lat': float(lat),
                'lng': float(lng),
                'website': website,
                'phone': phone,
                'category': category[:50] if category else category,  # Truncate category
                'subcategories': subcategories[:5] if subcategories else subcategories,  # Limit subcategories
                'price_level': price_level_mapped,
                'opening_hours': opening_hours_jsonb,
                'primary_photo_ref': primary_photo_ref,  # KISS V1: only photo reference, no URL
                'district_name': district_name,
                'neighbourhood_name': neighbourhood_name,
                'urban_area_mapped_at': datetime.now(timezone.utc).isoformat() if (district_name or neighbourhood_name) else None,
                'last_ingested_from_google_at': datetime.now(timezone.utc).isoformat()
            }
            
            return poi_data
            
        except Exception as e:
            logger.error(f"Error converting place data: {e}")
            return None
    
    def to_poi_row(self, search_result: Dict[str, Any], details_result: Optional[Dict[str, Any]] = None, city_slug: str = None) -> Optional[Dict[str, Any]]:
        """Convert Google Search result + optional Details to POI row (KISS V1)"""
        try:
            place_id = search_result.get('place_id')
            name = search_result.get('name')
            
            if not place_id or not name:
                return None
            
            # Extract location
            geometry = search_result.get('geometry', {})
            location = geometry.get('location', {})
            lat = location.get('lat')
            lng = location.get('lng')
            
            if not lat or not lng:
                return None
            
            # Check allowed types first
            types = search_result.get('types', [])
            if not self.is_type_allowed(types):
                return None  # Skip if not in whitelist
            
            # Get primary category from allowed types
            primary_category = self.get_primary_category(types)
            if not primary_category:
                return None
            
            # Get subcategories as raw Google types (minus primary)
            subcategories = self.filter_subcategories(types, primary_category)
            
            # Extract city and country from address_components (prefer details if available)
            result_to_parse = details_result if details_result else search_result
            city, country = self.extract_address_components(result_to_parse)
            
            # Fallback for city
            if not city:
                if city_slug:
                    city = city_slug
                else:
                    city = self._extract_city_from_address(search_result.get('formatted_address', ''))
            
            # Fallback for country
            if not country:
                formatted_address = search_result.get('formatted_address', '')
                if 'France' in formatted_address or 'Paris' in formatted_address:
                    country = 'France'
                else:
                    country = 'Unknown'
            
            # Build city_slug
            city_slug_resolved = city.lower().replace(' ', '_').replace('-', '_') if city else 'unknown'
            
            # Extract fields from details if available, otherwise from search
            source = details_result if details_result else search_result
            
            # Address
            address_street = source.get('formatted_address', '')[:500]  # Truncate
            
            # Contact info (details only)
            website = ''
            phone = ''
            if details_result:
                website = details_result.get('website', '')[:500]
                phone = details_result.get('international_phone_number', '')[:50]
            
            # Opening hours (details preferred) - keep only periods
            opening_hours_jsonb = None
            opening_hours = source.get('opening_hours')
            if opening_hours and isinstance(opening_hours, dict):
                # Keep only periods, remove open_now
                clean_hours = {}
                if 'periods' in opening_hours:
                    clean_hours['periods'] = opening_hours['periods']
                
                if clean_hours:  # Only save if we have periods
                    opening_hours_str = json.dumps(clean_hours)
                    if len(opening_hours_str) <= 500:
                        opening_hours_jsonb = opening_hours_str
            
            # Primary photo reference (no URL generation)
            primary_photo_ref = None
            photos = source.get('photos', [])
            if photos and len(photos) > 0:
                primary_photo_ref = photos[0].get('photo_reference')
                # Truncate photo reference if too long
                if primary_photo_ref and len(primary_photo_ref) > 500:
                    primary_photo_ref = primary_photo_ref[:500]
            
            poi_data = {
                'google_place_id': place_id,
                'name': name[:255],  # Truncate
                'category': primary_category,
                'address_street': address_street[:500] if address_street else address_street,  # Ensure truncation
                'city': city[:100] if city else city,  # Truncate city
                'country': country[:100] if country else country,  # Truncate country
                'lat': float(lat),
                'lng': float(lng),
                'opening_hours': opening_hours_jsonb,
                'phone': phone[:50] if phone else phone,  # Ensure phone truncation
                'website': website[:500] if website else website,  # Ensure website truncation
                'primary_photo_ref': primary_photo_ref,
                'eligibility_status': 'hold',  # V1: always hold
                'subcategories': subcategories[:5] if subcategories else subcategories,  # Limit subcategories
                'district_name': district_name,
                'neighbourhood_name': neighbourhood_name,
                'urban_area_mapped_at': datetime.now(timezone.utc).isoformat() if (district_name or neighbourhood_name) else None,
                'last_ingested_from_google_at': datetime.now(timezone.utc).isoformat()
            }
            
            return poi_data
            
        except Exception as e:
            logger.error(f"Error converting place data: {e}")
            return None
    
    def upsert_poi(self, row: Dict[str, Any]) -> Optional[str]:
        """Upsert POI and return poi_id"""
        try:
            google_place_id = row.get('google_place_id')
            if not google_place_id:
                return None
            
            # Check if POI already exists
            result = self.db.client.table('poi')\
                .select('id')\
                .eq('google_place_id', google_place_id)\
                .execute()
            
            if result.data:
                # Update existing POI
                poi_id = result.data[0]['id']
                update_data = row.copy()
                del update_data['google_place_id']  # Don't update the key
                
                self.db.client.table('poi').update(update_data).eq('id', poi_id).execute()
                logger.debug(f"poi_updated: {row.get('name')} (id: {poi_id})")
                return poi_id
            else:
                # Insert new POI
                insert_result = self.db.client.table('poi').insert(row).execute()
                if insert_result.data:
                    poi_id = insert_result.data[0]['id']
                    logger.debug(f"poi_created: {row.get('name')} (id: {poi_id})")
                    return poi_id
                return None
        
        except Exception as e:
            logger.error(f"Error upserting POI: {e}")
            return None
    
    def insert_rating_snapshot(self, poi_id: str, rating: float, count: int) -> None:
        """Insert rating snapshot for POI"""
        try:
            snapshot_data = {
                'poi_id': poi_id,
                'source_id': 'google',
                'rating_value': rating,
                'reviews_count': count,
                'captured_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.db.client.table('rating_snapshot').insert(snapshot_data).execute()
            logger.debug(f"snapshot_created: poi_id={poi_id}, rating={rating}, reviews={count}")
            
        except Exception as e:
            logger.error(f"Error creating rating snapshot: {e}")
    
    # Removed _map_google_type_to_category - replaced by map_category method
    
    def _extract_city_from_address(self, formatted_address: str) -> str:
        """Extract city from formatted address"""
        if not formatted_address:
            return 'Unknown'
        
        parts = [part.strip() for part in formatted_address.split(',')]
        if len(parts) >= 2:
            # Usually the second-to-last part is the city
            return parts[-2] if len(parts) > 2 else parts[-1]
        
        return 'Unknown'
    
    def ingest_poi_to_db(self, poi_data: Dict[str, Any], allow_fuzzy_upsert: bool = False, place_data: Dict[str, Any] = None) -> Optional[str]:
        """Minimal upsert POI (identity+display fields only, preserve Gatto scores)"""
        try:
            google_place_id = poi_data.get('google_place_id')
            
            # Dry run mode - don't persist to database
            if self.dry_run:
                logger.info(f"DRY RUN: Would ingest POI: {poi_data['name']} (category: {poi_data.get('category')})")
                self.ingested_count += 1
                return f"dry-run-{google_place_id}"
            
            # Mock mode - simulate database operations
            if self.mock_mode:
                logger.info(f"MOCK: Ingesting POI: {poi_data['name']} (category: {poi_data.get('category')})")
                self.ingested_count += 1
                return f"mock-{google_place_id}"
            
            # Check if POI already exists by google_place_id
            result = self.db.client.table('poi')\
                .select('id')\
                .eq('google_place_id', google_place_id)\
                .execute()
            
            # Extract rating for snapshot logic (not stored in poi table anymore)
            # Use place_data if provided, otherwise try to get from original Google response
            source_data = place_data if place_data else {}
            current_rating = source_data.get('rating')
            current_reviews = source_data.get('user_ratings_total')
            
            if result.data:
                # Update existing POI (only identity+display fields)
                poi_id = result.data[0]['id']
                
                # Remove google_place_id from update (it's the key)
                update_data = poi_data.copy()
                if 'google_place_id' in update_data:
                    del update_data['google_place_id']
                
                self.db.client.table('poi').update(update_data).eq('id', poi_id).execute()
                
                # Check if rating snapshot needed
                snapshot_created = False
                if current_rating is not None or current_reviews is not None:
                    snapshot_created = self._needs_rating_snapshot(poi_id, current_rating, current_reviews)
                    if snapshot_created:
                        self._create_rating_snapshot(poi_id, current_rating, current_reviews)
                
                logger.info(f"poi_updated: {poi_data['name']} | snapshot: {'written' if snapshot_created else 'skipped'}")
                self.ingested_count += 1
                
                return poi_id
            else:
                # Fuzzy fallback: try to match by city_slug + name (if enabled)
                fuzzy_poi_id = None
                if allow_fuzzy_upsert and poi_data.get('city_slug') and poi_data.get('name'):
                    fuzzy_result = self.db.client.table('poi')\
                        .select('id')\
                        .eq('city_slug', poi_data['city_slug'])\
                        .ilike('name', f"%{poi_data['name']}%")\
                        .execute()
                    
                    if fuzzy_result.data:
                        # Found fuzzy match - update instead of insert
                        fuzzy_poi_id = fuzzy_result.data[0]['id']
                        
                        update_data = poi_data.copy()
                        if 'google_place_id' in update_data:
                            del update_data['google_place_id']
                        
                        self.db.client.table('poi').update(update_data).eq('id', fuzzy_poi_id).execute()
                        
                        # Check if rating snapshot needed
                        snapshot_created = False
                        if current_rating is not None or current_reviews is not None:
                            snapshot_created = self._needs_rating_snapshot(fuzzy_poi_id, current_rating, current_reviews)
                            if snapshot_created:
                                self._create_rating_snapshot(fuzzy_poi_id, current_rating, current_reviews)
                        
                        logger.info(f"poi_updated (fuzzy): {poi_data['name']} | snapshot: {'written' if snapshot_created else 'skipped'}")
                        self.ingested_count += 1
                        
                        return fuzzy_poi_id
                
                # Insert new POI (no match found)
                insert_result = self.db.client.table('poi').insert(poi_data).execute()
                
                if insert_result.data:
                    poi_id = insert_result.data[0]['id']
                    
                    # Create initial snapshot for new POI
                    snapshot_created = False
                    if current_rating is not None or current_reviews is not None:
                        self._create_rating_snapshot(poi_id, current_rating, current_reviews)
                        snapshot_created = True
                    
                    logger.info(f"poi_created: {poi_data['name']} | snapshot: {'written' if snapshot_created else 'skipped'}")
                    self.ingested_count += 1
                    
                    return poi_id
                
                return None
                
        except Exception as e:
            logger.error(f"db_error: Error ingesting POI to DB: {e}")
            self.skipped_count += 1
            return None
    
    def get_latest_google_snapshot(self, poi_id: str) -> Optional[dict]:
        """Get the latest Google Maps rating snapshot for a POI"""
        try:
            result = self.db.client.table('rating_snapshot')\
                .select('rating_value,reviews_count,captured_at')\
                .eq('poi_id', poi_id)\
                .eq('source_id', 'google')\
                .order('captured_at', desc=True)\
                .limit(1)\
                .execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.warning(f"Error getting latest Google Maps snapshot for POI {poi_id}: {e}")
            return None
    
    def create_google_snapshot(self, poi_id: str, rating: float, reviews_count: int) -> None:
        """Create a Google Maps rating snapshot"""
        try:
            snapshot_data = {
                'poi_id': poi_id,
                'source_id': 'google',
                'rating_value': rating,
                'reviews_count': reviews_count,
                'captured_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.db.client.table('rating_snapshot').insert(snapshot_data).execute()
            
        except Exception as e:
            logger.error(f"Error creating Google Maps snapshot: {e}")
    
    def _needs_rating_snapshot(self, poi_id: str, current_rating: Optional[float], current_reviews: Optional[int]) -> bool:
        """Check if rating snapshot is needed based on simplified policy"""
        try:
            # Don't create snapshot if no rating data
            if current_rating is None and current_reviews is None:
                return False
            
            # Get latest Google Maps snapshot
            latest_snapshot = self.get_latest_google_snapshot(poi_id)
            
            if not latest_snapshot:
                return True  # First snapshot
            
            # Check if snapshot is older than interval days
            captured_at_str = latest_snapshot.get('captured_at')
            if captured_at_str:
                try:
                    captured_at = datetime.fromisoformat(captured_at_str.replace('Z', '+00:00'))
                    days_old = (datetime.now(timezone.utc) - captured_at).days
                    
                    if days_old >= self.rating_snapshot_days_interval:
                        return True  # Snapshot is stale
                except Exception:
                    return True  # Error parsing date, create new snapshot
            else:
                return True  # No captured_at date, create new snapshot
            
            # Check if rating changed significantly (>= 0.1) or reviews changed
            last_rating = latest_snapshot.get('rating_value')
            last_reviews = latest_snapshot.get('reviews_count')
            
            rating_changed = False
            if current_rating is not None and last_rating is not None:
                rating_changed = abs(current_rating - last_rating) >= 0.1
            elif current_rating != last_rating:  # One is None, other is not
                rating_changed = True
            
            reviews_changed = current_reviews != last_reviews
            
            return rating_changed or reviews_changed
            
        except Exception as e:
            logger.warning(f"Error checking rating snapshot need: {e}")
            return False
    
    def _create_rating_snapshot(self, poi_id: str, rating: Optional[float], reviews_count: Optional[int]):
        """Create rating snapshot with google_maps source"""
        try:
            snapshot_data = {
                'poi_id': poi_id,
                'source_id': 'google',  # Use google_maps as source_id
                'rating_value': rating,
                'reviews_count': reviews_count,
                'captured_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.db.client.table('rating_snapshot').insert(snapshot_data).execute()
            logger.debug(f"snapshot_written: poi_id={poi_id}, rating={rating}, reviews={reviews_count}")
            
        except Exception as e:
            logger.error(f"Error creating rating snapshot: {e}")
    
    def search_places_textsearch(self, query: str, location: str = None) -> List[Dict[str, Any]]:
        """Search places using Google Places Text Search API"""
        if not self.api_key:
            return []
        
        # Mock mode - return fake search results
        if self.mock_mode:
            return [
                {
                    'place_id': f'mock-place-{i}',
                    'name': f'Mock {query.split()[0].title()} {i}',
                    'geometry': {'location': {'lat': 48.8566 + i*0.001, 'lng': 2.3522 + i*0.001}},
                    'types': ['restaurant' if 'restaurant' in query else 'bar'],
                    'rating': 4.0 + (i * 0.1),
                    'user_ratings_total': 50 + (i * 10),
                    'formatted_address': f'{100 + i} Mock St, Paris, France'
                } for i in range(5)
            ]
        
        if not self._consume_token('basic'):
            return []
        
        try:
            url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {
                'query': query,
                'key': self.api_key
            }
            
            if location:
                params['location'] = location
                params['radius'] = 5000  # 5km radius
            
            logger.debug(f"Searching places with query: '{query}'")
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            logger.debug(f"Search returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Text search failed for '{query}': {e}")
            return []
    
    def run_seed_ingestion(self, city_slug: str = 'paris', neighborhood: str = None, category: str = None, limit: int = None) -> Dict[str, Any]:
        """Run KISS V1 seed ingestion with economic Details calls"""
        logger.info(f"🔍 KISS V1 Ingestion: city={city_slug}, rating≥{self.rating_min}, reviews≥{self.min_reviews}")
        
        # Only process allowed types (whitelist)
        categories = [category] if category else list(ALLOWED_GOOGLE_TYPES)
        neighborhoods = [neighborhood] if neighborhood else ['1er arrondissement', '2ème arrondissement']
        
        total_processed = 0
        quality_passed = 0
        details_calls_saved = 0
        
        for cat in categories:
            for neigh in neighborhoods:
                if limit and total_processed >= limit:
                    logger.info(f"Reached limit of {limit} POIs")
                    break
                    
                query = f"{cat} {neigh} {city_slug}"
                places = self.search_places_textsearch(query)
                
                places_to_process = places[:5]  # Default limit per query
                if limit:
                    remaining = limit - total_processed
                    places_to_process = places[:min(5, remaining)]
                
                for place in places_to_process:
                    # Filter by allowed types first
                    types = place.get('types', [])
                    if not self.is_type_allowed(types):
                        logger.debug(f"Skipped {place.get('name', 'unknown')}: type not in whitelist")
                        self.skipped_count += 1
                        continue
                    
                    # Check quality gate from search data
                    rating = place.get('rating')
                    reviews_count = place.get('user_ratings_total')
                    passes_quality = self.pass_quality_gate(rating, reviews_count)
                    
                    # Convert search result to POI (no details yet)
                    poi_data = self.to_poi_row(place, details_result=None, city_slug=city_slug)
                    if not poi_data:
                        self.skipped_count += 1
                        continue
                    
                    # Only get Details for quality candidates OR if configured for hold POIs
                    details_result = None
                    if passes_quality or self.enable_details_for_hold:
                        details_result = self.get_place_details(place.get('place_id'), include_contact=True)
                        if details_result:
                            # Update POI data with details
                            poi_data = self.to_poi_row(place, details_result, city_slug=city_slug)
                    else:
                        details_calls_saved += 1
                        logger.debug(f"💰 Details call saved for {place.get('name', 'unknown')} (rating={rating}, reviews={reviews_count})")
                    
                    if poi_data:
                        # Pass the rating data for snapshot creation
                        source_data = details_result if details_result else place
                        poi_id = self.ingest_poi_to_db(poi_data, place_data=source_data)
                        if poi_id:
                            total_processed += 1
                            if passes_quality:
                                quality_passed += 1
                        else:
                            self.skipped_count += 1
                    else:
                        self.skipped_count += 1
                
                if not self.mock_mode and not self.dry_run:
                    time.sleep(0.5)  # Rate limiting
            
            if limit and total_processed >= limit:
                break
        
        cost_estimate = self.get_cost_estimate()
        
        # Enhanced summary with savings
        summary = {
            "ingested": self.ingested_count,
            "quality_passed": quality_passed,
            "skipped": self.skipped_count,
            "details_calls_saved": details_calls_saved,
            "cost_estimate": f"${cost_estimate['estimate_usd']}",
            "tokens_left": cost_estimate['tokens_remaining']
        }
        logger.info(f"KISS V1 summary: {json.dumps(summary)}")
        
        return {
            'total_ingested': self.ingested_count,
            'quality_passed': quality_passed,
            'total_skipped': self.skipped_count,
            'details_calls_saved': details_calls_saved,
            'cost_estimate': cost_estimate
        }
    
    def ingest_poi_by_place_id(self, place_id: str, city_slug: str = None, dry_run: bool = True, json_output: bool = False, allow_fuzzy_upsert: bool = False) -> Dict[str, Any]:
        """Ingest a specific POI by Google Place ID"""
        logger.info(f"🎯 Targeting POI by Place ID: {place_id}")
        
        # Get place details
        place_details = self.get_place_details(place_id, include_contact=True)
        if not place_details:
            error_msg = f"Failed to get place details for {place_id}"
            if json_output:
                self._output_json_error(error_msg, "place_details_failed")
            else:
                logger.error(error_msg)
            return {'success': False, 'error': 'Failed to fetch place details'}
        
        # Normalize city_slug
        if not city_slug and place_details.get('formatted_address'):
            city_slug = self._extract_city_from_address(place_details['formatted_address'])
        
        # Convert to POI data
        poi_data = self.convert_place_data(place_details, city_slug=city_slug)
        if not poi_data:
            error_msg = f"Failed to convert place data for {place_id}"
            if json_output:
                self._output_json_error(error_msg, "place_data_conversion_failed")
            else:
                logger.error(error_msg)
            return {'success': False, 'error': 'Failed to convert place data'}
        
        # The google_place_id is already set in convert_place_data method
        # No need to set external_ids since we use google_place_id directly
        
        if dry_run:
            if json_output:
                self._output_json_dry_run(poi_data)
            else:
                logger.info("DRY-RUN: Ready to upsert POI")
                self._print_poi_summary(poi_data)
            return {'success': True, 'dry_run': True, 'poi_data': poi_data}
        else:
            poi_id = self.ingest_poi_to_db(poi_data, allow_fuzzy_upsert, place_data=place_details)
            if poi_id:
                if json_output:
                    self._output_json_upserted(poi_id, poi_data)
                else:
                    logger.info(f"UPSERT ok: {poi_id} | {poi_data.get('city', 'unknown')} | {poi_data.get('name', 'unnamed')}")
                return {'success': True, 'poi_id': poi_id, 'poi_data': poi_data}
            else:
                error_msg = "Failed to upsert POI to database"
                if json_output:
                    self._output_json_error(error_msg, "database_upsert_failed")
                else:
                    logger.error(error_msg)
                return {'success': False, 'error': 'Database upsert failed'}
    
    def ingest_poi_by_name(self, poi_name: str, city: str, lat: float = None, lng: float = None, category: str = None, dry_run: bool = True, json_output: bool = False, allow_fuzzy_upsert: bool = False) -> Dict[str, Any]:
        """Ingest a specific POI by name search"""
        logger.info(f"🎯 Targeting POI by name: '{poi_name}' in {city}")
        
        # Construct search query
        query = f"{poi_name} {city}"
        location = f"{lat},{lng}" if lat and lng else None
        
        # Search for places
        places = self.search_places_textsearch(query, location)
        if not places:
            logger.info(f"No candidates found for '{poi_name}' in {city}")
            return {'success': True, 'candidates_found': 0}
        
        # Find best match
        best_candidate = self._find_best_candidate(poi_name, places, lat, lng, category)
        if not best_candidate:
            logger.info(f"No suitable candidate found for '{poi_name}' in {city}")
            return {'success': True, 'candidates_found': len(places), 'best_match': None}
        
        logger.info(f"Best candidate: {best_candidate['name']} (place_id: {best_candidate['place_id']})")
        
        # Get full details and proceed with place ID path
        city_slug = city.lower().replace(' ', '_')
        return self.ingest_poi_by_place_id(best_candidate['place_id'], city_slug, dry_run, json_output, allow_fuzzy_upsert)
    
    def _convert_price_level(self, price_level: int) -> str:
        """Convert Google price level (0-4) to database enum"""
        if price_level is None:
            return None
        
        # Map Google price levels to meaningful values
        price_map = {
            0: 'free',
            1: 'inexpensive', 
            2: 'moderate',
            3: 'expensive',
            4: 'very_expensive'
        }
        return price_map.get(price_level, None)
    
    def _extract_city_from_address(self, address: str) -> Optional[str]:
        """Extract city slug from formatted address (French format)"""
        if not address:
            return None
            
        import re
        address_lower = address.lower()
        
        # French address format: usually "street, postal_code city, country"
        # Look for common French cities with postal codes
        city_patterns = [
            (r'75\d{3}\s+paris', 'paris'),
            (r'69\d{3}\s+lyon', 'lyon'), 
            (r'13\d{3}\s+marseille', 'marseille'),
            (r'33\d{3}\s+bordeaux', 'bordeaux'),
            (r'31\d{3}\s+toulouse', 'toulouse'),
            (r'59\d{3}\s+lille', 'lille'),
            (r'67\d{3}\s+strasbourg', 'strasbourg'),
            (r'44\d{3}\s+nantes', 'nantes'),
            (r'34\d{3}\s+montpellier', 'montpellier'),
            (r'35\d{3}\s+rennes', 'rennes')
        ]
        
        for pattern, city in city_patterns:
            if re.search(pattern, address_lower):
                return city
        
        # Fallback: simple city name matching
        simple_cities = ['paris', 'lyon', 'marseille', 'bordeaux', 'toulouse', 
                        'lille', 'strasbourg', 'nantes', 'montpellier', 'rennes']
        for city in simple_cities:
            if city in address_lower:
                return city
                
        return None
    
    def _find_best_candidate(self, target_name: str, candidates: List[Dict[str, Any]], lat: float = None, lng: float = None, category: str = None) -> Optional[Dict[str, Any]]:
        """Find the best matching candidate from search results with enhanced scoring"""
        import unicodedata
        
        def normalize_name(name: str) -> str:
            """Normalize name by removing diacritics and lowercasing"""
            name = unicodedata.normalize('NFD', name.lower().strip())
            return ''.join(c for c in name if unicodedata.category(c) != 'Mn')
        
        target_name_normalized = normalize_name(target_name)
        best_candidates = []
        
        for candidate in candidates:
            candidate_name = candidate.get('name', '')
            candidate_name_normalized = normalize_name(candidate_name)
            score = 0
            
            # Exact match on normalized name = +2
            if candidate_name_normalized == target_name_normalized:
                score += 2
            elif target_name_normalized in candidate_name_normalized or candidate_name_normalized in target_name_normalized:
                score += 1  # Partial match
            
            # Category match = +1 
            if category:
                candidate_types = candidate.get('types', [])
                category_map = {
                    'restaurant': ['restaurant', 'food', 'meal_takeaway'],
                    'bar': ['bar', 'night_club', 'liquor_store'],
                    'cafe': ['cafe', 'bakery', 'coffee_shop']
                }
                if category in category_map:
                    if any(t in candidate_types for t in category_map[category]):
                        score += 1
                elif category in candidate_types:
                    score += 1
            
            # Distance < 10km from bias = +1
            if lat and lng:
                try:
                    candidate_lat = candidate['geometry']['location']['lat']
                    candidate_lng = candidate['geometry']['location']['lng']
                    distance = self._calculate_distance(lat, lng, candidate_lat, candidate_lng)
                    if distance <= 10:  # 10km radius
                        score += 1
                    else:
                        score = 0  # Too far, reject completely
                except:
                    pass  # No geometry data, skip distance check
            
            # Skip candidates with score 0
            if score > 0:
                # Use user_ratings_total as tie-breaker
                ratings_count = candidate.get('user_ratings_total', 0)
                best_candidates.append((score, ratings_count, candidate))
        
        if not best_candidates:
            return None
        
        # Sort by score (desc), then by ratings count (desc)
        best_candidates.sort(key=lambda x: (-x[0], -x[1]))
        return best_candidates[0][2]
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate approximate distance in km between two coordinates"""
        from math import radians, sin, cos, sqrt, asin
        
        # Haversine formula
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
        return 2 * asin(sqrt(a)) * 6371  # Earth radius in km
    
    def _print_poi_summary(self, poi_data: Dict[str, Any]):
        """Print POI data summary for dry-run"""
        print("\n📋 POI Summary:")
        print(f"  Name: {poi_data.get('name', 'N/A')}")
        print(f"  City: {poi_data.get('city', 'N/A')}")
        print(f"  Address: {poi_data.get('address_street', 'N/A')}")
        print(f"  Category: {poi_data.get('category', 'N/A')}")
        print(f"  Rating: {poi_data.get('rating', 'N/A')}")
        print(f"  Website: {poi_data.get('website', 'N/A')}")
        print(f"  Google Place ID: {poi_data.get('google_place_id', 'N/A')}")
        print(f"  Coordinates: {poi_data.get('lat', 'N/A')}, {poi_data.get('lng', 'N/A')}")
    
    def _output_json_dry_run(self, poi_data: Dict[str, Any]):
        """Output dry-run JSON format"""
        output = {
            "status": "dry_run",
            "poi_preview": {
                "name": poi_data.get('name'),
                "city_slug": poi_data.get('city'),
                "google_place_id": poi_data.get('google_place_id'),
                "lat": poi_data.get('lat'),
                "lng": poi_data.get('lng')
            }
        }
        print(json.dumps(output, separators=(',', ':')))
    
    def _output_json_upserted(self, poi_id: str, poi_data: Dict[str, Any]):
        """Output upserted JSON format"""
        output = {
            "status": "upserted",
            "poi": {
                "id": poi_id,
                "name": poi_data.get('name'),
                "city_slug": poi_data.get('city'),
                "google_place_id": poi_data.get('google_place_id')
            }
        }
        print(json.dumps(output, separators=(',', ':')))
    
    def _output_json_error(self, message: str, code: str):
        """Output error JSON format"""
        output = {
            "status": "error",
            "message": message,
            "code": code
        }
        print(json.dumps(output, separators=(',', ':')))
    
    def run_test_ingestion(self, city_slug: str = 'paris', neighborhood: str = None, category: str = None, limit: int = 3) -> Dict[str, Any]:
        """Run light test ingestion"""
        logger.info(f"Starting test ingestion: city={city_slug}, neighborhood={neighborhood}, category={category}, limit={limit}")
        
        query = f"{category or 'restaurant'} {neighborhood or '1er arrondissement'} {city_slug}"
        places = self.search_places_textsearch(query)
        
        for place in places[:limit]:
            poi_data = self.convert_place_data(place, city_slug=city_slug)
            if poi_data:
                poi_id = self.ingest_poi_to_db(poi_data, place_data=place)
                if not poi_id:
                    self.skipped_count += 1
            else:
                self.skipped_count += 1
        
        cost_estimate = self.get_cost_estimate()
        
        # Log final summary as JSON
        summary = {
            "ingested": self.ingested_count,
            "skipped": self.skipped_count,
            "cost_estimate": f"${cost_estimate['estimate_usd']}",
            "tokens_left": cost_estimate['tokens_remaining']
        }
        logger.info(f"Test ingestion summary: {json.dumps(summary)}")
        
        return {
            'total_ingested': self.ingested_count,
            'total_skipped': self.skipped_count,
            'cost_estimate': cost_estimate
        }

# MOCK TESTS - Integrated in same file, no network calls
def run_mock_tests():
    """Run mock tests without network calls"""
    logger.info("🧪 Running Sprint 2 Mock Tests...")
    
    test_count = 0
    
    try:
        ingester = GooglePlacesIngesterV2(mock_mode=True)
        
        # Test 1: Token bucket initialization
        test_count += 1
        assert ingester.tokens_remaining == int(os.environ.get('PLACES_DAILY_TOKENS', '5000'))
        logger.info(f"✅ Test {test_count}: Token bucket initialization")
        
        # Test 2: Token consumption
        test_count += 1
        initial_tokens = ingester.tokens_remaining
        success = ingester._consume_token('basic')
        assert success == True
        assert ingester.tokens_remaining == initial_tokens - 1
        assert ingester.basic_calls == 1
        logger.info(f"✅ Test {test_count}: Token consumption")
        
        # Test 3: Cost estimation
        test_count += 1
        cost = ingester.get_cost_estimate()
        assert cost['basic_calls'] == 1
        assert cost['estimate_usd'] > 0
        logger.info(f"✅ Test {test_count}: Cost estimation (${cost['estimate_usd']})")
        
        # Test 4: Mock API call
        test_count += 1
        place_details = ingester.get_place_details('mock-place-123', include_contact=True)
        assert place_details is not None
        assert 'place_id' in place_details
        assert 'website' in place_details
        logger.info(f"✅ Test {test_count}: Mock API call")
        
        # Test 5: POI data conversion
        test_count += 1
        poi_data = ingester.convert_place_data(place_details)
        assert poi_data is not None
        assert poi_data['name'] == place_details['name']
        assert poi_data['category'] is not None
        logger.info(f"✅ Test {test_count}: POI data conversion")
        
        # Test 6: Mock ingestion
        test_count += 1
        poi_id = ingester.ingest_poi_to_db(poi_data, place_data=place_details)
        assert poi_id is not None
        assert ingester.ingested_count == 1
        logger.info(f"✅ Test {test_count}: Mock POI ingestion")
        
        # Test 7: Field mask verification
        test_count += 1
        basic_fields = ingester.basic_fields
        contact_fields = ingester.contact_fields
        assert 'place_id' in basic_fields
        assert 'website' in contact_fields
        logger.info(f"✅ Test {test_count}: Field mask verification")
        
        # Test 8: Mock search
        test_count += 1
        places = ingester.search_places_textsearch('restaurant paris')
        assert len(places) > 0
        logger.info(f"✅ Test {test_count}: Mock search ({len(places)} results)")
        
        # Test 9: Complete mock run
        test_count += 1
        result = ingester.run_seed_ingestion('paris', limit=3)
        assert result['total_ingested'] > 0
        logger.info(f"✅ Test {test_count}: Complete mock run ({result['total_ingested']} ingested)")
        
        # Final summary
        final_cost = ingester.get_cost_estimate()
        summary = {
            "ingested": ingester.ingested_count,
            "skipped": ingester.skipped_count,
            "cost_estimate": f"${final_cost['estimate_usd']}",
            "tokens_left": final_cost['tokens_remaining']
        }
        logger.info(f"Mock test summary: {json.dumps(summary)}")
        
        logger.info(f"🎉 All {test_count} tests passed!")
        logger.info("S2_MOCKS_OK")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Mock tests failed: {e}")
        return 1

def main():
    """CLI interface for Google Places Ingester V2"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Google Places Ingester V2 - Sprint 2')
    
    # Main operation flags
    parser.add_argument('--run-mocks', action='store_true', help='Execute mock dataset without API calls')
    parser.add_argument('--seed', action='store_true', help='Run initial seed ingestion or light refresh')
    parser.add_argument('--test', action='store_true', help='Run test ingestion')
    
    # Targeted ingestion flags
    parser.add_argument('--poi-name', type=str, help='Target specific POI by name (requires --city)')
    parser.add_argument('--place-id', type=str, help='Target specific POI by Google Place ID')
    parser.add_argument('--city', type=str, help='City name for POI search (required with --poi-name)')
    parser.add_argument('--lat', type=float, help='Latitude for geographic bias (optional)')
    parser.add_argument('--lng', type=float, help='Longitude for geographic bias (optional)')
    parser.add_argument('--commit', action='store_true', help='Actually persist data (default is dry-run for targeted ingestion)')
    parser.add_argument('--stdout-json', action='store_true', help='Output single JSON line to stdout instead of regular logs')
    parser.add_argument('--allow-fuzzy-upsert', action='store_true', help='Allow fuzzy upsert by city_slug + name if google_place_id not found (default: False)')
    
    # Configuration flags
    parser.add_argument('--city-slug', default='paris', help='City slug (e.g., paris)')
    parser.add_argument('--dry-run', action='store_true', help='Do not persist data, print summary only')
    parser.add_argument('--limit', type=int, help='Maximum number of POIs to process')
    parser.add_argument('--debug', action='store_true', help='Enable detailed logging')
    
    # Optional filters
    parser.add_argument('--neighborhood', help='Specific neighborhood')
    parser.add_argument('--category', help='POI category for disambiguation (restaurant|bar|cafe|etc.)')
    
    args = parser.parse_args()
    
    # Configure logging based on debug flag and stdout-json mode
    if args.stdout_json:
        # Suppress all logging when stdout-json is enabled
        logging.basicConfig(level=logging.CRITICAL)
    else:
        log_level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    try:
        if args.run_mocks:
            exit_code = run_mock_tests()
            sys.exit(exit_code)
        
        # Handle targeted ingestion
        if args.place_id or args.poi_name:
            # Validation
            if args.poi_name and not args.city:
                logger.error("--poi-name requires --city to be specified")
                sys.exit(1)
            
            # For targeted ingestion, default to dry-run unless --commit is specified
            dry_run_mode = not args.commit
            
            ingester = GooglePlacesIngesterV2(
                dry_run=False,  # We handle dry-run in the targeted methods
                mock_mode=False
            )
            
            if args.place_id:
                # Place ID path (priority)
                result = ingester.ingest_poi_by_place_id(
                    place_id=args.place_id,
                    city_slug=args.city.lower() if args.city else None,
                    dry_run=dry_run_mode,
                    json_output=args.stdout_json,
                    allow_fuzzy_upsert=args.allow_fuzzy_upsert
                )
            else:
                # POI name path
                result = ingester.ingest_poi_by_name(
                    poi_name=args.poi_name,
                    city=args.city,
                    lat=args.lat,
                    lng=args.lng,
                    category=args.category,
                    dry_run=dry_run_mode,
                    json_output=args.stdout_json,
                    allow_fuzzy_upsert=args.allow_fuzzy_upsert
                )
            
            # Handle targeted ingestion results
            if result['success']:
                if result.get('dry_run') or result.get('poi_id'):
                    # JSON output already handled in the methods
                    if not args.stdout_json:
                        if result.get('dry_run'):
                            print("\n✅ DRY-RUN: Ready to upsert")
                        elif result.get('poi_id'):
                            poi_data = result['poi_data']
                            print(f"\n✅ UPSERT ok: {result['poi_id']} | {poi_data.get('city', 'unknown')} | {poi_data.get('name', 'unnamed')}")
                    sys.exit(0)
                elif result.get('candidates_found') == 0:
                    if args.stdout_json:
                        ingester._output_json_error(f"No candidates found for '{args.poi_name}' in {args.city}", "no_candidates")
                    else:
                        print(f"\nℹ️  No candidates found for '{args.poi_name}' in {args.city}")
                    sys.exit(0)
                elif result.get('best_match') is None:
                    if args.stdout_json:
                        ingester._output_json_error(f"No suitable candidate found for '{args.poi_name}' in {args.city}", "no_suitable_candidate")
                    else:
                        print(f"\nℹ️  No suitable candidate found for '{args.poi_name}' in {args.city}")
                    sys.exit(0)
            else:
                error_msg = f"Targeted ingestion failed: {result.get('error', 'Unknown error')}"
                if args.stdout_json:
                    ingester._output_json_error(error_msg, "ingestion_failed")
                else:
                    logger.error(error_msg)
                sys.exit(1)
        
        # Initialize ingester with appropriate modes for bulk operations
        ingester = GooglePlacesIngesterV2(
            dry_run=args.dry_run,
            mock_mode=False
        )
        
        if args.test:
            result = ingester.run_test_ingestion(
                city_slug=args.city_slug,
                neighborhood=args.neighborhood,
                category=args.category,
                limit=args.limit or 3
            )
        elif args.seed:
            result = ingester.run_seed_ingestion(
                city_slug=args.city_slug,
                neighborhood=args.neighborhood,
                category=args.category,
                limit=args.limit
            )
        else:
            # Default behavior - run seed
            result = ingester.run_seed_ingestion(
                city_slug=args.city_slug,
                neighborhood=args.neighborhood,
                category=args.category,
                limit=args.limit
            )
        
        # Print final results
        mode = "DRY RUN" if args.dry_run else "LIVE"
        operation = "TEST" if args.test else "SEED"
        
        print(f"\n🎯 {operation} Results ({mode}):")
        print(f"   Ingested: {result['total_ingested']} POIs")
        print(f"   Skipped: {result['total_skipped']} POIs")
        print(f"💰 Cost Estimate: ${result['cost_estimate']['estimate_usd']}")
        print(f"🪣 Tokens Remaining: {result['cost_estimate']['tokens_remaining']}")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

# Alias for backward compatibility with run_pipeline.py
GooglePlacesIngester = GooglePlacesIngesterV2

if __name__ == "__main__":
    main()
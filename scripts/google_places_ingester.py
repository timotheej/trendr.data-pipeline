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
    
    def __init__(self, dry_run: bool = False, mock_mode: bool = False):
        self.dry_run = dry_run
        self.mock_mode = mock_mode
        self.config = get_config()
        
        if not mock_mode:
            self.db = SupabaseManager()
            if not self.config.google_places_api_key:
                logger.error("Missing GOOGLE_PLACES_API_KEY")
                sys.exit(1)
            self.api_key = self.config.google_places_api_key
        else:
            self.db = None
            self.api_key = 'mock-key'
        
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
        """Get place details"""
        if self.mock_mode:
            return {
                'place_id': place_id,
                'rating': 4.2,
                'user_ratings_total': 100
            }
        
        if not self._consume_token('basic'):
            return None
        
        try:
            url = "https://maps.googleapis.com/maps/api/place/details/json"
            params = {
                'place_id': place_id,
                'fields': 'formatted_address,formatted_phone_number,international_phone_number,website,opening_hours,photos,rating,user_ratings_total,price_level',
                'key': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'OK':
                return data.get('result')
            return None
                
        except Exception as e:
            logger.error(f"Place details API error: {e}")
            return None
    
    def search_places_nearby(self, location: str, radius: int, place_type: str) -> List[Dict[str, Any]]:
        """Search places using Google Places Nearby Search API"""
        if self.mock_mode:
            return [
                {
                    'place_id': f'mock-{place_type}-{i}',
                    'name': f'Mock {place_type.title()} {i}',
                    'geometry': {'location': {'lat': 48.8566 + i*0.0005, 'lng': 2.3522 + i*0.0005}},
                    'types': [place_type],
                    'rating': 4.0 + i*0.1,
                    'user_ratings_total': 50 + i*10,
                    'formatted_address': f'Address {i}, Paris, France'
                } for i in range(1, 10)
            ]
        
        if not self._consume_token('basic'):
            return []
        
        try:
            url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                'location': location,
                'radius': radius,
                'type': place_type,
                'key': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return data.get('results', [])
            
        except Exception as e:
            logger.error(f"Nearby search failed for '{place_type}' at {location}: {e}")
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
            
            # Extract rating data for snapshot and eligibility
            rating = row.pop('_rating', None)
            user_ratings_total = row.pop('_user_ratings_total', None)
            
            # Determine eligibility status based on quality thresholds
            # The key requirement: rating thresholds don't block POI insertion, only affect eligibility
            if rating is not None and user_ratings_total is not None:
                if rating >= self.rating_min and user_ratings_total >= self.min_reviews:
                    row['eligibility_status'] = 'eligible'
                else:
                    row['eligibility_status'] = 'hold'
            else:
                # No rating data available - mark as hold until we get rating info
                row['eligibility_status'] = 'hold'
            
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
                # Insert new POI - ALWAYS successful regardless of rating
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
    parser.add_argument('--h3-ingest', action='store_true', help='Run H3-based ingestion')
    parser.add_argument('--limit-cells', type=int, default=300, help='Max H3 cells to process')
    parser.add_argument('--update-interval-days', type=int, default=7, help='TTL for cell rescanning')
    parser.add_argument('--debug-cell', type=str, help='Debug: scan specific H3 cell')
    parser.add_argument('--city-slug', default='paris', help='City slug')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        ingester = GooglePlacesIngester(dry_run=args.dry_run, mock_mode=False)
        
        if args.h3_ingest:
            result = ingester.run_h3_ingestion(
                city_slug=args.city_slug,
                limit_cells=args.limit_cells,
                update_interval_days=args.update_interval_days,
                debug_cell=args.debug_cell
            )
        else:
            result = ingester.run_h3_ingestion(city_slug=args.city_slug)
        
        # Print results
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
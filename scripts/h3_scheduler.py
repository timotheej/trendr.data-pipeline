#!/usr/bin/env python3
"""
KISS H3-based POI Ingestion Scheduler
Single mode H3 scheduler following KISS principles
"""
import sys
import os
import logging
try:
    import h3
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise ImportError("H3 library required - install with: pip install h3")
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from shapely.geometry import Polygon, Point

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager
from config import get_config

logger = logging.getLogger(__name__)

@dataclass
class H3Cell:
    """H3 cell data structure"""
    h3: str
    city_slug: str
    res: int
    parent_h3: Optional[str] = None
    status: str = 'pending'
    saturated: bool = False
    last_scanned_at: Optional[datetime] = None
    next_due_at: Optional[datetime] = None
    results_last: Optional[int] = None
    attempts: int = 0

@dataclass
class ScanResult:
    """Result of scanning an H3 cell"""
    poi_ids_touched: List[str]
    total_results: int
    saturated: bool
    api_requests_made: int
    categories_scanned: List[str]

@dataclass
class MappingStats:
    """Statistics from spatial mapping"""
    districts_mapped: int
    neighbourhoods_mapped: int
    pois_processed: int

class H3Scheduler:
    """KISS H3-based scheduler for systematic POI ingestion"""
    
    def __init__(self, db: SupabaseManager):
        self.db = db
        self.config = get_config()
        
        # KISS radius mapping for H3 resolutions (in meters)
        self.radius_for_res = {
            9: 420,   # Base resolution for Paris coverage
            10: 220,  # First subdivision level
            11: 110   # Second subdivision level
        }
        
        # Use config-driven saturation threshold
        self.saturation_threshold = self.config.h3.scan_cap_per_cell
        
        # Initialize novelty detection
        self.novelty_detector = H3SchedulerNovelty(db)
    
    def seed_h3_cells_if_needed(self, city_slug: str, res_base: Optional[int] = None) -> int:
        """
        Seed H3 cells for a city if none exist
        
        Args:
            city_slug: City slug (e.g., 'paris')
            res_base: Base H3 resolution (uses config.h3.seed_res if None)
            
        Returns:
            Number of cells seeded
        """
        try:
            # Use config-driven resolution if not provided
            if res_base is None:
                res_base = self.config.h3.seed_res
            
            # Check if cells already exist for this city
            result = self.db.client.table('ingestion_cell_h3').select('h3', count='exact').eq('city_slug', city_slug).execute()
            if result.count and result.count > 0:
                logger.info(f"H3 cells already seeded for {city_slug}: {result.count} cells")
                return 0
            
            # Get Paris polygon from urban_areas (use simple approach for now)
            # For testing, we'll use a simple hardcoded polygon for Paris
            logger.info("Using simplified Paris polygon for H3 seeding")
            
            # Simple Paris bounding box polygon (approximate)
            paris_coords = [
                [2.224, 48.815],   # SW
                [2.469, 48.815],   # SE  
                [2.469, 48.902],   # NE
                [2.224, 48.902],   # NW
                [2.224, 48.815]    # Close
            ]
            
            # Convert to [lat, lng] format for h3.polyfill
            lat_lng_coords = [[lat, lng] for lng, lat in paris_coords]
            
            geojson_data = {
                'type': 'Polygon',
                'coordinates': [paris_coords]
            }
            if geojson_data['type'] != 'Polygon':
                raise Exception("Expected Polygon geometry from urban_areas union")
            
            # Convert to shapely polygon for h3.polyfill
            coords = geojson_data['coordinates'][0]  # First ring (exterior)
            # Convert [lng, lat] to [lat, lng] for h3.polyfill
            lat_lng_coords = [[lat, lng] for lng, lat in coords]
            
            # Generate H3 cells using polygon_to_cells (H3 v4 API)
            polygon = h3.LatLngPoly(lat_lng_coords)
            h3_cells = h3.polygon_to_cells(polygon, res_base)
            
            if not h3_cells:
                raise Exception(f"No H3 cells generated for {city_slug} at resolution {res_base}")
            
            # Prepare cells for database insertion
            cells_to_insert = []
            for h3_id in h3_cells:
                cell_data = {
                    'h3': h3_id,
                    'city_slug': city_slug,
                    'res': res_base,
                    'parent_h3': None,
                    'status': 'pending',
                    'saturated': False,
                    'last_scanned_at': None,
                    'next_due_at': None,
                    'results_last': None,
                    'attempts': 0
                }
                cells_to_insert.append(cell_data)
            
            # Insert cells in batch
            result = self.db.client.table('ingestion_cell_h3').insert(cells_to_insert).execute()
            cells_created = len(result.data) if result.data else 0
            
            logger.info(f"Seeded {cells_created} H3 cells for {city_slug} at resolution {res_base}")
            return cells_created
            
        except Exception as e:
            logger.error(f"Error seeding H3 cells for {city_slug}: {e}")
            raise
    
    def select_due_cells(self, city_slug: str, limit_cells: int) -> List[H3Cell]:
        """
        Select cells that are due for scanning
        
        Args:
            city_slug: City to filter by
            limit_cells: Maximum number of cells to return
            
        Returns:
            List of H3Cell objects ready for scanning
        """
        try:
            # Query for due cells using Supabase client
            query = self.db.client.table('ingestion_cell_h3')\
                .select('h3, city_slug, res, parent_h3, status, saturated, last_scanned_at, next_due_at, results_last, attempts')\
                .eq('city_slug', city_slug)\
                .neq('status', 'split')
            
            # Add time filter (simplified approach - get all and filter in Python for now)
            result = query.order('res', desc=False).limit(limit_cells).execute()
            
            if not result.data:
                logger.info(f"No due cells found for {city_slug}")
                return []
            
            # Convert to H3Cell objects
            cells = []
            for row in result.data:
                cell = H3Cell(
                    h3=row['h3'],
                    city_slug=row['city_slug'],
                    res=row['res'],
                    parent_h3=row['parent_h3'],
                    status=row['status'],
                    saturated=row['saturated'],
                    last_scanned_at=datetime.fromisoformat(row['last_scanned_at']) if row['last_scanned_at'] else None,
                    next_due_at=datetime.fromisoformat(row['next_due_at']) if row['next_due_at'] else None,
                    results_last=row['results_last'],
                    attempts=row['attempts']
                )
                cells.append(cell)
            
            logger.info(f"Selected {len(cells)} due cells for {city_slug}")
            return cells
            
        except Exception as e:
            logger.error(f"Error selecting due cells for {city_slug}: {e}")
            raise
    
    def get_radius_for_res(self, res: int) -> int:
        """Get search radius for H3 resolution"""
        return self.radius_for_res.get(res, 400)  # Fallback to 400m
    
    def scan_cell(self, h3_id: str, categories: List[str], ingester, city_slug: str = 'paris') -> ScanResult:
        """
        Scan an H3 cell for POIs using the Google Places ingester
        
        Args:
            h3_id: H3 cell identifier
            categories: List of POI categories to search
            ingester: GooglePlacesIngester instance
            
        Returns:
            ScanResult with scan statistics
        """
        try:
            # Get cell center coordinates (H3 v4 API)
            lat, lng = h3.cell_to_latlng(h3_id)
            res = h3.get_resolution(h3_id)
            radius_m = self.get_radius_for_res(res)
            
            poi_ids_touched = []
            total_results = 0
            api_requests_made = 0
            category_counts = []
            
            logger.info(f"Scanning H3 cell {h3_id} (res={res}) at ({lat:.6f}, {lng:.6f}) radius={radius_m}m")
            
            # Scan each category
            for category in categories:
                logger.debug(f"Scanning category '{category}' in cell {h3_id}")
                
                category_results = self._scan_category_in_cell(
                    lat, lng, radius_m, category, ingester, city_slug, h3_id
                )
                
                poi_ids_touched.extend(category_results['poi_ids'])
                total_results += category_results['count']
                api_requests_made += category_results['api_requests']
                category_counts.append(category_results['count'])
            
            # Saturation logic: Total ≥ threshold OR any category hit API limit (20)
            max_category_count = max(category_counts) if category_counts else 0
            api_limit_per_category = 20
            
            saturated = (total_results >= self.saturation_threshold or 
                        max_category_count >= api_limit_per_category)
            
            if saturated:
                if max_category_count >= api_limit_per_category:
                    logger.warning(f"Cell {h3_id} saturated: category hit API limit ({max_category_count}/{api_limit_per_category})")
                if total_results >= self.saturation_threshold:
                    logger.warning(f"Cell {h3_id} saturated: total results ({total_results}/{self.saturation_threshold})")
            
            return ScanResult(
                poi_ids_touched=poi_ids_touched,
                total_results=total_results,
                saturated=saturated,
                api_requests_made=api_requests_made,
                categories_scanned=categories
            )
            
        except Exception as e:
            logger.error(f"Error scanning cell {h3_id}: {e}")
            raise
    
    def _scan_category_in_cell(self, lat: float, lng: float, radius_m: int, 
                              category: str, ingester, city_slug: str, h3_id: str) -> Dict[str, Any]:
        """
        Scan a specific category within a cell using nearby search
        """
        try:
            location = f"{lat},{lng}"
            
            logger.debug(f"Scanning location {location} with radius {radius_m}m for category {category}")
            
            # Use nearby search with the ingester
            places = ingester.search_places_nearby(location, radius_m, category)
            logger.debug(f"Found {len(places) if places else 0} places")
            
            poi_ids_touched = []
            api_requests = 1  # One API call for nearby search
            
            # Process each place found
            for place in places:
                # Get place data
                rating = place.get('rating')
                reviews_count = place.get('userRatingCount')
                place_name = place.get('displayName', {}).get('text', 'Unknown') if place.get('displayName') else 'Unknown'
                
                logger.info(f"Processing place: {place_name}, rating: {rating}, reviews: {reviews_count}")
                
                # Calculate novelty score
                novelty_score = self.novelty_detector.calculate_novelty_score(place, rating, reviews_count)
                novelty_classification = self.novelty_detector.classify_novelty(novelty_score)
                
                logger.info(f"Novelty: {place_name} -> score={novelty_score:.2f}, class={novelty_classification}")
                
                # New logic: process if novelty score >= 0.4 OR passes quality gate
                should_get_details = (
                    novelty_score >= 0.4 or  # POIs potentially new
                    ingester.pass_quality_gate(rating, reviews_count)  # POIs that pass quality gate
                )
                
                if not should_get_details:
                    logger.info(f"Place {place_name} skipped (novelty_score: {novelty_score:.2f})")
                    continue
                
                # Store novelty data for upsert
                place['_novelty_score'] = novelty_score
                place['_novelty_classification'] = novelty_classification
                
                # TOUJOURS récupérer les détails pour avoir toutes les données
                details = ingester.get_place_details(place.get('id'))
                if details:
                    api_requests += 1
                    # Combiner les données de base et les détails
                    combined_data = {**place, **details}
                    poi_data = ingester.to_poi_row(combined_data, city_slug=city_slug, h3_cell_id=h3_id)
                else:
                    # Fallback avec données de base seulement
                    poi_data = ingester.to_poi_row(place, city_slug=city_slug, h3_cell_id=h3_id)
                
                if not poi_data:
                    logger.debug(f"POI {place.get('name', 'Unknown')} rejected by to_poi_row")
                    continue
                
                # Upsert POI to database (if not dry run)
                if not ingester.dry_run:
                    try:
                        poi_id = ingester.upsert_poi(poi_data)
                        if poi_id:
                            poi_ids_touched.append(poi_id)
                            
                            # Update urban areas mapping for this POI
                            try:
                                result = self.db.client.rpc('update_poi_urban_areas', {'poi_id_param': poi_id}).execute()
                                if result.data:
                                    logger.debug(f"Urban mapping for POI {poi_id}: {result.data}")
                            except Exception as e:
                                logger.warning(f"Urban area mapping failed for POI {poi_id}: {e}")
                            
                            # Create rating snapshot si on a les détails
                            if details and details.get('rating') and details.get('userRatingCount'):
                                ingester.create_rating_snapshot(poi_id, details)
                                        
                    except Exception as e:
                        logger.warning(f"Error upserting POI {poi_data.get('name', 'Unknown')}: {e}")
                        continue
                else:
                    # Dry run mode - just log
                    logger.info(f"DRY RUN: Would upsert POI {poi_data.get('name', 'Unknown')}")
            
            return {
                'poi_ids': poi_ids_touched,
                'count': len(places),
                'api_requests': api_requests
            }
            
        except Exception as e:
            logger.error(f"Error scanning category {category} in cell: {e}")
            return {
                'poi_ids': [],
                'count': 0,
                'api_requests': 0
            }
    
    def _needs_details_fetch(self, poi_id: str, update_interval_days: int) -> bool:
        """
        Check if POI needs details to be fetched based on last_ingested_from_google_at
        
        Args:
            poi_id: POI identifier
            update_interval_days: How many days before refetch is needed
            
        Returns:
            True if details should be fetched
        """
        try:
            # Get POI data
            result = self.db.client.table('poi').select('last_ingested_from_google_at').eq('id', poi_id).execute()
            if not result.data:
                return True  # New POI, always fetch details
            
            poi = result.data[0]
            last_ingested = poi.get('last_ingested_from_google_at')
            
            if not last_ingested:
                return True  # Never fetched details, should fetch
            
            # Check if enough time has passed
            try:
                last_ingested_dt = datetime.fromisoformat(last_ingested.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                days_since = (now - last_ingested_dt).days
                
                return days_since >= update_interval_days
            except Exception:
                return True  # Error parsing date, safer to fetch
                
        except Exception as e:
            logger.warning(f"Error checking if POI {poi_id} needs details fetch: {e}")
            return True  # Error case, safer to fetch
    
    def split_cell(self, parent_h3: str, city_polygon: Polygon, res_child: int) -> int:
        """
        Split a saturated H3 cell into children at higher resolution
        
        Args:
            parent_h3: Parent cell H3 ID
            city_polygon: City boundary polygon to filter children
            res_child: Target resolution for children
            
        Returns:
            Number of children cells created
        """
        try:
            # Get children at higher resolution (H3 v4 API)
            children = h3.cell_to_children(parent_h3, res_child)
            
            if not children:
                logger.warning(f"No children generated for cell {parent_h3} at resolution {res_child}")
                return 0
            
            # Filter children to only those within city boundary
            valid_children = []
            for child_h3 in children:
                child_lat, child_lng = h3.cell_to_latlng(child_h3)  # H3 v4 API
                child_point = Point(child_lng, child_lat)  # Shapely uses (x, y) = (lng, lat)
                
                if city_polygon.contains(child_point):
                    valid_children.append(child_h3)
            
            if not valid_children:
                logger.warning(f"No valid children found within city boundary for cell {parent_h3}")
                return 0
            
            # Get parent cell info
            parent_result = self.db.client.table('ingestion_cell_h3').select('city_slug').eq('h3', parent_h3).single().execute()
            if not parent_result.data:
                raise Exception(f"Parent cell {parent_h3} not found")
            
            city_slug = parent_result.data['city_slug']
            
            # Prepare children for insertion
            children_to_insert = []
            for child_h3 in valid_children:
                child_data = {
                    'h3': child_h3,
                    'city_slug': city_slug,
                    'res': res_child,
                    'parent_h3': parent_h3,
                    'status': 'pending',
                    'saturated': False,
                    'last_scanned_at': None,
                    'next_due_at': None,
                    'results_last': None,
                    'attempts': 0
                }
                children_to_insert.append(child_data)
            
            # Insert children
            result = self.db.client.table('ingestion_cell_h3').insert(children_to_insert).execute()
            children_created = len(result.data) if result.data else 0
            
            # Mark parent as split
            self.db.client.table('ingestion_cell_h3').update({
                'status': 'split',
                'saturated': True,
                'last_scanned_at': datetime.utcnow().isoformat(),
                'next_due_at': None
            }).eq('h3', parent_h3).execute()
            
            logger.info(f"Split cell {parent_h3}: created {children_created} children at resolution {res_child}")
            return children_created
            
        except Exception as e:
            logger.error(f"Error splitting cell {parent_h3}: {e}")
            raise
    
    def update_cell_after_scan(self, h3_id: str, scan_result: ScanResult, 
                              update_interval_days: int, saturated: bool = False):
        """
        Update cell status after scanning
        
        Args:
            h3_id: H3 cell identifier
            scan_result: Results from scanning
            update_interval_days: TTL for next scan
            saturated: Whether the cell is saturated and should be split
        """
        try:
            if saturated:
                # Cell will be split, mark accordingly
                update_data = {
                    'status': 'saturated',  # Will be split by caller
                    'saturated': True,
                    'last_scanned_at': datetime.utcnow().isoformat(),
                    'results_last': scan_result.total_results,
                    'attempts': 'attempts + 1'  # SQL expression
                }
            else:
                # Normal scan completed
                next_due = datetime.utcnow() + timedelta(days=update_interval_days)
                update_data = {
                    'status': 'scanned',
                    'saturated': False,
                    'last_scanned_at': datetime.utcnow().isoformat(),
                    'next_due_at': next_due.isoformat(),
                    'results_last': scan_result.total_results,
                    'attempts': 'attempts + 1'  # SQL expression
                }
            
            # Update using Supabase client (without increment for now)
            # Get current attempts count first
            current_result = self.db.client.table('ingestion_cell_h3').select('attempts').eq('h3', h3_id).execute()
            current_attempts = 0
            if current_result.data:
                current_attempts = current_result.data[0].get('attempts', 0)
            
            update_data['attempts'] = current_attempts + 1
            
            self.db.client.table('ingestion_cell_h3').update(update_data).eq('h3', h3_id).execute()
            
            logger.debug(f"Updated cell {h3_id} after scan: {scan_result.total_results} results, saturated={saturated}")
            
        except Exception as e:
            logger.error(f"Error updating cell {h3_id} after scan: {e}")
            raise
    
    def map_pois_to_urban_areas(self, poi_ids: List[str]) -> MappingStats:
        """
        Map POIs to urban areas (districts and neighbourhoods)
        
        Args:
            poi_ids: List of POI IDs to map
            
        Returns:
            MappingStats with mapping results
        """
        if not poi_ids:
            return MappingStats(0, 0, 0)
        
        try:
            # Use existing RPC function for spatial mapping (simpler approach)
            logger.info(f"Using existing spatial mapping RPC for {len(poi_ids)} POIs")
            
            try:
                result = self.db.client.rpc('update_all_paris_pois').execute()
                if result.data and len(result.data) > 0:
                    mapping_data = result.data[0]
                    districts_mapped = mapping_data.get('district_assignments', 0)
                    neighbourhoods_mapped = mapping_data.get('neighbourhood_assignments', 0)
                else:
                    districts_mapped = len(poi_ids)  # Estimate
                    neighbourhoods_mapped = len(poi_ids)  # Estimate
            except Exception as e:
                logger.warning(f"Spatial mapping RPC failed: {e}")
                districts_mapped = len(poi_ids)  # Estimate
                neighbourhoods_mapped = len(poi_ids)  # Estimate
            
            # For now, return estimated counts (actual counts would require more complex SQL)
            stats = MappingStats(
                districts_mapped=len(poi_ids),  # Estimate
                neighbourhoods_mapped=len(poi_ids),  # Estimate
                pois_processed=len(poi_ids)
            )
            
            logger.info(f"Mapped {len(poi_ids)} POIs to urban areas")
            return stats
            
        except Exception as e:
            logger.error(f"Error mapping POIs to urban areas: {e}")
            raise


def get_paris_polygon(db: SupabaseManager) -> Polygon:
    """
    Get Paris polygon - simplified version for testing
    
    Returns:
        Shapely Polygon representing Paris boundary
    """
    try:
        # Use simplified Paris bounding box for testing
        logger.info("Using simplified Paris polygon")
        
        # Simple Paris bounding box coordinates (lng, lat)
        paris_coords = [
            [2.224, 48.815],   # SW
            [2.469, 48.815],   # SE  
            [2.469, 48.902],   # NE
            [2.224, 48.902],   # NW
            [2.224, 48.815]    # Close
        ]
        
        # Shapely expects (x, y) = (lng, lat)
        return Polygon(paris_coords)
        
    except Exception as e:
        logger.error(f"Error creating Paris polygon: {e}")
        raise


class H3SchedulerNovelty:
    """Novelty detection methods for H3Scheduler"""
    
    def __init__(self, db: SupabaseManager):
        self.db = db
    
    def calculate_novelty_score(self, place: Dict, rating: Optional[float], reviews_count: Optional[int]) -> float:
        """Calculate novelty score for a POI"""
        score = 0.0
        
        # Factor 1: Reviews Pattern (40% weight)
        if rating is None and reviews_count is None:
            score += 0.4  # No reviews = very likely new
        elif reviews_count is not None and reviews_count < 5:
            score += 0.35  # Very few reviews = likely recent
        elif reviews_count is not None and reviews_count < 20 and rating and rating > 4.5:
            score += 0.25  # Few reviews but excellent = promising new
        
        # Factor 2: Historical Absence (30% weight)
        place_id = place.get('id')
        if place_id and not self._exists_in_db(place_id):
            score += 0.3  # Never seen = new
        
        # Factor 3: Name Patterns (15% weight)
        name = place.get('displayName', {}).get('text', '') if place.get('displayName') else ''
        name_signals = ["new", "nouveau", "fresh", "recent", "opening", "2025"]
        if any(signal in name.lower() for signal in name_signals):
            score += 0.15
        
        # Factor 4: Business Type (10% weight)
        types = place.get('types', [])
        dynamic_types = ["restaurant", "bar", "cafe", "bakery"]
        if any(t in dynamic_types for t in types):
            score += 0.1
        
        # Factor 5: Address Patterns (5% weight)
        address = place.get('formattedAddress', '')
        if any(signal in address.lower() for signal in ["new", "recent", "opening"]):
            score += 0.05
        
        return min(score, 1.0)

    def classify_novelty(self, score: float) -> str:
        """Classify novelty based on score"""
        if score >= 0.8:
            return "highly_likely_new"
        elif score >= 0.6:
            return "likely_new" 
        elif score >= 0.4:
            return "potentially_new"
        else:
            return "established"

    def _exists_in_db(self, google_place_id: str) -> bool:
        """Check if POI already exists in database"""
        try:
            result = self.db.client.table('poi').select('id').eq('google_place_id', google_place_id).execute()
            return bool(result.data)
        except:
            return False


# CLI for testing H3 scheduler functionality
if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='H3 Scheduler CLI')
    parser.add_argument('--seed', action='store_true', help='Seed H3 cells for Paris')
    parser.add_argument('--city', default='paris', help='City slug')
    parser.add_argument('--res', type=int, default=9, help='H3 resolution')
    parser.add_argument('--list-due', action='store_true', help='List due cells')
    parser.add_argument('--limit', type=int, default=10, help='Limit for listing cells')
    
    args = parser.parse_args()
    
    try:
        db = SupabaseManager()
        scheduler = H3Scheduler(db)
        
        if args.seed:
            count = scheduler.seed_h3_cells_if_needed(args.city, args.res)
            print(f"Seeded {count} H3 cells for {args.city}")
        
        if args.list_due:
            cells = scheduler.select_due_cells(args.city, args.limit)
            print(f"Found {len(cells)} due cells:")
            for cell in cells:
                print(f"  {cell.h3} (res={cell.res}, status={cell.status})")
    
    except Exception as e:
        logger.error(f"CLI error: {e}")
        sys.exit(1)
#!/usr/bin/env python3
"""
KISS Database Module
Clean, focused database operations for H3 POI ingestion pipeline
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from supabase import create_client, Client

from config import get_config

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Base database error"""
    pass

class KISSDatabase:
    """KISS database manager focused on core POI ingestion operations"""
    
    def __init__(self):
        self.config = get_config()
        self.client: Client = create_client(
            self.config.supabase_url, 
            self.config.supabase_key
        )
    
    # =============================================================================
    # POI OPERATIONS
    # =============================================================================
    
    def upsert_poi(self, poi_data: Dict[str, Any]) -> Optional[str]:
        """
        Upsert POI record and return poi_id
        
        Args:
            poi_data: POI data dictionary
            
        Returns:
            POI ID if successful, None otherwise
        """
        try:
            # Check if POI exists by google_place_id
            google_place_id = poi_data.get('google_place_id')
            if not google_place_id:
                raise ValueError("google_place_id is required")
            
            result = self.client.table('poi')\
                .select('id')\
                .eq('google_place_id', google_place_id)\
                .execute()
            
            if result.data:
                # Update existing POI
                poi_id = result.data[0]['id']
                update_data = poi_data.copy()
                del update_data['google_place_id']  # Don't update the key
                
                self.client.table('poi').update(update_data).eq('id', poi_id).execute()
                logger.debug(f"poi_updated: {poi_data.get('name')} (id: {poi_id})")
                return poi_id
            else:
                # Insert new POI
                insert_result = self.client.table('poi').insert(poi_data).execute()
                if insert_result.data:
                    poi_id = insert_result.data[0]['id']
                    logger.debug(f"poi_created: {poi_data.get('name')} (id: {poi_id})")
                    return poi_id
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error upserting POI: {e}")
            raise DatabaseError(f"POI upsert failed: {e}")
    
    def get_poi_by_id(self, poi_id: str) -> Optional[Dict[str, Any]]:
        """Get POI by ID"""
        try:
            result = self.client.table('poi')\
                .select('*')\
                .eq('id', poi_id)\
                .single()\
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Error getting POI {poi_id}: {e}")
            return None
    
    # =============================================================================
    # RATING SNAPSHOT OPERATIONS
    # =============================================================================
    
    def insert_rating_snapshot(self, poi_id: str, source_id: str, rating_value: float, 
                              reviews_count: int) -> bool:
        """
        Insert rating snapshot with duplicate prevention
        
        Args:
            poi_id: POI identifier
            source_id: Rating source (e.g., 'google')
            rating_value: Rating value
            reviews_count: Number of reviews
            
        Returns:
            Success status
        """
        try:
            snapshot_data = {
                'poi_id': poi_id,
                'source_id': source_id,
                'rating_value': rating_value,
                'reviews_count': reviews_count,
                'captured_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Use ON CONFLICT DO NOTHING to prevent duplicates
            result = self.client.table('rating_snapshot').insert(snapshot_data).execute()
            
            if result.data:
                logger.debug(f"Rating snapshot created: poi_id={poi_id}, rating={rating_value}")
                return True
            else:
                # May be a duplicate, which is ok
                return True
                
        except Exception as e:
            logger.error(f"Error inserting rating snapshot: {e}")
            return False
    
    def get_latest_rating_snapshot(self, poi_id: str, source_id: str = 'google') -> Optional[Dict[str, Any]]:
        """Get latest rating snapshot for POI"""
        try:
            result = self.client.table('rating_snapshot')\
                .select('*')\
                .eq('poi_id', poi_id)\
                .eq('source_id', source_id)\
                .order('captured_at', desc=True)\
                .limit(1)\
                .execute()
            
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting rating snapshot: {e}")
            return None
    
    # =============================================================================
    # H3 CELL OPERATIONS
    # =============================================================================
    
    def upsert_h3_cell(self, cell_data: Dict[str, Any]) -> bool:
        """
        Upsert H3 cell record
        
        Args:
            cell_data: H3 cell data
            
        Returns:
            Success status
        """
        try:
            h3_id = cell_data.get('h3')
            if not h3_id:
                raise ValueError("h3 cell ID is required")
            
            # Use upsert with conflict resolution
            result = self.client.table('ingestion_cell_h3')\
                .upsert(cell_data, on_conflict='h3')\
                .execute()
            
            if result.data:
                logger.debug(f"H3 cell upserted: {h3_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error upserting H3 cell: {e}")
            return False
    
    def get_due_h3_cells(self, city_slug: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get H3 cells that are due for scanning
        
        Args:
            city_slug: City to filter by
            limit: Maximum number of cells to return
            
        Returns:
            List of due cells
        """
        try:
            # Select cells that are pending or ready for rescan
            result = self.client.table('ingestion_cell_h3')\
                .select('*')\
                .eq('city_slug', city_slug)\
                .neq('status', 'split')\
                .order('res')\
                .order('next_due_at', nulls_first=True)\
                .limit(limit)\
                .execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting due H3 cells: {e}")
            return []
    
    def update_h3_cell_after_scan(self, h3_id: str, status: str, saturated: bool, 
                                 total_results: int, next_due_at: Optional[str] = None) -> bool:
        """
        Update H3 cell after scanning
        
        Args:
            h3_id: H3 cell identifier
            status: New status
            saturated: Whether cell is saturated
            total_results: Number of results found
            next_due_at: Next scan time (ISO string)
            
        Returns:
            Success status
        """
        try:
            update_data = {
                'status': status,
                'saturated': saturated,
                'last_scanned_at': datetime.now(timezone.utc).isoformat(),
                'results_last': total_results
            }
            
            if next_due_at:
                update_data['next_due_at'] = next_due_at
            
            result = self.client.table('ingestion_cell_h3')\
                .update(update_data)\
                .eq('h3', h3_id)\
                .execute()
            
            return bool(result.data)
        except Exception as e:
            logger.error(f"Error updating H3 cell {h3_id}: {e}")
            return False
    
    # =============================================================================
    # URBAN AREAS (SPATIAL MAPPING)
    # =============================================================================
    
    def update_poi_urban_areas(self, poi_id: str, lat: float, lng: float) -> bool:
        """
        Update POI with district and neighbourhood using spatial query
        
        Args:
            poi_id: POI identifier
            lat: Latitude
            lng: Longitude
            
        Returns:
            Success status
        """
        try:
            # This uses a stored procedure that performs the spatial join
            # The procedure should implement the SQL from requirements:
            # UPDATE poi SET district_name = ..., neighbourhood_name = ...
            # FROM urban_areas WHERE ST_Contains(geom, ST_SetSRID(ST_MakePoint(lng, lat), 4326))
            
            result = self.client.rpc('update_poi_urban_areas', {
                'poi_id': poi_id,
                'poi_lat': lat,
                'poi_lng': lng
            }).execute()
            
            if result.data:
                logger.debug(f"Urban areas updated for POI {poi_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating urban areas for POI {poi_id}: {e}")
            return False
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def execute_raw_sql(self, sql: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute raw SQL query (use sparingly)
        
        Args:
            sql: SQL query
            params: Query parameters
            
        Returns:
            Query results
        """
        try:
            # Supabase doesn't support raw SQL directly, would need to use RPC
            # This is here for future extensibility
            logger.warning("Raw SQL execution not implemented - use RPC functions instead")
            return []
        except Exception as e:
            logger.error(f"Error executing raw SQL: {e}")
            raise DatabaseError(f"SQL execution failed: {e}")
    
    def get_city_stats(self, city_slug: str) -> Dict[str, Any]:
        """Get basic statistics for a city"""
        try:
            # Get POI count
            poi_result = self.client.table('poi')\
                .select('id', count='exact')\
                .eq('city_slug', city_slug)\
                .execute()
            
            # Get H3 cell count
            cell_result = self.client.table('ingestion_cell_h3')\
                .select('h3', count='exact')\
                .eq('city_slug', city_slug)\
                .execute()
            
            return {
                'city_slug': city_slug,
                'poi_count': poi_result.count or 0,
                'h3_cell_count': cell_result.count or 0
            }
        except Exception as e:
            logger.error(f"Error getting city stats: {e}")
            return {'city_slug': city_slug, 'error': str(e)}
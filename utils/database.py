import logging
from typing import List, Dict, Any, Optional, Tuple
from supabase import create_client, Client
from tenacity import retry, stop_after_attempt, wait_exponential
import config
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseManager:
    """Extended Supabase manager with support for Collections, Neighborhoods, and SEO pages"""
    
    def __init__(self):
        self.client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    
    # =============================================================================
    # NEIGHBORHOOD METHODS
    # =============================================================================
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def insert_neighborhood(self, neighborhood_data: Dict[str, Any]) -> Optional[str]:
        """Insert a neighborhood record and return the ID."""
        try:
            result = self.client.table('neighborhoods').upsert(
                neighborhood_data,
                on_conflict='name,city,country'
            ).execute()
            
            if result.data:
                logger.info(f"Inserted/Updated neighborhood: {neighborhood_data['name']}")
                return result.data[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error inserting neighborhood {neighborhood_data['name']}: {e}")
            raise
    
    def get_neighborhood_by_name(self, name: str, city: str) -> Optional[Dict[str, Any]]:
        """Get neighborhood by name and city."""
        try:
            result = self.client.table('neighborhoods')\
                .select('*')\
                .eq('name', name)\
                .eq('city', city)\
                .execute()
            
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting neighborhood {name}: {e}")
            return None
    
    def get_neighborhoods_for_city(self, city: str) -> List[Dict[str, Any]]:
        """Get all neighborhoods for a city."""
        try:
            result = self.client.table('neighborhoods')\
                .select('*')\
                .eq('city', city)\
                .order('name')\
                .execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting neighborhoods for {city}: {e}")
            return []
    
    # =============================================================================
    # POI METHODS (Enhanced with neighborhood support)
    # =============================================================================
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def insert_poi(self, poi_data: Dict[str, Any]) -> Optional[str]:
        """Insert a POI record and return the ID. Now supports neighborhood_id."""
        try:
            # If neighborhood name is provided but no neighborhood_id, try to resolve it
            if 'neighborhood' in poi_data and 'neighborhood_id' not in poi_data:
                neighborhood = self.get_neighborhood_by_name(
                    poi_data['neighborhood'], 
                    poi_data.get('city', 'Montreal')
                )
                if neighborhood:
                    poi_data['neighborhood_id'] = neighborhood['id']
            
            result = self.client.table('poi').upsert(
                poi_data,
                on_conflict='name,address,city'
            ).execute()
            
            if result.data:
                logger.info(f"Inserted/Updated POI: {poi_data['name']}")
                return result.data[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error inserting POI {poi_data['name']}: {e}")
            raise

    def get_pois_for_city(self, city: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get POIs for a city with neighborhood information."""
        try:
            result = self.client.table('poi')\
                .select('*, neighborhoods!poi_neighborhood_id_fkey(name, description)')\
                .eq('city', city)\
                .limit(limit)\
                .execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting POIs for {city}: {e}")
            return []
    
    def get_pois_by_name(self, poi_name: str, city: str) -> List[Dict[str, Any]]:
        """Get POIs by name and city."""
        try:
            result = self.client.table('poi')\
                .select('*')\
                .ilike('name', f'%{poi_name}%')\
                .eq('city', city)\
                .execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting POIs by name {poi_name}: {e}")
            return []

    def get_pois_for_neighborhood(self, neighborhood_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get POIs for a specific neighborhood."""
        try:
            result = self.client.table('poi')\
                .select('*')\
                .eq('neighborhood_id', neighborhood_id)\
                .limit(limit)\
                .execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting POIs for neighborhood {neighborhood_id}: {e}")
            return []
    
    def get_pois_within_radius(self, lat: float, lng: float, radius_km: float = 5.0, limit: int = 20) -> List[Dict[str, Any]]:
        """Get POIs within a radius using the PostgreSQL function."""
        try:
            result = self.client.rpc('get_pois_within_radius', {
                'center_lat': lat,
                'center_lng': lng,
                'radius_km': radius_km,
                'limit_count': limit
            }).execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting POIs within radius: {e}")
            return []

    # =============================================================================
    # COLLECTION METHODS
    # =============================================================================
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def insert_collection(self, collection_data: Dict[str, Any]) -> Optional[str]:
        """Insert a collection/playlist record and return the ID."""
        try:
            result = self.client.table('collections').insert(collection_data).execute()
            
            if result.data:
                logger.info(f"Inserted collection: {collection_data['title']}")
                return result.data[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error inserting collection {collection_data['title']}: {e}")
            raise

    def get_collections_for_city(self, city: str, collection_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get collections for a city, optionally filtered by type."""
        try:
            query = self.client.table('collections')\
                .select('*, neighborhoods!collections_neighborhood_id_fkey(name)')\
                .eq('city', city)\
                .order('updated_at', desc=True)
            
            if collection_type:
                query = query.eq('type', collection_type)
                
            result = query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting collections for {city}: {e}")
            return []

    def get_collection_with_pois(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get a collection with its POIs populated."""
        try:
            # Get collection
            collection_result = self.client.table('collections')\
                .select('*')\
                .eq('id', collection_id)\
                .single()\
                .execute()
            
            if not collection_result.data:
                return None
                
            collection = collection_result.data
            
            # Get POIs for this collection
            if collection['poi_ids']:
                pois_result = self.client.table('poi')\
                    .select('*')\
                    .in_('id', collection['poi_ids'])\
                    .execute()
                
                collection['pois'] = pois_result.data or []
            else:
                collection['pois'] = []
                
            return collection
        except Exception as e:
            logger.error(f"Error getting collection with POIs {collection_id}: {e}")
            return None

    def update_collection_pois(self, collection_id: str, poi_ids: List[str]) -> bool:
        """Update the POI list for a collection."""
        try:
            result = self.client.table('collections')\
                .update({
                    'poi_ids': poi_ids,
                    'updated_at': datetime.utcnow().isoformat()
                })\
                .eq('id', collection_id)\
                .execute()
            
            return bool(result.data)
        except Exception as e:
            logger.error(f"Error updating collection POIs {collection_id}: {e}")
            return False

    def update_collection(self, collection_id: str, collection_data: Dict[str, Any]) -> bool:
        """Update an existing collection with new data."""
        try:
            # Add updated timestamp
            update_data = collection_data.copy()
            update_data['updated_at'] = datetime.utcnow().isoformat()
            
            result = self.client.table('collections')\
                .update(update_data)\
                .eq('id', collection_id)\
                .execute()
            
            return bool(result.data)
        except Exception as e:
            logger.error(f"Error updating collection {collection_id}: {e}")
            return False

    # =============================================================================
    # PROOF SOURCES METHODS (Enhanced)
    # =============================================================================
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def insert_proof_sources(self, sources: List[Dict[str, Any]]) -> int:
        """Insert multiple proof sources and return count of inserted records."""
        if not sources:
            return 0
            
        try:
            result = self.client.table('proof_sources').upsert(
                sources,
                on_conflict='poi_id,url'
            ).execute()
            
            count = len(result.data) if result.data else 0
            logger.info(f"Inserted/Updated {count} proof sources")
            return count
        except Exception as e:
            logger.error(f"Error inserting proof sources: {e}")
            raise

    def get_proof_sources_for_poi(self, poi_id: str) -> List[Dict[str, Any]]:
        """Get all proof sources for a POI."""
        try:
            result = self.client.table('proof_sources')\
                .select('*')\
                .eq('poi_id', poi_id)\
                .order('authority_score', desc=True)\
                .execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting proof sources for POI {poi_id}: {e}")
            return []

    def get_top_proof_sources_by_domain(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get top domains by proof source count."""
        try:
            result = self.client.rpc('get_proof_sources_by_domain_count', {
                'limit_count': limit
            }).execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting proof sources by domain: {e}")
            return []

    # =============================================================================
    # SEO PAGES METHODS  
    # =============================================================================
    
    def insert_seo_page(self, page_data: Dict[str, Any]) -> Optional[str]:
        """Insert an SEO page record."""
        try:
            result = self.client.table('seo_pages').upsert(
                page_data,
                on_conflict='slug'
            ).execute()
            
            if result.data:
                logger.info(f"Inserted/Updated SEO page: {page_data['slug']}")
                return result.data[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error inserting SEO page {page_data['slug']}: {e}")
            raise

    def get_seo_page_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get SEO page by slug."""
        try:
            result = self.client.table('seo_pages')\
                .select('*')\
                .eq('slug', slug)\
                .single()\
                .execute()
            
            return result.data
        except Exception as e:
            logger.error(f"Error getting SEO page {slug}: {e}")
            return None

    # =============================================================================
    # ANALYTICS & REPORTING
    # =============================================================================
    
    def get_city_statistics(self, city: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a city."""
        try:
            # Get POI count by category
            poi_stats = self.client.table('poi')\
                .select('category', count='*')\
                .eq('city', city)\
                .execute()
            
            # Get neighborhood count
            neighborhood_count = self.client.table('neighborhoods')\
                .select('id', count='exact')\
                .eq('city', city)\
                .execute()
            
            # Get collection count  
            collection_count = self.client.table('collections')\
                .select('id', count='exact')\
                .eq('city', city)\
                .execute()
            
            # Get proof sources count
            proof_count = self.client.rpc('get_proof_sources_count_for_city', {
                'city_name': city
            }).execute()
            
            return {
                'city': city,
                'poi_count': len(poi_stats.data) if poi_stats.data else 0,
                'neighborhood_count': neighborhood_count.count or 0,
                'collection_count': collection_count.count or 0,
                'proof_sources_count': proof_count.data[0]['count'] if proof_count.data else 0,
                'poi_by_category': {item['category']: item['count'] for item in poi_stats.data} if poi_stats.data else {}
            }
        except Exception as e:
            logger.error(f"Error getting city statistics for {city}: {e}")
            return {'city': city, 'error': str(e)}

    # =============================================================================
    # MIGRATION HELPERS
    # =============================================================================
    
    def migrate_poi_neighborhoods(self) -> int:
        """Migrate existing POI neighborhood text fields to neighborhood_id references."""
        try:
            # Get all POIs with neighborhood text but no neighborhood_id
            pois = self.client.table('poi')\
                .select('id, neighborhood, city')\
                .is_('neighborhood_id', 'null')\
                .neq('neighborhood', 'null')\
                .execute()
            
            updated_count = 0
            for poi in pois.data or []:
                neighborhood = self.get_neighborhood_by_name(poi['neighborhood'], poi['city'])
                if neighborhood:
                    self.client.table('poi')\
                        .update({'neighborhood_id': neighborhood['id']})\
                        .eq('id', poi['id'])\
                        .execute()
                    updated_count += 1
            
            logger.info(f"Migrated {updated_count} POIs to use neighborhood_id references")
            return updated_count
        except Exception as e:
            logger.error(f"Error migrating POI neighborhoods: {e}")
            return 0
    
    # =============================================================================
    # ORCHESTRATOR SUPPORT METHODS
    # =============================================================================
    
    def get_pois_needing_proof_sources(self, city: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get POIs that need proof sources (have few or none)."""
        try:
            # Get POIs with less than 3 proof sources or none at all
            result = self.client.table('poi')\
                .select('id, name, category, city')\
                .eq('city', city)\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            # Filter to only those with few proof sources
            pois_needing_proof = []
            for poi in result.data or []:
                proof_count = len(self.get_proof_sources_for_poi(poi['id']))
                if proof_count < 3:  # Threshold for needing more proof
                    pois_needing_proof.append(poi)
                
                if len(pois_needing_proof) >= limit:
                    break
            
            return pois_needing_proof
        except Exception as e:
            logger.error(f"Error getting POIs needing proof sources: {e}")
            return []
    
    def get_pois_without_neighborhood(self, city: str) -> List[Dict[str, Any]]:
        """Get POIs that don't have neighborhood attribution."""
        try:
            result = self.client.table('poi')\
                .select('id, name, latitude, longitude, city')\
                .eq('city', city)\
                .is_('neighborhood_id', 'null')\
                .not_.is_('latitude', 'null')\
                .not_.is_('longitude', 'null')\
                .execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting POIs without neighborhood: {e}")
            return []
    
    def calculate_neighborhood_mood_distribution(self, neighborhood_id: str) -> Optional[Dict[str, int]]:
        """Calculate mood distribution for a neighborhood based on its POIs."""
        try:
            # Get all POIs in the neighborhood
            result = self.client.table('poi')\
                .select('mood_tag')\
                .eq('neighborhood_id', neighborhood_id)\
                .execute()
            
            if not result.data:
                return None
            
            # Count mood tags
            mood_counts = {}
            total_pois = len(result.data)
            
            for poi in result.data:
                mood = poi.get('mood_tag', 'Trendy')
                mood_counts[mood] = mood_counts.get(mood, 0) + 1
            
            # Convert to percentages
            mood_distribution = {}
            for mood, count in mood_counts.items():
                percentage = round((count / total_pois) * 100)
                mood_key = mood.lower().replace(' ', '_').replace('gem', '')
                if mood_key == 'hidden':
                    mood_key = 'hidden'
                elif mood_key == 'chill':
                    mood_key = 'chill'  
                else:
                    mood_key = 'trendy'  # Default fallback
                
                mood_distribution[mood_key] = percentage
            
            # Ensure we have the three main moods
            for mood in ['chill', 'trendy', 'hidden']:
                if mood not in mood_distribution:
                    mood_distribution[mood] = 0
            
            return mood_distribution
            
        except Exception as e:
            logger.error(f"Error calculating mood distribution for neighborhood {neighborhood_id}: {e}")
            return None
    
    def update_neighborhood_mood_distribution(self, neighborhood_id: str, mood_distribution: Dict[str, int]) -> bool:
        """Update neighborhood mood distribution."""
        try:
            result = self.client.table('neighborhoods')\
                .update({
                    'mood_distribution': mood_distribution,
                    'updated_at': datetime.utcnow().isoformat()
                })\
                .eq('id', neighborhood_id)\
                .execute()
            
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Error updating mood distribution for neighborhood {neighborhood_id}: {e}")
            return False
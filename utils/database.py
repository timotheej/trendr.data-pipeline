import logging
from typing import List, Dict, Any, Optional, Tuple
from supabase import create_client, Client
from tenacity import retry, stop_after_attempt, wait_exponential
import config
import json
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database-specific exceptions for better error handling
class DatabaseError(Exception):
    """Base database error"""
    pass

class ConnectionError(DatabaseError):
    """Database connection failed"""
    pass

class QueryError(DatabaseError):
    """Query execution failed"""
    pass

class RetryableError(DatabaseError):
    """Temporary error that can be retried"""
    pass

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
        """Insert a POI record and return the ID. Filters out deprecated fields."""
        try:
            # Remove deprecated fields to keep database clean
            deprecated_fields = {'business_status', 'last_google_sync', 'last_updated', 'maps_url', 'website_url'}
            clean_data = {k: v for k, v in poi_data.items() 
                         if k not in deprecated_fields}
            
            result = self.client.table('poi').upsert(
                clean_data,
                on_conflict='name,address,city'
            ).execute()
            
            if result.data:
                logger.info(f"Inserted/Updated POI: {clean_data['name']}")
                return result.data[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error inserting POI {poi_data.get('name', 'Unknown')}: {e}")
            raise

    def get_pois_for_city(self, city: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get POIs for a city."""
        try:
            result = self.client.table('poi')\
                .select('*')\
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def get_pois_by_names_batch(self, poi_names: List[str], city: str) -> Dict[str, Dict[str, Any]]:
        """
        Batch get POIs by multiple names - solves N+1 problem
        
        Args:
            poi_names: List of POI names to search
            city: City to filter by
            
        Returns:
            Dict mapping poi_name -> poi_data (first match only)
        """
        if not poi_names:
            return {}
            
        try:
            # Use individual queries combined - more reliable than complex OR syntax
            poi_map = {}
            
            for poi_name in poi_names:
                try:
                    query = self.client.table('poi').select('*').ilike('name', f'%{poi_name}%')
                    if city:
                        # Try city_slug first (lowercase like 'paris'), then fallback to city (title case like 'Paris')
                        if city.islower():
                            query = query.eq('city_slug', city)
                        else:
                            query = query.eq('city', city)
                    
                    result = query.execute()
                    if result.data:
                        # Take first match for this POI name
                        poi_map[poi_name] = result.data[0]
                except Exception as single_error:
                    logger.debug(f"Failed to query POI '{poi_name}': {single_error}")
                    continue
            
            logger.debug(f"Batch POI query: {len(poi_names)} names → {len(poi_map)} matches")
            return poi_map
            
        except Exception as e:
            # Classify error and handle accordingly
            error_str = str(e).lower()
            
            if 'connection' in error_str or 'network' in error_str or 'timeout' in error_str:
                logger.warning(f"Connection error in batch POI query: {e}")
                raise RetryableError(f"Database connection issue: {e}")
            elif 'rate limit' in error_str or '429' in error_str:
                logger.warning(f"Rate limit hit in batch POI query: {e}")
                raise RetryableError(f"Database rate limit: {e}")
            else:
                logger.error(f"Query error in batch POI query for {len(poi_names)} names: {e}")
                # Don't retry for syntax/schema errors, fallback to individual queries
                poi_map = {}
                for name in poi_names:
                    try:
                        pois = self.get_pois_by_name(name, city)
                        if pois:
                            poi_map[name] = pois[0]
                    except Exception as fallback_error:
                        logger.debug(f"Fallback query failed for {name}: {fallback_error}")
                logger.warning(f"Batch query failed, fallback completed: {len(poi_map)}/{len(poi_names)} POIs found")
                return poi_map

    # DEPRECATED: Using string neighborhoods instead
    def get_pois_for_neighborhood(self, neighborhood_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """DEPRECATED: Get POIs for a specific neighborhood.""" 
        logger.warning("get_pois_for_neighborhood is deprecated - use string neighborhood filtering")
        return []
    
    def get_pois_by_neighborhood_name(self, neighborhood_name: str, city: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get POIs for a specific neighborhood by name."""
        try:
            result = self.client.table('poi')\
                .select('*')\
                .eq('neighborhood', neighborhood_name)\
                .eq('city', city)\
                .limit(limit)\
                .execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting POIs for neighborhood {neighborhood_name}: {e}")
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
                .select('*')\
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
    # SOURCE MENTION METHODS
    # =============================================================================
    
    # Intelligent cache for source_catalog with TTL
    _source_catalog_cache = None
    _source_catalog_cache_time = None
    _CACHE_TTL_SECONDS = 300  # 5 minutes TTL
    
    def _load_source_catalog(self, force_refresh: bool = False):
        """
        Load source_catalog from database with intelligent caching
        
        Args:
            force_refresh: Force cache refresh even if TTL not expired
            
        Returns:
            List of source_catalog entries
        """
        import time
        current_time = time.time()
        
        # Check if cache is expired or force refresh requested
        cache_expired = (
            self._source_catalog_cache is None or 
            self._source_catalog_cache_time is None or
            (current_time - self._source_catalog_cache_time) > self._CACHE_TTL_SECONDS or
            force_refresh
        )
        
        if cache_expired:
            try:
                logger.debug("Refreshing source_catalog cache (TTL expired or forced)")
                result = self.client.table('source_catalog').select('*').execute()
                if result.data:
                    self._source_catalog_cache = result.data
                    self._source_catalog_cache_time = current_time
                    logger.info(f"Cached {len(result.data)} sources from source_catalog (TTL={self._CACHE_TTL_SECONDS}s)")
                else:
                    self._source_catalog_cache = []
                    self._source_catalog_cache_time = current_time
                    logger.warning("source_catalog table is empty")
            except Exception as e:
                # On error, keep old cache if available, otherwise empty list
                if self._source_catalog_cache is None:
                    self._source_catalog_cache = []
                logger.error(f"Failed to refresh source_catalog cache, using stale data: {e}")
        else:
            logger.debug(f"Using cached source_catalog ({len(self._source_catalog_cache)} sources)")
        
        return self._source_catalog_cache
    
    def invalidate_source_catalog_cache(self):
        """Manually invalidate source_catalog cache"""
        self._source_catalog_cache = None
        self._source_catalog_cache_time = None
        logger.debug("source_catalog cache invalidated manually")
    
    # === DISCOVERED SOURCES MANAGEMENT ===
    
    def get_or_create_discovered_source(self, domain: str, language: str = 'fr', 
                                       geographic_scope: str = 'paris') -> Optional[str]:
        """Get discovered_source_id for domain, create if doesn't exist"""
        try:
            # Clean domain 
            domain = domain.lower().replace('www.', '')
            
            # Try to find existing discovered source
            result = self.client.table('discovered_sources').select('id').eq('domain', domain).execute()
            
            if result.data:
                return result.data[0]['id']
            
            # Create new discovered source
            source_data = {
                'domain': domain,
                'language': language,
                'geographic_scope': geographic_scope,
                'auto_authority_weight': 0.4  # Default for discovered sources
            }
            
            result = self.client.table('discovered_sources').insert(source_data).execute()
            if result.data:
                discovered_id = result.data[0]['id']
                logger.info(f"🆕 DISCOVERED SOURCE: {domain} → created with id={discovered_id}")
                return discovered_id
                
        except Exception as e:
            logger.error(f"Error managing discovered source {domain}: {e}")
            
        return None
    
    def resolve_source_type(self, domain: str) -> Dict[str, Any]:
        """Determine if domain is cataloged or discovered, return appropriate reference"""
        # Handle full URLs by extracting domain
        if '://' in domain:
            from urllib.parse import urlparse
            try:
                domain = urlparse(domain).netloc.lower()
            except:
                pass
        
        # Remove www. prefix
        domain = domain.lower().replace('www.', '')
        
        # First check if it's in official catalog
        cataloged_source_id = self._find_cataloged_source(domain)
        if cataloged_source_id:
            return {
                'type': 'cataloged',
                'source_id': cataloged_source_id,
                'discovered_source_id': None,
                'domain': domain
            }
        
        # Not cataloged, treat as discovered
        discovered_id = self.get_or_create_discovered_source(domain)
        return {
            'type': 'discovered', 
            'source_id': None,
            'discovered_source_id': discovered_id,
            'domain': domain
        }
    
    def _find_cataloged_source(self, domain: str) -> Optional[str]:
        """Find source_id in official catalog for exact domain match"""
        sources = self._load_source_catalog()
        
        # Try exact match first
        for source in sources:
            base_url = source.get('base_url', '')
            if base_url:
                try:
                    from urllib.parse import urlparse
                    catalog_domain = urlparse(base_url).netloc.lower().replace('www.', '')
                    if domain == catalog_domain:
                        return source.get('source_id')
                except:
                    continue
        
        # Try partial matches for subdomains
        for source in sources:
            base_url = source.get('base_url', '')
            if base_url:
                try:
                    from urllib.parse import urlparse
                    catalog_domain = urlparse(base_url).netloc.lower().replace('www.', '')
                    if domain.endswith(catalog_domain) or catalog_domain.endswith(domain):
                        return source.get('source_id')
                except:
                    continue
        
        return None
    
    def get_source_id_from_domain(self, domain: str) -> Optional[str]:
        """Get source_id from domain using source_catalog database"""
        # Handle full URLs by extracting domain
        if '://' in domain:
            from urllib.parse import urlparse
            try:
                domain = urlparse(domain).netloc.lower()
            except:
                pass
        
        # Remove www. prefix and clean up
        domain = domain.lower().replace('www.', '')
        
        # Load source catalog from DB
        sources = self._load_source_catalog()
        
        # Try exact match first
        for source in sources:
            base_url = source.get('base_url', '')
            if base_url:
                try:
                    from urllib.parse import urlparse
                    catalog_domain = urlparse(base_url).netloc.lower().replace('www.', '')
                    if domain == catalog_domain:
                        return source.get('source_id')
                except:
                    continue
        
        # Try partial matches for subdomains
        for source in sources:
            base_url = source.get('base_url', '')
            if base_url:
                try:
                    from urllib.parse import urlparse
                    catalog_domain = urlparse(base_url).netloc.lower().replace('www.', '')
                    if domain.endswith(catalog_domain) or catalog_domain.endswith(domain):
                        return source.get('source_id')
                except:
                    continue
        
        # No match found - use generic unknown source (ENUM constraint workaround)
        return self._handle_unknown_domain(domain)
    
    def _handle_unknown_domain(self, domain: str) -> str:
        """
        Handle unknown domain with ENUM constraint workaround
        
        Strategy:
        1. Use existing 'unknown' source_id if available
        2. Log domain for future catalog expansion  
        3. Store domain info in source_mention.domain field for analysis
        
        Args:
            domain: Domain to handle (e.g., 'septime-charonne.fr')
            
        Returns:
            str: Valid source_id from existing ENUM
        """
        # Check if we have an 'unknown' or generic source_id in catalog
        sources = self._load_source_catalog()
        
        # Look for dedicated unknown/generic source
        for source in sources:
            source_type = source.get('type', '').lower() 
            if source_type in ['unknown', 'generic', 'other']:
                logger.info(f"🔍 UNKNOWN DOMAIN: '{domain}' → using generic source_id: {source.get('source_id')}")
                return source.get('source_id')
        
        # Fallback to lowest authority press source for unknown domains
        press_sources = [(s.get('source_id'), s.get('authority_weight', 1.0)) 
                        for s in sources if s.get('type', '').lower() == 'press']
        
        if press_sources:
            # Use the press source with lowest authority
            lowest_authority_source = min(press_sources, key=lambda x: x[1])
            source_id = lowest_authority_source[0]
            logger.info(f"🔍 UNKNOWN DOMAIN: '{domain}' → using lowest-authority press: {source_id} (auth={lowest_authority_source[1]})")
            return source_id
        
        # Ultimate fallback - use any available source (this shouldn't happen in well-configured system)
        if sources:
            fallback_source = sources[0].get('source_id')
            logger.warning(f"🚨 UNKNOWN DOMAIN: '{domain}' → emergency fallback: {fallback_source}")
            return fallback_source
        
        # No sources available - should not happen
        logger.error(f"No sources available in catalog for unknown domain: {domain}")
        return None
    
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def upsert_source_mention(self, poi_id: str, source_id: str, url: str, excerpt: str, 
                             title: str, domain: str, query: str, serp_position: int,
                             final_score: float, score_components: dict) -> bool:
        """
        Upsert a source mention (OPTIMIZED: no duplication - type/authority_weight from source_catalog FK)
        
        Args:
            poi_id: POI UUID
            source_id: Source identifier from source_catalog (FK - type/authority_weight via JOIN)
            url: Final URL of the mention
            excerpt: Snippet text from SERP
            title: Page title from SERP
            domain: Domain extracted from URL
            query: Search query that found this result
            serp_position: Position in SERP results
            final_score: Final matching score
            score_components: Breakdown of scoring components
            
        Returns:
            bool: Success status
        """
        try:
            # Prepare optimized data for upsert (NO duplication - type/authority_weight via FK)
            mention_data = {
                'id': str(uuid.uuid4()),  # Generate UUID for required id field
                'poi_id': poi_id,
                'source_id': source_id,  # FK to source_catalog (type/authority_weight via JOIN)
                'url': url[:500] if url else None,  # Truncate to schema limit
                'excerpt': excerpt[:180] if excerpt else None,  # Truncate to schema limit
                'title': title[:200] if title else None,  # Reasonable title limit
                'domain': domain[:100] if domain else None,
                'query': query[:300] if query else None,
                'final_score': final_score if final_score is not None else None,
                'score_components': score_components if score_components else None,
                'accepted': True,  # Only accepted mentions are persisted
                'last_seen_at': datetime.utcnow().isoformat()
            }
            
            # Try simple insert first (in case table doesn't have the expected constraints)
            try:
                result = self.client.table('source_mention').insert(
                    mention_data
                ).execute()
            except Exception as insert_error:
                # If insert fails (likely due to duplicate), try update based on unique constraint
                logger.debug(f"Insert failed, attempting update: {insert_error}")
                try:
                    result = self.client.table('source_mention')\
                        .update({
                            'source_id': mention_data['source_id'],
                            'excerpt': mention_data['excerpt'],
                            'final_score': mention_data['final_score'],
                            'score_components': mention_data['score_components'],
                            'last_seen_at': mention_data['last_seen_at'],
                            'title': mention_data['title'],
                            'domain': mention_data['domain'],
                            'query': mention_data['query']
                        })\
                        .eq('poi_id', mention_data['poi_id'])\
                        .eq('url', mention_data['url'])\
                        .execute()
                except Exception as update_error:
                    logger.error(f"Both insert and update failed: insert={insert_error}, update={update_error}")
                    raise update_error
            
            # Success is indicated by no exception, even if no data returned
            # UPDATE operations often don't return data when updating existing records
            logger.info(f"💾 DB UPSERT: poi={poi_id[:8]}, domain={source_id}, url={url[:50]}...")
            return True
                
        except Exception as e:
            logger.error(f"Error upserting source_mention for POI {poi_id}: {e}")
            raise

    def upsert_source_mention_new(self, poi_id: str, url: str, excerpt: str, title: str, domain: str, 
                                 query: str, final_score: float, score_components: dict,
                                 source_id: str = None, discovered_source_id: str = None,
                                 published_at: str = None, published_at_confidence: str = None, 
                                 published_at_method: str = None) -> bool:
        """
        Upsert source mention supporting both cataloged sources and discovered sources
        
        Args:
            poi_id: POI UUID
            url: Final URL of the mention
            excerpt, title, domain, query, final_score, score_components: Mention data
            source_id: Source identifier from source_catalog (for cataloged sources)
            discovered_source_id: Discovered source ID (for non-cataloged sources)
            
        Note: Exactly one of source_id OR discovered_source_id must be provided
        """
        if not ((source_id is None) ^ (discovered_source_id is None)):
            raise ValueError("Exactly one of source_id or discovered_source_id must be provided")
            
        try:
            mention_data = {
                'id': str(uuid.uuid4()),
                'poi_id': poi_id,
                'url': url[:500] if url else None,
                'excerpt': excerpt[:180] if excerpt else None, 
                'title': title[:200] if title else None,
                'domain': domain[:100] if domain else None,
                'query': query[:300] if query else None,
                'final_score': final_score if final_score is not None else None,
                'score_components': score_components if score_components else None,
                'published_at': published_at,
                # TODO: Add when columns exist - 'published_at_confidence': published_at_confidence,
                # TODO: Add when columns exist - 'published_at_method': published_at_method,
                'accepted': True,
                'last_seen_at': datetime.utcnow().isoformat()
            }
            
            # Add the appropriate source reference
            if source_id:
                mention_data['source_id'] = source_id
                source_label = f"cataloged:{source_id}"
            else:
                mention_data['discovered_source_id'] = discovered_source_id
                source_label = f"discovered:{discovered_source_id[:8]}"
            
            # Try insert first
            try:
                result = self.client.table('source_mention').insert(mention_data).execute()
                logger.info(f"💾 DB INSERT: poi={poi_id[:8]}, source={source_label}, url={url[:50]}...")
            except Exception as insert_error:
                # Insert failed, try update
                logger.debug(f"Insert failed, attempting update: {insert_error}")
                
                update_data = {
                    'excerpt': mention_data['excerpt'],
                    'final_score': mention_data['final_score'], 
                    'score_components': mention_data['score_components'],
                    'published_at': mention_data['published_at'],
                    # TODO: Add when columns exist - 'published_at_confidence': mention_data['published_at_confidence'],
                    # TODO: Add when columns exist - 'published_at_method': mention_data['published_at_method'],
                    'last_seen_at': mention_data['last_seen_at'],
                    'title': mention_data['title'],
                    'domain': mention_data['domain'],
                    'query': mention_data['query']
                }
                
                if source_id:
                    update_data['source_id'] = source_id
                else:
                    update_data['discovered_source_id'] = discovered_source_id
                
                result = self.client.table('source_mention')\
                    .update(update_data)\
                    .eq('poi_id', mention_data['poi_id'])\
                    .eq('url', mention_data['url'])\
                    .execute()
                    
                logger.info(f"💾 DB UPDATE: poi={poi_id[:8]}, source={source_label}, url={url[:50]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Error upserting source_mention: {e}")
            raise
    
    def upsert_source_mentions_batch(self, mentions: List[Dict[str, Any]]) -> int:
        """
        Batch upsert multiple source mentions
        
        Args:
            mentions: List of mention dicts with required fields
            
        Returns:
            int: Number of successfully processed mentions
        """
        if not mentions:
            return 0
        
        try:
            # Prepare all mentions for batch upsert
            prepared_mentions = []
            for mention in mentions:
                mention_data = {
                    'poi_id': mention['poi_id'],
                    'source_id': mention['source_id'], 
                    'type': mention.get('type', 'mention'),
                    'url': mention['url'][:500] if mention.get('url') else None,
                    'excerpt': mention['excerpt'][:180] if mention.get('excerpt') else None,
                    'authority_weight': max(0.0, min(1.0, mention.get('authority_weight', 0.5))),
                    'last_seen_at': datetime.utcnow().isoformat()
                }
                prepared_mentions.append(mention_data)
            
            # Execute batch upsert
            result = self.client.table('source_mention').upsert(
                prepared_mentions,
                on_conflict='poi_id,source_id'
            ).execute()
            
            count = len(result.data) if result.data else 0
            logger.info(f"💾 DB BATCH UPSERT: {count}/{len(mentions)} source_mentions processed")
            return count
            
        except Exception as e:
            logger.error(f"Error batch upserting source_mentions: {e}")
            raise
    
    def get_source_mentions_for_poi(self, poi_id: str) -> List[Dict[str, Any]]:
        """Get all source mentions for a POI"""
        try:
            result = self.client.table('source_mention')\
                .select('*')\
                .eq('poi_id', poi_id)\
                .order('last_seen_at', desc=True)\
                .execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting source mentions for POI {poi_id}: {e}")
            return []

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
                .is_('neighborhood', 'null')\
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
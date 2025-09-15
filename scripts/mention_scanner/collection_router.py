#!/usr/bin/env python3
"""
Removed unused code: legacy SERP strategies, complex query builders, multi-source resolvers

Collection Router for GATTO Scanner - KISS Implementation
Handles the 3 simplified modes: balanced, serp-only, open
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class CollectionRouter:
    """Routes collection logic for the 3 modes: balanced/serp-only/open"""
    
    def __init__(self, config: Dict[str, Any], db_manager, cse_searcher, city_manager):
        self.config = config
        self.db = db_manager
        self.cse_searcher = cse_searcher
        self.city_manager = city_manager
        
        # Get configuration
        self.mention_config = config.get('mention_scanner', {}) if config else {}
        self.limits_config = self.mention_config.get('limits', {})
        self.cse_num = self.limits_config.get('cse_num', 30)
        
    def collect_from_catalog_active_sources(self, poi: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect from all active catalog sources (balanced mode part 1)"""
        if not self.db:
            logger.warning("No database connection for catalog collection")
            return []
            
        try:
            # Load all active sources from catalog
            source_catalog = self.db._load_source_catalog()
            active_domains = []
            
            for source in source_catalog:
                if source.get('is_active', False):  # Only active sources
                    base_url = source.get('base_url')
                    if base_url:
                        try:
                            domain = urlparse(base_url).netloc.replace('www.', '')
                            if domain:
                                active_domains.append(domain)
                        except Exception as e:
                            logger.debug(f"Failed to parse domain from {base_url}: {e}")
            
            logger.info(f"Catalog active sources: {len(active_domains)} domains")
            
            # Use standard CSE collection with site: filtering
            return self._collect_from_cse_with_sites(poi, active_domains)
            
        except Exception as e:
            logger.error(f"Failed to collect from catalog active sources: {e}")
            return []
    
    def collect_from_catalog_filtered(self, poi: Dict[str, Any], sources: List[str]) -> List[Dict[str, Any]]:
        """Collect from specified catalog sources only (serp-only mode)
        
        SERP-only mode = sources spécifiées uniquement via site: (pas de requêtes ouvertes)
        Only queries sites/domains listed in sources using site: operator - no open CSE calls
        """
        if not sources:
            logger.warning("No sources specified for filtered catalog collection")
            return []
            
        # Resolve source_ids to domains if needed
        domains = self._resolve_sources_to_domains(sources)
        logger.info(f"Filtered catalog sources: {len(domains)} domains from {len(sources)} source specs")
        
        # Use standard CSE collection with site: filtering
        return self._collect_from_cse_with_sites(poi, domains)
    
    def collect_from_cse(self, poi: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect from CSE without site: filtering (open mode)"""
        if not self.cse_searcher:
            logger.warning("CSE searcher not available")
            return []
            
        candidates = []
        queries = self._build_cse_queries(poi, use_site_filter=False)
        
        for query in queries:
            try:
                # Get locale parameters
                city_slug = poi.get('city_slug', 'paris')
                locale_params = self.city_manager.get_search_locale(city_slug)
                
                # Search CSE
                results = self.cse_searcher.search(
                    query=query,
                    cse_num=self.cse_num,
                    gl=locale_params['gl'],
                    hl=locale_params['hl'],
                    cr=locale_params['cr']
                )
                
                logger.info(f"CSE open query '{query}' returned {len(results)} results")
                
                # Convert results to candidates
                for result in results:
                    candidate = self._result_to_candidate(result, query, poi)
                    candidates.append(candidate)
                    
            except Exception as e:
                logger.error(f"CSE query failed: {query} - {e}")
                continue
        
        return candidates
    
    def _collect_from_cse_with_sites(self, poi: Dict[str, Any], domains: List[str]) -> List[Dict[str, Any]]:
        """Collect from CSE with site: filtering
        
        Used by SERP-only mode: no open CSE - site: only
        Only searches within specified domains using site: operator
        """
        if not self.cse_searcher or not domains:
            return []
            
        candidates = []
        queries = self._build_cse_queries(poi, use_site_filter=True, domains=domains)
        
        for query in queries:
            try:
                # Get locale parameters  
                city_slug = poi.get('city_slug', 'paris')
                locale_params = self.city_manager.get_search_locale(city_slug)
                
                # Search CSE
                results = self.cse_searcher.search(
                    query=query,
                    cse_num=self.cse_num,
                    gl=locale_params['gl'],
                    hl=locale_params['hl'],
                    cr=locale_params['cr']
                )
                
                logger.info(f"CSE site query '{query}' returned {len(results)} results")
                
                # Convert results to candidates
                for result in results:
                    candidate = self._result_to_candidate(result, query, poi)
                    candidates.append(candidate)
                    
            except Exception as e:
                logger.error(f"CSE site query failed: {query} - {e}")
                continue
        
        return candidates
    
    def _build_cse_queries(self, poi: Dict[str, Any], use_site_filter: bool = False, domains: List[str] = None) -> List[str]:
        """Build CSE queries - ALWAYS include {poi_name} {city_name} {category}"""
        poi_name = poi.get('name', '')
        poi_category = poi.get('category', 'restaurant')
        city_slug = poi.get('city_slug', 'paris')
        
        # Get city name from profile
        city_profile = self.city_manager.get_profile(city_slug)
        city_name = city_profile.city_names_aliases[0].title() if city_profile else city_slug.title()
        
        # Base context - ALWAYS includes poi_name, city_name, category
        context = {
            'poi_name': poi_name,
            'city_name': city_name,
            'category': poi_category
        }
        
        queries = []
        
        if use_site_filter and domains:
            # Create batched site queries
            batch_size = 8
            domain_batches = [domains[i:i + batch_size] for i in range(0, len(domains), batch_size)]
            
            for batch in domain_batches:
                site_filter = " OR ".join([f"site:{domain}" for domain in batch])
                context['site_filter'] = site_filter
                
                # Templates with site filtering - GOLDEN RULE: always include poi_name, city_name, category
                templates = [
                    '({site_filter}) "{poi_name}" {city_name} {category}',
                    '({site_filter}) {poi_name} {city_name} {category}'
                ]
                
                for template in templates:
                    try:
                        query = template.format(**context)
                        queries.append(query)
                    except KeyError as e:
                        logger.debug(f"Skipping template due to missing key {e}: {template}")
        else:
            # Open queries without site filtering - GOLDEN RULE: always include poi_name, city_name, category
            templates = [
                '"{poi_name}" {city_name} {category}',
                '{poi_name} {city_name} {category}'
            ]
            
            for template in templates:
                try:
                    query = template.format(**context)
                    queries.append(query)
                except KeyError as e:
                    logger.debug(f"Skipping template due to missing key {e}: {template}")
        
        # Limit queries and log
        queries = queries[:6]  # Max 6 queries per POI
        logger.info(f"Built {len(queries)} CSE queries for POI '{poi_name}' (use_site_filter={use_site_filter})")
        
        return queries
    
    def _resolve_sources_to_domains(self, sources: List[str]) -> List[str]:
        """Convert source_ids to domains or return domains as-is"""
        domains = []
        
        if not self.db:
            logger.warning("No database - treating sources as domains")
            return sources
            
        try:
            source_catalog = self.db._load_source_catalog()
            
            for source_spec in sources:
                found_domain = None
                
                # Check if it's already a domain (contains dot)
                if '.' in source_spec and not source_spec.startswith('http'):
                    found_domain = source_spec
                else:
                    # Look up in catalog by source_id
                    for source in source_catalog:
                        if source.get('source_id') == source_spec:
                            # Check for CSE override first
                            if source.get('cse_site_override'):
                                found_domain = source.get('cse_site_override')
                            elif source.get('base_url'):
                                try:
                                    parsed = urlparse(source['base_url'])
                                    found_domain = parsed.netloc.replace('www.', '')
                                except:
                                    pass
                            break
                
                if found_domain:
                    domains.append(found_domain)
                    logger.debug(f"Resolved '{source_spec}' -> '{found_domain}'")
                else:
                    logger.warning(f"Could not resolve source '{source_spec}'")
        
        except Exception as e:
            logger.error(f"Error resolving sources to domains: {e}")
            return sources  # Fallback to original
            
        return domains
    
    def _result_to_candidate(self, result: Dict[str, Any], query: str, poi: Dict[str, Any]) -> Dict[str, Any]:
        """Convert CSE result to candidate object"""
        try:
            from .domains import domain_of
        except ImportError:
            from domains import domain_of
        
        url = result.get('link', result.get('formattedUrl', ''))
        title = result.get('title', '')
        snippet = result.get('snippet', '')
        
        return {
            'url': url,
            'title': title,
            'snippet': snippet,
            'domain': domain_of(url),
            'query_used': query,
            'poi_id': poi.get('id', ''),
            'poi_name': poi.get('name', ''),
            'displayLink': result.get('displayLink', ''),
            'formattedUrl': result.get('formattedUrl', ''),
            'published_at': result.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time')
        }
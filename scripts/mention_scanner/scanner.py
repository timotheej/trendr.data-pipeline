#!/usr/bin/env python3
"""
Removed unused code: strategy-based scanning, complex query builders, legacy match pipelines

GATTO Mention Scanner - KISS Implementation
Main scanner class with 3 modes: balanced, serp-only, open
"""
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

# Set up logger early to avoid NameError in imports
logger = logging.getLogger(__name__)

@contextmanager
def performance_timer(operation: str, poi_name: str = None, extra_context: dict = None):
    """Context manager for structured performance logging"""
    start_time = time.perf_counter()
    context = {"operation": operation}
    if poi_name:
        context["poi_name"] = poi_name
    if extra_context:
        context.update(extra_context)
    
    try:
        yield context
    except Exception as e:
        context["error"] = str(e)
        context["success"] = False
        raise
    else:
        context["success"] = True
    finally:
        end_time = time.perf_counter()
        context["duration_ms"] = round((end_time - start_time) * 1000, 2)
        logger.info(f"[PERF] {operation}: {context['duration_ms']}ms", extra=context)

# Custom exceptions for better error handling
class MentionScannerError(Exception):
    """Base exception for mention scanner errors"""
    pass

class ConfigurationError(MentionScannerError):
    """Configuration-related errors"""
    pass

class CSEError(MentionScannerError):
    """CSE service-related errors"""
    pass

# Add path to utils for database access
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'utils'))

try:
    from database import SupabaseManager
except ImportError:
    logger.warning("Could not import SupabaseManager - database features disabled")
    SupabaseManager = None

# KISS imports with fallbacks
try:
    from .config_resolver import resolve_config
    from .cse_client import CSESearcher
    from .matching import MentionMatcher
    from .dedup import MentionDeduplicator
    from .logging_ext import JSONLWriter
    from .city_profiles import CityProfileManager
    from .scoring import final_score, make_tabular_decision
    from .domains import domain_of
    from .collection_router import CollectionRouter
except ImportError:
    # Fallback for direct execution
    try:
        from config_resolver import resolve_config
        from cse_client import CSESearcher
        from matching import MentionMatcher
        from dedup import MentionDeduplicator
        from logging_ext import JSONLWriter
        from city_profiles import CityProfileManager
        from scoring import final_score, make_tabular_decision
        from domains import domain_of
        from collection_router import CollectionRouter
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        raise ConfigurationError(f"Module imports failed - check dependencies: {e}")

# Database imports with fallbacks
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.database import SupabaseManager
    from utils.api_usage import get_caps, inc_api_usage
except ImportError:
    # Mock classes for development
    class SupabaseManager:
        def __init__(self):
            pass
    
    def get_caps():
        return {"daily": 1000}
    
    def inc_api_usage(calls=1):
        pass

logger = logging.getLogger(__name__)

# Constants (imported from modules or defaults)
CSE_DAILY_CAP = int(os.getenv('CSE_DAILY_CAP', 1000))

class GattoMentionScanner:
    """KISS Scanner - 3 modes: balanced, serp-only, open"""
    
    def __init__(self, debug: bool = False, allow_no_cse: bool = False, 
                 jsonl_out: str = None, log_drop_reasons: bool = False):
        self.db = SupabaseManager()
        self.debug = debug
        self._allow_no_cse = allow_no_cse
        self.config = resolve_config()  # Use unified config resolver
        
        # Initialize KISS components 
        self.matcher = MentionMatcher()
        self.deduplicator = MentionDeduplicator(config=self.config)
        self.city_manager = CityProfileManager()
        
        # Get thresholds from unified config (match_score structure from config.json)
        thresholds = self.config['mention_scanner']['match_score']
        self.high_threshold = thresholds['high']
        self.mid_threshold = thresholds['mid']
        
        # JSONL output configuration
        self.jsonl_writer = None
        if jsonl_out:
            self.jsonl_writer = JSONLWriter(jsonl_out, config=self.config, cli_jsonl=True)
            self.jsonl_writer.log_drop_reasons = log_drop_reasons
            self.jsonl_writer.initialize()
        
        # Debug mode DROP logging (enabled with SCAN_DEBUG=1 or log_drop_reasons config)
        self.debug_drop_logging = (
            os.getenv('SCAN_DEBUG') == '1' or
            log_drop_reasons or
            (self.config and self.config.get('mention_scanner', {}).get('logging', {}).get('log_drop_reasons', False))
        )
        
        # Audit JSONL output for debugging
        self.audit_jsonl_file = None
        if os.getenv('SCAN_DEBUG') == '1':
            os.makedirs('out', exist_ok=True)
            audit_filename = f"out/audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            self.audit_jsonl_file = open(audit_filename, 'w', encoding='utf-8')
            logger.info(f"🔍 Audit JSONL output: {audit_filename}")
        
        # Metrics tracking
        self.pois_loaded = 0
        self.pois_processed = 0
        self.pois_with_candidates = 0
        self.pois_with_accepts = 0
        self.stopped_reason = "ok"
        
        # Summary statistics
        self.total_cse_calls = 0
        self.total_candidates = 0
        self.total_accepted = 0
        self.total_rejected = 0
        
        # Initialize CSE searcher
        self._initialize_cse()
        
        # Initialize collection router
        self.collection_router = CollectionRouter(self.config, self.db, self.cse_searcher, self.city_manager)
    
    def _enrich_poi_with_coords(self, poi: dict) -> dict:
        """Enrichir un POI avec lat/lng depuis la DB"""
        if poi.get("lat") is not None and poi.get("lng") is not None:
            return poi
        try:
            from utils.database import SupabaseManager
        except Exception:
            logger.warning(f"Database module not found, skipping POI enrichment for {poi.get('name')}")
            return poi
        
        try:
            db = SupabaseManager()

            # Try to find POI by name and city using available methods
            results = []
            try:
                results = db.get_pois_by_name(poi["name"], poi.get("city_slug", ""))
                if not results:
                    # Try without city filter
                    all_results = db.get_pois_by_name(poi["name"], "")
                    # Filter by city manually if city_slug is provided
                    if poi.get("city_slug"):
                        results = [r for r in all_results if r.get("city_slug") == poi.get("city_slug")]
                    else:
                        results = all_results
            except Exception as e:
                logger.debug(f"Error querying POI by name: {e}")
            
            if results:
                row = results[0]  # Take first match
                try:
                    poi["lat"] = float(row["lat"]) if row.get("lat") is not None else None
                    poi["lng"] = float(row["lng"]) if row.get("lng") is not None else None
                    poi.setdefault("city_slug", row.get("city_slug"))
                    poi.setdefault("category", row.get("category"))
                    poi["id"] = row.get("id")  # Use real POI ID from database
                    logger.info(f"Enriched POI '{poi['name']}' with coords: lat={poi['lat']}, lng={poi['lng']}")
                except Exception:
                    logger.debug("Coords not castable for %r", poi.get("name"))
        except Exception as e:
            logger.warning(f"Failed to enrich POI {poi.get('name')} with coordinates: {e}")
        return poi

    def _enrich_pois_batch(self, pois: List[dict]) -> List[dict]:
        """
        Batch enrich multiple POIs with coordinates - solves N+1 problem
        
        Args:
            pois: List of POI dicts with at least 'name' and 'city_slug'
            
        Returns:
            List of enriched POI dicts
        """
        # Separate POIs that need enrichment from those that don't
        needs_enrichment = []
        already_enriched = []
        
        for poi in pois:
            if poi.get("lat") is not None and poi.get("lng") is not None:
                already_enriched.append(poi)
            else:
                needs_enrichment.append(poi)
        
        if not needs_enrichment:
            return pois  # All already enriched
        
        try:
            from utils.database import SupabaseManager
        except Exception:
            logger.warning("Database module not found, skipping batch POI enrichment")
            return pois
        
        with performance_timer("poi_batch_enrichment", 
                             extra_context={"poi_count": len(needs_enrichment), "already_enriched": len(already_enriched)}):
            try:
                db = SupabaseManager()
                
                # Extract names and city for batch query
                poi_names = [poi["name"] for poi in needs_enrichment]
                city_slug = needs_enrichment[0].get("city_slug", "") if needs_enrichment else ""
                
                # Batch query - single DB call instead of N calls
                with performance_timer("database_batch_query", 
                                     extra_context={"query_type": "poi_names_batch", "poi_count": len(poi_names)}):
                    poi_data_map = db.get_pois_by_names_batch(poi_names, city_slug)
                
                # Enrich POIs with database data
                enriched_pois = []
                for poi in needs_enrichment:
                    poi_name = poi["name"]
                    if poi_name in poi_data_map:
                        db_poi = poi_data_map[poi_name]
                        try:
                            poi["lat"] = float(db_poi["lat"]) if db_poi.get("lat") is not None else None
                            poi["lng"] = float(db_poi["lng"]) if db_poi.get("lng") is not None else None
                            poi.setdefault("city_slug", db_poi.get("city_slug"))
                            poi.setdefault("category", db_poi.get("category"))
                            poi["id"] = db_poi.get("id")  # Use real POI ID from database
                            logger.debug(f"Batch enriched POI '{poi_name}' with coords: lat={poi['lat']}, lng={poi['lng']}")
                        except Exception:
                            logger.debug(f"Coords not castable for POI '{poi_name}'")
                    enriched_pois.append(poi)
                
                logger.info(f"Batch enriched {len(poi_data_map)}/{len(needs_enrichment)} POIs in single query")
                return already_enriched + enriched_pois
                
            except Exception as e:
                logger.warning(f"Batch POI enrichment failed, falling back to individual enrichment: {e}")
                # Fallback to individual enrichment to maintain functionality
                for i, poi in enumerate(needs_enrichment):
                    needs_enrichment[i] = self._enrich_poi_with_coords(poi)
                return already_enriched + needs_enrichment
    
    def _initialize_cse(self):
        """Initialize CSE searcher with config"""
        api_key = os.getenv('GOOGLE_CUSTOM_SEARCH_API_KEY')
        search_engine_id = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID')
        
        if api_key and search_engine_id:
            self.cse_searcher = CSESearcher(api_key, search_engine_id, self.config)
            logger.info(f"CSE initialized with API key: ***REDACTED*** and CX: {search_engine_id}")
        elif not self._allow_no_cse:
            logger.error("CSE credentials not configured and --allow-no-cse not specified")
            raise CSEError("CSE configuration required - provide GOOGLE_CSE_API_KEY and GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
        else:
            logger.warning("CSE not configured, running in no-CSE mode")
    
    
    def scan_balanced_mode(self, poi_names: List[str], city_slug: str = 'paris', 
                          limit_per_poi: int = None, **kwargs) -> Dict[str, Any]:
        """KISS Balanced mode: collect_from_catalog_active_sources() + collect_from_cse()"""
        results = {'accepted': 0, 'rejected': 0, 'total_mentions': 0}
        
        try:
            # Prepare POIs
            pois = []
            for poi_name in poi_names:
                pois.append({
                    "id": f"poi_{poi_name}", 
                    "name": poi_name, 
                    "category": "restaurant", 
                    "city_slug": city_slug
                })
            
            # Batch enrich POIs
            enriched_pois = self._enrich_pois_batch(pois)
            
            # Process each POI with balanced collection
            for poi in enriched_pois:
                logger.info(f"🔄 Processing POI: {poi['name']} (balanced mode)")
                
                # Step 1: Collect from catalog active sources
                catalog_candidates = self.collection_router.collect_from_catalog_active_sources(poi)
                logger.info(f"  📚 Catalog active sources: {len(catalog_candidates)} candidates")
                
                # Step 2: Collect from CSE (open queries)
                cse_candidates = self.collection_router.collect_from_cse(poi)
                logger.info(f"  🔍 CSE open queries: {len(cse_candidates)} candidates")
                
                # Combine candidates
                all_candidates = catalog_candidates + cse_candidates
                
                # Process candidates through KISS pipeline
                poi_results = self._process_candidates_kiss(poi, all_candidates, limit_per_poi)
                
                results['accepted'] += poi_results['accepted']
                results['rejected'] += poi_results['rejected'] 
                results['total_mentions'] += poi_results['total_mentions']
            
            logger.info(f"✅ Balanced mode completed: {results}")
            return results
        
        except Exception as e:
            logger.error(f"Balanced mode scan failed: {e}")
            results['error'] = str(e)
            return results
    
    def scan_open_mode(self, poi_names: List[str], city_slug: str = 'paris', 
                      limit_per_poi: int = None, **kwargs) -> Dict[str, Any]:
        """KISS Open mode: collect_from_cse() only (no site: filtering)"""
        results = {'accepted': 0, 'rejected': 0, 'total_mentions': 0}
        
        try:
            # Prepare POIs
            pois = []
            for poi_name in poi_names:
                pois.append({
                    "id": f"poi_{poi_name}", 
                    "name": poi_name, 
                    "category": "restaurant", 
                    "city_slug": city_slug
                })
            
            # Batch enrich POIs
            enriched_pois = self._enrich_pois_batch(pois)
            
            # Process each POI with open collection
            for poi in enriched_pois:
                logger.info(f"🔄 Processing POI: {poi['name']} (open mode)")
                
                # Collect from CSE only (no site: filtering)
                candidates = self.collection_router.collect_from_cse(poi)
                logger.info(f"  🔍 CSE open queries: {len(candidates)} candidates")
                
                # Process candidates through KISS pipeline
                poi_results = self._process_candidates_kiss(poi, candidates, limit_per_poi)
                
                results['accepted'] += poi_results['accepted']
                results['rejected'] += poi_results['rejected']
                results['total_mentions'] += poi_results['total_mentions']
            
            logger.info(f"✅ Open mode completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Open mode scan failed: {e}")
            results['error'] = str(e)
            return results

    def scan_serp_only(self, poi_names: List[str], source_ids: List[str], city_slug: str = 'paris', 
                      limit_per_poi: int = None, **kwargs) -> Dict[str, Any]:
        """KISS SERP-only mode: collect_from_catalog_filtered() - no CSE calls"""
        results = {'accepted': 0, 'rejected': 0, 'total_mentions': 0}
        
        try:
            # Prepare POIs
            pois = []
            for poi_name in poi_names:
                pois.append({
                    "id": f"poi_{poi_name}", 
                    "name": poi_name, 
                    "category": "restaurant", 
                    "city_slug": city_slug
                })
            
            # Batch enrich POIs
            enriched_pois = self._enrich_pois_batch(pois)
            
            # Process each POI with filtered catalog collection
            for poi in enriched_pois:
                logger.info(f"🔄 Processing POI: {poi['name']} (serp-only mode)")
                
                # Collect from filtered catalog sources only (no CSE)
                candidates = self.collection_router.collect_from_catalog_filtered(poi, source_ids)
                logger.info(f"  📋 Filtered catalog sources: {len(candidates)} candidates")
                
                # Process candidates through KISS pipeline
                poi_results = self._process_candidates_kiss(poi, candidates, limit_per_poi)
                
                results['accepted'] += poi_results['accepted']
                results['rejected'] += poi_results['rejected']
                results['total_mentions'] += poi_results['total_mentions']
            
            logger.info(f"✅ SERP-only mode completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"SERP-only scan failed: {e}")
            results['error'] = str(e)
            return results
    
    def _process_candidates_kiss(self, poi: Dict[str, Any], candidates: List[Dict[str, Any]], 
                                limit_per_poi: int = None) -> Dict[str, Any]:
        """KISS candidate processing pipeline: normalize, dedup, score, decide, log"""
        results = {'accepted': 0, 'rejected': 0, 'total_mentions': len(candidates)}
        
        if not candidates:
            return results
        
        poi_name = poi.get('name', '')
        city_slug = poi.get('city_slug', 'paris')
        
        # Step 1: Normalize and dedup immediately
        normalized_candidates = []
        seen_urls = set()
        seen_domains_titles = set()
        
        for candidate in candidates:
            url = candidate.get('url', '')
            domain = candidate.get('domain', '')
            title = candidate.get('title', '')
            
            # Normalize URL (remove utm, #, trailing slash)
            normalized_url = self._normalize_url_kiss(url)
            normalized_title = self._normalize_title_kiss(title)
            
            # Dedup by (domain, normalized_url) or fallback (domain, normalized_title)
            dedup_key_primary = (domain, normalized_url)
            dedup_key_fallback = (domain, normalized_title)
            
            if dedup_key_primary not in seen_urls and dedup_key_fallback not in seen_domains_titles:
                seen_urls.add(dedup_key_primary)
                seen_domains_titles.add(dedup_key_fallback)
                normalized_candidates.append(candidate)
        
        logger.info(f"  📍 Dedup: {len(candidates)} → {len(normalized_candidates)} candidates")
        
        # Step 1.5: Apply domain exclusions from config (nested in mention_scanner)
        exclusions_config = self.config['mention_scanner'].get('domain_exclusions', {})
        if exclusions_config.get('exclude_from_mentions', False):
            excluded_domains = set()
            excluded_domains.update(exclusions_config.get('social_networks', []))
            excluded_domains.update(exclusions_config.get('review_sites', []))
            
            filtered_candidates = []
            excluded_count = 0
            
            for candidate in normalized_candidates:
                domain = candidate.get('domain', '')
                if domain in excluded_domains:
                    excluded_count += 1
                    logger.debug(f"Excluded domain: {domain}")
                else:
                    filtered_candidates.append(candidate)
            
            normalized_candidates = filtered_candidates
            logger.info(f"  🚫 Domain exclusions: removed {excluded_count} candidates, {len(normalized_candidates)} remaining")
        
        # Step 2: Score and decide for each candidate
        accepted_candidates = []
        
        for candidate in normalized_candidates:
            try:
                # Calculate final score
                score, explain = final_score(
                    poi_name=poi_name,
                    title=candidate.get('title', ''),
                    snippet=candidate.get('snippet', ''),
                    url=candidate.get('url', ''),
                    poi_category=poi.get('category', 'restaurant'),
                    config=self.config,
                    debug=True,
                    city_slug=city_slug,
                    poi_coords=(poi.get('lat'), poi.get('lng')) if poi.get('lat') and poi.get('lng') else None,
                    db_manager=self.db
                )
                
                # Make tabular decision
                decision, accepted_by, drop_reasons = make_tabular_decision(
                    score, explain, candidate, self.high_threshold, self.mid_threshold
                )
                
                # Log decision in JSON format
                self._log_decision_json(poi, candidate, score, explain, decision, accepted_by, drop_reasons)
                
                if decision in ["ACCEPT", "REVIEW"]:
                    accepted_candidates.append({
                        **candidate,
                        'score': score,
                        'accepted_by': accepted_by,
                        'explain': explain
                    })
                    results['accepted'] += 1
                    
                    # Check limit per POI
                    if limit_per_poi and len(accepted_candidates) >= limit_per_poi:
                        break
                elif decision == "REJECT":
                    results['rejected'] += 1
                else:
                    # Unknown decision type - count as rejected
                    results['rejected'] += 1
                    logger.warning(f"Unknown decision type: {decision}")
                    
            except Exception as e:
                logger.error(f"Candidate processing failed: {candidate.get('url', 'unknown')} - {e}")
                results['rejected'] += 1
        
        # Step 3: Persist to database if accepted
        if accepted_candidates and self.db:
            try:
                upserted = self._persist_accepted_mentions(poi, accepted_candidates)
                logger.info(f"  💾 Persisted: {upserted} mentions to database")
            except Exception as e:
                logger.error(f"Database persistence failed: {e}")
        
        return results
    
    def _normalize_url_kiss(self, url: str) -> str:
        """Simple URL normalization: remove utm, #, trailing slash"""
        if not url:
            return url
            
        # Remove fragments
        if '#' in url:
            url = url.split('#')[0]
            
        # Remove UTM parameters
        utm_params = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term']
        for param in utm_params:
            if param in url:
                import re
                url = re.sub(f'[?&]{param}=[^&]*', '', url)
        
        # Clean up double ?/?& and trailing slash
        url = url.replace('?&', '?').rstrip('/')
        
        return url
    
    def _normalize_title_kiss(self, title: str) -> str:
        """Simple title normalization for dedup"""
        if not title:
            return title
        return title.lower().strip()
    
    def _log_decision_json(self, poi: Dict[str, Any], candidate: Dict[str, Any], score: float,
                          explain: Dict[str, Any], decision: str, accepted_by: str, drop_reasons: List[str]):
        """Log decision in JSON format (minimal but useful)"""
        import json
        
        # Minimal JSON log
        log_entry = {
            'decision': decision,
            'accepted_by': accepted_by,
            'drop_reasons': drop_reasons,
            'poi_name': poi.get('name'),
            'domain': candidate.get('domain'),
            'url': candidate.get('url'),
            'final_score': round(score, 3),
            'components': {
                'name': explain['components']['name_match'],
                'geo': explain['components']['geo_score'],
                'authority': explain['components']['authority']
            }
        }
        
        # Log in debug mode
        if self.debug:
            log_entry['thresholds_used'] = {
                'high': self.high_threshold,
                'mid': self.mid_threshold
            }
            log_entry['weights_used'] = explain['weights']
        
        logger.info(json.dumps(log_entry, ensure_ascii=False))

    def _resolve_source_ids_to_domains(self, source_ids: List[str]) -> List[str]:
        """Convert source_ids to actual domains for CSE queries"""
        if not source_ids:
            return []
        
        domains = []
        if self.db:
            try:
                self.db._load_source_catalog()
                catalog = self.db._source_catalog_cache or []
                
                for source_id in source_ids:
                    found_domain = None
                    for source in catalog:
                        if source.get('source_id') == source_id:
                            # Check for CSE override first
                            cse_override = source.get('cse_site_override')
                            if cse_override:
                                found_domain = cse_override
                            else:
                                # Extract domain from base_url
                                base_url = source.get('base_url', '')
                                if base_url:
                                    from urllib.parse import urlparse
                                    parsed = urlparse(base_url)
                                    found_domain = parsed.netloc.replace('www.', '')
                            break
                    
                    if found_domain:
                        domains.append(found_domain)
                        logger.debug(f"Resolved source_id '{source_id}' -> domain '{found_domain}'")
                    else:
                        logger.warning(f"Could not resolve source_id '{source_id}' to domain, skipping")
                        
            except Exception as e:
                logger.error(f"Failed to resolve source_ids to domains: {e}")
                return source_ids  # Fallback to original
        else:
            logger.warning("No database connection, using source_ids as domains (fallback)")
            return source_ids  # Fallback to original
            
        return domains


    def _filter_excluded_domains(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out excluded domains (social networks, review sites) based on configuration"""
        if not candidates or not self.config:
            return candidates
        
        exclusions = self.config.get('mention_scanner', {}).get('domain_exclusions', {})
        if not exclusions.get('exclude_from_mentions', False):
            return candidates  # Exclusion disabled
        
        social_networks = exclusions.get('social_networks', [])
        review_sites = exclusions.get('review_sites', [])
        excluded_domains = set(social_networks + review_sites)
        
        filtered_candidates = []
        for candidate in candidates:
            url = candidate.get('url', '')
            domain = candidate.get('domain', '')
            
            # Extract domain from URL if not present
            if not domain and url:
                from urllib.parse import urlparse
                try:
                    parsed = urlparse(url)
                    domain = parsed.netloc.replace('www.', '')
                except:
                    domain = ''
            
            # Check if domain should be excluded
            is_excluded = False
            for excluded_domain in excluded_domains:
                if excluded_domain in domain:
                    is_excluded = True
                    logger.debug(f"Excluding {url}: matches excluded domain {excluded_domain}")
                    break
            
            if not is_excluded:
                filtered_candidates.append(candidate)
        
        return filtered_candidates

    def run_serp_pipeline(self, poi: Dict[str, Any], config: Dict[str, Any], summary, jsonl_writer, 
                         *, sources: Optional[List[str]] = None, strategy_label: str, cli_overrides=None, 
                         cse_num: int = None, limit_per_poi: int = None, no_cache: bool = False, dump_serp_dir: str = None,
                         sources_are_domains: bool = False, city_slug: str = 'paris') -> None:
        """Reusable SERP pipeline for POI scanning with configurable source filtering"""
        from .config_resolver import _resolve_thresholds
        from .domains import domain_of
        from .scoring import final_score
        from .dedup import MentionDeduplicator
        
        poi_name = poi.get('name', '')
        poi_id = poi.get('id', '')
        poi_category = poi.get('category', 'restaurant')
        
        if not poi_name:
            logger.warning(f"POI {poi_id} has no name, skipping")
            return
        
        # Convert source_ids to domains for CSE queries (skip if already domains)
        if sources and not sources_are_domains:
            sources = self._resolve_source_ids_to_domains(sources)
        
        # Resolve thresholds from CLI > config > defaults
        high_threshold, mid_threshold = _resolve_thresholds(cli_overrides, config)
        
        # Get config parameters with fallbacks
        mention_scanner_config = config.get('mention_scanner', {}) if config else {}
        
        # Apply DEBUG mode threshold overrides
        debug_override = False
        if os.getenv('SCAN_DEBUG') == '1':
            debug_config = mention_scanner_config.get('debug_mode', {})
            high_threshold = debug_config.get('threshold_high', 0.30)
            mid_threshold = debug_config.get('threshold_mid', 0.15)
            debug_override = True
            logger.info("🐛 DEBUG MODE: thresholds_effective high=%.2f, mid=%.2f (debug_override=True)", 
                       high_threshold, mid_threshold)
            
            # Override token_required in debug mode to allow more matches
            cli_overrides.token_required_for_mid = debug_config.get('token_required_for_mid', False)
            # Override thresholds for matcher too
            cli_overrides.threshold_high = high_threshold  
            cli_overrides.threshold_mid = mid_threshold
            logger.info("🐛 DEBUG MODE: token_required_for_mid=%s, thresholds updated for matcher (debug_override)", 
                       cli_overrides.token_required_for_mid)
        else:
            logger.info("📊 NORMAL MODE: thresholds_effective high=%.2f, mid=%.2f (debug_override=False)", 
                       high_threshold, mid_threshold)
        
        query_config = mention_scanner_config.get('query_strategy', {})
        limits_config = mention_scanner_config.get('limits', {})
        
        # Build context for templates with dynamic city
        poi_name_normalized = self.matcher.normalize(poi_name)
        
        # Get city name from city profile for dynamic templates
        city_profile = self.city_manager.get_profile(city_slug)
        city_name = city_profile.city_names_aliases[0].title() if city_profile else city_slug.title()
        
        # Build queries from templates
        templates = query_config.get('templates', ['"{poi_name}"'])
        global_templates = query_config.get('global_templates', [])
        max_templates = query_config.get('max_templates_per_poi', 6)
        geo_hints = query_config.get('geo_hints', [city_name])
        category_synonyms = query_config.get('category_synonyms', {}).get(poi_category, [poi_category])
        
        # CSE parameters from config
        if cse_num is None:
            cse_num = limits_config.get('cse_num', 10)
        max_cse_num = limits_config.get('cse_num', 10)
        cse_num = min(cse_num, max_cse_num)
        
        context = {
            'poi_name': poi_name,
            'poi_name_normalized': poi_name_normalized,
            'geo_hint': geo_hints[0] if geo_hints else city_name,
            'category_synonym': category_synonyms[0] if category_synonyms else poi_category,
            'city_name': city_name
        }
        
        # Generate queries
        queries = []
        
        if sources is None:
            # Global queries without site: prefix - only use global_templates
            for template in global_templates[:max_templates]:
                try:
                    query = template.format(**context)
                    # Normalize quotes to ASCII
                    query = query.replace('"', '"').replace('"', '"')
                    if query not in queries:
                        queries.append(query)
                except KeyError as e:
                    logger.debug(f"Skipping template '{template}' in open mode: missing key {e}")
                    continue
        else:
            # Create multi-site queries by grouping domains
            batch_size = 8  # Group domains in batches of 8 for OR queries
            domain_batches = [sources[i:i + batch_size] for i in range(0, len(sources), batch_size)]
            
            for batch in domain_batches:
                # Create multi-site query: "site:a.com OR site:b.com OR site:c.com"
                multi_site = " OR ".join([f"site:{domain}" for domain in batch])
                context['multi_site_query'] = multi_site
                
                for template in templates:
                    try:
                        query = template.format(**context)
                        # Normalize quotes to ASCII  
                        query = query.replace('"', '"').replace('"', '"')
                        if query not in queries:
                            queries.append(query)
                    except KeyError as e:
                        logger.debug(f"Skipping template '{template}' for batch {batch}: missing key {e}")
                        continue
            
            # Add global templates for broader coverage
            for template in global_templates:
                try:
                    query = template.format(**context)
                    query = query.replace('"', '"').replace('"', '"')
                    if query not in queries:
                        queries.append(query)
                except KeyError as e:
                    logger.debug(f"Skipping global template '{template}': missing key {e}")
                    continue
        
        # Limit total queries per POI
        queries = queries[:max_templates]
        
        # Collect all candidates
        candidates = []
        accepted_count = 0
        
        # Step-by-step counters for debugging
        serp_items_total = 0
        candidates_created = 0
        candidates_scored = 0
        drop_reasons_seen = []
        
        # Debug scoring for first N items
        debug_scoring_limit = 5  # Could be made configurable
        items_detailed = 0
        
        for query in queries:
            if not self.cse_searcher:
                logger.warning("CSE searcher not available, skipping query")
                continue
                
            # Get locale-specific search parameters
            locale_params = self.city_manager.get_search_locale(city_slug)
            
            # Search with CSE (pass summary for cache stats)
            with performance_timer("cse_search", poi_name=poi_name, 
                                 extra_context={"query": query, "cse_num": cse_num, "locale": locale_params}):
                results = self.cse_searcher.search(query, debug=self.debug, cse_num=cse_num, summary=summary, 
                                                 no_cache=no_cache, dump_serp_dir=dump_serp_dir,
                                                 gl=locale_params['gl'], hl=locale_params['hl'], cr=locale_params['cr'])
                logger.info(f"CSE search for '{query}' returned {len(results) if results else 0} results")
            summary.increment_cse_call()
            
            # Count SERP items
            if hasattr(summary, 'serp_items_total'):
                summary.serp_items_total += len(results)
            
            logger.info(f"Query '{query}' returned {len(results)} results")
            if not results:
                logger.info(f"No results for query '{query}', skipping")
                continue
            
            # Count SERP items for this query
            serp_items_total += len(results)
            
            for item in results:
                # Create candidate object
                url = item.get('link', item.get('formattedUrl', ''))
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                source_domain = domain_of(url)
                
                # Log SERP item processing
                if self.debug_drop_logging:
                    logger.info("SERP[%s]: %s | url=%s | title=%s | snippet=%s | poi_context(name='%s', city=%s, lat=%s, lng=%s)",
                               source_domain, url, url, title[:100], snippet[:100], 
                               poi.get('name'), poi.get('city_slug') or poi.get('city'), poi.get('lat'), poi.get('lng'))
                
                candidate = {
                    'url': url,
                    'displayLink': item.get('displayLink', ''),
                    'formattedUrl': item.get('formattedUrl', ''),
                    'title': title,
                    'snippet': snippet,
                    'published_at': item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time'),
                    'domain': source_domain,
                    'source_domain': source_domain,
                    'query_used': query,
                    'poi_id': poi_id,
                    'poi_name': poi_name
                }
                
                # Count candidate creation
                candidates_created += 1
                
                # Matching score
                # LOG: ce qu'on envoie vraiment au matcher
                logger.info("POI context → name='%s', lat=%s, lng=%s, city=%s",
                           poi.get('name'), poi.get('lat'), poi.get('lng'), poi.get('city_slug') or poi.get('city'))
                match_result = self.matcher.match_poi_to_article(poi, candidate['title'], config=config, cli_overrides=cli_overrides)
                if not match_result:
                    if self.debug_drop_logging:
                        # Get detailed reason from matcher by calling it again in debug mode
                        poi_name = poi.get('name', '')
                        poi_norm = self.matcher.normalize(poi_name) 
                        title_norm = self.matcher.normalize(candidate['title'])
                        trigram_score = self.matcher.trigram_score(poi_norm, title_norm)
                        
                        # Use already resolved thresholds (no need to re-resolve here)
                        
                        # Get trigram_min from config
                        trigram_min = 0.15  # Default fallback
                        if config and 'mention_scanner' in config:
                            trigram_min = config['mention_scanner'].get('match_score', {}).get('trigram_min', 0.15)
                        
                        if trigram_score < trigram_min:
                            reason = f"trigram_too_low: {trigram_score:.3f} < {trigram_min}"
                        elif trigram_score < high_threshold:
                            # Check if this is specifically a token requirement failure
                            poi_tokens = set(self.matcher.extract_tokens(poi.get('name', '')))
                            title_tokens = set(self.matcher.extract_tokens(candidate['title']))
                            has_discriminant = bool(poi_tokens & title_tokens)
                            
                            # Get token_required setting
                            token_required = True
                            if config and 'mention_scanner' in config:
                                name_match_cfg = config['mention_scanner'].get('name_match', {})
                                token_required = name_match_cfg.get('require_token_for_mid', True)
                            
                            if token_required and not has_discriminant:
                                reason = f"token_required_mid: trigram={trigram_score:.3f} >= {mid_threshold} but no token match"
                            else:
                                reason = f"trigram_mid_range: {trigram_score:.3f}, failed other requirements (geo/etc)"
                        else:
                            reason = "unknown_matcher_rejection"
                            
                        # Collect drop reasons for summary
                        drop_reasons_seen.append(reason)
                            
                        # Use specific DROP category for token requirements
                        drop_category = "token_required_mid" if "token_required_mid" in reason else "matching_failed"
                        logger.info("DROP[%s]: %s | %s | trigram=%.3f | title='%s'", 
                                   drop_category, url, reason, trigram_score, candidate['title'][:100])
                    continue
                    
                summary.increment_candidate()
                
                # Calculate final score (with debug for first N items, but always get components for intelligent acceptance)
                show_debug = self.debug_drop_logging and items_detailed < debug_scoring_limit
                
                # Always get explain for intelligent acceptance (not just for debug display)
                city_slug = poi.get('city_slug', 'paris')
                poi_coords = None
                if poi.get('lat') is not None and poi.get('lng') is not None:
                    poi_coords = (poi['lat'], poi['lng'])
                
                with performance_timer("scoring_computation", poi_name=poi_name, 
                                     extra_context={"url": candidate['url'], "candidate_count": candidates_scored + 1}):
                    score, explain = final_score(poi_name, candidate['title'], candidate['snippet'], 
                                                candidate['url'], poi_category, config, debug=True,
                                                city_slug=city_slug, poi_coords=poi_coords, db_manager=self.db)
                    
                # Ensure we always have a float score
                if isinstance(score, tuple):
                    score = score[0]  # Extract float from (score, explain) tuple if needed
                    
                candidate['score'] = score
                candidate['match_score'] = match_result.get('score', 0.0) 
                
                # Use scoring components (now always available)
                candidate['geo_score'] = explain['components']['geo_score']
                
                # Count candidate scoring
                candidates_scored += 1
                
                # Comprehensive instrumentation for all candidates (not just first N)
                audit_data = self._audit_candidate(candidate, poi, match_result, explain, high_threshold, mid_threshold, config)
                
                # Log detailed scoring breakdown for first N items
                if show_debug and items_detailed < debug_scoring_limit:
                    items_detailed += 1
                    self._log_detailed_audit(audit_data, items_detailed, url)
                
                # Check acceptance with type safety
                drop_reasons = []
                
                # Type safety assertions
                if not isinstance(score, (int, float)):
                    logger.error("FATAL: score is not numeric: %s (type: %s)", score, type(score))
                    raise ValueError(f"Score must be numeric, got {type(score)}: {score}")
                    
                geo_score_value = candidate.get('geo_score', 0.0)
                if not isinstance(geo_score_value, (int, float)):
                    logger.error("FATAL: geo_score is not numeric: %s (type: %s)", geo_score_value, type(geo_score_value))
                    raise ValueError(f"Geo score must be numeric, got {type(geo_score_value)}: {geo_score_value}")
                
                # Use intelligent acceptance rules
                from .scoring import is_acceptable_intelligent
                
                name_match = explain['components']['name_match']
                authority = explain['components']['authority']
                poi_coords = None
                if poi.get('lat') is not None and poi.get('lng') is not None:
                    poi_coords = (poi['lat'], poi['lng'])
                
                acceptable, drop_reasons = is_acceptable_intelligent(
                    name_match, geo_score_value, authority, score, 
                    high_threshold, mid_threshold, poi_coords, config, candidate.get('domain', '')
                )
                
                if not acceptable:
                    # Collect drop reasons for summary
                    drop_reasons_seen.extend(drop_reasons)
                    
                    # Log DROP for threshold failures
                    if self.debug_drop_logging:
                        logger.info("DROP[threshold_failed]: %s | reasons=%s | score=%.3f | geo_score=%.3f", 
                                   url, ', '.join(drop_reasons), score, candidate.get('geo_score', 0.0))
                
                candidate['drop_reasons'] = drop_reasons
                candidate['acceptable'] = acceptable
                
                # Store audit data in candidate for potential JSONL output
                candidate['audit_data'] = audit_data
                
                # Write to audit JSONL if enabled
                if self.audit_jsonl_file:
                    self._write_audit_jsonl(audit_data)
                
                # Write to JSONL if enabled
                if jsonl_writer and jsonl_writer.enabled:
                    decision = "accept" if acceptable else "reject"
                    threshold_used = f"high={high_threshold},mid={mid_threshold}"
                    jsonl_writer.write_mention(
                        poi_id, poi_name, query, candidate['domain'], candidate['url'],
                        score, threshold_used, decision, drop_reasons, strategy_label
                    )
                
                candidates.append(candidate)
                
                if acceptable:
                    accepted_count += 1
                    domain = candidate['domain']
                    summary.increment_accepted(domain)
                    
                    # Check limit per POI
                    if limit_per_poi and accepted_count >= limit_per_poi:
                        break
                else:
                    summary.increment_rejected()
            
            if limit_per_poi and accepted_count >= limit_per_poi:
                break
        
        # Apply deduplication
        if candidates:
            accepted_candidates = [c for c in candidates if c.get('acceptable')]
            
            # Filter out excluded domains (social networks, review sites)
            filtered_candidates = self._filter_excluded_domains(accepted_candidates)
            excluded_count = len(accepted_candidates) - len(filtered_candidates)
            if excluded_count > 0:
                logger.info(f"Domain exclusion: {len(accepted_candidates)} → {len(filtered_candidates)} (-{excluded_count} excluded domains)")
            accepted_candidates = filtered_candidates
            
            # Apply intelligent deduplication with multi-language support
            source_catalog_cache = None
            if self.db:
                try:
                    self.db._load_source_catalog()
                    source_catalog_cache = self.db._source_catalog_cache
                except Exception as e:
                    logger.debug(f"Failed to load source catalog for dedup: {e}")
            
            deduplicated_accepted = self.deduplicator.filter(accepted_candidates, source_catalog_cache)
            
            # Log deduplication stats
            if len(accepted_candidates) > len(deduplicated_accepted):
                removed_count = len(accepted_candidates) - len(deduplicated_accepted)
                logger.info(f"Deduplication: {len(accepted_candidates)} → {len(deduplicated_accepted)} (-{removed_count} duplicates)")
            
            # Persist accepted mentions to database
            if deduplicated_accepted and self.db:
                upserted_count = self._persist_accepted_mentions(poi, deduplicated_accepted)
                logger.info(f"  • Upserted to DB: {upserted_count}")
            
            # Note: summary counters already incremented during processing
            
        # Log step-by-step counters for debugging
        rejected_count = candidates_scored - accepted_count
        logger.info(f"POI {poi_name} processing summary:")
        logger.info(f"  • Queries executed: {len(queries)}")
        logger.info(f"  • SERP items returned: {serp_items_total}")
        logger.info(f"  • Candidates created: {candidates_created}")
        logger.info(f"  • Candidates scored: {candidates_scored}")
        logger.info(f"  • Accepted: {accepted_count}")
        logger.info(f"  • Rejected: {rejected_count}")
        
        # If no candidates were created, log top 3 distinct drop reasons
        if candidates_created == 0 and drop_reasons_seen:
            from collections import Counter
            reason_counts = Counter(drop_reasons_seen)
            top_reasons = reason_counts.most_common(3)
            logger.info(f"  • Top 3 drop reasons (no candidates created):")
            for i, (reason, count) in enumerate(top_reasons, 1):
                logger.info(f"    {i}. {reason} ({count} occurrences)")

    def run_serp_only_for_poi(self, poi: Dict[str, Any], source_ids: List[str], config: Dict[str, Any], 
                             summary, jsonl_writer, cli_overrides, cse_num: int, limit_per_poi: int = None, city_slug: str = 'paris'):
        """Run SERP-only scanning for a single POI using reusable pipeline"""
        self.run_serp_pipeline(
            poi=poi,
            config=config,
            summary=summary,
            jsonl_writer=jsonl_writer,
            sources=source_ids,
            strategy_label="serp_only",
            cli_overrides=cli_overrides,
            cse_num=cse_num,
            limit_per_poi=limit_per_poi,
            city_slug=city_slug
        )
    
    def _persist_accepted_mentions(self, poi: Dict[str, Any], accepted_candidates: List[Dict[str, Any]]) -> int:
        """
        Persist accepted mentions to source_mention table
        
        Args:
            poi: POI information with id
            accepted_candidates: List of accepted candidate dicts (with audit_data containing scoring components)
            
        Returns:
            int: Number of successfully upserted mentions
        """
        if not self.db or not accepted_candidates:
            return 0
        
        poi_id = poi.get('id')
        poi_name = poi.get('name', 'unknown')
        if not poi_id:
            logger.error(f"Cannot persist mentions: POI missing id field")
            return 0
        
        # Skip POIs with fake IDs (not from database)
        if poi_id.startswith('poi_'):
            logger.info(f"Skipping persistence for fake POI ID: {poi_id}")
            return 0
        
        with performance_timer("database_persistence", poi_name=poi_name, 
                             extra_context={"mention_count": len(accepted_candidates), "poi_id": poi_id}):
            upserted_count = 0
            for candidate in accepted_candidates:
                try:
                    # Extract required fields
                    url = candidate.get('url', '')
                    if not url:
                        continue
                    
                    # Get domain and map to source_id
                    domain = candidate.get('domain', '')
                    if not domain:
                        from .domains import domain_of
                        domain = domain_of(url)
                    
                    # Resolve source type (cataloged vs discovered)
                    source_info = self.db.resolve_source_type(domain)
                    if not source_info:
                        logger.warning(f"Failed to resolve source type for domain: {domain}")
                        continue
                        
                    # Extract other fields
                    excerpt = candidate.get('snippet', '')
                    title = candidate.get('title', '')
                    query = candidate.get('query_used', '')  
                    serp_position = None  # TODO: add position tracking in candidate creation
                    final_score = candidate.get('score', 0.0)
                    
                    # Get score components from audit data if available
                    score_components = {}
                    if 'audit_data' in candidate and candidate['audit_data']:
                        scoring = candidate['audit_data'].get('scoring', {})
                        score_components = scoring.get('components', {})
                    
                    # Upsert using new method that handles both cataloged and discovered sources
                    success = self.db.upsert_source_mention_new(
                        poi_id=poi_id,
                        url=url,
                        excerpt=excerpt,
                        title=title,
                        domain=domain,
                        query=query,
                        serp_position=serp_position,
                        final_score=final_score,
                        score_components=score_components,
                        source_id=source_info.get('source_id'),
                        discovered_source_id=source_info.get('discovered_source_id')
                    )
                    
                    if success:
                        upserted_count += 1
                        
                except Exception as e:
                    logger.error(f"Error persisting mention {candidate.get('url', 'unknown')}: {e}")
                    continue
            
            return upserted_count
    
    def _audit_candidate(self, candidate: Dict[str, Any], poi: Dict[str, Any], match_result: Dict[str, Any], 
                        explain: Optional[Dict[str, Any]], high_threshold: float, mid_threshold: float, config: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive candidate audit with detailed instrumentation"""
        from .scoring import geo_hint_detailed, name_match_detailed
        
        url = candidate.get('url', '')
        poi_name = poi.get('name', '')
        
        # GEO instrumentation  
        city_slug = poi.get('city_slug', 'paris')  # Fallback to paris if no city_slug
        poi_coords = None
        if poi.get('lat') is not None and poi.get('lng') is not None:
            poi_coords = (poi['lat'], poi['lng'])
        geo_audit = geo_hint_detailed(candidate['title'], candidate['snippet'], candidate['url'], city_slug, poi_coords, config)
        
        # NAME instrumentation
        article_text = candidate['title'] + " " + candidate['snippet']
        name_audit = name_match_detailed(poi_name, article_text)
        
        # Authority analysis
        from .domains import domain_of
        domain = domain_of(url)
        is_official_domain = domain.lower() in [poi_name.lower().replace(' ', ''), poi_name.lower().replace(' ', '-')]
        
        # Sanity checks
        geo_used = candidate.get('geo_score', 0.0)
        geo_components = geo_audit['score']
        geo_inconsistency = abs(geo_used - geo_components) > 0.01
        
        name_used = explain['components']['name_match'] if explain else 0.0
        name_score_calculated = name_audit['score']
        name_inconsistency = abs(name_used - name_score_calculated) > 0.01
        
        return {
            'url': url,
            'title': candidate['title'][:80],
            'domain': domain,
            'poi_context': {
                'name': poi_name,
                'coords': (poi.get('lat'), poi.get('lng')),
                'city_slug': poi.get('city_slug')
            },
            'geo_audit': geo_audit,
            'name_audit': name_audit,
            'scoring': {
                'components': explain['components'] if explain else {},
                'weighted': explain['weighted_components'] if explain else {},
                'final_score': explain['final_score'] if explain else candidate.get('score', 0.0),
                'thresholds': {'high': high_threshold, 'mid': mid_threshold}
            },
            'matcher': {
                'match_score': match_result.get('match_score', 0.0),
                'trigram_score': match_result.get('trigram_score', 0.0),
                'geo_score': match_result.get('geo_score', 0.0),
                'has_discriminant': match_result.get('has_discriminant', False)
            },
            'sanity_checks': {
                'geo_inconsistency': geo_inconsistency,
                'name_inconsistency': name_inconsistency,
                'is_official_domain': is_official_domain
            },
            'candidate': candidate,  # Store full candidate data for JSONL
            'acceptable': candidate.get('acceptable', False),
            'drop_reasons': candidate.get('drop_reasons', [])
        }
    
    def _log_detailed_audit(self, audit_data: Dict[str, Any], item_num: int, url: str):
        """Log detailed audit information in structured format"""
        logger.info("🔍 AUDIT #%d: %s", item_num, url[:60])
        logger.info("  Title: '%s'", audit_data['title'])
        
        # GEO Instrumentation (obligatoire)
        geo = audit_data['geo_audit']
        # Handle both old and new signal formats
        if geo['signals_found'] and isinstance(geo['signals_found'][0], str):
            # New format: list of strings
            logger.info("  🗺️ GEO SIGNALS: %s", geo['signals_found'])
        else:
            # Old format: list of dicts
            logger.info("  🗺️ GEO SIGNALS: %s", 
                       [f"{s['signal']}({s['weight']}) from {','.join(s['sources'])}" for s in geo['signals_found']])
        logger.info("    poi_coords: %s | poi_city: %s", 
                   audit_data['poi_context']['coords'], audit_data['poi_context']['city_slug'])
        logger.info("    geo_components: %s | geo_score_used: %.3f", geo['components'], geo['score'])
        if not geo['signals_found']:
            logger.info("    reason: %s", geo['reason'])
        
        # NAME Instrumentation (obligatoire)
        name = audit_data['name_audit']
        logger.info("  📝 NAME: poi_norm='%s' | title_norm='%s'", 
                   name['poi_norm'], name['text_norm'])
        logger.info("    exact_substring: %s | trigram: %.3f | fuzzy: %.3f", 
                   name['exact_substring'], name['trigram_score'], name['fuzzy_score'])
        logger.info("    token_overlap: poi=%s | text=%s | overlap=%s", 
                   name['token_overlap']['poi_tokens'], 
                   name['token_overlap']['text_tokens'], 
                   name['token_overlap']['overlap'])
        if name['normalization_warning']:
            logger.warning("    WARNING: trigram/fuzzy score divergence detected - check normalization")
        
        # Authority & domain
        if audit_data['sanity_checks']['is_official_domain']:
            logger.info("    OFFICIAL DOMAIN detected: %s", audit_data['domain'])
        
        # Sanity checks (obligatoire)
        checks = audit_data['sanity_checks']
        if checks['geo_inconsistency'] or checks['name_inconsistency']:
            logger.error("  ❌ INCONSISTENCY detected:")
            if checks['geo_inconsistency']:
                logger.error("    GEO: components=%.3f vs used=%.3f", 
                           geo['score'], audit_data['candidate'].get('geo_score', 0.0))
            if checks['name_inconsistency']:
                logger.error("    NAME: calculated=%.3f vs used=%.3f", 
                           name['score'], audit_data['scoring']['components'].get('name_match', 0.0))
        
        # Final synthesis (obligatoire)
        scoring = audit_data['scoring']
        decision = "ACCEPTED" if audit_data['acceptable'] else "REJECTED"
        reason = f" | reason={audit_data['drop_reasons']}" if audit_data['drop_reasons'] else ""
        
        logger.info("  🎯 DECISION: domain=%s, name=%.3f, geo=%.3f (used=%.3f), cat=%.3f, auth=%.3f → final=%.3f | thresholds high=%.2f mid=%.2f | %s%s",
                   audit_data['domain'],
                   scoring['components'].get('name_match', 0.0),
                   scoring['components'].get('geo_score', 0.0),
                   audit_data['candidate'].get('geo_score', 0.0),
                   scoring['components'].get('cat_score', 0.0),
                   scoring['components'].get('authority', 0.0),
                   scoring['final_score'],
                   scoring['thresholds']['high'],
                   scoring['thresholds']['mid'],
                   decision,
                   reason)
    
    def _write_audit_jsonl(self, audit_data: Dict[str, Any]):
        """Write audit data to JSONL file for analysis"""
        import json
        
        # Create flattened audit record
        record = {
            'timestamp': datetime.now().isoformat(),
            'url': audit_data['url'],
            'title': audit_data['title'],
            'domain': audit_data['domain'],
            'poi_name': audit_data['poi_context']['name'],
            'poi_coords': audit_data['poi_context']['coords'],
            'poi_city': audit_data['poi_context']['city_slug'],
            'geo_signals': audit_data['geo_audit']['signals_found'],
            'geo_components': audit_data['geo_audit']['components'],
            'geo_score_used': audit_data['candidate'].get('geo_score', 0.0),
            'name_poi_norm': audit_data['name_audit']['poi_norm'],
            'name_text_norm': audit_data['name_audit']['text_norm'],
            'name_exact_substring': audit_data['name_audit']['exact_substring'],
            'name_trigram': audit_data['name_audit']['trigram_score'],
            'name_fuzzy': audit_data['name_audit']['fuzzy_score'],
            'name_token_overlap': audit_data['name_audit']['token_overlap'],
            'name_score_used': audit_data['scoring']['components'].get('name_match', 0.0),
            'scoring_components': audit_data['scoring']['components'],
            'scoring_weighted': audit_data['scoring']['weighted'],
            'final_score': audit_data['scoring']['final_score'],
            'thresholds': audit_data['scoring']['thresholds'],
            'matcher_scores': audit_data['matcher'],
            'sanity_checks': audit_data['sanity_checks'],
            'acceptable': audit_data['acceptable'],
            'drop_reasons': audit_data['drop_reasons']
        }
        
        # Write to JSONL
        self.audit_jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.audit_jsonl_file.flush()
    
    def __del__(self):
        """Cleanup on destruction"""
        if self.jsonl_writer:
            self.jsonl_writer.close()
        if self.audit_jsonl_file:
            self.audit_jsonl_file.close()


def main():
    """KISS CLI interface for GATTO Mention Scanner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GATTO Mention Scanner - KISS Edition (3 modes: balanced/serp-only/open)')
    parser.add_argument('--mode', choices=['balanced', 'open', 'serp-only'], default='balanced', 
                        help='Mode: balanced=catalog+CSE, open=CSE only, serp-only=specified sources only')
    parser.add_argument('--poi-name', help='Single POI name (e.g. "Le Rigmarole")')
    parser.add_argument('--poi-names', help='Multiple POI names (e.g. "Septime,Le Chateaubriand")')
    parser.add_argument('--sources', help='Source list for serp-only mode (e.g. "lefooding.com,timeout.fr")')
    parser.add_argument('--city-slug', default='paris', help='City slug (default: paris)')
    parser.add_argument('--limit-per-poi', type=int, help='Max accepted mentions per POI')
    
    # Output options
    parser.add_argument('--jsonl-out', help='Output JSONL file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Misc options
    parser.add_argument('--allow-no-cse', action='store_true', help='Allow running without CSE')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create KISS scanner instance
        scanner = GattoMentionScanner(
            debug=args.debug,
            allow_no_cse=args.allow_no_cse,
            jsonl_out=args.jsonl_out
        )
        
        # Parse POI names
        poi_names = []
        if args.poi_name:
            poi_names.append(args.poi_name)
        elif args.poi_names:
            poi_names = [name.strip() for name in args.poi_names.split(',')]
        
        if not poi_names:
            logger.error(f"{args.mode.title()} mode requires --poi-name or --poi-names")
            return 1
        
        # Mode-specific execution
        if args.mode == 'open':
            results = scanner.scan_open_mode(
                poi_names=poi_names,
                city_slug=args.city_slug,
                limit_per_poi=args.limit_per_poi
            )
            
        elif args.mode == 'serp-only':
            # Get sources from CLI or config
            source_ids = []
            if args.sources:
                source_ids = [src.strip() for src in args.sources.split(',')]
            else:
                serp_only_config = scanner.config['mention_scanner']['serp_only']
                source_ids = serp_only_config.get('sources', [])
            
            if not source_ids:
                logger.error("SERP-only mode requires --sources or config serp_only.sources")
                return 1
                
            results = scanner.scan_serp_only(
                poi_names=poi_names,
                source_ids=source_ids,
                city_slug=args.city_slug,
                limit_per_poi=args.limit_per_poi
            )
            
        else:  # balanced mode (default)
            results = scanner.scan_balanced_mode(
                poi_names=poi_names,
                city_slug=args.city_slug,
                limit_per_poi=args.limit_per_poi
            )
        
        # Print results
        print(f"\n🎯 {args.mode.title()} Mode Results:")
        print(f"  • Total mentions: {results.get('total_mentions', 0)}")
        print(f"  • Accepted: {results.get('accepted', 0)}")
        print(f"  • Rejected: {results.get('rejected', 0)}")
        
        if results.get('error'):
            print(f"  • Error: {results['error']}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Scanner error: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
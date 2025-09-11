#!/usr/bin/env python3
"""
Gatto Mention Scanner - Public API
Main scanner class with strategy-based scanning capabilities
"""
import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional

# Robust imports with fallbacks
try:
    from .config_resolver import load_config, _resolve_thresholds
    from .cse_client import CSESearcher
    from .content_fetcher import ContentFetcher
    from .matching import MentionMatcher
    from .dedup import MentionDeduplicator
    from .logging_ext import JSONLWriter, print_stable_summary
    from .scoring import *
    from .domains import *
except ImportError:
    # Fallback for direct execution
    try:
        from config_resolver import load_config, _resolve_thresholds
        from cse_client import CSESearcher
        from content_fetcher import ContentFetcher
        from matching import MentionMatcher
        from dedup import MentionDeduplicator
        from logging_ext import JSONLWriter, print_stable_summary
        from scoring import *
        from domains import *
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        raise RuntimeError("Module imports failed - check dependencies")

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
    """Consolidated mention scanner V2 - Sprint 3 - S3 SERP Fix"""
    
    def __init__(self, debug: bool = False, allow_no_cse: bool = False, 
                 jsonl_out: str = None, log_drop_reasons: bool = False, 
                 high_threshold: float = None, mid_threshold: float = None):
        self.db = SupabaseManager()
        self.debug = debug
        self._allow_no_cse = allow_no_cse
        self.config = load_config()
        
        # Initialize components with config
        self.fetcher = ContentFetcher(config=self.config)
        self.matcher = MentionMatcher()
        self.deduplicator = MentionDeduplicator()
        
        # Store configured thresholds (using config_resolver defaults)
        self.high_threshold = high_threshold or 0.82  # Matches DEFAULT_HIGH in config_resolver
        self.mid_threshold = mid_threshold or 0.33    # Matches DEFAULT_MID in config_resolver
        
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
            (config and config.get('mention_scanner', {}).get('logging', {}).get('log_drop_reasons', False))
        )
        
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
    
    def _initialize_cse(self):
        """Initialize CSE searcher with config"""
        api_key = os.getenv('GOOGLE_CUSTOM_SEARCH_API_KEY')
        search_engine_id = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID')
        
        if api_key and search_engine_id:
            self.cse_searcher = CSESearcher(api_key, search_engine_id, self.config)
            logger.info(f"CSE initialized with API key: {api_key[:20]}... and CX: {search_engine_id}")
        elif not self._allow_no_cse:
            logger.error("CSE credentials not configured and --allow-no-cse not specified")
            raise RuntimeError("CSE configuration required")
        else:
            logger.warning("CSE not configured, running in no-CSE mode")
    
    def scan_strategy_based(self, city_slug: str, strategy: str, poi_cse_budget: int, 
                           run_cse_cap: int, poi_limit: int = None) -> Dict[str, Any]:
        """Main entry point for strategy-based scanning"""
        logger.error("Strategy-based scanning not yet implemented in modular version")
        return {
            "error": "Strategy-based scanning not implemented",
            "strategy": strategy,
            "accepted": 0,
            "rejected": 0,
            "cse_calls": 0
        }
    
    def scan_open_mode(self, poi_names: List[str], city_slug: str = 'paris', 
                      limit_per_poi: int = None, threshold_high: float = None, threshold_mid: float = None,
                      cse_num: int = 10, no_cache: bool = False, dump_serp: bool = False) -> Dict[str, Any]:
        """Open mode scanning - global queries without site: prefix"""
        from .logging_ext import RunSummary
        
        summary = RunSummary()
        
        # Create CLI override object for threshold resolution
        cli_overrides = type('Args', (), {})()
        cli_overrides.threshold_high = threshold_high
        cli_overrides.threshold_mid = threshold_mid
        
        try:
            for poi_name in poi_names:
                poi = {"id": f"poi_{poi_name}", "name": poi_name, "category": "restaurant", "city_slug": city_slug}
                # enrichir chaque POI avec coords DB avant tout matching
                poi = self._enrich_poi_with_coords(poi)
                summary.increment_poi()
                self.run_serp_pipeline(
                    poi=poi,
                    config=self.config,
                    summary=summary,
                    jsonl_writer=self.jsonl_writer,
                    sources=None,  # No site: prefix
                    strategy_label="open",
                    cli_overrides=cli_overrides,
                    cse_num=cse_num,
                    limit_per_poi=limit_per_poi,
                    no_cache=no_cache,
                    dump_serp_dir=dump_serp
                )
                
            summary.print_summary()
            
            return {
                "total_mentions": summary.accepted_total + summary.rejected_total,
                "accepted": summary.accepted_total,
                "rejected": summary.rejected_total,
                "cse_calls": summary.cse_calls,
                "cache_hits": summary.cache_hits,
                "domains_accepted": dict(summary.top_domains),
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Open scan failed: {e}")
            return {
                "total_mentions": 0,
                "accepted": 0,
                "rejected": 0, 
                "cse_calls": 0,
                "cache_hits": 0,
                "domains_accepted": {},
                "error": str(e)
            }

    def scan_serp_only(self, poi_names: List[str], source_ids: List[str], city_slug: str = 'paris', 
                      limit_per_poi: int = None, threshold_high: float = None, threshold_mid: float = None,
                      token_required_for_mid: bool = None, cse_num: int = 10, no_cache: bool = False, dump_serp: bool = False) -> Dict[str, Any]:
        """SERP-only scanning with CLI threshold overrides"""
        from .logging_ext import RunSummary
        
        # Create summary object to track results
        summary = RunSummary()
        
        # Create CLI override object for threshold resolution
        cli_overrides = type('Args', (), {})()
        cli_overrides.threshold_high = threshold_high
        cli_overrides.threshold_mid = threshold_mid
        
        try:
            # Get POIs from database (mock for now - in real implementation would query DB)
            for poi_name in poi_names:
                poi = {"id": f"poi_{poi_name}", "name": poi_name, "category": "restaurant", "city_slug": city_slug}
                # enrichir chaque POI avec coords DB avant tout matching
                poi = self._enrich_poi_with_coords(poi)
                summary.increment_poi()
                self.run_serp_pipeline(
                    poi=poi,
                    config=self.config,
                    summary=summary,
                    jsonl_writer=self.jsonl_writer,
                    sources=source_ids,  # With site: prefix
                    strategy_label="serp_only",
                    cli_overrides=cli_overrides,
                    cse_num=cse_num,
                    limit_per_poi=limit_per_poi,
                    no_cache=no_cache,
                    dump_serp_dir=dump_serp
                )
                
            # Print summary
            summary.print_summary()
            
            # Return dict format for compatibility
            return {
                "total_mentions": summary.accepted_total + summary.rejected_total,
                "accepted": summary.accepted_total,
                "rejected": summary.rejected_total,
                "cse_calls": summary.cse_calls,
                "cache_hits": summary.cache_hits,
                "domains_accepted": dict(summary.top_domains),
                "error": None
            }
            
        except Exception as e:
            logger.error(f"SERP scan failed: {e}")
            return {
                "total_mentions": 0,
                "accepted": 0,
                "rejected": 0, 
                "cse_calls": 0,
                "cache_hits": 0,
                "domains_accepted": {},
                "error": str(e)
            }

    def run_serp_pipeline(self, poi: Dict[str, Any], config: Dict[str, Any], summary, jsonl_writer, 
                         *, sources: Optional[List[str]] = None, strategy_label: str, cli_overrides=None, 
                         cse_num: int = None, limit_per_poi: int = None, no_cache: bool = False, dump_serp_dir: str = None) -> None:
        """Reusable SERP pipeline for POI scanning with configurable source filtering"""
        from .config_resolver import _resolve_thresholds
        from .domains import domain_of
        from .scoring import final_score, is_acceptable
        from .dedup import MentionDeduplicator
        
        poi_name = poi.get('name', '')
        poi_id = poi.get('id', '')
        poi_category = poi.get('category', 'restaurant')
        
        if not poi_name:
            logger.warning(f"POI {poi_id} has no name, skipping")
            return
        
        # Resolve thresholds from CLI > config > defaults
        high_threshold, mid_threshold = _resolve_thresholds(cli_overrides, config)
        
        # Get config parameters with fallbacks
        mention_scanner_config = config.get('mention_scanner', {}) if config else {}
        query_config = mention_scanner_config.get('query_strategy', {})
        limits_config = mention_scanner_config.get('limits', {})
        dedup_config = mention_scanner_config.get('dedup', {})
        
        # Build queries from templates
        templates = query_config.get('templates', ['"{poi_name}"'])
        global_templates = query_config.get('global_templates', [])
        max_templates = query_config.get('max_templates_per_poi', 6)
        geo_hints = query_config.get('geo_hints', ['Paris'])
        category_synonyms = query_config.get('category_synonyms', {}).get(poi_category, [poi_category])
        
        # CSE parameters from config
        if cse_num is None:
            cse_num = limits_config.get('cse_num', 10)
        max_cse_num = limits_config.get('cse_num', 10)
        cse_num = min(cse_num, max_cse_num)
        
        # Build context for templates
        poi_name_normalized = self.matcher.normalize(poi_name)
        context = {
            'poi_name': poi_name,
            'poi_name_normalized': poi_name_normalized,
            'geo_hint': geo_hints[0] if geo_hints else 'Paris',
            'category_synonym': category_synonyms[0] if category_synonyms else poi_category
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
            # Domain-specific queries with site: prefix
            for domain in sources:
                context['domain'] = domain
                for template in templates:
                    try:
                        query = template.format(**context)
                        # Normalize quotes to ASCII  
                        query = query.replace('"', '"').replace('"', '"')
                        if query not in queries:
                            queries.append(query)
                    except KeyError as e:
                        logger.debug(f"Skipping template '{template}' for domain {domain}: missing key {e}")
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
                
            # Search with CSE (pass summary for cache stats)
            results = self.cse_searcher.search(query, debug=self.debug, cse_num=cse_num, summary=summary, 
                                             no_cache=no_cache, dump_serp_dir=dump_serp_dir)
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
                        
                        if trigram_score < mid_threshold:
                            reason = f"trigram_too_low: {trigram_score:.3f} < {mid_threshold}"
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
                
                # Calculate final score (with debug for first N items)
                show_debug = self.debug_drop_logging and items_detailed < debug_scoring_limit
                
                if show_debug:
                    score, explain = final_score(poi_name, candidate['title'], candidate['snippet'], 
                                                candidate['url'], poi_category, config, debug=True)
                else:
                    score = final_score(poi_name, candidate['title'], candidate['snippet'], 
                                       candidate['url'], poi_category, config)
                
                candidate['score'] = score
                candidate['match_score'] = match_result.get('score', 0.0)
                candidate['geo_score'] = match_result.get('geo_score', 0.0)
                
                # Count candidate scoring
                candidates_scored += 1
                
                # Log detailed scoring breakdown for first N items
                if show_debug:
                    items_detailed += 1
                    logger.info("🔍 DETAILED SCORING #%d: %s", items_detailed, url[:60])
                    logger.info("  Title: '%s'", candidate['title'][:80])
                    logger.info("  Components: name=%.3f geo=%.3f cat=%.3f auth=%.3f pen=%.3f", 
                               explain['components']['name_match'],
                               explain['components']['geo_score'], 
                               explain['components']['cat_score'],
                               explain['components']['authority'],
                               explain['components']['penalties'])
                    logger.info("  Weighted: name=%.3f geo=%.3f cat=%.3f auth=%.3f pen=%.3f",
                               explain['weighted_components']['name_component'],
                               explain['weighted_components']['geo_component'],
                               explain['weighted_components']['cat_component'],
                               explain['weighted_components']['authority_component'], 
                               explain['weighted_components']['penalty_component'])
                    logger.info("  Score: raw=%.3f final=%.3f | Match: trigram=%.3f geo=%.3f",
                               explain['raw_score'], explain['final_score'],
                               candidate['match_score'], candidate['geo_score'])
                    logger.info("  Thresholds: high=%.3f mid=%.3f | Domain: %s", 
                               high_threshold, mid_threshold, explain['domain'])
                
                # Check acceptance
                drop_reasons = []
                acceptable = score >= high_threshold and candidate.get('geo_score', 0.0) >= mid_threshold
                
                if not acceptable:
                    if score < high_threshold:
                        drop_reasons.append(f"score_too_low: {score:.3f} < {high_threshold}")
                    if candidate.get('geo_score', 0.0) < mid_threshold:
                        drop_reasons.append(f"geo_score_too_low: {candidate.get('geo_score', 0.0):.3f} < {mid_threshold}")
                    
                    # Collect drop reasons for summary
                    drop_reasons_seen.extend(drop_reasons)
                    
                    # Log DROP for threshold failures
                    if self.debug_drop_logging:
                        logger.info("DROP[threshold_failed]: %s | reasons=%s | score=%.3f | geo_score=%.3f", 
                                   url, ', '.join(drop_reasons), score, candidate.get('geo_score', 0.0))
                
                candidate['drop_reasons'] = drop_reasons
                candidate['acceptable'] = acceptable
                
                # Log decision for detailed items
                if show_debug:
                    decision = "ACCEPTED" if acceptable else "REJECTED"
                    reasons_text = f" | reasons: {', '.join(drop_reasons)}" if drop_reasons else ""
                    logger.info("  Decision: %s%s", decision, reasons_text)
                
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
            window_days = dedup_config.get('window_days', 21)
            
            accepted_candidates = [c for c in candidates if c.get('acceptable')]
            rejected_candidates = [c for c in candidates if not c.get('acceptable')]
            
            # Simple dedup by URL (full implementation would use database)
            seen_urls = set()
            deduplicated_accepted = []
            for candidate in accepted_candidates:
                url = candidate.get('url', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    deduplicated_accepted.append(candidate)
                else:
                    # Log DROP for duplicate URLs
                    if self.debug_drop_logging:
                        logger.info("DROP[duplicate_url]: %s | reason=URL already seen in this scan", url)
            
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
                             summary, jsonl_writer, cli_overrides, cse_num: int, limit_per_poi: int = None):
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
            limit_per_poi=limit_per_poi
        )
    
    def __del__(self):
        """Cleanup on destruction"""
        if self.jsonl_writer:
            self.jsonl_writer.close()


def main():
    """CLI interface for Gatto Mention Scanner V2"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gatto Mention Scanner V2 - Modular Edition')
    parser.add_argument('--mode', choices=['balanced', 'open', 'serp-only', 'strategy'], default='balanced', 
                        help='Scan mode: balanced (default), open (no site: prefix), serp-only (with site: prefix), or strategy')
    parser.add_argument('--poi-name', help='Specific POI name to scan (e.g. "Le Rigmarole")')
    parser.add_argument('--poi-names', help='Comma-separated POI names (e.g. "Septime,Le Chateaubriand")')
    parser.add_argument('--sources', help='Comma-separated source domains (e.g. "lefooding.com,timeout.fr")')
    parser.add_argument('--city-slug', default='paris', help='City to scan (default: paris)')
    parser.add_argument('--cse-num', type=int, default=10, help='Number of CSE results (1-10, default: 10)')
    parser.add_argument('--limit-per-poi', type=int, help='Limit accepted mentions per POI')
    
    # Thresholds
    parser.add_argument('--threshold-high', type=float, help='Override high match score threshold')
    parser.add_argument('--threshold-mid', type=float, help='Override mid match score threshold')
    
    # Output options
    parser.add_argument('--jsonl-out', help='Output JSONL file path')
    parser.add_argument('--log-drop-reasons', action='store_true', help='Log reasons for dropped candidates')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    # Strategy mode options
    parser.add_argument('--strategy', choices=['open', 'whitelist', 'hybrid'], default='hybrid', 
                        help='Search strategy for strategy mode')
    parser.add_argument('--poi-cse-budget', type=int, default=2, help='Max CSE calls per POI (strategy mode)')
    parser.add_argument('--run-cse-cap', type=int, default=200, help='Max total CSE calls (strategy mode)')
    parser.add_argument('--poi-limit', type=int, help='Max POIs to process (strategy mode)')
    
    # Misc options
    parser.add_argument('--allow-no-cse', action='store_true', help='Allow running without CSE configuration')
    parser.add_argument('--no-cache', action='store_true', help='Disable CSE cache for fresh results')
    parser.add_argument('--dump-serp', help='Directory to dump raw SERP results for debugging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Clamp cse_num
    args.cse_num = max(1, min(args.cse_num, 10))
    
    try:
        # Create scanner instance
        scanner = GattoMentionScanner(
            debug=args.debug,
            allow_no_cse=args.allow_no_cse,
            jsonl_out=args.jsonl_out,
            log_drop_reasons=args.log_drop_reasons,
            high_threshold=args.threshold_high,
            mid_threshold=args.threshold_mid
        )
        
        if args.mode == 'open':
            # Open mode - no site: prefix, global queries only
            poi_names = []
            if args.poi_name:
                poi_names.append(args.poi_name)
            elif args.poi_names:
                poi_names = [name.strip() for name in args.poi_names.split(',')]
            
            if not poi_names:
                logger.error("Open mode requires --poi-name or --poi-names")
                return 1
            
            results = scanner.scan_open_mode(
                poi_names=poi_names,
                city_slug=args.city_slug,
                limit_per_poi=args.limit_per_poi,
                threshold_high=args.threshold_high,
                threshold_mid=args.threshold_mid,
                cse_num=args.cse_num,
                no_cache=args.no_cache,
                dump_serp=args.dump_serp
            )
            
            print(f"\n🎯 Open Scan Results:")
            print(f"  • Total mentions: {results.get('total_mentions', 0)}")
            print(f"  • Accepted: {results.get('accepted', 0)}")
            print(f"  • Rejected: {results.get('rejected', 0)}")
            print(f"  • CSE calls: {results.get('cse_calls', 0)}")
            
        elif args.mode == 'serp-only' or (args.poi_name and args.sources):
            # SERP-only mode with domain filtering
            poi_names = []
            if args.poi_name:
                poi_names.append(args.poi_name)
            elif args.poi_names:
                poi_names = [name.strip() for name in args.poi_names.split(',')]
            
            # Get source domains from CLI or config
            source_ids = []
            if args.sources:
                source_ids = [src.strip() for src in args.sources.split(',')]
            else:
                # Try to get default sources from config
                mention_config = scanner.config.get('mention_scanner', {}) if scanner.config else {}
                default_sources = mention_config.get('default_sources', [])
                if default_sources:
                    source_ids = default_sources
                    logger.info(f"Using default sources from config: {source_ids}")
            
            if not poi_names:
                logger.error("SERP-only mode requires --poi-name or --poi-names")
                return 1
            if not source_ids:
                logger.error("SERP-only mode requires --sources or config default_sources")
                return 1
            
            results = scanner.scan_serp_only(
                poi_names=poi_names,
                source_ids=source_ids,
                city_slug=args.city_slug,
                limit_per_poi=args.limit_per_poi,
                threshold_high=args.threshold_high,
                threshold_mid=args.threshold_mid,
                cse_num=args.cse_num,
                no_cache=args.no_cache,
                dump_serp=args.dump_serp
            )
            
            print(f"\n🎯 SERP Scan Results:")
            print(f"  • Total mentions: {results.get('total_mentions', 0)}")
            print(f"  • Accepted: {results.get('accepted', 0)}")
            print(f"  • Rejected: {results.get('rejected', 0)}")
            print(f"  • CSE calls: {results.get('cse_calls', 0)}")
            
        elif args.mode == 'strategy':
            # Strategy-based mode
            results = scanner.scan_strategy_based(
                city_slug=args.city_slug,
                strategy=args.strategy,
                poi_cse_budget=args.poi_cse_budget,
                run_cse_cap=args.run_cse_cap,
                poi_limit=args.poi_limit
            )
            
            print(f"\n🎯 Strategy Scan Results:")
            print(f"  • Strategy: {results.get('strategy', args.strategy)}")
            print(f"  • POIs processed: {results.get('pois_processed', 0)}")
            print(f"  • Accepted: {results.get('accepted', 0)}")
            print(f"  • Rejected: {results.get('rejected', 0)}")
            print(f"  • CSE calls: {results.get('cse_calls', 0)}")
            
        else:  # balanced mode (default) - for now, use SERP-only as fallback
            logger.debug("BALANCED currently routes to SERP_ONLY (modular)")
            
            # Same logic as serp-only for now
            poi_names = []
            if args.poi_name:
                poi_names.append(args.poi_name)
            elif args.poi_names:
                poi_names = [name.strip() for name in args.poi_names.split(',')]
            
            source_ids = []
            if args.sources:
                source_ids = [src.strip() for src in args.sources.split(',')]
            
            if not poi_names:
                logger.error("Balanced mode requires --poi-name or --poi-names")
                return 1
            if not source_ids:
                logger.error("Balanced mode requires --sources")
                return 1
            
            results = scanner.scan_serp_only(
                poi_names=poi_names,
                source_ids=source_ids,
                city_slug=args.city_slug,
                limit_per_poi=args.limit_per_poi,
                threshold_high=args.threshold_high,
                threshold_mid=args.threshold_mid,
                cse_num=args.cse_num,
                no_cache=args.no_cache,
                dump_serp=args.dump_serp
            )
            
            print(f"\n🎯 Balanced Scan Results:")
            print(f"  • Total mentions: {results.get('total_mentions', 0)}")
            print(f"  • Accepted: {results.get('accepted', 0)}")
            print(f"  • Rejected: {results.get('rejected', 0)}")
            print(f"  • CSE calls: {results.get('cse_calls', 0)}")
            print(f"  • Cache hits: {results.get('cache_hits', 0)}")
            
            if results.get('domains_accepted'):
                print(f"  • Top domains: {dict(list(results['domains_accepted'].items())[:3])}")
                
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
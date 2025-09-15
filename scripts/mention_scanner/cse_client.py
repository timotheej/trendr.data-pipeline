#!/usr/bin/env python3
"""
Google Custom Search Engine client for Gatto Mention Scanner
Enhanced searcher with rate limiting, retry logic, and persistent caching
"""
import time
import json
import random
import logging
import requests
from typing import Dict, Any, List, Optional

# Try to import dependencies with fallbacks
try:
    from .config_resolver import _rate_limit_delay
except ImportError:
    try:
        from config_resolver import _rate_limit_delay
    except ImportError:
        # Fallback - will be handled in the class
        _rate_limit_delay = lambda config: 1.0

# Import API cache with fallback
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.api_cache import APICache
except ImportError:
    # Fallback - create a mock APICache for development
    class APICache:
        def __init__(self, cache_dir=None):
            self.cache = {}
            self.api_calls_saved = 0
            self.api_calls_made = 0
        
        def get(self, api_name, query, params=None, ttl=None):
            key = f"{api_name}:{query}:{params}"
            return self.cache.get(key)
        
        def set(self, api_name, query, results, params=None, ttl=None):
            key = f"{api_name}:{query}:{params}"
            self.cache[key] = results

logger = logging.getLogger(__name__)

class CSESearcher:
    """Enhanced Google Custom Search Engine searcher with rate limiting, retry, and caching"""
    
    def __init__(self, api_key: str, search_engine_id: str, config=None):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.config = config
        self.base_url = 'https://www.googleapis.com/customsearch/v1'
        self.api_cache = APICache(cache_dir="cache/cse_search")
        self.last_request_time = 0
        self.error_429_count = 0
        
        # Get configurable values exclusively from config with documented fallbacks
        mention_scanner_config = config.get('mention_scanner', {}) if config else {}
        
        # Cache TTL from config
        cache_config = mention_scanner_config.get('cache', {})
        self.ttl_serp_seconds = cache_config.get('ttl_serp_seconds', 86400)  # Default: 1 day
        
        # HTTP settings from config
        http_config = mention_scanner_config.get('http', {})
        self.user_agent = http_config.get('user_agent', "GattoScanner/1.0")  # Default user agent
        self.timeout_s = http_config.get('timeout_s', 15)  # Default: 15s timeout
        
    def _get_cache_key(self, query: str, start: int = 1, num: int = 10) -> str:
        """Generate cache key for CSE query with required format"""
        return f"CSE:{query}|start={start}|num={num}"
    
    def _rate_limit_delay(self):
        """Implement rate limiting using config or defaults"""
        now = time.time()
        time_since_last = now - self.last_request_time
        min_interval = _rate_limit_delay(self.config)  # Get from config
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            # Add jitter (100-300ms)
            jitter = random.uniform(0.1, 0.3)
            time.sleep(sleep_time + jitter)
        
        self.last_request_time = time.time()
    
    def _retry_request(self, params: dict, max_retries: int = 5) -> Optional[dict]:
        """Make request with exponential backoff on 429/5xx errors"""
        backoff_delays = [0.25, 0.5, 1.0, 2.0, 4.0]
        
        for attempt in range(max_retries):
            try:
                self._rate_limit_delay()
                response = requests.get(self.base_url, params=params, timeout=self.timeout_s, headers={'User-Agent': self.user_agent})
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    self.error_429_count += 1
                    retry_after = int(response.headers.get('Retry-After', backoff_delays[min(attempt, len(backoff_delays)-1)]))
                    logger.warning(f"Rate limited (429), retry after {retry_after}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_after)
                elif response.status_code >= 500:
                    delay = backoff_delays[min(attempt, len(backoff_delays)-1)]
                    logger.warning(f"Server error {response.status_code}, retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"CSE request failed with status {response.status_code}: {response.text}")
                    break
                    
            except requests.exceptions.Timeout:
                delay = backoff_delays[min(attempt, len(backoff_delays)-1)]
                logger.warning(f"Request timeout, retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Request failed: {e}")
                break
        
        return None
    
    def search(self, query: str, debug: bool = False, cse_num: int = 10, summary=None, no_cache: bool = False, dump_serp_dir: str = None, gl: str = 'fr', hl: str = 'fr', cr: str = 'countryFR') -> List[Dict[str, Any]]:
        """Search using Google CSE with persistent caching and retry logic"""
        # Generate cache params for the key format
        start = 1
        num = min(cse_num, 10)  # Google CSE API max per request is 10
        cache_params = {'start': start, 'num': num, 'gl': gl, 'hl': hl, 'cr': cr}
        
        # Check persistent cache first (unless no_cache is enabled)
        cached_results = None
        if not no_cache:
            cached_results = self.api_cache.get("google_cse", query, cache_params, ttl=self.ttl_serp_seconds)
            if cached_results is not None:
                if debug:
                    logger.info(f"LOG[QUERY_CACHED] {json.dumps({'q': query})}")
                if summary and hasattr(summary, 'increment_cache_hit'):
                    summary.increment_cache_hit()
                return cached_results
        
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': num,
            'start': start,
            'gl': gl,
            'hl': hl,
            'cr': cr
        }
        
        if debug:
            logger.info(f"LOG[QUERY] {json.dumps({'q': query})}")
        
        data = self._retry_request(params)
        if not data:
            if summary and hasattr(summary, 'increment_cache_miss'):
                summary.increment_cache_miss()
            return []
        
        # Increment cache miss counter for successful requests
        if summary and hasattr(summary, 'increment_cache_miss'):
            summary.increment_cache_miss()
        
        results = []
        for item in data.get('items', []):
            results.append({
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'displayLink': item.get('displayLink', ''),
                'snippet': item.get('snippet', '')
            })
        
        # Cache the results in persistent storage (unless no_cache is enabled)
        if not no_cache:
            self.api_cache.set("google_cse", query, results, cache_params, ttl=self.ttl_serp_seconds)
        
        # Dump SERP results if requested
        if dump_serp_dir and data:
            self._dump_serp_results(query, data, dump_serp_dir)
        
        return results
    
    def _dump_serp_results(self, query: str, data: dict, dump_dir: str):
        """Dump raw SERP results to file for debugging"""
        import os
        import time
        
        try:
            os.makedirs(dump_dir, exist_ok=True)
            
            # Create filename from query and timestamp
            safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_query = safe_query.replace(' ', '_')[:50]  # Limit length
            timestamp = int(time.time())
            filename = f"serp_{safe_query}_{timestamp}.json"
            filepath = os.path.join(dump_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': query,
                    'timestamp': timestamp,
                    'raw_response': data
                }, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"SERP results dumped to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to dump SERP results: {e}")
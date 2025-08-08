#!/usr/bin/env python3
"""
API Cache Manager - Optimizes expensive API calls
Intelligent cache system for Google Custom Search and other APIs
"""
import os
import json
import hashlib
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class APICache:
    """Intelligent cache for optimizing API calls"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.ensure_cache_dir()
        
        # Cache configuration
        self.default_ttl = 24 * 60 * 60  # 24 hours by default
        self.max_cache_size = 1000  # Maximum number of entries
        
        # Counters for monitoring
        self.api_calls_saved = 0
        self.api_calls_made = 0
        
    def ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_key(self, api_name: str, query: str, params: Dict[str, Any] = None) -> str:
        """Generate a unique key for the query"""
        key_data = f"{api_name}:{query}"
        if params:
            # Sort parameters to have a consistent key
            sorted_params = sorted(params.items())
            key_data += f":{json.dumps(sorted_params)}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> str:
        """Get the cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, api_name: str, query: str, params: Dict[str, Any] = None, ttl: int = None) -> Optional[Dict[str, Any]]:
        """Retrieve a cache entry if it exists and is valid"""
        cache_key = self._get_cache_key(api_name, query, params)
        cache_file = self._get_cache_file(cache_key)
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_entry = json.load(f)
            
            # Check if cache has expired
            cache_time = datetime.fromisoformat(cache_entry['timestamp'])
            cache_ttl = ttl or cache_entry.get('ttl', self.default_ttl)
            
            if datetime.now() - cache_time > timedelta(seconds=cache_ttl):
                # Cache expired, remove the file
                os.remove(cache_file)
                return None
            
            self.api_calls_saved += 1
            logger.info(f"Cache HIT for {api_name}: {query[:50]}...")
            return cache_entry['data']
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Corrupted cache for {cache_key}: {e}")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            return None
    
    def set(self, api_name: str, query: str, data: Dict[str, Any], params: Dict[str, Any] = None, ttl: int = None):
        """Save an API response in the cache"""
        cache_key = self._get_cache_key(api_name, query, params)
        cache_file = self._get_cache_file(cache_key)
        
        cache_entry = {
            'timestamp': datetime.now().isoformat(),
            'api_name': api_name,
            'query': query,
            'params': params,
            'ttl': ttl or self.default_ttl,
            'data': data
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, ensure_ascii=False, indent=2)
            
            self.api_calls_made += 1
            logger.info(f"Cache MISS for {api_name}: {query[:50]}... (saved)")
            
        except Exception as e:
            logger.error(f"Error saving cache {cache_key}: {e}")
    
    def invalidate(self, api_name: str, query: str, params: Dict[str, Any] = None):
        """Invalidate a specific cache entry"""
        cache_key = self._get_cache_key(api_name, query, params)
        cache_file = self._get_cache_file(cache_key)
        
        if os.path.exists(cache_file):
            os.remove(cache_file)
            logger.info(f"Cache invalidated: {api_name}:{query[:30]}...")
    
    def clear_expired(self):
        """Clean expired cache entries"""
        cleared_count = 0
        
        for filename in os.listdir(self.cache_dir):
            if not filename.endswith('.json'):
                continue
                
            cache_file = os.path.join(self.cache_dir, filename)
            
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_entry = json.load(f)
                
                cache_time = datetime.fromisoformat(cache_entry['timestamp'])
                cache_ttl = cache_entry.get('ttl', self.default_ttl)
                
                if datetime.now() - cache_time > timedelta(seconds=cache_ttl):
                    os.remove(cache_file)
                    cleared_count += 1
                    
            except Exception as e:
                logger.warning(f"Error cleaning cache {filename}: {e}")
                # Remove corrupted files
                try:
                    os.remove(cache_file)
                    cleared_count += 1
                except:
                    pass
        
        logger.info(f"Cache cleanup: {cleared_count} expired entries removed")
        return cleared_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
        
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f)) 
            for f in cache_files
        )
        
        return {
            'entries_count': len(cache_files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'api_calls_saved': self.api_calls_saved,
            'api_calls_made': self.api_calls_made,
            'cache_hit_rate': (
                self.api_calls_saved / (self.api_calls_saved + self.api_calls_made) * 100
                if (self.api_calls_saved + self.api_calls_made) > 0 else 0
            )
        }


class GoogleSearchCache(APICache):
    """Specialized cache for Google Custom Search"""
    
    def __init__(self):
        super().__init__(cache_dir="cache/google_search")
        # Longer TTL for searches (48 hours)
        self.default_ttl = 48 * 60 * 60
    
    def search_cached(self, query: str, **params) -> Optional[List[Dict[str, Any]]]:
        """Search with cache for Google Custom Search"""
        return self.get("google_custom_search", query, params)
    
    def cache_search_results(self, query: str, results: List[Dict[str, Any]], **params):
        """Save search results"""
        self.set("google_custom_search", query, results, params)


# Global cache instance for Google Search
google_search_cache = GoogleSearchCache()
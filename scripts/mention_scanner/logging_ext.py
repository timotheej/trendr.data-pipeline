#!/usr/bin/env python3
"""
Logging extensions for Gatto Mention Scanner
JSONL writer and summary helpers for structured output and reporting
"""
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, TextIO
from collections import Counter

logger = logging.getLogger(__name__)

class RunSummary:
    """Run summary with counters for mention scanner statistics"""
    
    def __init__(self):
        self.poi_count = 0
        self.cse_calls = 0
        self.serp_items_total = 0
        self.candidates_total = 0
        self.accepted_total = 0
        self.rejected_total = 0
        self.upserts_total = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.top_domains = Counter()
    
    def increment_poi(self):
        """Increment POI count"""
        self.poi_count += 1
    
    def increment_cse_call(self):
        """Increment CSE calls count"""
        self.cse_calls += 1
    
    def increment_candidate(self):
        """Increment candidates count"""
        self.candidates_total += 1
    
    def increment_accepted(self, domain: str = None):
        """Increment accepted count and optionally domain counter"""
        self.accepted_total += 1
        if domain:
            self.top_domains[domain] += 1
    
    def increment_rejected(self):
        """Increment rejected count"""
        self.rejected_total += 1
    
    def increment_upsert(self):
        """Increment upserts count"""
        self.upserts_total += 1
    
    def increment_cache_hit(self):
        """Increment cache hits count"""
        self.cache_hits += 1
    
    def increment_cache_miss(self):
        """Increment cache misses count"""
        self.cache_misses += 1
    
    def print_summary(self):
        """Print formatted summary to console"""
        print("\n" + "="*50)
        print("GATTO MENTION SCANNER - RUN SUMMARY")
        print("="*50)
        print(f"POIs processed: {self.poi_count}")
        print(f"CSE queries: {self.cse_calls}")
        print(f"SERP items returned: {self.serp_items_total}")
        
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0
        print(f"Cache hits/misses: {self.cache_hits}/{self.cache_misses} ({cache_hit_rate:.1f}% hit rate)")
        
        print(f"Candidates found: {self.candidates_total}")
        print(f"Accepted: {self.accepted_total}")
        print(f"Rejected: {self.rejected_total}")
        print(f"Upserts: {self.upserts_total}")
        
        if self.top_domains:
            print("Top domains (count):")
            for domain, count in self.top_domains.most_common(10):
                print(f"  {domain}: {count}")
        
        print("="*50)

class JSONLWriter:
    """JSONL writer with append-only mode and immediate flushing"""
    
    def __init__(self, filepath: str, config: Optional[Dict[str, Any]] = None, cli_jsonl: bool = False):
        self.filepath = filepath
        self.config = config
        self.jsonl_file: Optional[TextIO] = None
        self.enabled = False
        self.log_drop_reasons = False
        
        if cli_jsonl:
            self.enabled = True
        elif config and 'mention_scanner' in config:
            logging_config = config['mention_scanner'].get('logging', {})
            self.enabled = logging_config.get('jsonl', False)
            self.log_drop_reasons = logging_config.get('log_drop_reasons', False)
        
    def initialize(self):
        """Initialize JSONL output file if enabled"""
        if not self.enabled:
            return
        
        try:
            self.jsonl_file = open(self.filepath, 'a', encoding='utf-8')
            logger.debug(f"JSONL output initialized: {self.filepath}")
        except Exception as e:
            logger.error(f"Failed to initialize JSONL output {self.filepath}: {e}")
            self.jsonl_file = None
    
    def write_mention(self, poi_id: str, poi_name: str, query: str, 
                      domain: str, url: str, score: float, threshold_used: str,
                      decision: str, drop_reasons: list, strategy: str):
        """Write mention to JSONL file with append and flush"""
        if not self.enabled or not self.jsonl_file:
            return
        
        try:
            entry = {
                'poi_id': poi_id,
                'poi_name': poi_name,
                'query': query,
                'domain': domain,
                'url': url,
                'score': score,
                'threshold_used': threshold_used,
                'decision': decision,
                'drop_reasons': drop_reasons if self.log_drop_reasons else [],
                'strategy': strategy,
                'ts': datetime.now(timezone.utc).isoformat()
            }
            
            json_line = json.dumps(entry, ensure_ascii=False)
            self.jsonl_file.write(json_line + '\n')
            self.jsonl_file.flush()  # Ensure immediate write
            
        except Exception as e:
            logger.error(f"Failed to write JSONL entry: {e}")
    
    def close(self):
        """Close JSONL output file"""
        if self.jsonl_file:
            try:
                self.jsonl_file.close()
                self.jsonl_file = None
            except Exception as e:
                logger.error(f"Error closing JSONL file: {e}")

def print_stable_summary(summary: Dict[str, Any]):
    """Print stable summary block to console"""
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"POIs processed: {summary.get('pois_processed', 0)}")
    print(f"CSE queries: {summary.get('total_cse_queries', 0)}")
    
    cache_hits = summary.get('cache_hits', 0)
    cache_misses = summary.get('cache_misses', 0)
    cache_total = cache_hits + cache_misses
    cache_hit_rate = (cache_hits / cache_total * 100) if cache_total > 0 else 0
    print(f"Cache hits/misses: {cache_hits}/{cache_misses} ({cache_hit_rate:.1f}% hit rate)")
    
    print(f"Candidates found: {summary.get('total_candidates', 0)}")
    print(f"Accepted: {summary.get('total_accepted', 0)}")
    print(f"Rejected: {summary.get('total_rejected', 0)}")
    print(f"Upserts: {summary.get('total_upserts', 0)}")
    
    top_domains = summary.get('top_domains', {})
    if top_domains:
        print("Top domains (count):")
        for domain, count in sorted(top_domains.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {domain}: {count}")
    
    print("="*50)

def print_serp_summary(poi_stats: Dict[str, Any]):
    """Print formatted summary of SERP scan results"""
    print("\n===== SERP DEBUG SUMMARY =====")
    
    for poi_name, stats in poi_stats.items():
        print(f"POI: {poi_name}")
        print(f"  Queries sent: {stats['queries_sent']}")
        print(f"  Candidates: {stats['candidates']}")
        print(f"  Accepted: {stats['accepted']}")
        print(f"  Upserted: {stats['upserted']}")
        
        if stats['sources']:
            sources_str = ', '.join([f"{src}({count})" for src, count in stats['sources'].items()])
            print(f"  Sources: {sources_str}")
        else:
            print(f"  Sources: none")
        print()
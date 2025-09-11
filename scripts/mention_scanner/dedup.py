#!/usr/bin/env python3
"""
Deduplication utilities for Gatto Mention Scanner
Handles mention deduplication within time windows using dedupe keys
"""
import re
import logging
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# Pre-compiled regex patterns for deduplication
RE_FILE_EXTENSION = re.compile(r'\.[^/]*$')
RE_PATH_SUFFIXES = re.compile(r'-(part\d+|update|v\d+|\d+)$')

def dedupe_key(url: str) -> str:
    """Generate deduplication key from URL"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path.rstrip('/')
        # Remove file extension and version suffixes for better deduplication
        path_stem = RE_FILE_EXTENSION.sub('', path)
        # Also remove common suffixes like -part2, -update, etc.
        path_stem = RE_PATH_SUFFIXES.sub('', path_stem)
        return f"{domain}{path_stem}"
    except:
        return url

class MentionDeduplicator:
    """Deduplicate mentions within time windows"""
    
    def __init__(self, config=None):
        self.config = config
    
    def filter(self, mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter mentions to keep only the best per source/dedupe_key"""
        if not mentions:
            return []
        
        # Group by (source_id, dedupe_key)
        groups = {}
        for mention in mentions:
            key = (mention.get('source_id'), dedupe_key(mention.get('url', '')))
            if key not in groups:
                groups[key] = []
            groups[key].append(mention)
        
        # Keep best mention per group
        filtered = []
        for group_mentions in groups.values():
            # Sort by authority_weight * w_time descending
            group_mentions.sort(
                key=lambda m: (m.get('authority_weight_snapshot', 0) * m.get('w_time', 0)),
                reverse=True
            )
            
            # Get max_per_window from config with fallback
            max_per_window = 2
            if self.config and 'mention_scanner' in self.config:
                max_per_window = self.config['mention_scanner'].get('dedup', {}).get('max_per_window', 2)
            
            # Keep top mentions up to max_per_window
            filtered.extend(group_mentions[:max_per_window])
        
        logger.info(f"Dedup: {len(mentions)} → {len(filtered)}")
        return filtered
#!/usr/bin/env python3
"""
Matching utilities for Gatto Mention Scanner
Handles POI-article matching with trigram similarity, geo-distance, and token discrimination
"""
import re
import math
import unicodedata
import logging
from typing import Dict, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)
from .config_resolver import _resolve_thresholds

# Pre-compiled regex patterns for normalization
RE_PUNCTUATION_CLEANUP = re.compile(r'[^\w\s]')
RE_WHITESPACE_NORMALIZE = re.compile(r'\s+')

def _normalize_core(text: str) -> str:
    """Core text normalization - single source of truth"""
    if not text:
        return ""
    # Remove accents
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # Remove punctuation and normalize spaces
    text = RE_PUNCTUATION_CLEANUP.sub(' ', text)
    text = RE_WHITESPACE_NORMALIZE.sub(' ', text)
    return text.lower().strip()

def normalize(text: str) -> str:
    """Normalize text: lowercase, strip accents/punct, collapse spaces"""
    return _normalize_core(text)

class MentionMatcher:
    """Robust POI matching with trigram, geo-distance, and token discrimination"""
    
    def normalize(self, text: str) -> str:
        """Normalize text for matching"""
        return _normalize_core(text)
    
    def trigram_score(self, a: str, b: str) -> float:
        """Calculate trigram similarity score"""
        if not a or not b:
            return 0.0
        
        # Create trigrams
        def get_trigrams(s):
            s = '  ' + s + '  '  # Padding
            return set(s[i:i+3] for i in range(len(s)-2))
        
        tri_a = get_trigrams(a)
        tri_b = get_trigrams(b)
        
        if not tri_a or not tri_b:
            return 0.0
        
        intersection = len(tri_a & tri_b)
        union = len(tri_a | tri_b)
        
        return intersection / union if union > 0 else 0.0
    
    def geo_distance_m(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points in meters"""
        R = 6371000  # Earth radius in meters
        φ1 = math.radians(lat1)
        φ2 = math.radians(lat2)
        Δφ = math.radians(lat2 - lat1)
        Δλ = math.radians(lng2 - lng1)
        
        a = (math.sin(Δφ/2) * math.sin(Δφ/2) +
             math.cos(φ1) * math.cos(φ2) *
             math.sin(Δλ/2) * math.sin(Δλ/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def extract_tokens(self, text: str) -> Set[str]:
        """Extract discriminant tokens (>3 chars, not common words)"""
        stopwords = {'restaurant', 'café', 'bar', 'brasserie', 'bistro', 'the', 'une', 'des', 'les'}
        normalized = self.normalize(text)
        tokens = set()
        for word in normalized.split():
            if len(word) > 3 and word not in stopwords:
                tokens.add(word)
        return tokens
    
    def match_poi_to_article(self, poi: Dict[str, Any], article_title: str, article_location: Optional[Tuple[float, float]] = None, 
                            config: Optional[Dict[str, Any]] = None, cli_overrides: Any = None) -> Optional[Dict[str, Any]]:
        """Match POI against article with scoring using config resolver for thresholds"""
        poi_name = poi.get('name', '')
        if not poi_name or not article_title:
            return None
        
        # Resolve thresholds from config
        high_threshold, mid_threshold = _resolve_thresholds(cli_overrides, config)
        
        # Get other config values with fallbacks
        token_required = True
        max_distance = 400
        if config and 'mention_scanner' in config:
            name_match_cfg = config['mention_scanner'].get('name_match', {})
            token_required = name_match_cfg.get('require_token_for_mid', True)
            max_distance = name_match_cfg.get('max_distance_meters', 400)
        
        # CLI overrides have highest priority
        if cli_overrides and hasattr(cli_overrides, 'token_required_for_mid'):
            token_required = cli_overrides.token_required_for_mid
        
        # Normalize names
        poi_norm = self.normalize(poi_name)
        title_norm = self.normalize(article_title)
        
        # Trigram similarity
        trigram_score = self.trigram_score(poi_norm, title_norm)
        
        # Token discrimination
        poi_tokens = self.extract_tokens(poi_name)
        title_tokens = self.extract_tokens(article_title)
        has_discriminant = bool(poi_tokens & title_tokens)
        token_score = 1.0 if has_discriminant else 0.0
        
        # Geographic distance
        geo_score = 0.0
        distance_meters = None
        if (article_location and 
            poi.get('lat') is not None and poi.get('lng') is not None):
            distance_meters = self.geo_distance_m(
                float(poi['lat']), float(poi['lng']),
                article_location[0], article_location[1]
            )
            geo_score = 1.0 if distance_meters <= max_distance else 0.0
        
        # Combined match score
        match_score = 0.6 * trigram_score + 0.3 * geo_score + 0.1 * token_score
        
        # Acceptance rules
        accepted = False
        if match_score >= high_threshold:
            accepted = True
        elif match_score >= mid_threshold and (not token_required or has_discriminant):
            accepted = True
        
        # Debug logging when SCAN_DEBUG=1
        import os
        if os.getenv('SCAN_DEBUG') == '1':
            logger.info("🔍 MATCHER DEBUG: trigram=%.3f, geo=%.3f, token=%.3f → match_score=%.3f", 
                       trigram_score, geo_score, token_score, match_score)
            logger.info("   Thresholds: high=%.3f, mid=%.3f | token_required=%s, has_discriminant=%s", 
                       high_threshold, mid_threshold, token_required, has_discriminant)
            logger.info("   Conditions: (score>=high)=%s, (score>=mid AND token_ok)=%s → accepted=%s", 
                       match_score >= high_threshold, 
                       match_score >= mid_threshold and (not token_required or has_discriminant),
                       accepted)
        
        if accepted:
            return {
                'poi_id': poi.get('id', 'unknown'),  # Use .get() to handle missing id
                'match_score': round(match_score, 3),
                'trigram_score': round(trigram_score, 3),
                'geo_score': geo_score,
                'token_score': token_score,
                'distance_meters': distance_meters,
                'has_discriminant': has_discriminant
            }
        
        return None
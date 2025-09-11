#!/usr/bin/env python3
"""
Scoring utilities for Gatto Mention Scanner
Handles scoring calculations, acceptability checks, and candidate comparison
"""
import re
import math
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, Tuple, Union
import logging

# Try to import domains and matching to avoid circular imports
try:
    from .domains import domain_of
    from .matching import normalize
except ImportError:
    try:
        from domains import domain_of
        from matching import normalize
    except ImportError:
        # Fallback - will need to import these functions when used
        pass

logger = logging.getLogger(__name__)


# Pre-compiled regex patterns for geo scoring
RE_ARRONDISSEMENT = re.compile(r'\b(1er|2e|3e|4e|5e|6e|7e|8e|9e|10e|11e|12e|13e|14e|15e|16e|17e|18e|19e|20e)\b')
RE_ARRONDISSEMENT_WORD = re.compile(r'\barrondissement\b')
RE_PARIS_POSTAL = re.compile(r'\b750\d{2}\b')
RE_FRANCE_INDICATORS = re.compile(r'\bfrance\b|\bfr\b')

def time_decay(published_at_iso: Optional[str], config: Optional[Dict[str, Any]] = None) -> float:
    """Calculate time decay weight using exponential decay"""
    if not published_at_iso:
        return 1.0  # No penalty if no date
    
    # Get decay parameters from config with fallbacks
    decay_config = {}
    if config and 'mention_scanner' in config:
        decay_config = config['mention_scanner'].get('decay', {})
    
    half_life_days = decay_config.get('half_life_days', 30)
    floor_value = decay_config.get('floor', 0.1)
    
    try:
        # Parse ISO datetime
        if published_at_iso.endswith('Z'):
            published_at_iso = published_at_iso[:-1] + '+00:00'
        published_at = datetime.fromisoformat(published_at_iso)
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        
        # Calculate age in days
        now_utc = datetime.now(timezone.utc)
        delta_days = (now_utc - published_at).days
        
        # Exponential decay with half-life: w_time = exp(-ln(2)*Δ/τ)
        decay_constant = math.log(2) / half_life_days
        w_time = math.exp(-decay_constant * delta_days)
        return max(floor_value, min(1.0, w_time))
        
    except (ValueError, AttributeError):
        return 1.0

def fuzzy_score(a: str, b: str) -> float:
    """Calculate fuzzy similarity ratio in [0,1]"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

def geo_hint(title: str, snippet: str, url: str) -> float:
    """Return 0..1 score based on presence of Paris/France geo indicators"""
    text = f"{title} {snippet} {url}".lower()
    score = 0.0
    
    # Paris indicators
    if 'paris' in text:
        score += 0.4
    
    # Arrondissement indicators 
    if RE_ARRONDISSEMENT.search(text):
        score += 0.3
    elif RE_ARRONDISSEMENT_WORD.search(text):
        score += 0.2
    
    # Postal codes
    if RE_PARIS_POSTAL.search(text):
        score += 0.3
    
    # France indicators
    if RE_FRANCE_INDICATORS.search(text):
        score += 0.1
    
    return min(score, 1.0)

def cat_hint(title: str, snippet: str, category: str) -> float:
    """Lightweight category keyword matching"""
    text = f"{title} {snippet}".lower()
    
    # Category-specific keywords
    category_keywords = {
        'restaurant': ['restaurant', 'cuisine', 'chef', 'menu', 'plat', 'gastronomie', 'table'],
        'bar': ['bar', 'cocktail', 'drink', 'alcool', 'bière', 'vin', 'spiritueux'],
        'cafe': ['café', 'coffee', 'expresso', 'cappuccino', 'thé', 'petit déjeuner'],
        'bakery': ['boulangerie', 'pain', 'croissant', 'pâtisserie', 'viennoiserie'],
        'night_club': ['club', 'dj', 'musique', 'soirée', 'danse', 'nightlife']
    }
    
    keywords = category_keywords.get(category, [])
    matches = sum(1 for keyword in keywords if keyword in text)
    
    return min(matches / max(len(keywords), 1), 1.0)

def calculate_authority(domain: str, config: Optional[Dict[str, Any]] = None) -> float:
    """Calculate authority score for a domain using domain_groups mapping"""
    # Get domain groups from config
    domain_groups = {}
    if config and 'mention_scanner' in config:
        domain_groups = config['mention_scanner'].get('domain_groups', {})
    
    # Get apex domain
    apex_domain = domain_of(domain)
    
    # Return weight from domain_groups or fallback to 0.5
    return domain_groups.get(apex_domain, 0.5)

def calculate_penalties(domain: str, url: str, config: Optional[Dict[str, Any]] = None) -> float:
    """Calculate penalty score for domain/URL patterns"""
    penalties = 0.0
    
    # Get penalty config with fallbacks
    penalty_config = {}
    if config and 'mention_scanner' in config:
        penalty_config = config['mention_scanner'].get('penalties', {})
    
    social_domains = penalty_config.get('social_domains', {
        'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
        'youtube.com', 'tiktok.com', 'pinterest.com'
    })
    social_penalty = penalty_config.get('social_penalty', 0.15)
    low_quality_penalty = penalty_config.get('low_quality_penalty', 0.10)
    url_param_penalty = penalty_config.get('url_param_penalty', 0.05)
    
    # Social media penalty
    if domain in social_domains:
        penalties += social_penalty
    
    # Low-quality domain patterns
    if any(pattern in domain for pattern in ['blogspot', 'wordpress', 'wix', 'squarespace']):
        penalties += low_quality_penalty
    
    # Suspicious URL patterns
    if any(pattern in url.lower() for pattern in ['?', '&', '=', '%']):
        penalties += url_param_penalty
    
    return penalties

def final_score(poi_name: str, title: str, snippet: str, url: str, 
                poi_category: str, config: Optional[Dict[str, Any]] = None) -> float:
    """Calculate final score combining authority, boosts, and decay"""
    domain = domain_of(url)
    
    # Calculate components
    authority = calculate_authority(domain, config)
    name_match = fuzzy_score(poi_name, title + " " + snippet)
    geo_score = geo_hint(title, snippet, url)
    cat_score = cat_hint(title, snippet, poi_category)
    penalties = calculate_penalties(domain, url, config)
    
    # Get scoring weights from config with fallbacks
    scoring_config = {}
    if config and 'mention_scanner' in config:
        scoring_config = config['mention_scanner'].get('scoring', {})
    
    name_weight = scoring_config.get('name_weight', 0.55)
    geo_weight = scoring_config.get('geo_weight', 0.25)
    cat_weight = scoring_config.get('cat_weight', 0.05)
    authority_weight = scoring_config.get('authority_weight', 0.15)
    
    # Apply formula
    score = (name_weight * name_match + 
             geo_weight * geo_score + 
             cat_weight * cat_score + 
             authority_weight * authority - 
             penalties)
    
    return max(0.0, min(1.0, score))

def is_acceptable(candidate: Dict[str, Any], thresholds: Dict[str, float], rules: Dict[str, Any]) -> bool:
    """Check if candidate meets acceptance criteria using exact current logic"""
    score = candidate.get('score', 0.0)
    geo_score = candidate.get('geo_score', 0.0)
    authority_weight = candidate.get('authority_weight', 0.0)
    title = candidate.get('title', '')
    url = candidate.get('url', '')
    poi_name = candidate.get('poi_name', '')
    
    # Get thresholds
    high_threshold = thresholds.get('high', 0.82)
    mid_threshold = thresholds.get('mid', 0.33)
    authority_threshold = rules.get('authority_weight_threshold', 0.3)
    min_sources = rules.get('min_distinct_sources', 2)
    
    # Title/URL boost checks
    title_boost = rules.get('title_has_poi_name_boost', False)
    url_boost = rules.get('url_has_poi_name_boost', False)
    
    # Current acceptance logic: score >= high AND geo >= mid
    base_acceptable = score >= high_threshold and geo_score >= mid_threshold
    
    # Additional rules if configured
    if authority_threshold and authority_weight < authority_threshold:
        base_acceptable = False
    
    # Apply boosts if configured
    if title_boost and poi_name.lower() in title.lower():
        base_acceptable = True
    if url_boost and poi_name.lower() in url.lower():
        base_acceptable = True
    
    return base_acceptable

def is_better_candidate(candidate1: Dict[str, Any], candidate2: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
    """Compare two candidates using stable tie-breaker rules"""
    score1 = candidate1.get('score', 0.0)
    score2 = candidate2.get('score', 0.0)
    
    # Get epsilon from config
    epsilon = 0.01
    if config and 'mention_scanner' in config:
        epsilon = config['mention_scanner'].get('scoring', {}).get('tie_break_epsilon', 0.01)
    
    # If score difference is significant (>= epsilon), use score
    if abs(score1 - score2) >= epsilon:
        return score1 > score2
    
    # Scores are close, use tie-breaker: (-score, -authority_weight, domain lexicographic)
    auth1 = candidate1.get('authority_weight', 0.0)
    auth2 = candidate2.get('authority_weight', 0.0)
    
    if abs(auth1 - auth2) >= epsilon:
        return auth1 > auth2
    
    # Both score and authority are close, use domain lexicographic order
    domain1 = candidate1.get('domain', '')
    domain2 = candidate2.get('domain', '')
    return domain1 < domain2  # Lexicographically smaller domain wins
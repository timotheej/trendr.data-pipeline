#!/usr/bin/env python3
"""
Removed unused code: complex weighted systems, legacy matcher_scores, experimental algorithms

KISS Scoring System for GATTO Scanner - Fixed weights: 0.60*name + 0.25*geo + 0.15*authority
Simple penalty system with country/city mismatch detection
"""
import re
import math
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, Tuple, Union, List
import logging

# Try to import domains and city_profiles to avoid circular imports
try:
    from .domains import domain_of
    from .city_profiles import city_manager
except ImportError:
    try:
        from domains import domain_of
        from city_profiles import city_manager
    except ImportError:
        # Fallback - will need to import these functions when used
        pass

logger = logging.getLogger(__name__)


# KISS implementation with simplified functions

def calculate_authority(domain: str, config: Optional[Dict[str, Any]] = None, 
                      db_manager: Optional[Any] = None) -> float:
    """Calculate authority score using source_catalog.authority_weight (database-driven)"""
    # If database manager available, use source_catalog
    if db_manager:
        try:
            source_id = db_manager.get_source_id_from_domain(domain)
            if source_id:
                # Load source_catalog to get authority_weight
                sources = db_manager._load_source_catalog()
                for source in sources:
                    if source.get('source_id') == source_id:
                        authority_weight = source.get('authority_weight', 0.5)
                        return authority_weight
        except Exception:
            pass  # Fallback to config/default
    
    # Legacy fallback: use domain_groups from config  
    domain_groups = {}
    if config and 'mention_scanner' in config:
        domain_groups = config['mention_scanner'].get('domain_groups', {})
    
    # Get apex domain
    apex_domain = domain_of(domain)
    
    # Return weight from domain_groups or fallback to 0.5
    return domain_groups.get(apex_domain, 0.5)


def final_score(poi_name: str, title: str, snippet: str, url: str, 
                poi_category: str, config: Optional[Dict[str, Any]] = None, debug: bool = False,
                city_slug: str = 'paris', poi_coords: Optional[Tuple[float, float]] = None,
                db_manager: Optional[Any] = None, published_at: Optional[str] = None) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """KISS final score: fixed weights 0.60*name + 0.25*geo + 0.15*authority
    
    Returns:
        float: Final score (0.0-1.0), or tuple of (score, explain_dict) if debug=True
    """
    import os
    
    # Check if debug mode is enabled
    debug_enabled = debug or os.getenv('SCAN_DEBUG') == '1'
    
    domain = domain_of(url)
    
    # Calculate 3 components with KISS approach
    name_score = _calculate_name_score_kiss(poi_name, title, snippet)
    geo_score = _calculate_geo_score_kiss(title, snippet, url, city_slug, poi_coords, config, db_manager)
    authority = calculate_authority(domain, config, db_manager)
    
    # Apply FIXED weights (KISS requirement)
    WEIGHTS = {'name': 0.60, 'geo': 0.25, 'authority': 0.15}
    
    weighted_name = WEIGHTS['name'] * name_score
    weighted_geo = WEIGHTS['geo'] * geo_score  
    weighted_authority = WEIGHTS['authority'] * authority
    
    # Base score
    base_score = weighted_name + weighted_geo + weighted_authority
    
    # Apply penalties (country/city mismatch only)
    penalties = _calculate_kiss_penalties(poi_name, title, snippet, url, city_slug, config)
    score_after_penalties = max(0.0, min(1.0, base_score - penalties['total']))
    
    # Apply time decay if enabled and published_at is available
    time_decay_multiplier = 1.0
    time_decay_info = {"enabled": False, "multiplier": 1.0, "age_days": None}
    
    if config and published_at:
        time_decay_config = config.get('mention_scanner', {}).get('time_decay', {})
        if time_decay_config.get('enabled', False):
            decay_multiplier = _calculate_time_decay(published_at, time_decay_config)
            if decay_multiplier is not None:
                time_decay_multiplier = decay_multiplier
                time_decay_info = {
                    "enabled": True, 
                    "multiplier": decay_multiplier,
                    "age_days": _calculate_age_days(published_at)
                }
    
    final_score_value = score_after_penalties * time_decay_multiplier
    
    if debug_enabled:
        explain = {
            'components': {
                'name_match': round(name_score, 3),
                'geo_score': round(geo_score, 3), 
                'authority': round(authority, 3),
                'penalties': penalties
            },
            'weights': WEIGHTS,
            'weighted_components': {
                'name_component': round(weighted_name, 3),
                'geo_component': round(weighted_geo, 3),
                'authority_component': round(weighted_authority, 3),
            },
            'base_score': round(base_score, 3),
            'score_after_penalties': round(score_after_penalties, 3),
            'time_decay': time_decay_info,
            'final_score': round(final_score_value, 3),
            'domain': domain
        }
        return final_score_value, explain
    
    return final_score_value

def make_tabular_decision(final_score: float, explain: Dict[str, Any], candidate: Dict[str, Any], 
                         high_threshold: float = 0.35, mid_threshold: float = 0.20) -> Tuple[str, str, List[str]]:
    """
    KISS tabular decision logic with clear priority order:
    
    1. Auto-accept confirmed domain (no country alert) → accepted_by=confirmed_domain
    2. final_score ≥ 0.35 → ACCEPT (accepted_by=score_high) 
    3. 0.20 ≤ final_score < 0.35 AND (geo≥0.25 OR authority≥0.60) → REVIEW (accepted_by=mid_conditional)
    4. Otherwise → REJECT
    
    Returns: (decision, accepted_by, drop_reasons)
    """
    authority = explain['components']['authority']
    geo_score = explain['components']['geo_score'] 
    penalties = explain['components']['penalties']
    drop_reasons = []
    
    # Rule 1: Auto-accept confirmed domain (if no country mismatch)
    if authority >= 1.0 and penalties.get('country_mismatch', 0) == 0:
        return "ACCEPT", "confirmed_domain", []
    
    # Check hard rejects first
    if penalties.get('country_mismatch', 0) > 0:
        drop_reasons.append("country_mismatch")
        return "REJECT", "", drop_reasons
    
    # Rule 2: High score threshold
    if final_score >= high_threshold:
        return "ACCEPT", "score_high", []
    
    # Rule 3: Mid threshold with conditions
    if (mid_threshold <= final_score < high_threshold and 
        (geo_score >= 0.25 or authority >= 0.60)):
        return "REVIEW", "mid_conditional", []
    
    # Rule 4: Reject with reasons
    if final_score < mid_threshold:
        drop_reasons.append(f"score_too_low:{final_score:.3f}<{mid_threshold}")
    else:
        drop_reasons.append(f"mid_conditions_failed:geo={geo_score:.3f}<0.25 AND auth={authority:.3f}<0.60")
    
    return "REJECT", "", drop_reasons


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


def _calculate_name_score_kiss(poi_name: str, title: str, snippet: str) -> float:
    """KISS name scoring: 2 signals (fuzzy + trigram) with simple stopword normalization"""
    if not poi_name:
        return 0.0
        
    text = f"{title} {snippet}"
    if not text.strip():
        return 0.0
    
    # Two distinct signals
    fuzzy_score = SequenceMatcher(None, poi_name.lower(), text.lower()).ratio()
    trigram_score = _trigram_similarity_kiss(poi_name, text)
    
    # Simple stopword normalization boost
    normalized_text = _remove_stopwords_kiss(text)
    normalized_poi = _remove_stopwords_kiss(poi_name)
    
    if normalized_poi and normalized_text:
        normalized_fuzzy = SequenceMatcher(None, normalized_poi.lower(), normalized_text.lower()).ratio()
        fuzzy_score = max(fuzzy_score, normalized_fuzzy)
    
    return max(fuzzy_score, trigram_score)


def _calculate_geo_score_kiss(title: str, snippet: str, url: str, city_slug: str, poi_coords: Optional[Tuple[float, float]], config: Optional[Dict[str, Any]], db_manager: Optional[Any]) -> float:
    """KISS geo scoring using city_manager"""
    try:
        from .city_profiles import city_manager
        city_profile = city_manager.get_profile(city_slug)
        if not city_profile:
            return 0.0
        
        # Extract geo signals using city manager
        geo_result = city_manager.extract_geo_signals(title, snippet, url, city_profile, poi_coords, config)
        return geo_result['score']
        
    except Exception as e:
        logger.debug(f"Geo scoring failed: {e}")
        return 0.0


def _trigram_similarity_kiss(a: str, b: str) -> float:
    """Simple trigram similarity"""
    if not a or not b:
        return 0.0
        
    def get_trigrams(s):
        s = f"  {s.lower()}  "  # Padding
        return set(s[i:i+3] for i in range(len(s)-2))
    
    tri_a = get_trigrams(a)
    tri_b = get_trigrams(b)
    
    if not tri_a or not tri_b:
        return 0.0
    
    intersection = len(tri_a & tri_b)
    union = len(tri_a | tri_b)
    return intersection / union if union > 0 else 0.0


def _remove_stopwords_kiss(text: str) -> str:
    """Simple stopword removal"""
    stopwords = {
        'le', 'la', 'les', 'du', 'de', 'des', 'un', 'une', 'et', 'ou', 
        'restaurant', 'cafe', 'bar', 'chez', 'aux', 'au', 'paris'
    }
    
    words = text.lower().split()
    filtered = [w for w in words if w not in stopwords and len(w) > 2]
    return ' '.join(filtered)


def _calculate_kiss_penalties(poi_name: str, title: str, snippet: str, url: str, city_slug: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """KISS penalties: country mismatch (hard reject) + city mismatch (soft penalty)"""
    penalties = {
        'country_mismatch': 0.0,
        'city_mismatch': 0.0, 
        'total': 0.0
    }
    
    try:
        from .city_profiles import city_manager
        city_profile = city_manager.get_profile(city_slug)
        if not city_profile:
            return penalties
            
        text = f"{title} {snippet} {url}".lower()
        
        # Country mismatch (hard reject)
        expected_country = city_profile.country_code
        if _detect_wrong_country_kiss(text, expected_country):
            penalties['country_mismatch'] = 1.0  # Hard reject
            penalties['total'] = 1.0
            return penalties
        
        # City mismatch (soft penalty)
        if _detect_wrong_city_kiss(text, city_profile):
            penalties['city_mismatch'] = 0.15  # -0.15 penalty
            penalties['total'] = 0.15
            
    except Exception as e:
        logger.debug(f"Penalty calculation failed: {e}")
    
    return penalties


def _detect_wrong_country_kiss(text: str, expected_country: str) -> bool:
    """KISS country mismatch detection - only flag obvious conflicts"""
    # Only check for explicit country mentions, not TLDs (too restrictive)
    country_indicators = {
        'FR': [r'\benglish\b', r'\bunited states\b', r'\busa\b', r'\bgermany\b', r'\bspain\b', r'\bitaly\b'],
        'CA': [r'\bfrance\b', r'\bgermany\b', r'\bspain\b', r'\bitaly\b'],
        'US': [r'\bfrance\b', r'\bcanada\b', r'\bgermany\b', r'\bspain\b', r'\bitaly\b'],
        'GB': [r'\bfrance\b', r'\bcanada\b', r'\bgermany\b', r'\bspain\b', r'\bitaly\b'],
        'DE': [r'\bfrance\b', r'\bcanada\b', r'\busa\b', r'\bspain\b', r'\bitaly\b'],
        'ES': [r'\bfrance\b', r'\bcanada\b', r'\busa\b', r'\bgermany\b', r'\bitaly\b'],
        'IT': [r'\bfrance\b', r'\bcanada\b', r'\busa\b', r'\bgermany\b', r'\bspain\b']
    }
    
    # Only flag if explicit conflicting country is mentioned
    conflicting_patterns = country_indicators.get(expected_country, [])
    for pattern in conflicting_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            logger.debug(f"Country mismatch detected: pattern '{pattern}' found for expected country {expected_country}")
            return True
    
    return False


def _detect_wrong_city_kiss(text: str, city_profile) -> bool:
    """Simple city mismatch detection"""
    competing_cities = getattr(city_profile, 'competing_cities', [])
    for city in competing_cities:
        if city.lower() in text:
            return True
    return False


def _calculate_time_decay(published_at: str, time_decay_config: Dict[str, Any]) -> Optional[float]:
    """Calculate time decay multiplier based on published_at date
    
    Uses exponential decay: multiplier = exp(-age_days / tau_days)
    Where tau_days is the half-life parameter
    
    Args:
        published_at: ISO date string (e.g., "2024-03-15", "2024-03-15T12:00:00Z")
        time_decay_config: Config dict with tau_days and max_age_days
        
    Returns:
        float: Decay multiplier (0.0-1.0), or None if parsing fails
    """
    try:
        # Parse different date formats
        if 'T' in published_at:
            # ISO with time: "2024-03-15T12:00:00Z" 
            pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        else:
            # Date only: "2024-03-15"
            pub_date = datetime.strptime(published_at, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            
        # Calculate age in days
        now = datetime.now(timezone.utc)
        age_days = (now - pub_date).days
        
        # Check max age limit
        max_age = time_decay_config.get('max_age_days', 365)
        if age_days > max_age:
            return 0.0
            
        # Calculate exponential decay
        tau_days = time_decay_config.get('tau_days', 90)
        multiplier = math.exp(-age_days / tau_days)
        
        return max(0.0, min(1.0, multiplier))
        
    except Exception as e:
        logger.debug(f"Time decay calculation failed for '{published_at}': {e}")
        return None


def _calculate_age_days(published_at: str) -> Optional[int]:
    """Helper to calculate age in days from published_at string"""
    try:
        if 'T' in published_at:
            pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        else:
            pub_date = datetime.strptime(published_at, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            
        now = datetime.now(timezone.utc)
        return (now - pub_date).days
        
    except Exception:
        return None
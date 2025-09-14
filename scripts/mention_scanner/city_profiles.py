#!/usr/bin/env python3
"""
City Profile system for city-agnostic geo scoring
Provides city metadata for geo signal detection without hardcoded city-specific logic
"""
import re
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CityProfile:
    """City metadata for geo-scoring"""
    city_slug: str
    city_names_aliases: List[str]  # ["paris", "parís", "باريس", etc.]
    country_code: str  # "FR", "ES", etc.
    admin_names: List[str]  # ["Île-de-France", "Grand Paris", etc.]
    postal_prefixes: List[str]  # ["75", "750"] for Paris
    bbox: Optional[Tuple[float, float, float, float]]  # (lat_min, lat_max, lng_min, lng_max)
    centroid: Optional[Tuple[float, float]]  # (lat, lng)

class CityProfileManager:
    """Manages city profiles and provides geo scoring utilities"""
    
    def __init__(self):
        self._profiles: Dict[str, CityProfile] = {}
        self._load_default_profiles()
    
    def _load_default_profiles(self):
        """Load default city profiles - in production this could come from DB"""
        
        # Paris profile
        self._profiles['paris'] = CityProfile(
            city_slug='paris',
            city_names_aliases=['paris', 'parís', 'parigi', 'パリ'],
            country_code='FR',
            admin_names=['île-de-france', 'grand paris', 'region parisienne'],
            postal_prefixes=['75', '750'],
            bbox=(48.8155755, 48.9021449, 2.2247418, 2.4697602),
            centroid=(48.8566, 2.3522)
        )
        
        # Lyon profile  
        self._profiles['lyon'] = CityProfile(
            city_slug='lyon',
            city_names_aliases=['lyon', 'lyons', 'lione', 'リヨン'],
            country_code='FR',
            admin_names=['auvergne-rhône-alpes', 'rhône', 'région lyonnaise'],
            postal_prefixes=['69', '690'],
            bbox=(45.7078, 45.8084, 4.7847, 4.9228),
            centroid=(45.7640, 4.8357)
        )
        
        # Marseille profile
        self._profiles['marseille'] = CityProfile(
            city_slug='marseille',
            city_names_aliases=['marseille', 'marseilles', 'marsiglia', 'マルセイユ'],
            country_code='FR',
            admin_names=['provence-alpes-côte d\'azur', 'paca', 'bouches-du-rhône'],
            postal_prefixes=['13', '130'],
            bbox=(43.1691, 43.3559, 5.2683, 5.5952),
            centroid=(43.2965, 5.3698)
        )
        
        logger.info(f"Loaded {len(self._profiles)} city profiles: {list(self._profiles.keys())}")
    
    def get_profile(self, city_slug: str) -> Optional[CityProfile]:
        """Get city profile by slug"""
        return self._profiles.get(city_slug.lower())
    
    def get_search_locale(self, city_slug: str) -> Dict[str, str]:
        """Get Google CSE locale parameters (gl, hl, cr) for a city"""
        profile = self.get_profile(city_slug)
        if not profile:
            return {'gl': 'fr', 'hl': 'fr', 'cr': 'countryFR'}  # Default to France
        
        # Map country codes to Google CSE parameters
        locale_mapping = {
            'FR': {'gl': 'fr', 'hl': 'fr', 'cr': 'countryFR'},  # France
            'CA': {'gl': 'ca', 'hl': 'fr', 'cr': 'countryCA'},  # Canada
            'US': {'gl': 'us', 'hl': 'en', 'cr': 'countryUS'},  # United States
            'GB': {'gl': 'uk', 'hl': 'en', 'cr': 'countryGB'},  # United Kingdom
            'DE': {'gl': 'de', 'hl': 'de', 'cr': 'countryDE'},  # Germany
            'ES': {'gl': 'es', 'hl': 'es', 'cr': 'countryES'},  # Spain
            'IT': {'gl': 'it', 'hl': 'it', 'cr': 'countryIT'},  # Italy
        }
        
        return locale_mapping.get(profile.country_code, {'gl': 'fr', 'hl': 'fr', 'cr': 'countryFR'})
    
    def extract_geo_signals(self, title: str, snippet: str, url: str, 
                          city_profile: CityProfile, poi_coords: Optional[Tuple[float, float]] = None, 
                          config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract geo signals for any city using the city profile
        Returns detailed signal breakdown with sources
        """
        # Get scoring config with fallbacks
        geo_scoring = {}
        if config and config.get('mention_scanner'):
            geo_scoring = config['mention_scanner'].get('geo_scoring', {})
        
        city_score = geo_scoring.get('city_name_score', 0.4)
        postal_score = geo_scoring.get('postal_code_score', 0.3)
        admin_score = geo_scoring.get('admin_region_score', 0.2)
        country_score = geo_scoring.get('country_score', 0.1)
        url_city_score = geo_scoring.get('url_city_segment_score', 0.3)
        distance_full = geo_scoring.get('distance_full_score', 0.3)
        distance_half = geo_scoring.get('distance_half_score', 0.15)
        distance_full_km = geo_scoring.get('distance_full_threshold_km', 3)
        distance_half_km = geo_scoring.get('distance_half_threshold_km', 15)
        
        signals_found = []
        components = {
            'city_name': 0.0,
            'postal_code': 0.0, 
            'admin_region': 0.0,
            'country': 0.0,
            'url_city_segment': 0.0,
            'distance_to_poi': 0.0
        }
        
        # Combine all text for analysis
        all_text = f"{title} {snippet}".lower()
        url_lower = url.lower()
        
        # 1. City name detection
        city_sources = []
        for alias in city_profile.city_names_aliases:
            alias_lower = alias.lower()
            if alias_lower in title.lower():
                city_sources.append('title')
            if alias_lower in snippet.lower():
                city_sources.append('snippet')
        
        if city_sources:
            components['city_name'] = city_score
            unique_sources = list(set(city_sources))
            signals_found.append(f"{city_profile.city_slug}({city_score}) from {','.join(unique_sources)}")
        
        # 2. Postal code detection
        postal_sources = []
        for prefix in city_profile.postal_prefixes:
            # Match postal codes with this prefix (e.g., 75001-75020 for Paris, 69001-69009 for Lyon)
            postal_pattern = re.compile(rf'\b{re.escape(prefix)}\d{{1,3}}\b')
            if postal_pattern.search(title.lower()):
                postal_sources.append('title')
            if postal_pattern.search(snippet.lower()):
                postal_sources.append('snippet')
            if postal_pattern.search(url_lower):
                postal_sources.append('url')
        
        if postal_sources:
            components['postal_code'] = postal_score
            unique_sources = list(set(postal_sources))
            # Find the actual postal code for display
            postal_match = None
            for prefix in city_profile.postal_prefixes:
                postal_pattern = re.compile(rf'\b({re.escape(prefix)}\d{{1,3}})\b')
                match = postal_pattern.search(all_text)
                if match:
                    postal_match = match.group(1)
                    break
            if postal_match:
                signals_found.append(f"postal_{postal_match}({postal_score}) from {','.join(unique_sources)}")
            else:
                signals_found.append(f"postal_{city_profile.postal_prefixes[0]}xxx({postal_score}) from {','.join(unique_sources)}")
        
        # 3. Administrative region detection
        admin_sources = []
        for admin in city_profile.admin_names:
            admin_lower = admin.lower()
            if admin_lower in title.lower():
                admin_sources.append('title')
            if admin_lower in snippet.lower():
                admin_sources.append('snippet')
        
        if admin_sources:
            components['admin_region'] = admin_score
            unique_sources = list(set(admin_sources))
            signals_found.append(f"admin_region({admin_score}) from {','.join(unique_sources)}")
        
        # 4. Country detection
        country_indicators = ['france', 'fr'] if city_profile.country_code == 'FR' else [city_profile.country_code.lower()]
        country_sources = []
        
        for indicator in country_indicators:
            if indicator in title.lower():
                country_sources.append('title')
            if indicator in snippet.lower():
                country_sources.append('snippet')
        
        if country_sources:
            components['country'] = country_score
            unique_sources = list(set(country_sources))
            signals_found.append(f"country_{city_profile.country_code.lower()}({country_score}) from {','.join(unique_sources)}")
        
        # 5. URL city segment detection
        for alias in city_profile.city_names_aliases:
            alias_lower = alias.lower()
            if f'/{alias_lower}/' in url_lower or f'/{alias_lower}-' in url_lower or url_lower.endswith(f'/{alias_lower}'):
                components['url_city_segment'] = url_city_score
                signals_found.append(f"url_{alias_lower}_segment({url_city_score}) from url")
                break
        
        # 6. Distance to POI if coordinates available
        if poi_coords and city_profile.centroid:
            poi_lat, poi_lng = poi_coords
            city_lat, city_lng = city_profile.centroid
            
            # Calculate distance using Haversine formula
            distance_km = self._calculate_distance(poi_lat, poi_lng, city_lat, city_lng)
            
            if distance_km < distance_full_km:
                distance_score = distance_full  # Full score for <3km
            elif distance_km < distance_half_km:
                distance_score = distance_half  # Half score for 3-15km
            else:
                distance_score = 0.0  # No score for >15km
            
            if distance_score > 0:
                components['distance_to_poi'] = distance_score
                signals_found.append(f"distance_{distance_km:.1f}km({distance_score:.1f}) from coords")
        
        # Calculate total score
        total_score = sum(components.values())
        
        return {
            'score': total_score,
            'signals_found': signals_found,
            'components': components,
            'reason': 'geo_signals_detected' if signals_found else 'no_geo_signals_found',
            'city_profile_used': city_profile.city_slug
        }
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

# Global instance
city_manager = CityProfileManager()